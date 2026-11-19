package mp3

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"io"
	"runtime"
	"testing"
)

// baselineFillbuf is the original (pre-pooling) fillbuf implementation,
// preserved here as the reference the pooled decoder must match byte for
// byte.
func baselineFillbuf(d []byte, r io.Reader, l int) (res []byte, err error) {
	if len(d) >= l {
		return d, nil
	}

	missing := l - len(d)

	if cap(d) < l {
		if d == nil {
			d = make([]byte, l)
		} else {
			il := len(d)
			d = d[:cap(d)]
			d = append(d, make([]byte, l-cap(d))...)
			d = d[:il]
		}
	}

	d = d[:l]
	_, err = io.ReadFull(r, d[len(d)-missing:])

	return d, err
}

// baselineDecode is the original (pre-pooling) Decode implementation.
func baselineDecode(r io.Reader, v *Frame, skipped *int) (err error) {
	v.buf = v.buf[:0]

	hLen := 4
	*skipped = 0
	for {
		v.buf, err = baselineFillbuf(v.buf, r, hLen)
		if err != nil {
			return err
		}
		if v.buf[0] == 0xFF && (v.buf[1]&0xE0 == 0xE0) &&
			v.Header().Emphasis() != EmphReserved &&
			v.Header().Layer() != LayerReserved &&
			v.Header().Version() != MPEGReserved &&
			v.Header().SampleRate() != -1 &&
			v.Header().BitRate() != -1 {
			break
		}
		switch {
		case v.buf[1] == 0xFF:
			v.buf = v.buf[1:]
			*skipped++
		default:
			v.buf = v.buf[2:]
			*skipped += 2
		}
	}

	crcLen := 0
	if v.Header().Protection() {
		crcLen = 2
		v.buf, err = baselineFillbuf(v.buf, r, hLen+crcLen)
		if err != nil {
			return err
		}
	}

	sideLen, err := v.SideInfoLength()
	if err != nil {
		return err
	}

	v.buf, err = baselineFillbuf(v.buf, r, hLen+crcLen+sideLen)
	if err != nil {
		return err
	}

	dataLen := v.Size()
	v.buf, err = baselineFillbuf(v.buf, r, dataLen)
	if err != nil {
		return err
	}

	return nil
}

// junkStream returns a reader that yields count copies of the silent
// frame, each preceded by a deterministic run of 0-7 junk bytes, to
// exercise the resync/sliding-window paths as well as plain frames.
func junkStream(count int) io.Reader {
	var buf bytes.Buffer
	// deterministic xorshift PRNG, no imports needed
	state := uint32(0x9e3779b9)
	next := func() byte {
		state ^= state << 13
		state ^= state >> 17
		state ^= state << 5
		return byte(state)
	}
	for i := 0; i < count; i++ {
		for n := int(next() & 0x07); n > 0; n-- {
			buf.WriteByte(next())
		}
		buf.Write(SilentBytes)
	}
	return &buf
}

func hashFramesBaseline(t *testing.T, r io.Reader) []byte {
	t.Helper()
	h := sha256.New()
	var f Frame
	skipped := 0
	var sb [8]byte
	for {
		if err := baselineDecode(r, &f, &skipped); err != nil {
			break
		}
		h.Write(f.buf)
		binary.BigEndian.PutUint64(sb[:], uint64(skipped))
		h.Write(sb[:])
	}
	return h.Sum(nil)
}

// hashFramesPooled decodes r with a buffer-limited decoder. When
// freshFrame is set a new Frame is used for every decode and released
// afterwards, exercising the pool's get/put cycle.
func hashFramesPooled(t *testing.T, r io.Reader, limit int64, freshFrame bool) []byte {
	t.Helper()
	d := NewDecoderWithBufferLimit(r, limit)
	h := sha256.New()
	var f Frame
	skipped := 0
	var sb [8]byte
	for {
		if freshFrame {
			f = Frame{}
		}
		if err := d.Decode(&f, &skipped); err != nil {
			break
		}
		h.Write(f.buf)
		binary.BigEndian.PutUint64(sb[:], uint64(skipped))
		h.Write(sb[:])
		if freshFrame {
			f.Release()
		}
	}
	return h.Sum(nil)
}

// TestDecodeHashMatchesBaseline verifies that the pooled, buffer-limited
// decoder produces output byte-identical to the original implementation
// on long streams, across pool limits and frame-reuse strategies.
func TestDecodeHashMatchesBaseline(t *testing.T) {
	streams := map[string]func() io.Reader{
		// ~16 MiB of back-to-back frames, no resyncing
		"silence": func() io.Reader { return io.LimitReader(MakeSilence(), 16<<20) },
		// frames interleaved with junk bytes, exercising resync
		"junk": func() io.Reader { return junkStream(40000) },
	}
	limits := map[string]int64{
		"unpooled-retention": 0,
		"tiny":               3 << 10,
		"default":            DefaultFrameBufferLimit,
	}
	for sname, mk := range streams {
		want := hashFramesBaseline(t, mk())
		for lname, limit := range limits {
			for _, fresh := range []bool{false, true} {
				got := hashFramesPooled(t, mk(), limit, fresh)
				if !bytes.Equal(want, got) {
					t.Errorf("stream %s limit %s freshFrame=%v: hash mismatch with baseline", sname, lname, fresh)
				}
			}
		}
	}
}

func TestBufferPoolReuseByCapacity(t *testing.T) {
	p := NewBufferPool(1 << 20)

	a := p.Get(100)
	if cap(a) != 1<<minBufferClassShift {
		t.Fatalf("Get(100) cap = %d, want %d", cap(a), 1<<minBufferClassShift)
	}
	p.Put(a)
	if got := p.Retained(); got != int64(cap(a)) {
		t.Fatalf("Retained() = %d, want %d", got, cap(a))
	}

	b := p.Get(1000)
	if &a[:1][0] != &b[:1][0] {
		t.Fatal("expected buffer of the same capacity class to be reused")
	}
	if got := p.Retained(); got != 0 {
		t.Fatalf("Retained() after Get = %d, want 0", got)
	}

	// a larger request must use the next capacity class, not the 1 KiB buffer
	p.Put(b)
	c := p.Get(2000)
	if cap(c) != 2<<minBufferClassShift {
		t.Fatalf("Get(2000) cap = %d, want %d", cap(c), 2<<minBufferClassShift)
	}
	if got := p.Retained(); got != 1<<minBufferClassShift {
		t.Fatalf("Retained() = %d, want %d", got, 1<<minBufferClassShift)
	}
}

func TestBufferPoolLimit(t *testing.T) {
	p := NewBufferPool(2048)
	var bufs [][]byte
	for i := 0; i < 10; i++ {
		bufs = append(bufs, p.Get(100)) // 1 KiB class each
	}
	for _, b := range bufs {
		p.Put(b)
	}
	if got := p.Retained(); got != 2048 {
		t.Fatalf("Retained() = %d, want 2048 (limit)", got)
	}
	// the dropped buffers must not be handed out again
	for i := 0; i < 10; i++ {
		p.Get(100)
	}
	if got := p.Retained(); got != 0 {
		t.Fatalf("Retained() after draining = %d, want 0", got)
	}
}

func TestBufferPoolZeroLimit(t *testing.T) {
	p := NewBufferPool(0)
	b := p.Get(100)
	p.Put(b)
	if got := p.Retained(); got != 0 {
		t.Fatalf("Retained() = %d, want 0 with a zero limit", got)
	}
}

// TestDecodeMemoryBounded decodes a long stream frame by frame with a
// small buffer limit and verifies retained pool memory stays within the
// limit and the live heap does not grow with the stream length.
func TestDecodeMemoryBounded(t *testing.T) {
	const limit = 8 << 10

	var ms1, ms2 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&ms1)

	d := NewDecoderWithBufferLimit(io.LimitReader(MakeSilence(), 64<<20), limit)
	skipped := 0
	frames := 0
	for {
		var f Frame
		if err := d.Decode(&f, &skipped); err != nil {
			break
		}
		f.Release()
		frames++
	}
	if frames < 100000 {
		t.Fatalf("decoded only %d frames, stream too short", frames)
	}

	if got := d.pool.Retained(); got > limit {
		t.Fatalf("pool retained %d bytes, limit is %d", got, limit)
	}

	runtime.GC()
	runtime.ReadMemStats(&ms2)
	growth := int64(ms2.HeapAlloc) - int64(ms1.HeapAlloc)
	// A linear-in-duration leak would show ~64 MiB here; allow a few MiB
	// of runtime noise.
	if growth > 4<<20 {
		t.Fatalf("heap grew by %d bytes decoding a 64 MiB stream, want bounded growth", growth)
	}
}

// TestDecodeAllocsBounded verifies steady-state decoding does not
// allocate per frame, whether the caller reuses one Frame or releases
// fresh frames back to the pool.
func TestDecodeAllocsBounded(t *testing.T) {
	skipped := 0

	d := NewDecoderWithBufferLimit(MakeSilence(), 8<<10)
	var f Frame
	d.Decode(&f, &skipped) // warm up
	allocs := testing.AllocsPerRun(1000, func() {
		if err := d.Decode(&f, &skipped); err != nil {
			t.Fatal(err)
		}
	})
	if allocs > 0 {
		t.Errorf("reused-frame decode: %v allocs/op, want 0", allocs)
	}

	d2 := NewDecoderWithBufferLimit(MakeSilence(), 8<<10)
	var warm Frame
	d2.Decode(&warm, &skipped)
	warm.Release()
	allocs = testing.AllocsPerRun(1000, func() {
		var ff Frame
		if err := d2.Decode(&ff, &skipped); err != nil {
			t.Fatal(err)
		}
		ff.Release()
	})
	if allocs > 1 {
		t.Errorf("fresh-frame decode with release: %v allocs/op, want <= 1", allocs)
	}
}

func BenchmarkDecodePooledFreshFrame(b *testing.B) {
	skipped := 0
	b.ReportAllocs()
	d := NewDecoderWithBufferLimit(MakeSilence(), 8<<10)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var f Frame
		d.Decode(&f, &skipped)
		b.SetBytes(int64(len(f.buf)))
		f.Release()
	}
}
