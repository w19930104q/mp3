package mp3

import (
	"bytes"
	"errors"
	"io"
	"os"
	"testing"
)

// TestCRC16CheckValue verifies the CRC-16 implementation (polynomial
// 0x8005, initial value 0xFFFF, as used by MPEG audio) against its check
// value, confirmed with an independent bitwise implementation.
func TestCRC16CheckValue(t *testing.T) {
	if got := crc16mpeg(crc16Init, []byte("123456789")); got != 0xAEE7 {
		t.Fatalf("crc16mpeg check value = %#04x, want 0xAEE7", got)
	}
}

// TestNDataBegin verifies the 9 bit main_data_begin decoding.
func TestNDataBegin(t *testing.T) {
	for _, tc := range []struct {
		b0, b1 byte
		want   uint16
	}{
		{0x00, 0x00, 0},
		{0x00, 0x80, 1},
		{0x01, 0x00, 2},
		{0x01, 0x80, 3},
		{0xFF, 0x80, 511},
	} {
		if got := (FrameSideInfo{tc.b0, tc.b1}).NDataBegin(); got != tc.want {
			t.Errorf("NDataBegin(%#02x, %#02x) = %d, want %d", tc.b0, tc.b1, got, tc.want)
		}
	}
}

// frameSpec describes a synthetic MPEG-1 Layer III frame.
type frameSpec struct {
	bitrateIdx int    // bitrate index 1..14 (MPEG-1 Layer III)
	pad        bool   // padding bit
	protection bool   // prepend a valid CRC-16
	ndb        uint16 // main_data_begin
}

// buildFrame constructs a complete, well formed frame. The main data is
// zero filled; the CRC, when requested, is computed over the protected
// bytes so the frame passes validation.
func buildFrame(spec frameSpec) []byte {
	h := []byte{0xFF, 0xFA, byte(spec.bitrateIdx << 4), 0x00}
	if !spec.protection {
		h[1] |= 0x01
	}
	if spec.pad {
		h[2] |= 0x02
	}

	size := int(float64(1152)/8*float64(bitrates[MPEG1][Layer3][spec.bitrateIdx]*1000)) / 44100
	if spec.pad {
		size++
	}

	side := make([]byte, 32) // MPEG-1 stereo side information
	side[0] = byte(spec.ndb >> 1)
	side[1] = byte(spec.ndb&1) << 7

	frame := make([]byte, 0, size)
	frame = append(frame, h...)
	if spec.protection {
		crc := crc16mpeg(crc16Init, h[2:4])
		crc = crc16mpeg(crc, side)
		frame = append(frame, byte(crc>>8), byte(crc))
	}
	frame = append(frame, side...)
	return frame[:size]
}

func joinFrames(frames ...[]byte) []byte {
	return bytes.Join(frames, nil)
}

// decodeAll decodes every frame in r, returning the frame sizes and the
// terminating error (io.EOF on a clean stream).
func decodeAll(r io.Reader) ([]int, error) {
	d := NewDecoder(r)
	var f Frame
	var sizes []int
	skipped := 0
	for {
		if err := d.Decode(&f, &skipped); err != nil {
			return sizes, err
		}
		sizes = append(sizes, f.Size())
	}
}

// TestProtectedVBRStreamOK feeds a CRC protected stream with varying bit
// rates (a synthetic VBR stream) and expects clean decoding.
func TestProtectedVBRStreamOK(t *testing.T) {
	var frames [][]byte
	for _, idx := range []int{4, 9, 6, 12, 8, 14, 3} {
		frames = append(frames, buildFrame(frameSpec{bitrateIdx: idx, protection: true}))
	}
	sizes, err := decodeAll(bytes.NewReader(joinFrames(frames...)))
	if err != io.EOF {
		t.Fatalf("decode error: %v", err)
	}
	if len(sizes) != len(frames) {
		t.Fatalf("decoded %d frames, want %d", len(sizes), len(frames))
	}
	for i, s := range sizes {
		if s != len(frames[i]) {
			t.Errorf("frame %d: size %d, want %d", i, s, len(frames[i]))
		}
	}
}

// TestCRCMismatchClassified corrupts the stored CRC word of the third
// frame and expects a classified error carrying the frame index.
func TestCRCMismatchClassified(t *testing.T) {
	var frames [][]byte
	for i := 0; i < 5; i++ {
		frames = append(frames, buildFrame(frameSpec{bitrateIdx: 8, protection: true}))
	}
	frames[2][4] ^= 0x01 // flip a bit in the stored CRC word

	_, err := decodeAll(bytes.NewReader(joinFrames(frames...)))
	assertFrameError(t, err, ErrCRCMismatch, 2)
}

// TestCRCMismatchSideInfo corrupts a side information byte (not the CRC
// word itself) and expects the corruption to be detected and classified.
func TestCRCMismatchSideInfo(t *testing.T) {
	var frames [][]byte
	for i := 0; i < 4; i++ {
		frames = append(frames, buildFrame(frameSpec{bitrateIdx: 8, protection: true}))
	}
	frames[1][6] ^= 0xFF // first side info byte of frame 1

	_, err := decodeAll(bytes.NewReader(joinFrames(frames...)))
	assertFrameError(t, err, ErrCRCMismatch, 1)
}

// TestReservoirUnderflowClassified builds a stream whose second frame
// claims a main_data_begin beyond the data the first frame could have
// contributed to the bit reservoir.
func TestReservoirUnderflowClassified(t *testing.T) {
	// 32 kb/s frames hold 104 bytes, i.e. 68 bytes of main data.
	f0 := buildFrame(frameSpec{bitrateIdx: 1, ndb: 0})
	f1 := buildFrame(frameSpec{bitrateIdx: 1, ndb: 100})

	_, err := decodeAll(bytes.NewReader(joinFrames(f0, f1)))
	assertFrameError(t, err, ErrReservoirUnderflow, 1)
}

// TestReservoirChainedOK chains frames that legitimately reference the
// reservoir of their predecessors, including a 320 kb/s frame whose main
// data alone exceeds the 511 byte reservoir limit.
func TestReservoirChainedOK(t *testing.T) {
	frames := [][]byte{
		buildFrame(frameSpec{bitrateIdx: 1, ndb: 0}),   // reservoir: 68
		buildFrame(frameSpec{bitrateIdx: 1, ndb: 68}),  // drains and refills 68
		buildFrame(frameSpec{bitrateIdx: 14, ndb: 68}), // reservoir capped at 511
		buildFrame(frameSpec{bitrateIdx: 1, ndb: 511}), // valid: uses full reservoir
	}
	sizes, err := decodeAll(bytes.NewReader(joinFrames(frames...)))
	if err != io.EOF {
		t.Fatalf("decode error: %v", err)
	}
	if len(sizes) != len(frames) {
		t.Fatalf("decoded %d frames, want %d", len(sizes), len(frames))
	}
}

// TestReservoirResetOnResync ensures that skipped bytes (a loss of sync)
// reset the reservoir accounting instead of producing false failures.
func TestReservoirResetOnResync(t *testing.T) {
	garbage := []byte{0x00, 0x11, 0x22, 0x33}
	f0 := buildFrame(frameSpec{bitrateIdx: 1, ndb: 100}) // unverifiable after resync
	f1 := buildFrame(frameSpec{bitrateIdx: 1, ndb: 0})

	sizes, err := decodeAll(bytes.NewReader(joinFrames(garbage, f0, f1)))
	if err != io.EOF {
		t.Fatalf("decode error: %v", err)
	}
	if len(sizes) != 2 {
		t.Fatalf("decoded %d frames, want 2", len(sizes))
	}
}

// TestRealVBRFile decodes the VBR sample produced by LAME (via ffmpeg)
// and expects clean decoding with varying frame sizes.
func TestRealVBRFile(t *testing.T) {
	f, err := os.Open("testdata/vbr.mp3")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	sizes, err := decodeAll(f)
	if err != io.EOF && err != io.ErrUnexpectedEOF {
		t.Fatalf("decode error: %v", err)
	}
	if len(sizes) < 100 {
		t.Fatalf("decoded %d frames, want at least 100", len(sizes))
	}
	distinct := map[int]bool{}
	for _, s := range sizes {
		distinct[s] = true
	}
	if len(distinct) < 2 {
		t.Fatalf("expected varying frame sizes in VBR stream, got %v", distinct)
	}
}

// TestSilentStreamUnchanged ensures the validation chain does not alter
// decoding of a normal unprotected stream.
func TestSilentStreamUnchanged(t *testing.T) {
	r := MakeSilence()
	d := NewDecoder(r)
	f := Frame{}
	skipped := 0
	want := SilentFrame.Size()
	for i := 0; i < 200; i++ {
		if err := d.Decode(&f, &skipped); err != nil {
			t.Fatalf("frame %d: %v", i, err)
		}
		if skipped != 0 {
			t.Fatalf("frame %d: skipped %d bytes", i, skipped)
		}
		if f.Size() != want {
			t.Fatalf("frame %d: size %d, want %d", i, f.Size(), want)
		}
	}
}

func assertFrameError(t *testing.T, err error, kind error, index int) {
	t.Helper()
	if err == nil {
		t.Fatalf("expected %v at frame %d, got no error", kind, index)
	}
	if !errors.Is(err, kind) {
		t.Fatalf("error %v does not classify as %v", err, kind)
	}
	var fe *FrameError
	if !errors.As(err, &fe) {
		t.Fatalf("error %v is not a *FrameError", err)
	}
	if fe.Index != index {
		t.Fatalf("frame index = %d, want %d", fe.Index, index)
	}
}
