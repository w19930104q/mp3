package mp3

import "sync"

const (
	// minBufferClassShift is the shift of the smallest pooled buffer
	// class (1 << 10 == 1 KiB). Any valid mp3 frame is well under 2 KiB,
	// so even the smallest legitimate frames fit the lowest classes.
	minBufferClassShift = 10
	// maxBufferClassShift is the shift of the largest pooled buffer
	// class (1 << 21 == 2 MiB). Larger buffers are allocated on demand
	// and never retained by the pool.
	maxBufferClassShift = 21

	// DefaultFrameBufferLimit is the default cap, in bytes, on the total
	// size of frame buffers retained for reuse. Decoders created with
	// NewDecoder share a pool bounded by this limit, so the memory held
	// for frame-buffer reuse stays constant no matter how long the
	// decoded stream is.
	DefaultFrameBufferLimit = 4 << 20
)

// BufferPool recycles frame buffers by capacity class. Buffers are
// rounded up to power-of-two capacity classes, so a buffer used for a
// large frame is reused for later frames of the same or smaller class.
//
// The pool retains at most limit bytes of idle buffers; buffers returned
// while the pool is at its limit are dropped for the garbage collector.
// The limit bounds idle (retained) memory only: Get always returns a
// usable buffer, allocating a fresh, unpooled one when necessary.
//
// A BufferPool is safe for concurrent use by multiple decoders.
type BufferPool struct {
	limit int64

	mu       sync.Mutex
	free     map[uint][][]byte
	retained int64
}

// NewBufferPool returns a BufferPool that retains at most limit bytes of
// idle buffers. A limit of 0 disables retention entirely (every Get
// allocates, every Put is dropped). A negative limit is treated as 0.
func NewBufferPool(limit int64) *BufferPool {
	if limit < 0 {
		limit = 0
	}
	return &BufferPool{limit: limit, free: make(map[uint][][]byte)}
}

// bufferClass returns the shift of the smallest power-of-two capacity
// class that can hold n bytes, and whether that class is pooled.
func bufferClass(n int) (uint, bool) {
	for s := uint(minBufferClassShift); s <= maxBufferClassShift; s++ {
		if 1<<s >= n {
			return s, true
		}
	}
	return 0, false
}

// Get returns a buffer with capacity for at least n bytes, reusing a
// retained buffer of the matching capacity class when one is available.
// The returned slice's length equals its capacity; callers should
// reslice it as needed. The contents of the buffer are unspecified.
func (p *BufferPool) Get(n int) []byte {
	s, ok := bufferClass(n)
	if !ok {
		return make([]byte, n)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if q := p.free[s]; len(q) > 0 {
		b := q[len(q)-1]
		p.free[s] = q[:len(q)-1]
		p.retained -= int64(cap(b))
		return b
	}
	return make([]byte, 1<<s)
}

// Put returns b to the pool for reuse by future Get calls. b must have
// been obtained from Get and must not be referenced after Put returns.
// If retaining b would exceed the pool's limit, or b is not a pooled
// capacity class, b is dropped and left to the garbage collector.
func (p *BufferPool) Put(b []byte) {
	if b == nil {
		return
	}
	s, ok := bufferClass(cap(b))
	if !ok || cap(b) != 1<<s {
		// Not a buffer this pool hands out; do not retain it.
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.retained+int64(cap(b)) > p.limit {
		return
	}
	p.free[s] = append(p.free[s], b[:cap(b)])
	p.retained += int64(cap(b))
}

// Retained reports the total number of bytes currently held idle by the
// pool. It never exceeds the limit the pool was created with.
func (p *BufferPool) Retained() int64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.retained
}
