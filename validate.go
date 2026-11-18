package mp3

import (
	"errors"
	"fmt"
)

// Classified frame validation failures. A *FrameError wrapping one of
// these sentinels is returned from Decoder.Decode when a frame fails
// validation; use errors.Is to classify and errors.As to recover the
// frame index.
var (
	// ErrCRCMismatch indicates that the CRC-16 stored in a protected
	// frame does not match the CRC computed over the protected bytes
	// (the last two header bytes and the side information).
	ErrCRCMismatch = errors.New("frame CRC mismatch")

	// ErrReservoirUnderflow indicates that a frame's main_data_begin
	// points further back than the data made available by the preceding
	// frames (and the 511 byte bit reservoir limit), i.e. the bit
	// reservoir accounting is inconsistent with the frame lengths seen
	// so far.
	ErrReservoirUnderflow = errors.New("bit reservoir underflow")

	// ErrInvalidFrameLength indicates that the frame length implied by
	// the header cannot even hold the header, CRC and side information.
	ErrInvalidFrameLength = errors.New("invalid frame length")
)

// FrameError describes a validation failure on a specific frame of the
// decoded stream.
type FrameError struct {
	// Index is the zero-based index of the offending frame, counting
	// only frames that were successfully located since the decoder was
	// created.
	Index int
	// Err is one of the Err* sentinels above classifying the failure.
	Err error
}

// Error returns a human readable description of the failure, including
// the frame index.
func (e *FrameError) Error() string {
	return fmt.Sprintf("mp3: frame %d: %v", e.Index, e.Err)
}

// Unwrap returns the classification of the failure.
func (e *FrameError) Unwrap() error {
	return e.Err
}

// maxReservoir is the maximum bit reservoir size in bytes. The
// main_data_begin pointer is a 9 bit field, so main data can never
// begin more than 511 bytes before the current frame's main data.
const maxReservoir = 511

// validate runs the frame validation chain for v, which must hold a
// complete frame. Frames that pass update the decoder's bit reservoir
// accounting; frames that fail produce a classified *FrameError.
//
// Only Layer III frames are validated: the side information layout (and
// hence the CRC coverage and the main_data_begin pointer) used here is
// Layer III specific, which matches the rest of this package.
func (d *Decoder) validate(v *Frame) error {
	h := v.Header()
	if h.Layer() != Layer3 {
		return nil
	}

	crcLen := 0
	if h.Protection() {
		crcLen = 2
	}
	sideLen, err := v.SideInfoLength()
	if err != nil {
		return err
	}

	// Frame length consistency: the length derived from the header
	// must be large enough to hold the header, the optional CRC word
	// and the side information.
	overhead := 4 + crcLen + sideLen
	size := v.Size()
	if size < overhead {
		return &FrameError{Index: d.frames, Err: ErrInvalidFrameLength}
	}

	// CRC check: the CRC-16 covers the last two bytes of the header
	// and the whole side information block.
	if h.Protection() {
		stored, err := v.CRC()
		if err != nil {
			return err
		}
		crc := crc16mpeg(crc16Init, v.buf[2:4])
		crc = crc16mpeg(crc, v.buf[4+crcLen:overhead])
		if crc != stored {
			return &FrameError{Index: d.frames, Err: ErrCRCMismatch}
		}
	}

	// Bit reservoir accounting, linked to the frame lengths observed
	// so far: main_data_begin may not reach further back than the main
	// data bytes the preceding frames could have contributed. Without
	// decoding the audio data the exact reservoir level is unknowable,
	// so an optimistic upper bound is tracked: a frame's main data ends
	// no earlier than its main_data_begin pointer, so the next frame
	// can reach back at most main_data_begin + (main data bytes of this
	// frame), accumulated over previous frames and capped at the 511
	// byte reservoir limit. Valid streams — CBR, VBR or otherwise —
	// never fail this check.
	mainData := size - overhead
	ndb := int(v.SideInfo().NDataBegin())
	if d.reservoirValid && ndb > d.reservoir {
		return &FrameError{Index: d.frames, Err: ErrReservoirUnderflow}
	}
	reservoir := ndb + mainData
	if d.reservoirValid && d.reservoir+mainData < reservoir {
		reservoir = d.reservoir + mainData
	}
	if reservoir > maxReservoir {
		reservoir = maxReservoir
	}
	d.reservoir = reservoir
	d.reservoirValid = true

	return nil
}
