package mp3

// MPEG audio frames protect their header and side information with a
// CRC-16 using the polynomial x^16 + x^15 + x^2 + 1 (0x8005), an initial
// value of 0xFFFF, no input/output reflection and no final XOR. This is
// also known as CRC-16/BUYPASS (check value 0xFEE8 for "123456789").
const (
	crc16Poly = 0x8005
	crc16Init = 0xFFFF
)

var crc16tab [256]uint16

func init() {
	for i := 0; i < 256; i++ {
		crc := uint16(i) << 8
		for j := 0; j < 8; j++ {
			if crc&0x8000 != 0 {
				crc = (crc << 1) ^ crc16Poly
			} else {
				crc <<= 1
			}
		}
		crc16tab[i] = crc
	}
}

// crc16mpeg updates crc with the bytes of data.
func crc16mpeg(crc uint16, data []byte) uint16 {
	for _, b := range data {
		crc = crc16tab[byte(crc>>8)^b] ^ (crc << 8)
	}
	return crc
}
