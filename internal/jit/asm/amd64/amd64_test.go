package amd64

import (
	"encoding/binary"
	"testing"
)

// An address takes ten bytes whatever its value, so that code's length
// does not depend on where memory is: never MovImm's shorter forms for a
// value of 32 bits.
func TestMovAddrFixedLength(t *testing.T) {
	for _, v := range []uint64{0, 1, 0x7FFF_FFFF, 0xC000_0000_0000, 0xFFFF_FFFF_FFFF_FFFF} {
		for _, r := range []Reg{RAX, R11} {
			var a Asm
			a.MovAddr(r, v)
			if len(a.buf) != 10 || binary.LittleEndian.Uint64(a.buf[2:]) != v || a.buf[1] != 0xB8+byte(r)&7 {
				t.Fatalf("%#x into %d: % x", v, r, a.buf)
			}
		}
	}
}
