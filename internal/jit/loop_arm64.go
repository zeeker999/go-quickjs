//go:build quickjs_jit && !android && !ios && darwin

package jit

import "encoding/binary"

// The entry bridge supplies the state address in R0. R1 and F0-F2 are
// scratch; SP, FP, LR, R18, and Go's R28 are untouched by the kernel.
func loopInstructions() []byte {
	words := [...]uint32{
		0xfd400000, // ldr d0, [x0]
		0xfd400401, // ldr d1, [x0, #8]
		0xfd400802, // ldr d2, [x0, #16]
		0xf9400c01, // ldr x1, [x0, #24]
		0xb40000a1, // cbz x1, store
		0x1e602842, // loop: fadd d2, d2, d0
		0x1e612800, // fadd d0, d0, d1
		0xf1000421, // subs x1, x1, #1
		0x54ffffa1, // b.ne loop
		0xfd000000, // store: str d0, [x0]
		0xfd000802, // str d2, [x0, #16]
		0xd65f03c0, // ret
	}
	code := make([]byte, len(words)*4)
	for i, word := range words {
		binary.LittleEndian.PutUint32(code[i*4:], word)
	}
	return code
}
