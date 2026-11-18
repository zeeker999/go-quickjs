package nativegen

import . "github.com/go-quickjs/go-quickjs/internal/jit/asm/amd64"

// charCodeIndexW gives the intrinsic kernel its result in A and its
// argument in X. Exits still read the original method-call instruction.
func (g *nativeGen) charCodeIndexW() {
	a := g.a
	a.MovRR(R8, RAX)
	a.ShiftImm(Shr, R8, 16, true)
	a.OpImm(And, R8, 0xffff, true)
	a.OpImm(Add, R8, 2, true)
	a.ShiftImm(Shl, R8, 8, true)
	a.ShiftImm(Shr, RAX, 32, true)
	a.OpImm(And, RAX, 0xffff, true)
	a.ShiftImm(Shl, RAX, 16, true)
	a.Op(Or, RAX, R8, true)
}
