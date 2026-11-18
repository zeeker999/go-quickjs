package nativegen

import . "github.com/go-quickjs/go-quickjs/internal/jit/asm/arm64"

// charCodeIndexW is amd64's conversion to the intrinsic kernel's encoding.
func (g *armGen) charCodeIndexW() {
	a := g.a
	a.Ubfx(r8, rW, 16, 16)
	a.AddImm(r8, r8, 2, true)
	a.ShiftImm(Lsl, r8, r8, 8, true)
	a.Ubfx(rW, rW, 32, 16)
	a.ShiftImm(Lsl, rW, rW, 16, true)
	a.Op(Orr, rW, rW, r8, true)
}
