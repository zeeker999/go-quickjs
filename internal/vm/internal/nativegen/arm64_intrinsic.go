package nativegen

import . "github.com/go-quickjs/go-quickjs/internal/jit/asm/arm64"

// strIntrinsic is amd64's (amd64_intrinsic.go): call_method_w's call of
// a String.prototype method its site cached, known by its elemOp and
// answered by its fused form's kernel; no bound function has an elemOp.
// r9 is the funcData and rD its elemOp (push), rS the window, rW the
// instruction.
func (g *armGen) strIntrinsic(other, light Label) {
	a := g.a
	a.CmpImm(rD, int64(elemStrCharCodeAt), false)
	a.BCond(NE, other)
	// The argument a number, the receiver a string, which the kernel
	// takes in rS, not through the scratch.
	a.AddImm(rD, rS, 32, true)
	g.numOr(0, rD, light)
	a.Load(rD, rS, 16)
	g.cmpMem(rD, rCtx, ncStrBits)
	a.BCond(NE, light)
	a.Load(rS, rS, 24)
	g.charCodeUnit(light, g.storeB, true)
}

// charCodeUnit is amd64's: the code unit of the flat string in rS where
// inRS, else in the scratch's sV0, at the position the number in D0
// gives, stored by store, or NaN; anything else goes to miss.
func (g *armGen) charCodeUnit(miss Label, store func(FReg), inRS bool) {
	a := g.a
	nan, wide, have, idx := a.NewLabel(), a.NewLabel(), a.NewLabel(), a.NewLabel()
	a.FCmp(0, 0)
	a.BCond(VC, idx)
	a.FMovToF(0, ZR)
	a.Bind(idx)
	a.Fcvtzs(rC, 0)
	if !inRS {
		a.Load(rS, rCtx, scr(sV0))
	}
	g.cmpMem(rC, rS, offStrLen)
	a.BCond(HS, nan)
	g.cmpMemImm(rS, offStrLeft, 0)
	a.BCond(NE, miss)
	a.LoadU8(rD, rS, offStrASCII)
	a.Cbz(rD, wide, false)
	a.Load(rD, rS, offStrS)
	a.Op(Add, rD, rD, rC, true)
	a.LoadU8(rC, rD, 0)
	a.B(have)
	a.Bind(wide)
	a.Load(rD, rS, offStrAux)
	a.Cbz(rD, miss, true)
	a.AddLsl(rD, rD, rC, 1)
	a.LoadU16(rC, rD, 0)
	a.Bind(have)
	a.Scvtf(0, rC, true)
	store(0)
	a.Bind(nan)
	a.LoadF(0, rCtx, ncNaN)
	store(0)
}
