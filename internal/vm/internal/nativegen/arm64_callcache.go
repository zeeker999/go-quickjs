package nativegen

import . "github.com/go-quickjs/go-quickjs/internal/jit/asm/arm64"

// callPrefix reads the ordinary call cache and window, leaving its entry,
// window and callee ready for callHit on a hit, or branching to miss.
func (g *armGen) callPrefix(miss, bounds Label) {
	a := g.a
	// e := &calls[C].
	a.ShiftImm(Lsr, rC, rW, 48, true)
	a.Load(r9, rCtx, ncCC)
	g.cmpMem(rC, r9, offCCRegCalls+8)
	a.BCond(HS, bounds)
	a.MovImm(r8, uint64(sizeE))
	a.Mul(rC, rC, r8)
	g.addMem(rC, r9, offCCRegCalls)
	// The window, its callee the cache's.
	g.regA(rS)
	a.Load(r9, rS, 0)
	g.cmpMem(r9, rCtx, ncObjBits)
	a.BCond(NE, g.ex(2))
	a.Load(r9, rS, 8)
	g.cmpMem(r9, rC, offECallee)
	a.BCond(NE, miss)
}
