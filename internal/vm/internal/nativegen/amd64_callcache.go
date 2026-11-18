package nativegen

import . "github.com/go-quickjs/go-quickjs/internal/jit/asm/amd64"

// callPrefix reads the ordinary call cache and window, leaving its entry,
// window and callee ready for callHit on a hit, or branching to miss.
func (g *nativeGen) callPrefix(miss, bounds Label) {
	a := g.a
	// e := &calls[C].
	a.MovRR(RCX, RAX)
	a.ShiftImm(Shr, RCX, 48, true)
	a.Load(R9, RDI, ncCC)
	a.CmpMem(RCX, R9, offCCRegCalls+8, true)
	a.Jcc(CondAE, bounds)
	a.MovImm(R8, uint64(sizeE))
	a.Imul(RCX, R8)
	a.AddMem(RCX, R9, offCCRegCalls)
	// The window, its callee the cache's.
	g.decA(RDX)
	a.LeaIdx(RSI, R12, RDX, 1, 0)
	a.Load(R9, RSI, 0)
	a.CmpMem(R9, RDI, ncObjBits, true)
	a.Jcc(CondNE, g.ex(2))
	a.Load(R9, RSI, 8)
	a.CmpMem(R9, RCX, offECallee, true)
	a.Jcc(CondNE, miss)
}
