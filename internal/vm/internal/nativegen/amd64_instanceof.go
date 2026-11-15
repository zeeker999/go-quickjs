package nativegen

import (
	. "github.com/go-quickjs/go-quickjs/internal/jit/asm/amd64"
)

// instanceOf is instanceof: R[A] = R[B] instanceof R[C], answered as
// instanceOfAt answers it by ordinaryHasInstance, without running
// anything a script could see: R[C] an object whose Symbol.hasInstance
// the site's cache (Aux[pc]) finds is the realm's (the runtime's
// hasInstanceFn), a function not bound, whose own prototype property is a
// data property holding an object, found in its table; R[B] a primitive,
// false, or an object whose prototypes, walked as proto walks them, are
// no proxies. Anything else exits, before anything is written: Go fills
// the site's cache, makes a prototype not made yet, and throws.
func (g *nativeGen) instanceOf() {
	a := g.a
	scan, next, found, loop, yes, no, store := a.NewLabel(), a.NewLabel(), a.NewLabel(), a.NewLabel(), a.NewLabel(), a.NewLabel(), a.NewLabel()
	g.decC(RDX)
	g.object(RSI, RDX)
	a.Store(RDI, scr(sE), RSI)
	// The site, Aux[pc]: &ic[Aux[pc]].
	a.Load(R8, RDI, ncRC)
	a.Load(R8, R8, offRCAux)
	a.Op(Test, R8, R8, true)
	a.Jcc(CondE, g.exit)
	a.MovRR(RCX, RBX)
	a.Load(R9, RDI, ncCodeBase)
	a.Op(Sub, RCX, R9, true)
	a.ShiftImm(Shr, RCX, 1, true)
	a.Op(Add, R8, RCX, true)
	a.LoadU32(RCX, R8, 0)
	a.MovImm(R8, uint64(size("propCache")))
	a.Imul(RCX, R8)
	a.Op(Add, RCX, R15, true)
	// Symbol.hasInstance, the realm's.
	g.readSlot()
	a.Load(R8, RCX, 0)
	a.CmpMem(R8, RDI, ncObjBits, true)
	a.Jcc(CondNE, g.exit)
	// The realm running now's, the Runtime's first word, at the offset the
	// context gives (ncHasInst), as in's cache is read.
	a.Load(R8, RCX, 8)
	a.Load(R9, RDI, ncRT)
	a.Load(R9, R9, rtRel(0))
	a.AddMem(R9, RDI, ncHasInst)
	a.CmpMem(R8, R9, 0, true)
	a.Jcc(CondNE, g.exit)
	// A function, not bound.
	a.Load(RSI, RDI, scr(sE))
	a.LoadU16(RDX, RSI, offObjFlags)
	a.OpImm(And, RDX, int32(objHasData), false)
	a.Jcc(CondE, g.exit)
	a.Load(RDX, RSI, sizeObj+offExtData)
	a.CmpMem(RDX, RDI, ncFuncT, true)
	a.Jcc(CondNE, g.exit)
	a.Load(R9, RSI, sizeObj+offExtData+8)
	a.LoadU8(RDX, R9, offFDBound)
	a.Op(Test, RDX, RDX, false)
	a.Jcc(CondNE, g.exit)
	// Its own prototype, a data property holding an object: its key
	// found in its table, at most 16 keys in.
	a.LoadU32(RDX, RSI, off("Object.valn"))
	a.OpImm(Cmp, RDX, 16, true)
	a.Jcc(CondA, g.exit)
	a.Load(R9, RSI, offObjShape)
	a.Op(Test, R9, R9, true)
	a.Jcc(CondE, g.exit)
	a.Load(R9, R9, off("shape.keys"))
	a.Op(Xor, RCX, RCX, true)
	a.Bind(scan)
	a.Op(Cmp, RCX, RDX, true)
	a.Jcc(CondAE, g.exit)
	a.LeaIdx(R8, R9, RCX, uint8(size("propKey")), 0)
	a.LoadU32(RAX, R8, off("propKey.key"))
	a.OpImm(Cmp, RAX, atomPrototype, false)
	a.Jcc(CondNE, next)
	a.LoadU8(RAX, R8, off("propKey.flags"))
	a.OpImm(And, RAX, propOddKey, false)
	a.Jcc(CondE, found)
	a.Jmp(g.exit)
	a.Bind(next)
	a.OpImm(Add, RCX, 1, true)
	a.Jmp(scan)
	a.Bind(found)
	a.ShiftImm(Shl, RCX, 4, true)
	a.AddMem(RCX, RSI, off("Object.valp"))
	a.Load(R8, RCX, 0)
	a.CmpMem(R8, RDI, ncObjBits, true)
	a.Jcc(CondNE, g.exit)
	a.Load(RCX, RCX, 8)
	// The value: a primitive is no instance; an object's prototypes are
	// walked, a proxy's trap left to Go.
	a.Load(RAX, RBX, 0)
	g.decB(RDX)
	a.LoadIdx(R9, R12, RDX, 1, 0)
	a.CmpMem(R9, RDI, ncObjBits, true)
	a.Jcc(CondNE, no)
	a.LoadIdx(R8, R12, RDX, 1, 8)
	a.Bind(loop)
	a.LoadU8(RDX, R8, offObjClass)
	a.OpImm(Cmp, RDX, int32(ClassProxy), false)
	a.Jcc(CondE, g.exit)
	poly, have := a.NewLabel(), a.NewLabel()
	a.LoadU16(RDX, R8, offObjFlags)
	a.OpImm(And, RDX, int32(objPolyProto), false)
	a.Jcc(CondNE, poly)
	a.Load(R8, R8, offObjShape)
	a.Op(Test, R8, R8, true)
	a.Jcc(CondE, no)
	a.Load(R8, R8, offShapeProto)
	a.Jmp(have)
	a.Bind(poly)
	a.Load(R8, R8, off("Object.valp"))
	a.Load(R8, R8, 8)
	a.Bind(have)
	a.Op(Test, R8, R8, true)
	a.Jcc(CondE, no)
	a.Op(Cmp, R8, RCX, true)
	a.Jcc(CondE, yes)
	a.Jmp(loop)
	a.Bind(yes)
	a.Load(R9, RDI, ncTrue)
	a.Jmp(store)
	a.Bind(no)
	a.Load(R9, RDI, ncFalse)
	a.Bind(store)
	a.MovQToX(0, R9)
	g.storeA(0)
}
