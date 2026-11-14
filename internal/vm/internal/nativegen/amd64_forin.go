package nativegen

import (
	. "github.com/go-quickjs/go-quickjs/internal/jit/asm/amd64"
)

// A for-in's step and the read of an element by the key it hands out,
// natively (docs/register-interpreter.md, "Exceptions": for-of and for-in
// loops), as iterNext steps a for-in's cursor and stringKeyRead reads.

// forInStep is iter_next's step of a for-in's cursor, the cursor object in
// RSI, the instruction in RAX: an array's index key, walked by index
// (forInState.indexKeys), whose element is still in its dense storage
// and whose string the runtime keeps (intStrings), handed out as keyString
// hands it out -- remembered in keyAtoms, where stringKeyRead finds it --
// or the snapshot's next key while its object keeps the shape the keys
// came from; past both, the cursor done and the loop left. Anything else
// -- an element gone, a key to look up again, an index of no string yet
// -- exits before anything is written.
func (g *nativeGen) forInStep() {
	a := g.a
	keys, same, store, end := a.NewLabel(), a.NewLabel(), a.NewLabel(), a.NewLabel()
	a.Load(RCX, RSI, offItIdx)
	a.CmpMem(RCX, RSI, offFiIndexKeys, true)
	a.Jcc(CondGE, keys)
	// An index key: the element there in the dense storage, no hole.
	a.Load(R8, RSI, offFiObj)
	a.MovRR(R9, RCX)
	g.dense(R9, R8, 0, g.exit)
	// Its string, made already.
	a.MovImm(RDX, intStringsLen)
	a.Op(Cmp, RCX, RDX, true)
	a.Jcc(CondAE, g.exit)
	a.Load(R9, RDI, ncRT)
	a.Load(R9, R9, offRtIntStrs)
	a.Op(Test, R9, R9, true)
	a.Jcc(CondE, g.exit)
	a.LoadIdx(R9, R9, RCX, 8, 0)
	a.Op(Test, R9, R9, true)
	a.Jcc(CondE, g.exit)
	// keyAtoms[keySlot(s)] = {s, the index's atom}, where it is not.
	a.MovImm(RDX, keySlotMul)
	a.MovRR(R8, R9)
	a.Imul(R8, RDX)
	a.ShiftImm(Shr, R8, 56, true)
	a.MovImm(RDX, uint64(sizeKA))
	a.Imul(R8, RDX)
	a.Load(RDX, RDI, ncRT)
	a.LeaIdx(R8, RDX, R8, 1, offRtKeyAtoms)
	a.MovImm(RDX, uint64(uint32(atomIndexTag)))
	a.Op(Or, RDX, RCX, true)
	a.CmpMem(R9, R8, offKAS, true)
	a.Jcc(CondNE, store)
	a.LoadU32(RCX, R8, offKAAtom)
	a.Op(Cmp, RCX, RDX, false)
	a.Jcc(CondNE, store)
	a.LoadU8(RCX, R8, offKANone)
	a.Op(Test, RCX, RCX, false)
	a.Jcc(CondE, same)
	a.Bind(store)
	// Nothing is written before here.
	a.Store32(R8, offKAAtom, RDX)
	a.Op(Xor, RCX, RCX, false)
	a.Store8(R8, offKANone, RCX)
	a.Store(RDI, scr(sE), RSI)
	a.Store(RDI, scr(sI), R9)
	a.Lea(RAX, R8, offKAS)
	a.MovRR(RCX, R9)
	g.ptrStore()
	a.Load(RSI, RDI, scr(sE))
	a.Load(R9, RDI, scr(sI))
	a.Load(RAX, RBX, 0)
	a.Bind(same)
	// The cursor stepped; R[A] the string.
	a.IncMem(RSI, offItIdx)
	a.MovRR(R8, R9)
	a.Load(R9, RDI, ncStrBits)
	g.decA(RDX)
	a.LeaIdx(RAX, R12, RDX, 1, 0)
	g.storeValue()

	// The snapshot's next key, while its object keeps the shape the keys
	// came from, where they are its own: nothing deleted from it.
	a.Bind(keys)
	a.CmpMem(RCX, RSI, offFiKeys+8, true)
	a.Jcc(CondGE, end)
	have := a.NewLabel()
	a.Load(R8, RSI, offFiObj)
	a.Op(Test, R8, R8, true)
	a.Jcc(CondE, have)
	a.Load(R9, RSI, offFiShape)
	a.Op(Test, R9, R9, true)
	a.Jcc(CondE, g.exit)
	a.CmpMem(R9, R8, offObjShape, true)
	a.Jcc(CondNE, g.exit)
	a.Bind(have)
	a.ShiftImm(Shl, RCX, 4, true)
	a.AddMem(RCX, RSI, offFiKeys)
	a.IncMem(RSI, offItIdx)
	g.loadValue(RCX)
	g.decA(RDX)
	a.LeaIdx(RAX, R12, RDX, 1, 0)
	g.storeValue()

	// Past the end: the cursor done and the loop left.
	a.Bind(end)
	a.MovImm(RCX, 1)
	a.Store8(RSI, offItDone, RCX)
	g.immSC(RCX)
	g.taken(RCX)
}

// keyIndex puts in RCX the index that the string whose words are R9 (num)
// and at the window's offset off+8 names, where the runtime handed it
// out as an index's key (keyAtoms), as stringKeyRead finds it; anything
// else exits. RDX and R8 are its scratch.
func (g *nativeGen) keyIndex(off Reg) {
	a := g.a
	a.MovRR(RCX, R9)
	a.Op(Xor, RCX, R10, true)
	a.ShiftImm(Shr, RCX, 16, true)
	a.Jcc(CondNE, g.exit)
	a.MovZX8(RCX, R9)
	a.OpImm(Cmp, RCX, int32(KindString), false)
	a.Jcc(CondNE, g.exit)
	a.LoadIdx(R9, R12, off, 1, 8)
	a.MovImm(RDX, keySlotMul)
	a.MovRR(R8, R9)
	a.Imul(R8, RDX)
	a.ShiftImm(Shr, R8, 56, true)
	a.MovImm(RDX, uint64(sizeKA))
	a.Imul(R8, RDX)
	a.Load(RDX, RDI, ncRT)
	a.LeaIdx(R8, RDX, R8, 1, offRtKeyAtoms)
	a.CmpMem(R9, R8, offKAS, true)
	a.Jcc(CondNE, g.exit)
	a.LoadU8(RCX, R8, offKANone)
	a.Op(Test, RCX, RCX, false)
	a.Jcc(CondNE, g.exit)
	a.LoadU32(RCX, R8, offKAAtom)
	a.MovImm(RDX, uint64(uint32(atomIndexTag)))
	a.Op(Test, RCX, RDX, false)
	a.Jcc(CondE, g.exit)
	a.Op(Xor, RCX, RDX, false)
}
