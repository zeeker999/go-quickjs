package mir

import (
	"fmt"
	"math"
	"slices"

	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
	"github.com/go-quickjs/go-quickjs/internal/jit/asm/arm64"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
	"github.com/go-quickjs/go-quickjs/internal/jit/ssa"
)

// arm64's fixed registers. SP, X18 (the platform's), X28 (Go's g), X29 (the
// frame pointer) and X30 (the link register) are never touched; X27 is the
// encoder's temporary.
const (
	a64Ctx    = arm64.Reg(0)
	a64Locals = arm64.Reg(1)
	a64Stack  = arm64.Reg(2)
	a64A      = arm64.Reg(15)
	a64B      = arm64.Reg(16)
	a64C      = arm64.Reg(17)
	a64D      = arm64.Reg(26)
	a64F0     = arm64.FReg(0)
	a64F1     = arm64.FReg(1)
	a64F2     = arm64.FReg(2)
)

// a64Compiler is arm64's code generator over the shared core. It follows
// amd64's (amd64.go) step for step; where the two differ, the comments say
// why.
type a64Compiler struct {
	*core
	a       *arm64.Asm
	labels  []arm64.Label // by block ID
	stubs   map[stubKey]arm64.Label
	stubFor []a64Stub
	cold    []func()
	// tail is where exits that filled records go (recordsTail), if any
	// does; recorded marks the exit being emitted as one.
	tail               arm64.Label
	tailUsed, recorded bool
}

type a64Stub struct {
	key   stubKey
	label arm64.Label
}

// CompileARM64 compiles f for arm64, given the VM's value encoding.
func CompileARM64(f *ssa.Func, enc abi.Encoding) (*Code, error) { return compileARM64(nil, f, enc) }

func compileARM64(w *Workspace, f *ssa.Func, enc abi.Encoding) (code *Code, err error) {
	defer func() {
		if v := recover(); v != nil {
			code, err = nil, fmt.Errorf("%w: %v", ErrUnsupported, v)
		}
	}()
	var gprs, fprs []int
	for r := 3; r <= 14; r++ {
		gprs = append(gprs, r)
	}
	for r := 19; r <= 25; r++ {
		gprs = append(gprs, r)
	}
	for r := 3; r <= 31; r++ {
		fprs = append(fprs, r)
	}
	k, err := prepare(w, f, enc, gprs, fprs)
	if err != nil {
		return nil, err
	}
	c := &a64Compiler{core: k}
	if w != nil {
		g := &w.arm64
		if g.stubs == nil {
			g.stubs = map[stubKey]arm64.Label{}
		}
		g.a.Reset()
		clear(g.stubs)
		c.a, c.stubs, c.stubFor, c.cold = &g.a, g.stubs, g.stubFor[:0], g.cold[:0]
		c.labels = slices.Grow(g.labels[:0], numBlocks(f))[:numBlocks(f)]
		defer func() { g.labels, g.stubFor, g.cold = c.labels[:0], c.stubFor[:0], clearFuncs(c.cold) }()
	} else {
		c.a, c.stubs = &arm64.Asm{}, map[stubKey]arm64.Label{}
		c.labels = make([]arm64.Label, numBlocks(f))
	}
	entries := map[int]int{}
	for _, b := range c.order {
		c.labels[b.ID] = c.a.NewLabel()
	}
	for i, b := range c.order {
		c.a.Bind(c.labels[b.ID])
		if b.PC < 0 {
			for _, e := range f.Entries {
				if e.Block == b {
					entries[e.PC] = c.a.Len()
				}
			}
		}
		var next *ssa.Block
		if i+1 < len(c.order) {
			next = c.order[i+1]
		}
		c.block(b, next)
	}
	for i := 0; i < len(c.stubFor); i++ {
		s := c.stubFor[i]
		c.a.Bind(s.label)
		c.exitTo(s.key.state, s.key.kind)
	}
	for i := 0; i < len(c.cold); i++ {
		c.cold[i]()
	}
	// Last: cold code's exits may come here too.
	if c.tailUsed {
		c.a.Bind(c.tail)
		c.recordsTail()
	}
	bytes, err := c.a.Finish()
	if err != nil {
		return nil, err
	}
	return &Code{Bytes: bytes, Entries: entries, core: c.core}, nil
}

// slotAddr is the base register and displacement of a frame slot's word.
func (c *a64Compiler) slotAddr(slot int, ref bool, scratch arm64.Reg) (arm64.Reg, int32) {
	off := c.enc.NumOffset
	if ref {
		off = c.enc.RefOffset
	}
	switch {
	case slot < c.f.FrameLocals:
		return a64Locals, int32(slot)*c.enc.ValueSize + off
	case slot == c.f.ThisSlot:
		return a64Ctx, abi.OffThis + off
	case slot < c.f.Locals:
		c.a.Load(scratch, a64Ctx, abi.OffUpvalues)
		c.a.Load(scratch, scratch, int32(slot-c.f.FrameLocals)*8)
		c.a.Load(scratch, scratch, c.enc.UpvalueSlot)
		return scratch, off
	}
	return a64Stack, int32(slot-c.f.Locals)*c.enc.ValueSize + off
}

// gpr returns a register holding v's word, loading a spilled or lazy value
// into scratch.
func (c *a64Compiler) gpr(v *ssa.Value, scratch arm64.Reg) arm64.Reg {
	if c.isLazy(v) {
		c.materialize(v, scratch)
		return scratch
	}
	l, ok := c.loc(v)
	if !ok {
		panic(fmt.Sprintf("no location for %v (%v)", v, v.Op))
	}
	if l.reg >= 0 {
		return arm64.Reg(l.reg)
	}
	c.a.Load(scratch, a64Ctx, c.spillDisp(l.spill))
	return scratch
}

// fpr returns a floating-point register holding v.
func (c *a64Compiler) fpr(v *ssa.Value, scratch arm64.FReg) arm64.FReg {
	l, ok := c.loc(v)
	if !ok {
		panic(fmt.Sprintf("no location for %v (%v)", v, v.Op))
	}
	if l.reg >= 0 {
		return arm64.FReg(l.reg)
	}
	c.a.LoadF(scratch, a64Ctx, c.spillDisp(l.spill))
	return scratch
}

// gdst and fdst are the register an instruction computing v writes: v's
// own, or scratch A or F0 when v is spilled, which setG and setF then
// store. A64's instructions read their operands before they write, so the
// register may be an operand's.
func (c *a64Compiler) gdst(v *ssa.Value) arm64.Reg {
	if l := c.locAt(v); l.reg >= 0 {
		return arm64.Reg(l.reg)
	}
	return a64A
}

func (c *a64Compiler) fdst(v *ssa.Value) arm64.FReg {
	if l := c.locAt(v); l.reg >= 0 {
		return arm64.FReg(l.reg)
	}
	return a64F0
}

// constF64 puts a double's bits in dst: from the zero register, as an
// FMOV immediate, or through A.
func (c *a64Compiler) constF64(dst arm64.FReg, bits uint64) {
	if bits == 0 {
		c.a.FMovToF(dst, arm64.ZR)
		return
	}
	if imm, ok := arm64.FloatImm(bits); ok {
		c.a.FMovImm(dst, imm)
		return
	}
	c.a.MovImm(a64A, bits)
	c.a.FMovToF(dst, a64A)
}

// setG stores src into v's location.
func (c *a64Compiler) setG(v *ssa.Value, src arm64.Reg) {
	l := c.locAt(v)
	if l.reg >= 0 {
		if arm64.Reg(l.reg) != src {
			c.a.MovRR(arm64.Reg(l.reg), src)
		}
		return
	}
	c.a.Store(a64Ctx, c.spillDisp(l.spill), src)
}

// setF stores src into v's location.
func (c *a64Compiler) setF(v *ssa.Value, src arm64.FReg) {
	l := c.locAt(v)
	if l.reg >= 0 {
		if arm64.FReg(l.reg) != src {
			c.a.FMov(arm64.FReg(l.reg), src)
		}
		return
	}
	c.a.StoreF(a64Ctx, c.spillDisp(l.spill), src)
}

// materialize computes a lazy value's word into dst.
func (c *a64Compiler) materialize(v *ssa.Value, dst arm64.Reg) {
	switch v.Op {
	case ssa.OpConst:
		c.a.MovImm(dst, c.constWord(v.Const))
	case ssa.OpConstSource:
		c.a.MovImm(dst, uint64(int64(v.Aux)))
	case ssa.OpBoxF64:
		c.boxF64(c.fpr(v.Args[0], a64F1), dst)
	case ssa.OpBoxBool:
		c.boxBool(c.gpr(v.Args[0], a64B), dst)
	default:
		panic("materialize " + v.Op.String())
	}
}

// boxF64 puts x's bits in dst, with NaN canonical.
func (c *a64Compiler) boxF64(x arm64.FReg, dst arm64.Reg) {
	done := c.a.NewLabel()
	c.a.FMovFromF(dst, x)
	c.a.FCmp(x, x)
	c.a.BCond(arm64.VC, done)
	c.a.MovImm(dst, c.enc.CanonicalNaN)
	c.a.Bind(done)
}

// boxBool puts the word for b (0 or 1) in dst. dst may be b.
func (c *a64Compiler) boxBool(b, dst arm64.Reg) {
	tr, done := c.a.NewLabel(), c.a.NewLabel()
	c.a.Cbnz(b, tr, false)
	c.a.MovImm(dst, c.enc.False)
	c.a.B(done)
	c.a.Bind(tr)
	c.a.MovImm(dst, c.enc.True)
	c.a.Bind(done)
}

// stubLabel returns the exit stub for a state and kind.
func (c *a64Compiler) stubLabel(s *ssa.FrameState, kind uint64) arm64.Label {
	k := stubKey{s, kind}
	if l, ok := c.stubs[k]; ok {
		return l
	}
	l := c.a.NewLabel()
	c.stubs[k] = l
	c.stubFor = append(c.stubFor, a64Stub{k, l})
	return l
}

// exitTo writes a frame state into the frame and returns with an exit
// record, as amd64's does.
func (c *a64Compiler) exitTo(s *ssa.FrameState, kind uint64) {
	c.exitThen(s, kind, nil)
}

// exitThen is exitTo, going on at then, if it is not nil, instead of
// returning to Go, as amd64's does.
func (c *a64Compiler) exitThen(s *ssa.FrameState, kind uint64, then *arm64.Label) {
	c.recorded = false
	c.a.Store(a64Ctx, abi.OffRecords, arm64.ZR)
	for i, v := range s.Slots {
		if v == nil || v.Op == ssa.OpLoadSlot && v.Aux == i || c.captured(i) {
			continue
		}
		var w arm64.Reg
		if remat(v) {
			c.materialize(v, a64A)
			w = a64A
		} else {
			w = c.gpr(v, a64A)
		}
		next := c.a.NewLabel()
		if v.Shadow != nil {
			// As amd64's: a record unless the value is a primitive or this
			// slot's reference.
			scalar := c.a.NewLabel()
			from := c.gpr(v.Shadow, a64B)
			c.a.CmpImm(from, -1, true)
			c.a.BCond(arm64.EQ, scalar)
			c.a.CmpImm(from, int64(i), true)
			c.a.BCond(arm64.EQ, next)
			c.appendRecord(uint64(i)|abi.RecordMaybe, c.gprAfter(v.Shadow), 0, false, &w)
			c.a.B(next)
			c.a.Bind(scalar)
		} else if o, ok := c.origin.Of(v); ok && o >= 0 {
			scalar := c.a.NewLabel()
			c.isReference(v, w, o, scalar)
			if o != i {
				c.appendRecord(uint64(i), nil, uint64(o), true, nil)
			}
			c.a.B(next)
			c.a.Bind(scalar)
		}
		record := c.a.NewLabel()
		base, disp := c.slotAddr(i, true, a64C)
		c.a.Load(a64C, base, disp)
		c.a.Cbnz(a64C, record, true)
		base, disp = c.slotAddr(i, false, a64C)
		c.a.Store(base, disp, w)
		c.a.B(next)
		c.a.Bind(record)
		c.appendRecord(uint64(i)|abi.RecordScalar, nil, 0, true, &w)
		c.a.Bind(next)
	}
	if in := s.Inline; in != nil {
		// As amd64's: the callee's frame is said in the next context.
		c.a.AddImm(a64B, a64Ctx, int64(abi.ContextSize), true)
		for _, f := range []struct {
			off int32
			v   uint64
		}{{abi.OffInlineClosure, uint64(in.Closure)}, {abi.OffInlineLocals, uint64(in.Locals)}, {abi.OffInlineThis, uint64(in.ThisSlot + 1)},
			{abi.OffExitKind, kind}, {abi.OffExitPC, uint64(in.PC)}, {abi.OffExitDepth, uint64(in.Depth)}, {abi.OffExitSite, uint64(int64(in.Site))},
			{abi.OffLive, abi.LiveInline}} {
			c.a.MovImm(a64A, f.v)
			c.a.Store(a64B, f.off, a64A)
		}
		c.a.Load(a64C, a64Ctx, abi.OffStackBase)
		c.a.Op(arm64.Sub, a64A, a64Stack, a64C, true)
		c.a.ShiftImm(arm64.Lsr, a64A, a64A, 4, true)
		c.a.AddImm(a64A, a64A, int64(in.Base-c.f.Locals), true)
		c.a.Store(a64B, abi.OffBase, a64A)
		kind = abi.ExitHost
	}
	c.record(kind, uint64(s.PC), uint64(s.Depth), uint64(int64(s.Site)), then)
}

// isReference falls through when v, whose word is in w, is the reference
// slot o held at entry, and jumps to primitive when it is not. It uses C.
func (c *a64Compiler) isReference(v *ssa.Value, w arm64.Reg, o int, primitive arm64.Label) {
	base, disp := c.slotAddr(o, true, a64C)
	c.a.Load(a64C, base, disp)
	c.a.Cbz(a64C, primitive, true)
	if v.Op != ssa.OpLoadSlot {
		base, disp = c.slotAddr(o, false, a64C)
		c.a.Load(a64C, base, disp)
		c.a.Cmp(w, a64C, true)
		c.a.BCond(arm64.NE, primitive)
	}
}

// appendRecord adds an abi.Record for slot, as amd64's does. It uses B
// and C.
func (c *a64Compiler) appendRecord(slot uint64, arg func() arm64.Reg, imm uint64, useImm bool, word *arm64.Reg) {
	c.recorded = true
	c.a.Load(a64C, a64Ctx, abi.OffRecords)
	c.a.ShiftImm(arm64.Lsl, a64C, a64C, 5, true)
	c.a.Op(arm64.Add, a64C, a64C, a64Ctx, true)
	if word != nil {
		c.a.Store(a64C, abi.OffRecord+16, *word)
	}
	c.a.MovImm(a64B, slot)
	c.a.Store(a64C, abi.OffRecord, a64B)
	if useImm {
		c.a.MovImm(a64B, imm)
		c.a.Store(a64C, abi.OffRecord+8, a64B)
	} else {
		c.a.Store(a64C, abi.OffRecord+8, arg())
	}
	c.a.Load(a64C, a64Ctx, abi.OffRecords)
	c.a.AddImm(a64C, a64C, 1, true)
	c.a.Store(a64Ctx, abi.OffRecords, a64C)
}

// gprAfter returns a function yielding a register holding v, loading it
// into B if it has none.
func (c *a64Compiler) gprAfter(v *ssa.Value) func() arm64.Reg {
	return func() arm64.Reg { return c.gpr(v, a64B) }
}

// record fills the exit record and returns to Go, or goes on at then, as
// amd64's does.
func (c *a64Compiler) record(kind, pc, depth, site uint64, then *arm64.Label) {
	c.a.MovImm(a64A, kind)
	c.a.Store(a64Ctx, abi.OffExitKind, a64A)
	c.a.MovImm(a64A, pc)
	c.a.Store(a64Ctx, abi.OffExitPC, a64A)
	c.a.MovImm(a64A, depth)
	c.a.Store(a64Ctx, abi.OffExitDepth, a64A)
	c.a.MovImm(a64A, site)
	c.a.Store(a64Ctx, abi.OffExitSite, a64A)
	switch {
	case !c.recorded && then == nil:
		c.a.Ret()
		return
	case !c.recorded:
		c.a.B(*then)
		return
	case then != nil:
		c.a.Adr(a64A, *then)
		c.a.Store(a64Ctx, abi.OffTailReturn, a64A)
	}
	if !c.tailUsed {
		c.tail, c.tailUsed = c.a.NewLabel(), true
	}
	c.a.B(c.tail)
}

// recordsTail applies an exit's records natively while the collector is
// not marking, as amd64's does. D counts the records; F1 keeps how many.
func (c *a64Compiler) recordsTail() {
	ret, read, write, maybe, scalar, copyValue, next := c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel(),
		c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel()
	c.a.MovImm(a64B, c.enc.WriteBarrier)
	c.a.LoadU8(a64B, a64B, 0)
	c.a.Cbnz(a64B, ret, false)
	c.a.Load(a64D, a64Ctx, abi.OffRecords)
	c.a.Cbz(a64D, ret, true)
	c.a.FMovToF(a64F1, a64D)
	c.a.Bind(read)
	c.a.AddImm(a64D, a64D, -1, true)
	c.a.ShiftImm(arm64.Lsl, a64A, a64D, 5, true)
	c.a.Op(arm64.Add, a64A, a64A, a64Ctx, true)
	c.a.Load(a64C, a64A, abi.OffRecord)
	c.a.MovImm(a64B, abi.RecordScalar)
	c.a.Tst(a64C, a64B, true)
	c.a.BCond(arm64.NE, scalar)
	c.a.MovImm(a64B, abi.RecordMaybe)
	c.a.Tst(a64C, a64B, true)
	c.a.BCond(arm64.NE, maybe)
	c.a.Load(a64C, a64A, abi.OffRecord+8)
	c.sourceAddr()
	c.a.Bind(copyValue)
	c.a.Load(a64B, a64C, c.enc.NumOffset)
	c.a.Store(a64A, abi.OffRecord+16, a64B)
	c.a.Load(a64B, a64C, c.enc.RefOffset)
	c.a.Store(a64A, abi.OffRecord+8, a64B)
	c.a.B(next)
	c.a.Bind(maybe)
	c.a.Load(a64C, a64A, abi.OffRecord+8)
	c.a.CmpImm(a64C, 0, true)
	c.a.BCond(arm64.MI, scalar)
	c.sourceAddr()
	c.a.Load(a64B, a64C, c.enc.RefOffset)
	c.a.Cbnz(a64B, copyValue, true)
	c.a.Bind(scalar)
	c.a.Store(a64A, abi.OffRecord+8, arm64.ZR)
	c.a.Bind(next)
	c.a.Cbnz(a64D, read, true)
	c.a.FMovFromF(a64D, a64F1)
	c.a.Bind(write)
	c.a.AddImm(a64D, a64D, -1, true)
	c.a.ShiftImm(arm64.Lsl, a64A, a64D, 5, true)
	c.a.Op(arm64.Add, a64A, a64A, a64Ctx, true)
	c.a.Load(a64C, a64A, abi.OffRecord)
	c.a.MovImm(a64B, ^uint64(abi.RecordScalar|abi.RecordMaybe))
	c.a.Op(arm64.And, a64C, a64C, a64B, true)
	c.sourceAddr()
	c.a.Load(a64B, a64A, abi.OffRecord+16)
	c.a.Store(a64C, c.enc.NumOffset, a64B)
	c.a.Load(a64B, a64A, abi.OffRecord+8)
	c.a.Store(a64C, c.enc.RefOffset, a64B)
	c.a.Cbnz(a64D, write, true)
	c.a.Store(a64Ctx, abi.OffRecords, arm64.ZR)
	c.a.Bind(ret)
	// A native call's writing of its caller's state goes on in the
	// caller's code (exitThen).
	toGo := c.a.NewLabel()
	c.a.Load(a64B, a64Ctx, abi.OffTailReturn)
	c.a.Cbz(a64B, toGo, true)
	c.a.Store(a64Ctx, abi.OffTailReturn, arm64.ZR)
	c.a.Br(a64B)
	c.a.Bind(toGo)
	c.a.Ret()
}

// returnNative returns to the native caller, if there is one, with the
// result, as amd64's does; otherwise it falls through to return to Go.
func (c *a64Compiler) returnNative() {
	toGo, word := c.a.NewLabel(), c.a.NewLabel()
	c.a.Load(a64A, a64Ctx, abi.OffReturnTo)
	c.a.Cbz(a64A, toGo, true)
	c.a.Load(a64A, a64Ctx, abi.OffRet)
	c.a.Store(a64Ctx, abi.OffRetValue+c.enc.NumOffset, a64A)
	c.a.Store(a64Ctx, abi.OffRetValue+c.enc.RefOffset, arm64.ZR)
	c.a.Load(a64C, a64Ctx, abi.OffRetFrom)
	c.a.Cbz(a64C, word, true)
	c.a.AddImm(a64C, a64C, -1, true)
	c.sourceAddr()
	c.a.Load(a64C, a64C, c.enc.RefOffset)
	c.a.Store(a64Ctx, abi.OffRetValue+c.enc.RefOffset, a64C)
	c.a.Bind(word)
	c.a.Load(a64A, a64Ctx, abi.OffReturnTo)
	c.a.Br(a64A)
	c.a.Bind(toGo)
}

// nativeCall calls the function the call at b's end calls natively, if it
// is one of those it was seen to call, as amd64's does. Once the caller's
// state is written nothing is in a register, and the frame is made in
// registers the allocator gives values, X3 to X8.
func (c *a64Compiler) nativeCall(b *ssa.Block) {
	sites, s := b.Calls, b.State
	site := sites[0]
	host := c.a.NewLabel()
	toHost := func(cond arm64.Cond) { c.a.BCond(cond, host) }
	var cont *ssa.Block
	for _, e := range c.f.Entries {
		if e.PC == site.Cont {
			cont = e.Block
		}
	}
	vs := int32(c.enc.ValueSize)
	sp := len(s.Slots)
	calleeSlot := sp - site.Argc - 1
	operand := func(slot int) int32 { return int32(slot-c.f.Locals) * vs }
	if cont == nil || calleeSlot < c.f.Locals || site.Method && calleeSlot-1 < c.f.Locals {
		return
	}
	c.a.MovImm(a64B, c.enc.WriteBarrier)
	c.a.LoadU8(a64B, a64B, 0)
	c.a.Cbnz(a64B, host, false)
	c.a.Load(a64B, a64Ctx, abi.OffLevel)
	c.a.AddImm(a64B, a64B, 1, true)
	c.a.Load(a64C, a64Ctx, abi.OffLevelLimit)
	c.a.Cmp(a64B, a64C, true)
	toHost(arm64.HS)
	callee := s.Slots[calleeSlot]
	if callee.Shadow == nil && c.origin.At(callee) < 0 {
		c.a.B(host)
		c.a.Bind(host)
		return
	}
	c.a.MovImm(a64B, c.enc.Object)
	c.a.Cmp(c.gpr(callee, a64A), a64B, true)
	toHost(arm64.NE)
	c.sourceRef(callee, toHost)
	checked := c.a.NewLabel()
	for _, t := range sites {
		next := c.a.NewLabel()
		c.a.MovImm(a64B, uint64(t.Callee))
		c.a.Cmp(a64C, a64B, true)
		c.a.BCond(arm64.NE, next)
		c.a.MovImm(a64A, uint64(t.Entry))
		c.a.Load(a64A, a64A, 0)
		c.a.Cbz(a64A, host, true)
		if t.Coerce {
			c.a.MovImm(a64B, c.enc.Object)
			c.a.Cmp(c.gpr(s.Slots[calleeSlot-1], a64A), a64B, true)
			toHost(arm64.NE)
		}
		c.a.Load(a64B, a64Ctx, abi.OffStackTop)
		c.a.Load(a64B, a64B, 0)
		c.a.AddImm(a64B, a64B, int64(t.LocalCount+t.MaxStack), true)
		c.a.Load(a64C, a64Ctx, abi.OffStackEnd)
		c.a.Cmp(a64B, a64C, true)
		toHost(arm64.HI)
		c.a.B(checked)
		c.a.Bind(next)
	}
	c.a.B(host)
	c.a.Bind(checked)
	written := c.a.NewLabel()
	c.exitThen(s, abi.ExitHost, &written)
	c.a.Bind(written)
	const calleeCtx, base, locals, tmp, top, high = arm64.Reg(3), arm64.Reg(4), arm64.Reg(5), arm64.Reg(6), arm64.Reg(7), arm64.Reg(8)
	c.a.AddImm(calleeCtx, a64Ctx, int64(abi.ContextSize), true)
	c.a.Load(base, a64Ctx, abi.OffStackTop)
	c.a.Load(base, base, 0)
	c.a.Load(locals, a64Ctx, abi.OffStackBase)
	c.a.ShiftImm(arm64.Lsl, tmp, base, 4, true)
	c.a.Op(arm64.Add, locals, locals, tmp, true)
	c.a.Store(calleeCtx, abi.OffLocals, locals)
	for _, off := range []int32{abi.OffBackEdges, abi.OffGlobal, abi.OffLexNames, abi.OffLevelLimit,
		abi.OffStackBase, abi.OffStackEnd, abi.OffStackTop, abi.OffStackHigh} {
		c.a.Load(tmp, a64Ctx, off)
		c.a.Store(calleeCtx, off, tmp)
	}
	c.a.Store(calleeCtx, abi.OffUpvalues, arm64.ZR)
	c.a.Store(calleeCtx, abi.OffTailReturn, arm64.ZR)
	c.a.Load(tmp, a64Ctx, abi.OffLevel)
	c.a.AddImm(tmp, tmp, 1, true)
	c.a.Store(calleeCtx, abi.OffLevel, tmp)
	c.a.Store(calleeCtx, abi.OffBase, base)
	c.a.MovImm(tmp, 1)
	c.a.Store(calleeCtx, abi.OffLive, tmp)
	back := c.a.NewLabel()
	c.a.Adr(tmp, back)
	c.a.Store(calleeCtx, abi.OffReturnTo, tmp)
	c.a.Load(tmp, a64Stack, operand(calleeSlot)+c.enc.RefOffset)
	for i, t := range sites {
		next := c.a.NewLabel()
		if i < len(sites)-1 {
			c.a.MovImm(top, uint64(t.Callee))
			c.a.Cmp(tmp, top, true)
			c.a.BCond(arm64.NE, next)
		}
		c.a.AddImm(top, locals, int64(t.LocalCount)*int64(vs), true)
		c.a.Store(calleeCtx, abi.OffStack, top)
		for i := 0; i < t.LocalCount; i++ {
			at := int32(i) * vs
			if i < t.Params && i < t.Argc {
				from := operand(sp - t.Argc + i)
				c.a.Load(top, a64Stack, from+c.enc.NumOffset)
				c.a.Store(locals, at+c.enc.NumOffset, top)
				c.a.Load(top, a64Stack, from+c.enc.RefOffset)
				c.a.Store(locals, at+c.enc.RefOffset, top)
				continue
			}
			c.a.MovImm(top, c.enc.Undefined)
			c.a.Store(locals, at+c.enc.NumOffset, top)
			c.a.Store(locals, at+c.enc.RefOffset, arm64.ZR)
		}
		if t.ThisSlot >= 0 {
			from := operand(calleeSlot - 1)
			c.a.Load(top, a64Stack, from+c.enc.NumOffset)
			c.a.Store(calleeCtx, abi.OffThis+c.enc.NumOffset, top)
			c.a.Load(top, a64Stack, from+c.enc.RefOffset)
			c.a.Store(calleeCtx, abi.OffThis+c.enc.RefOffset, top)
		}
		c.a.MovImm(top, uint64(t.Closure))
		c.a.Store(calleeCtx, abi.OffClosure, top)
		if t.Count != 0 {
			c.a.MovImm(top, uint64(t.Count))
			c.a.Load(high, top, 0)
			c.a.AddImm(high, high, 1, true)
			c.a.Store(top, 0, high)
		}
		same := c.a.NewLabel()
		c.a.AddImm(top, base, int64(t.LocalCount+t.MaxStack), true)
		c.a.Load(tmp, a64Ctx, abi.OffStackTop)
		c.a.Store(tmp, 0, top)
		c.a.Load(tmp, a64Ctx, abi.OffStackHigh)
		c.a.Load(high, tmp, 0)
		c.a.Cmp(top, high, true)
		c.a.BCond(arm64.LS, same)
		c.a.Store(tmp, 0, top)
		c.a.Bind(same)
		c.a.MovImm(a64A, uint64(t.Entry))
		c.a.Load(a64A, a64A, 0)
		c.a.MovRR(a64Ctx, calleeCtx)
		c.a.Br(a64A)
		c.a.Bind(next)
	}
	c.a.Bind(back)
	c.a.MovRR(calleeCtx, a64Ctx)
	c.a.AddImm(a64Ctx, a64Ctx, -int64(abi.ContextSize), true)
	c.a.Load(a64Stack, a64Ctx, abi.OffStack)
	to := operand(site.Result)
	c.a.Load(tmp, calleeCtx, abi.OffRetValue+c.enc.NumOffset)
	c.a.Store(a64Stack, to+c.enc.NumOffset, tmp)
	c.a.Load(tmp, calleeCtx, abi.OffRetValue+c.enc.RefOffset)
	c.a.Store(a64Stack, to+c.enc.RefOffset, tmp)
	c.a.Load(tmp, a64Ctx, abi.OffStackTop)
	c.a.Load(top, calleeCtx, abi.OffBase)
	c.a.Store(tmp, 0, top)
	c.a.Store(calleeCtx, abi.OffLive, arm64.ZR)
	c.a.Store(calleeCtx, abi.OffReturnTo, arm64.ZR)
	c.a.B(c.labels[cont.ID])
	c.a.Bind(host)
}

// block emits one block. next is the block laid out after it.
func (c *a64Compiler) block(b *ssa.Block, next *ssa.Block) {
	if b.PC < 0 {
		c.a.Load(a64Locals, a64Ctx, abi.OffLocals)
		c.a.Load(a64Stack, a64Ctx, abi.OffStack)
	}
	for _, v := range b.Values {
		if v.Op == ssa.OpPhi || c.isLazy(v) {
			continue
		}
		c.value(v, b)
	}
	switch b.Kind {
	case ssa.BlockPlain:
		c.edge(b, b.Succs[0], next)
	case ssa.BlockIf:
		c.branch(b, next)
	case ssa.BlockReturn:
		r := c.gpr(b.Control, a64A)
		c.a.Store(a64Ctx, abi.OffRet, r)
		c.a.Store(a64Ctx, abi.OffRetFrom, arm64.ZR)
		if s := b.Control.Shadow; s != nil {
			// RetFrom is the source plus one, and 0 for a primitive's -1.
			c.a.AddImm(a64C, c.gpr(s, a64C), 1, true)
			c.a.Store(a64Ctx, abi.OffRetFrom, a64C)
		} else if o, ok := c.origin.Of(b.Control); ok && o >= 0 {
			done := c.a.NewLabel()
			c.isReference(b.Control, r, o, done)
			c.a.MovImm(a64C, uint64(o)+1)
			c.a.Store(a64Ctx, abi.OffRetFrom, a64C)
			c.a.Bind(done)
		}
		c.a.MovImm(a64A, abi.ExitReturn)
		c.a.Store(a64Ctx, abi.OffExitKind, a64A)
		c.a.MovImm(a64A, uint64(b.PC))
		c.a.Store(a64Ctx, abi.OffExitSite, a64A)
		c.returnNative()
		c.a.Ret()
	case ssa.BlockExit:
		if len(b.Calls) != 0 {
			c.nativeCall(b)
		}
		c.exitTo(b.State, exitKind(int(b.ExitKind)))
	}
}

// edge emits the moves for an edge's phis, a poll on a back-edge, and a jump
// unless the target is laid out next.
func (c *a64Compiler) edge(from, to, next *ssa.Block) {
	idx := -1
	for i, p := range to.Preds {
		if p == from {
			idx = i
		}
	}
	c.phiMoves(to, idx)
	if to.LoopHeader && to.Backedge[idx] {
		poll := c.a.NewLabel()
		c.a.Load(a64B, a64Ctx, abi.OffBackEdges)
		c.a.Load(a64A, a64B, 0)
		c.a.AddImm(a64A, a64A, -1, true)
		c.a.Store(a64B, 0, a64A)
		c.a.CmpImm(a64A, 0, true)
		c.a.BCond(arm64.LE, poll)
		header := to.Header
		c.cold = append(c.cold, func() {
			c.a.Bind(poll)
			c.exitTo(header, abi.ExitPoll)
		})
	}
	if to != next {
		c.a.B(c.labels[to.ID])
	}
}

// phiMoves performs the parallel move of an edge's phi arguments, as the
// core schedules it.
func (c *a64Compiler) phiMoves(to *ssa.Block, idx int) {
	for _, st := range c.phiSchedule(to, idx) {
		switch {
		case st.park == nil:
			c.moveValue(st.dst, st.src, st.parked)
		case isFloat(st.park):
			c.a.FMov(a64F0, c.fpr(st.park, a64F0))
		default:
			c.a.MovRR(a64C, c.gpr(st.park, a64C))
		}
	}
}

// moveValue copies src into dst's location; parked sources are in scratch.
func (c *a64Compiler) moveValue(dst, src *ssa.Value, parked bool) {
	if isFloat(dst) {
		x := a64F0
		if !parked {
			x = c.fpr(src, a64F1)
		}
		c.setF(dst, x)
		return
	}
	r := a64C
	if !parked {
		r = c.gpr(src, a64A)
	}
	c.setG(dst, r)
}

// a64Compare is the condition an FCMP of x with y leaves for a comparison
// operator: each is false when unordered, but NE, which JavaScript's !=
// wants true for NaN.
func a64Compare(op ir.Operator) arm64.Cond {
	switch op {
	case ir.Lt:
		return arm64.MI
	case ir.Le:
		return arm64.LS
	case ir.Gt:
		return arm64.GT
	case ir.Ge:
		return arm64.GE
	case ir.Eq:
		return arm64.EQ
	}
	return arm64.NE
}

// branch emits an If block's end, fusing a float comparison into the jump.
func (c *a64Compiler) branch(b *ssa.Block, next *ssa.Block) {
	yes, no := c.a.NewLabel(), c.a.NewLabel()
	ctl := b.Control
	if ctl.Op == ssa.OpCmpF64 && ctl.Uses == 1 && ctl.Block == b {
		c.a.FCmp(c.fpr(ctl.Args[0], a64F0), c.fpr(ctl.Args[1], a64F1))
		c.a.BCond(a64Compare(ir.Operator(ctl.Aux)), yes)
	} else {
		c.a.Cbnz(c.gpr(ctl, a64A), yes, false)
	}
	// The false edge follows.
	c.a.Bind(no)
	c.edge(b, b.Succs[1], nil)
	c.a.Bind(yes)
	c.edge(b, b.Succs[0], next)
}

// looseNullish is x == null, as amd64's: null's or undefined's word, or an
// object with [[IsHTMLDDA]].
func (c *a64Compiler) looseNullish(v *ssa.Value, guard func(arm64.Cond)) {
	a := v.Args[0]
	yes, no, done := c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel()
	w := c.gpr(a, a64B)
	for _, word := range []uint64{c.enc.Null, c.enc.Undefined} {
		c.a.MovImm(a64A, word)
		c.a.Cmp(w, a64A, true)
		c.a.BCond(arm64.EQ, yes)
	}
	c.a.MovImm(a64A, c.enc.Object)
	c.a.Cmp(w, a64A, true)
	c.a.BCond(arm64.NE, no)
	if c.reference(v, a, c.enc.Object, guard) {
		c.a.LoadU8(a64A, a64C, c.enc.ObjectFlags)
		c.a.MovImm(a64B, uint64(c.enc.FlagHTMLDDA))
		c.a.Tst(a64A, a64B, false)
		c.a.BCond(arm64.NE, yes)
	}
	c.a.Bind(no)
	c.a.MovImm(a64A, 0)
	c.a.B(done)
	c.a.Bind(yes)
	c.a.MovImm(a64A, 1)
	c.a.Bind(done)
	c.setG(v, a64A)
}

// eqTagged is OpEqTagged, as amd64's.
func (c *a64Compiler) eqTagged(v *ssa.Value, guard func(arm64.Cond)) {
	x, y := v.Args[0], v.Args[1]
	yes, no, done, differ, number := c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel()
	exit := c.stubLabel(v.State, exitKind(v.Aux))
	wx, wy := c.gpr(x, a64A), c.gpr(y, a64B)
	c.a.Cmp(wx, wy, true)
	c.a.BCond(arm64.NE, differ)
	c.a.MovImm(a64C, abi.NumberLimit)
	c.a.Cmp(wx, a64C, true)
	c.a.BCond(arm64.LO, number)
	for _, w := range []uint64{c.enc.Null, c.enc.Undefined, c.enc.True, c.enc.False} {
		c.a.MovImm(a64C, w)
		c.a.Cmp(wx, a64C, true)
		c.a.BCond(arm64.EQ, yes)
	}
	strings := c.a.NewLabel()
	c.a.MovImm(a64C, c.enc.String)
	c.a.Cmp(wx, a64C, true)
	c.a.BCond(arm64.EQ, strings)
	c.a.MovImm(a64C, c.enc.Object)
	c.a.Cmp(wx, a64C, true)
	c.a.BCond(arm64.NE, exit)
	if c.reference(v, x, c.enc.Object, guard) {
		c.a.FMovToF(a64F1, a64C)
		if c.reference(v, y, c.enc.Object, guard) {
			c.a.FMovFromF(a64A, a64F1)
			c.a.Cmp(a64A, a64C, true)
			c.a.BCond(arm64.EQ, yes)
			c.a.B(no)
		}
	}
	c.a.Bind(strings)
	if c.reference(v, x, c.enc.String, guard) {
		c.a.FMovToF(a64F1, a64C)
		if c.reference(v, y, c.enc.String, guard) {
			c.a.FMovFromF(a64A, a64F1)
			c.a.Cmp(a64A, a64C, true)
			c.a.BCond(arm64.EQ, yes)
			c.a.Load(a64A, a64A, c.enc.StringLength)
			c.a.Load(a64B, a64C, c.enc.StringLength)
			c.a.Cmp(a64A, a64B, true)
			c.a.BCond(arm64.NE, no)
			c.stringBytes(exit, yes, no)
		}
	}
	c.a.Bind(number)
	c.a.FMovToF(a64F0, wx)
	c.a.FCmp(a64F0, a64F0)
	c.a.BCond(arm64.VS, no)
	c.a.B(yes)
	c.a.Bind(differ)
	other := no
	if v.Index != 1 {
		other = exit
	}
	c.a.MovImm(a64C, abi.NumberLimit)
	c.a.Cmp(wx, a64C, true)
	c.a.BCond(arm64.HS, other)
	c.a.Cmp(wy, a64C, true)
	c.a.BCond(arm64.HS, other)
	c.a.FMovToF(a64F0, wx)
	c.a.FMovToF(a64F1, wy)
	c.a.FCmp(a64F0, a64F1)
	c.a.BCond(arm64.NE, no)
	c.a.BCond(arm64.VS, no)
	c.a.Bind(yes)
	c.a.MovImm(a64A, 1)
	c.a.B(done)
	c.a.Bind(no)
	c.a.MovImm(a64A, 0)
	c.a.Bind(done)
	c.setG(v, a64A)
}

// arrayOf finds the array a value is and checks its class.
func (c *a64Compiler) arrayOf(v *ssa.Value, guard func(arm64.Cond)) {
	if !c.objectOf(v, guard) {
		return
	}
	c.a.LoadU8(a64B, a64C, c.enc.ObjectClass)
	c.a.CmpImm(a64B, int64(c.enc.ClassArray), false)
	guard(arm64.NE)
	c.setG(v, a64C)
}

// objectOf finds the object a value is, into C.
func (c *a64Compiler) objectOf(v *ssa.Value, guard func(arm64.Cond)) bool {
	return c.reference(v, v.Args[0], c.enc.Object, guard)
}

// reference finds the reference a is, of the kind whose word is word, into
// C, as amd64's does. It uses A and B.
func (c *a64Compiler) reference(v, a *ssa.Value, word uint64, guard func(arm64.Cond)) bool {
	fail := c.stubLabel(v.State, exitKind(v.Aux))
	o := c.origin.At(a)
	if a.Shadow == nil && o < 0 {
		c.a.B(fail)
		return false
	}
	w := c.gpr(a, a64A)
	c.a.MovImm(a64B, word)
	c.a.Cmp(w, a64B, true)
	guard(arm64.NE)
	c.sourceRef(a, guard)
	c.a.Cbz(a64C, fail, true)
	return true
}

// sourceRef loads the pointer word of a, a value with a source, into C, as
// amd64's does. It uses A and B.
func (c *a64Compiler) sourceRef(a *ssa.Value, guard func(arm64.Cond)) {
	o := c.origin.At(a)
	if s := a.Shadow; s != nil {
		c.a.MovRR(a64C, c.gpr(s, a64C))
		c.a.CmpImm(a64C, 0, true)
		guard(arm64.MI)
		c.sourceAddr()
		c.a.Load(a64C, a64C, c.enc.RefOffset)
	} else {
		base, disp := c.slotAddr(o, true, a64C)
		c.a.Load(a64C, base, disp)
	}
}

// sourceAddr turns the source in C into the address of its value, as
// amd64's does. It uses B.
func (c *a64Compiler) sourceAddr() {
	captured, stack, found := c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel()
	c.a.CmpImm(a64C, abi.MaxRecords, true)
	c.a.BCond(arm64.HS, found)
	c.a.CmpImm(a64C, int64(c.f.FrameLocals), true)
	c.a.BCond(arm64.HS, captured)
	c.a.ShiftImm(arm64.Lsl, a64C, a64C, 4, true)
	c.a.Op(arm64.Add, a64C, a64C, a64Locals, true)
	c.a.B(found)
	c.a.Bind(captured)
	c.a.CmpImm(a64C, int64(c.f.Locals), true)
	c.a.BCond(arm64.HS, stack)
	if t := c.f.ThisSlot; t >= 0 {
		notThis := c.a.NewLabel()
		c.a.CmpImm(a64C, int64(t), true)
		c.a.BCond(arm64.NE, notThis)
		c.a.AddImm(a64C, a64Ctx, int64(abi.OffThis), true)
		c.a.B(found)
		c.a.Bind(notThis)
	}
	c.a.AddImm(a64C, a64C, -int64(c.f.FrameLocals), true)
	c.a.ShiftImm(arm64.Lsl, a64C, a64C, 3, true)
	c.a.Load(a64B, a64Ctx, abi.OffUpvalues)
	c.a.Op(arm64.Add, a64C, a64C, a64B, true)
	c.a.Load(a64C, a64C, 0)
	c.a.Load(a64C, a64C, c.enc.UpvalueSlot)
	c.a.B(found)
	c.a.Bind(stack)
	c.a.AddImm(a64C, a64C, -int64(c.f.Locals), true)
	c.a.ShiftImm(arm64.Lsl, a64C, a64C, 4, true)
	c.a.Op(arm64.Add, a64C, a64C, a64Stack, true)
	c.a.Bind(found)
}

// propStore stores a value in a property, as amd64's does. It uses every
// scratch register and F1.
func (c *a64Compiler) propStore(v *ssa.Value, guard func(arm64.Cond)) {
	x := v.Args[1]
	none, have := c.a.NewLabel(), c.a.NewLabel()
	w := c.gpr(x, a64A)
	c.a.MovImm(a64B, abi.NumberLimit)
	c.a.Cmp(w, a64B, true)
	c.a.BCond(arm64.LO, none)
	for _, p := range []uint64{c.enc.Undefined, c.enc.Null, c.enc.True, c.enc.False, c.enc.Uninitialized} {
		c.a.MovImm(a64B, p)
		c.a.Cmp(w, a64B, true)
		c.a.BCond(arm64.EQ, none)
	}
	if x.Shadow == nil && c.origin.At(x) < 0 {
		c.a.B(c.stubLabel(v.State, exitKind(v.Aux)))
	} else {
		c.sourceRef(x, guard)
		c.a.B(have)
	}
	c.a.Bind(none)
	c.a.MovImm(a64C, 0)
	c.a.Bind(have)
	c.a.FMovToF(a64F1, a64C)
	c.property(v, guard)
	for _, s := range v.Args[2:] {
		// As amd64's: only a pointer word there is lost.
		other := c.a.NewLabel()
		c.a.Cmp(c.gpr(s, a64B), a64A, true)
		c.a.BCond(arm64.NE, other)
		c.a.Load(a64B, a64A, c.enc.RefOffset)
		c.a.Cbnz(a64B, c.stubLabel(v.State, exitKind(v.Aux)), true)
		c.a.Bind(other)
	}
	scalar := c.a.NewLabel()
	c.a.FMovFromF(a64C, a64F1)
	c.a.Load(a64B, a64A, c.enc.RefOffset)
	c.a.Op(arm64.Orr, a64B, a64B, a64C, true)
	c.a.Cbz(a64B, scalar, true)
	c.a.MovImm(a64B, c.enc.WriteBarrier)
	c.a.LoadU8(a64B, a64B, 0)
	c.a.CmpImm(a64B, 0, false)
	guard(arm64.NE)
	c.a.Bind(scalar)
	c.a.Store(a64A, c.enc.NumOffset, c.gpr(x, a64B))
	c.a.Store(a64A, c.enc.RefOffset, a64C)
}

// property finds the property a property operation names and leaves the
// address of its value in A, as amd64's does. It uses B, C and D.
func (c *a64Compiler) property(v *ssa.Value, guard func(arm64.Cond)) {
	if v.Holders != nil || v.Cases != nil {
		c.holder(v, guard)
		return
	}
	found, scan := c.a.NewLabel(), c.a.NewLabel()
	if v.Const.Bits != 0 {
		p := c.gpr(v.Args[0], a64A)
		c.a.Load(a64B, p, c.enc.ObjectShape)
		c.a.MovImm(a64C, v.Const.Bits)
		c.a.Cmp(a64B, a64C, true)
		c.a.BCond(arm64.NE, scan)
		c.a.Load(a64A, p, c.enc.ObjectProps)
		c.a.AddImm(a64A, a64A, int64(int32(v.Index)*c.enc.PropertySize+c.enc.PropertyValue), true)
		c.a.B(found)
	}
	c.a.Bind(scan)
	p := c.gpr(v.Args[0], a64A)
	c.a.LoadU8(a64B, p, c.enc.ObjectClass)
	c.a.CmpImm(a64B, int64(c.enc.ClassObject), false)
	guard(arm64.NE)
	c.a.Load(a64B, p, c.enc.ObjectProps+8)
	c.a.CmpImm(a64B, abi.MaxScan, true)
	guard(arm64.HI)
	c.a.Load(a64A, p, c.enc.ObjectProps)
	mask, want := int64(c.enc.PropNotData), int64(0)
	if v.Op == ssa.OpPropWrite {
		mask, want = int64(c.enc.PropNotWritable), int64(c.enc.PropWritable)
	}
	c.a.MovImm(a64D, uint64(mask))
	for k := int32(0); k < abi.MaxScan; k++ {
		next := c.a.NewLabel()
		entry := k * c.enc.PropertySize
		c.a.CmpImm(a64B, int64(k), true)
		guard(arm64.LS)
		c.a.LoadU32(a64C, a64A, entry+c.enc.PropertyKey)
		c.a.CmpImm(a64C, int64(v.Key), false)
		c.a.BCond(arm64.NE, next)
		c.a.LoadU8(a64C, a64A, entry+c.enc.PropertyFlags)
		c.a.Op(arm64.And, a64C, a64C, a64D, false)
		c.a.CmpImm(a64C, want, false)
		guard(arm64.NE)
		c.a.AddImm(a64A, a64A, int64(entry+c.enc.PropertyValue), true)
		c.a.B(found)
		c.a.Bind(next)
	}
	c.a.B(c.stubLabel(v.State, exitKind(v.Aux)))
	c.a.Bind(found)
}

// stringBytes compares two strings of one length by their bytes, as
// amd64's does: their pointers in F1 and C, the length in A. The loop
// borrows the frame's operand base, which it reloads from the context
// before it leaves.
func (c *a64Compiler) stringBytes(exit, yes, no arm64.Label) {
	c.a.CmpImm(a64A, abi.MaxEqualUnits, true)
	c.a.BCond(arm64.HI, exit)
	c.a.FMovFromF(a64A, a64F1)
	for _, s := range []arm64.Reg{a64A, a64C} {
		c.a.Load(a64B, s, c.enc.StringLeft)
		c.a.Cbnz(a64B, exit, true)
	}
	c.a.Load(a64B, a64C, c.enc.StringData+8)
	c.a.Load(a64D, a64A, c.enc.StringData+8)
	c.a.Cmp(a64D, a64B, true)
	c.a.BCond(arm64.NE, no)
	c.a.Load(a64A, a64A, c.enc.StringData)
	c.a.Load(a64C, a64C, c.enc.StringData)
	words, bytes, same, differ := c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel()
	c.a.Bind(words)
	c.a.CmpImm(a64B, 8, true)
	c.a.BCond(arm64.LO, bytes)
	c.a.Load(a64D, a64A, 0)
	c.a.Load(a64Stack, a64C, 0)
	c.a.Cmp(a64D, a64Stack, true)
	c.a.BCond(arm64.NE, differ)
	c.a.AddImm(a64A, a64A, 8, true)
	c.a.AddImm(a64C, a64C, 8, true)
	c.a.AddImm(a64B, a64B, -8, true)
	c.a.B(words)
	c.a.Bind(bytes)
	c.a.Cbz(a64B, same, true)
	c.a.LoadU8(a64D, a64A, 0)
	c.a.LoadU8(a64Stack, a64C, 0)
	c.a.Cmp(a64D, a64Stack, false)
	c.a.BCond(arm64.NE, differ)
	c.a.AddImm(a64A, a64A, 1, true)
	c.a.AddImm(a64C, a64C, 1, true)
	c.a.AddImm(a64B, a64B, -1, true)
	c.a.B(bytes)
	for _, l := range []struct{ at, to arm64.Label }{{same, yes}, {differ, no}} {
		c.a.Bind(l.at)
		c.a.Load(a64Stack, a64Ctx, abi.OffStack)
		c.a.B(l.to)
	}
}

// keepSource leaves in C where a tagged value came from, as amd64's does.
// It uses A.
func (c *a64Compiler) keepSource(x *ssa.Value) {
	switch o, static := c.origin.Of(x); {
	case x.Shadow != nil:
		if r := c.gpr(x.Shadow, a64C); r != a64C {
			c.a.MovRR(a64C, r)
		}
	case static && o >= 0:
		scalar, done := c.a.NewLabel(), c.a.NewLabel()
		c.isReference(x, c.gpr(x, a64A), o, scalar)
		c.a.MovImm(a64C, uint64(o))
		c.a.B(done)
		c.a.Bind(scalar)
		c.a.MovImm(a64C, ^uint64(0))
		c.a.Bind(done)
	default:
		c.a.MovImm(a64C, ^uint64(0))
	}
}

// keepRef is a tagged value's pointer word, as amd64's.
func (c *a64Compiler) keepRef(v *ssa.Value) {
	x := v.Args[0]
	scalar := c.a.NewLabel()
	c.keepSource(x)
	c.a.MovImm(a64A, 0)
	c.a.MovImm(a64B, c.enc.WriteBarrier)
	c.a.LoadU8(a64B, a64B, 0)
	c.a.Cbnz(a64B, scalar, false)
	c.a.CmpImm(a64C, -1, true)
	c.a.BCond(arm64.EQ, scalar)
	c.sourceAddr()
	c.a.Load(a64A, a64C, c.enc.RefOffset)
	c.a.Bind(scalar)
	c.setG(v, a64A)
}

// keep copies a value into a keep cell, as amd64's.
func (c *a64Compiler) keep(v *ssa.Value) {
	x := v.Args[0]
	done := c.a.NewLabel()
	c.keepSource(x)
	c.a.MovRR(a64A, a64C)
	c.a.MovImm(a64B, c.enc.WriteBarrier)
	c.a.LoadU8(a64B, a64B, 0)
	c.a.Cbnz(a64B, done, false)
	at := abi.OffKeep + int32(v.Index)*int32(c.enc.ValueSize)
	c.a.Store(a64Ctx, at+c.enc.RefOffset, c.gpr(v.Args[1], a64B))
	c.a.Store(a64Ctx, at+c.enc.NumOffset, c.gpr(x, a64B))
	c.a.AddImm(a64A, a64Ctx, int64(at), true)
	c.a.Bind(done)
	c.setG(v, a64A)
}

// holder finds the property a read whose receiver's shape it knows names,
// for whichever of the shapes it met the receiver's is, into A, as amd64's
// does.
func (c *a64Compiler) holder(v *ssa.Value, guard func(arm64.Cond)) {
	var first [2]ssa.Holder
	if v.Holders != nil {
		first = *v.Holders
	}
	cases := append([]ssa.PropertyCase{{Shape: uintptr(v.Const.Bits), Index: int32(v.Index), Holders: first}}, v.Cases...)
	done := c.a.NewLabel()
	for i, k := range cases {
		next := c.a.NewLabel()
		p := c.gpr(v.Args[0], a64A)
		c.a.Load(a64B, p, c.enc.ObjectShape)
		c.a.MovImm(a64C, uint64(k.Shape))
		c.a.Cmp(a64B, a64C, true)
		if i == len(cases)-1 {
			guard(arm64.NE)
		} else {
			c.a.BCond(arm64.NE, next)
		}
		if p != a64A {
			c.a.MovRR(a64A, p)
		}
		c.a.Load(a64B, a64A, c.enc.ObjectProto)
		for _, h := range k.Holders {
			if h.Object == 0 {
				break
			}
			c.a.MovImm(a64A, uint64(h.Object))
			c.a.Cmp(a64B, a64A, true)
			guard(arm64.NE)
			c.a.Load(a64B, a64A, c.enc.ObjectShape)
			c.a.MovImm(a64C, uint64(h.Shape))
			c.a.Cmp(a64B, a64C, true)
			guard(arm64.NE)
			c.a.Load(a64B, a64A, c.enc.ObjectProto)
		}
		c.a.Load(a64A, a64A, c.enc.ObjectProps)
		c.a.AddImm(a64A, a64A, int64(k.Index*c.enc.PropertySize+c.enc.PropertyValue), true)
		if i < len(cases)-1 {
			c.a.B(done)
			c.a.Bind(next)
		}
	}
	c.a.Bind(done)
}

// integer converts the double x to an integer in r, failing unless it is
// one an int64 holds exactly: FCVTZS saturates, so a result of 2**63-1,
// which no double converts to exactly, means x is 2**63 or more. It uses
// D and F2.
func (c *a64Compiler) integer(r arm64.Reg, x arm64.FReg, guard func(arm64.Cond)) {
	c.a.Fcvtzs(r, x)
	c.a.Scvtf(a64F2, r, true)
	c.a.FCmp(a64F2, x)
	guard(arm64.NE)
	c.a.MovImm(a64D, math.MaxInt64)
	c.a.Cmp(r, a64D, true)
	guard(arm64.EQ)
}

// remainder is JavaScript's % of two integers, as amd64's is. SDIV never
// faults -- -2**63 / -1 is -2**63, whose MSUB remainder is 0 -- so no
// divisor needs a case of its own.
func (c *a64Compiler) remainder(v *ssa.Value, guard func(arm64.Cond)) {
	x, y := c.fpr(v.Args[0], a64F0), c.fpr(v.Args[1], a64F1)
	c.integer(a64A, x, guard)
	c.integer(a64C, y, guard)
	c.a.Cbz(a64C, c.stubLabel(v.State, exitKind(v.Aux)), true)
	zero, done := c.a.NewLabel(), c.a.NewLabel()
	c.a.Sdiv(a64B, a64A, a64C)
	c.a.Msub(a64B, a64B, a64C, a64A)
	c.a.Cbz(a64B, zero, true)
	c.a.Scvtf(a64F0, a64B, true)
	c.a.B(done)
	c.a.Bind(zero)
	c.a.FMovFromF(a64A, x)
	c.a.MovImm(a64B, 1<<63)
	c.a.Op(arm64.And, a64A, a64A, a64B, true)
	c.a.FMovToF(a64F0, a64A)
	c.a.Bind(done)
	c.setF(v, a64F0)
}

// length is x.length, as amd64's is.
func (c *a64Compiler) length(v *ssa.Value, guard func(arm64.Cond)) {
	a := v.Args[0]
	if a.Shadow == nil && c.origin.At(a) < 0 {
		c.a.B(c.stubLabel(v.State, exitKind(v.Aux)))
		return
	}
	array, have := c.a.NewLabel(), c.a.NewLabel()
	c.a.MovImm(a64B, c.enc.String)
	c.a.Cmp(c.gpr(a, a64A), a64B, true)
	c.a.BCond(arm64.NE, array)
	c.reference(v, a, c.enc.String, guard)
	c.a.Load(a64C, a64C, c.enc.StringLength)
	c.a.B(have)
	c.a.Bind(array)
	c.reference(v, a, c.enc.Object, guard)
	c.a.LoadU8(a64B, a64C, c.enc.ObjectClass)
	c.a.CmpImm(a64B, int64(c.enc.ClassArray), false)
	guard(arm64.NE)
	p := a64A
	c.a.MovRR(p, a64C)
	c.a.Load(a64C, p, c.enc.ObjectElems+8)
	c.a.LoadU8(a64B, p, c.enc.ObjectFlags)
	c.a.MovImm(a64D, uint64(c.enc.FlagSparse))
	c.a.Tst(a64B, a64D, false)
	c.a.BCond(arm64.EQ, have)
	c.a.LoadU32(a64B, p, c.enc.ObjectArrayLen)
	c.a.Cmp(a64B, a64C, true)
	c.a.BCond(arm64.LS, have)
	c.a.MovRR(a64C, a64B)
	c.a.Bind(have)
	c.a.Scvtf(a64F0, a64C, true)
	c.setF(v, a64F0)
}

// stringCode is charCodeAt called on a string, as amd64's is.
func (c *a64Compiler) stringCode(v *ssa.Value, guard func(arm64.Cond)) {
	if !c.reference(v, v.Args[0], c.enc.Object, guard) {
		return
	}
	c.a.Load(a64B, a64Ctx, abi.OffCharCode+c.enc.RefOffset)
	c.a.Cmp(a64C, a64B, true)
	guard(arm64.NE)
	if !c.reference(v, v.Args[1], c.enc.String, guard) {
		return
	}
	fail := c.stubLabel(v.State, exitKind(v.Aux))
	c.a.Load(a64B, a64C, c.enc.StringLeft)
	c.a.Cbnz(a64B, fail, true)
	c.index(v.Args[2], guard)
	c.a.Load(a64B, a64C, c.enc.StringLength)
	c.a.Cmp(a64A, a64B, true)
	guard(arm64.HS)
	units, done := c.a.NewLabel(), c.a.NewLabel()
	c.a.LoadU8(a64B, a64C, c.enc.StringASCII)
	c.a.Cbz(a64B, units, false)
	c.a.Load(a64B, a64C, c.enc.StringData)
	c.a.Op(arm64.Add, a64B, a64B, a64A, true)
	c.a.LoadU8(a64A, a64B, 0)
	c.a.B(done)
	c.a.Bind(units)
	c.a.Load(a64B, a64C, c.enc.StringU16)
	c.a.Cbz(a64B, fail, true)
	c.a.Op(arm64.Add, a64A, a64A, a64A, true)
	c.a.Op(arm64.Add, a64B, a64B, a64A, true)
	c.a.LoadU16(a64A, a64B, 0)
	c.a.Bind(done)
	c.a.Scvtf(a64F0, a64A, true)
	c.setF(v, a64F0)
}

// index converts an element's key to an index in A, failing unless it is an
// integer in [0, 2**32). FCVTZS saturates, so a key of 2**63 or more gives
// 2**63-1, whose top half is not clear. It uses B and F1.
func (c *a64Compiler) index(key *ssa.Value, guard func(arm64.Cond)) {
	k := c.fpr(key, a64F0)
	c.a.Fcvtzs(a64A, k)
	c.a.Scvtf(a64F1, a64A, true)
	c.a.FCmp(a64F1, k)
	guard(arm64.NE)
	c.a.ShiftImm(arm64.Lsr, a64B, a64A, 32, true)
	c.a.CmpImm(a64B, 0, true)
	guard(arm64.NE)
}

// element turns the index in A into the address of an array's element in
// A, and its number word in B, as amd64's does. It uses C.
func (c *a64Compiler) element(array *ssa.Value, guard func(arm64.Cond)) {
	p := c.gpr(array, a64C)
	c.a.Load(a64B, p, c.enc.ObjectElems+8)
	c.a.Cmp(a64A, a64B, true)
	guard(arm64.HS)
	c.a.Load(a64B, p, c.enc.ObjectElems)
	c.a.ShiftImm(arm64.Lsl, a64A, a64A, 4, true)
	c.a.Op(arm64.Add, a64A, a64A, a64B, true)
	c.a.Load(a64B, a64A, c.enc.NumOffset)
	c.a.MovImm(a64C, abi.NumberLimit)
	c.a.Cmp(a64B, a64C, true)
	guard(arm64.HS)
}

// elementCell turns the index in A into the address of an array's element
// there, in A, failing unless it is within the dense elements and not a
// hole, as amd64's does. It uses B and C.
func (c *a64Compiler) elementCell(array *ssa.Value, guard func(arm64.Cond)) {
	p := c.gpr(array, a64C)
	c.a.Load(a64B, p, c.enc.ObjectElems+8)
	c.a.Cmp(a64A, a64B, true)
	guard(arm64.HS)
	c.a.Load(a64B, p, c.enc.ObjectElems)
	c.a.ShiftImm(arm64.Lsl, a64A, a64A, 4, true)
	c.a.Op(arm64.Add, a64A, a64A, a64B, true)
	c.a.Load(a64B, a64A, c.enc.NumOffset)
	c.a.MovImm(a64C, c.enc.Uninitialized)
	c.a.Cmp(a64B, a64C, true)
	guard(arm64.EQ)
}

// numberCell checks the word at A's cell is a number, into B.
func (c *a64Compiler) numberCell(guard func(arm64.Cond)) {
	c.a.Load(a64B, a64A, c.enc.NumOffset)
	c.a.MovImm(a64C, abi.NumberLimit)
	c.a.Cmp(a64B, a64C, true)
	guard(arm64.HS)
}

// value emits one value.
func (c *a64Compiler) value(v *ssa.Value, b *ssa.Block) {
	arg := func(i int) *ssa.Value { return v.Args[i] }
	guard := func(cond arm64.Cond) {
		c.a.BCond(cond, c.stubLabel(v.State, exitKind(v.Aux)))
	}
	switch v.Op {
	case ssa.OpLoadSlot:
		base, disp := c.slotAddr(v.Aux, false, a64A)
		c.a.Load(a64A, base, disp)
		c.setG(v, a64A)
	case ssa.OpConst:
		c.a.MovImm(a64A, c.constWord(v.Const))
		c.setG(v, a64A)
	case ssa.OpConstF64:
		d := c.fdst(v)
		c.constF64(d, v.Const.Bits)
		c.setF(v, d)
	case ssa.OpUnboxF64:
		r := c.gpr(arg(0), a64B)
		c.a.ShiftImm(arm64.Lsr, a64A, r, 51, true)
		c.a.CmpImm(a64A, 0x1FFF, false)
		guard(arm64.EQ)
		d := c.fdst(v)
		c.a.FMovToF(d, r)
		c.setF(v, d)
	case ssa.OpCheckInit:
		r := c.gpr(arg(0), a64B)
		c.a.MovImm(a64A, c.enc.Uninitialized)
		c.a.Cmp(r, a64A, true)
		guard(arm64.EQ)
	case ssa.OpTruth:
		c.truth(v, guard)
	case ssa.OpArrayOf:
		c.arrayOf(v, guard)
	case ssa.OpObjectOf:
		if c.objectOf(v, guard) {
			c.setG(v, a64C)
		}
	case ssa.OpKeepRef:
		c.keepRef(v)
	case ssa.OpKeep:
		c.keep(v)
	case ssa.OpKept:
		c.setG(v, c.gpr(arg(1), a64A))
	case ssa.OpFrameRoom:
		c.a.Load(a64B, a64Ctx, abi.OffStackBase)
		c.a.Op(arm64.Sub, a64A, a64Stack, a64B, true)
		c.a.ShiftImm(arm64.Lsr, a64A, a64A, 4, true)
		c.a.AddImm(a64A, a64A, int64(v.Index), true)
		c.a.Load(a64B, a64Ctx, abi.OffStackEnd)
		c.a.Cmp(a64A, a64B, true)
		guard(arm64.HI)
		c.a.Load(a64A, a64Ctx, abi.OffLevel)
		c.a.AddImm(a64A, a64A, 1, true)
		c.a.Load(a64B, a64Ctx, abi.OffLevelLimit)
		c.a.Cmp(a64A, a64B, true)
		guard(arm64.HS)
	case ssa.OpSameObject:
		c.a.MovImm(a64B, v.Const.Bits)
		c.a.Cmp(c.gpr(arg(0), a64A), a64B, true)
		guard(arm64.NE)
	case ssa.OpPropRead:
		c.property(v, guard)
		c.numberCell(guard)
		c.a.FMovToF(a64F0, a64B)
		c.setF(v, a64F0)
	case ssa.OpPropCell:
		c.property(v, guard)
		c.setG(v, a64A)
	case ssa.OpGlobalCell:
		unshadowed := c.a.NewLabel()
		word := int64(v.Key >> 6)
		c.a.Load(a64B, a64Ctx, abi.OffLexNames)
		c.a.Load(a64A, a64B, 8)
		c.a.CmpImm(a64A, word, true)
		c.a.BCond(arm64.LS, unshadowed)
		c.a.Load(a64B, a64B, 0)
		c.a.Load(a64B, a64B, int32(word*8))
		c.a.MovImm(a64A, 1<<(v.Key&63))
		c.a.Tst(a64B, a64A, true)
		guard(arm64.NE)
		c.a.Bind(unshadowed)
		c.a.Load(a64A, a64Ctx, abi.OffGlobal)
		c.a.Cbz(a64A, c.stubLabel(v.State, exitKind(v.Aux)), true)
		c.a.Load(a64B, a64A, c.enc.ObjectProps+8)
		c.a.CmpImm(a64B, int64(v.Index), true)
		guard(arm64.LS)
		c.a.Load(a64A, a64A, c.enc.ObjectProps)
		c.a.AddImm(a64A, a64A, int64(int32(v.Index)*c.enc.PropertySize), true)
		c.a.LoadU32(a64B, a64A, c.enc.PropertyKey)
		c.a.CmpImm(a64B, int64(v.Key), false)
		guard(arm64.NE)
		c.a.LoadU8(a64B, a64A, c.enc.PropertyFlags)
		c.a.MovImm(a64C, uint64(c.enc.PropNotData|c.enc.PropUninit))
		c.a.Tst(a64B, a64C, false)
		guard(arm64.NE)
		c.a.AddImm(a64A, a64A, int64(c.enc.PropertyValue), true)
		c.setG(v, a64A)
	case ssa.OpStringMethod:
		c.a.MovImm(a64B, c.enc.String)
		c.a.Cmp(c.gpr(arg(0), a64A), a64B, true)
		guard(arm64.NE)
		c.a.Load(a64B, a64Ctx, abi.OffCharCode+c.enc.RefOffset)
		c.a.Cbz(a64B, c.stubLabel(v.State, exitKind(v.Aux)), true)
		c.a.AddImm(a64A, a64Ctx, int64(abi.OffCharCode), true)
		c.setG(v, a64A)
	case ssa.OpStringCode:
		c.stringCode(v, guard)
	case ssa.OpLoadCell:
		c.a.Load(a64A, c.gpr(arg(0), a64A), c.enc.NumOffset)
		c.setG(v, a64A)
	case ssa.OpPropWrite:
		c.propStore(v, guard)
	case ssa.OpLength:
		c.length(v, guard)
	case ssa.OpElemKey:
		c.index(arg(0), guard)
	case ssa.OpElemRead:
		c.index(arg(1), guard)
		c.element(arg(0), guard)
		c.a.FMovToF(a64F0, a64B)
		c.setF(v, a64F0)
	case ssa.OpElemCell:
		c.index(arg(1), guard)
		c.elementCell(arg(0), guard)
		c.setG(v, a64A)
	case ssa.OpElemWrite:
		c.index(arg(1), guard)
		c.element(arg(0), guard)
		c.boxF64(c.fpr(arg(2), a64F0), a64B)
		c.a.Store(a64A, c.enc.NumOffset, a64B)
	case ssa.OpBoxF64:
		c.boxF64(c.fpr(arg(0), a64F0), a64A)
		c.setG(v, a64A)
	case ssa.OpBoxBool:
		c.boxBool(c.gpr(arg(0), a64B), a64A)
		c.setG(v, a64A)
	case ssa.OpAddF64, ssa.OpSubF64, ssa.OpMulF64, ssa.OpDivF64:
		d := c.fdst(v)
		c.a.FArith(a64FOp(v.Op), d, c.fpr(arg(0), a64F0), c.fpr(arg(1), a64F1))
		c.setF(v, d)
	case ssa.OpModF64:
		c.remainder(v, guard)
	case ssa.OpNegF64:
		d := c.fdst(v)
		c.a.FNeg(d, c.fpr(arg(0), a64F0))
		c.setF(v, d)
	case ssa.OpCmpF64:
		if v.Uses == 1 && b.Control == v && b.Kind == ssa.BlockIf {
			return // fused into the branch
		}
		c.a.FCmp(c.fpr(arg(0), a64F0), c.fpr(arg(1), a64F1))
		d := c.gdst(v)
		c.a.Cset(d, a64Compare(ir.Operator(v.Aux)))
		c.setG(v, d)
	case ssa.OpStrictNullish:
		word := c.enc.Null
		if v.Aux == 1 {
			word = c.enc.Undefined
		}
		w := c.gpr(arg(0), a64B)
		c.a.MovImm(a64A, word)
		c.a.Cmp(w, a64A, true)
		d := c.gdst(v)
		c.a.Cset(d, arm64.EQ)
		c.setG(v, d)
	case ssa.OpLooseNullish:
		c.looseNullish(v, guard)
	case ssa.OpEqTagged:
		c.eqTagged(v, guard)
	case ssa.OpNot:
		c.a.CmpImm(c.gpr(arg(0), a64A), 0, false)
		d := c.gdst(v)
		c.a.Cset(d, arm64.EQ)
		c.setG(v, d)
	case ssa.OpToInt32:
		c.toInt32(v)
	case ssa.OpAndI32, ssa.OpOrI32, ssa.OpXorI32:
		d := c.gdst(v)
		c.a.Op(a64ALU(v.Op), d, c.gpr(arg(0), a64A), c.gpr(arg(1), a64B), false)
		c.setG(v, d)
	case ssa.OpShlI32, ssa.OpSarI32, ssa.OpShrU32:
		// The W forms take the count modulo 32, as JavaScript does.
		d := c.gdst(v)
		c.a.ShiftReg(a64Shift(v.Op), d, c.gpr(arg(0), a64A), c.gpr(arg(1), a64C), false)
		c.setG(v, d)
	case ssa.OpNotI32:
		c.a.Mvn(a64A, c.gpr(arg(0), a64A), false)
		c.setG(v, a64A)
	case ssa.OpI32ToF64:
		c.a.Scvtf(a64F0, c.gpr(arg(0), a64A), false)
		c.setF(v, a64F0)
	case ssa.OpU32ToF64:
		c.a.Ucvtf(a64F0, c.gpr(arg(0), a64A), false)
		c.setF(v, a64F0)
	default:
		panic("value " + v.Op.String())
	}
}

// truth computes JavaScript truthiness for the kinds the slot IR decides it
// for, exiting for the rest.
func (c *a64Compiler) truth(v *ssa.Value, guard func(arm64.Cond)) {
	r := c.gpr(v.Args[0], a64B)
	number, yes, no, done := c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel()
	c.a.ShiftImm(arm64.Lsr, a64A, r, 51, true)
	c.a.CmpImm(a64A, 0x1FFF, false)
	c.a.BCond(arm64.NE, number)
	for _, w := range []struct {
		word uint64
		to   arm64.Label
	}{{c.enc.True, yes}, {c.enc.False, no}, {c.enc.Undefined, no}, {c.enc.Null, no}} {
		c.a.MovImm(a64A, w.word)
		c.a.Cmp(r, a64A, true)
		c.a.BCond(arm64.EQ, w.to)
	}
	// An object is true unless it is [[IsHTMLDDA]], a string unless it is
	// empty, as amd64's are; anything else is not decided here.
	notObject := c.a.NewLabel()
	c.a.MovImm(a64A, c.enc.Object)
	c.a.Cmp(r, a64A, true)
	c.a.BCond(arm64.NE, notObject)
	if c.reference(v, v.Args[0], c.enc.Object, guard) {
		c.a.LoadU8(a64A, a64C, c.enc.ObjectFlags)
		c.a.MovImm(a64D, uint64(c.enc.FlagHTMLDDA))
		c.a.Tst(a64A, a64D, false)
		c.a.BCond(arm64.NE, no)
		c.a.B(yes)
	}
	c.a.Bind(notObject)
	c.a.MovImm(a64A, c.enc.String)
	c.a.Cmp(r, a64A, true)
	guard(arm64.NE)
	if c.reference(v, v.Args[0], c.enc.String, guard) {
		c.a.Load(a64A, a64C, c.enc.StringLength)
		c.a.Cbnz(a64A, yes, true)
		c.a.B(no)
	}
	c.a.Bind(number)
	// A number is true unless zero or NaN: not equal to zero, and ordered.
	c.a.FMovToF(a64F0, r)
	c.a.FMovToF(a64F1, arm64.ZR)
	c.a.FCmp(a64F0, a64F1)
	c.a.Cset(a64A, arm64.NE)
	c.a.Cset(a64C, arm64.VC)
	c.a.Op(arm64.And, a64A, a64A, a64C, false)
	c.a.B(done)
	c.a.Bind(yes)
	c.a.MovImm(a64A, 1)
	c.a.B(done)
	c.a.Bind(no)
	c.a.MovImm(a64A, 0)
	c.a.Bind(done)
	c.setG(v, a64A)
}

// toInt32 is ToUint32's bits. FCVTZS is exact for |x| < 2**63 and gives 0
// for NaN, as ToUint32 does; it saturates past that, to 2**63-1 or -2**63,
// which a cold path redoes from the exponent and significand, as amd64's
// does.
func (c *a64Compiler) toInt32(v *ssa.Value) {
	xv := v.Args[0]
	slow, back := c.a.NewLabel(), c.a.NewLabel()
	x := c.fpr(xv, a64F0)
	c.a.Fcvtzs(a64A, x)
	c.a.MovImm(a64B, math.MaxInt64)
	c.a.Cmp(a64A, a64B, true)
	c.a.BCond(arm64.EQ, slow)
	c.a.MovImm(a64B, 1<<63)
	c.a.Cmp(a64A, a64B, true)
	c.a.BCond(arm64.EQ, slow)
	c.a.Bind(back)
	c.a.MovRR32(a64A, a64A)
	c.setG(v, a64A)
	c.cold = append(c.cold, func() {
		zero, positive := c.a.NewLabel(), c.a.NewLabel()
		c.a.Bind(slow)
		c.a.FMovFromF(a64A, c.fpr(xv, a64F0))
		// e = exponent; NaN and the infinities give 0.
		c.a.ShiftImm(arm64.Lsr, a64C, a64A, 52, true)
		c.a.MovImm(a64D, 0x7FF)
		c.a.Op(arm64.And, a64C, a64C, a64D, false)
		c.a.CmpImm(a64C, 0x7FF, false)
		c.a.BCond(arm64.EQ, zero)
		c.a.AddImm(a64C, a64C, -1075, false)
		c.a.CmpImm(a64C, 32, false)
		c.a.BCond(arm64.HS, zero)
		c.a.MovRR(a64B, a64A) // the sign is bit 63
		c.a.ShiftImm(arm64.Lsl, a64A, a64A, 12, true)
		c.a.ShiftImm(arm64.Lsr, a64A, a64A, 12, true)
		c.a.ShiftReg(arm64.Lsl, a64A, a64A, a64C, true)
		c.a.CmpImm(a64B, 0, true)
		c.a.BCond(arm64.PL, positive)
		c.a.Neg(a64A, a64A, false)
		c.a.Bind(positive)
		c.a.B(back)
		c.a.Bind(zero)
		c.a.MovImm(a64A, 0)
		c.a.B(back)
	})
}

// a64FOp, a64ALU and a64Shift are the instructions for SSA's arithmetic,
// bitwise and shift operations.
func a64FOp(op ssa.Op) arm64.FOp {
	switch op {
	case ssa.OpAddF64:
		return arm64.FAdd
	case ssa.OpSubF64:
		return arm64.FSub
	case ssa.OpMulF64:
		return arm64.FMul
	}
	return arm64.FDiv
}

func a64ALU(op ssa.Op) arm64.ALU {
	switch op {
	case ssa.OpAndI32:
		return arm64.And
	case ssa.OpOrI32:
		return arm64.Orr
	}
	return arm64.Eor
}

func a64Shift(op ssa.Op) arm64.Shift {
	switch op {
	case ssa.OpShlI32:
		return arm64.Lsl
	case ssa.OpSarI32:
		return arm64.Asr
	}
	return arm64.Lsr
}
