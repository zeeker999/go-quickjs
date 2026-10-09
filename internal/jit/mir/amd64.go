package mir

import (
	"fmt"
	"slices"

	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
	"github.com/go-quickjs/go-quickjs/internal/jit/asm/amd64"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
	"github.com/go-quickjs/go-quickjs/internal/jit/ssa"
)

// The fixed registers. RSP, RBP and R14 (Go's g) are never touched.
const (
	regCtx    = amd64.RDI
	regLocals = amd64.RSI
	regStack  = amd64.RDX
	scratchA  = amd64.RAX
	scratchC  = amd64.RCX // also shift counts
	scratchB  = amd64.R11
	xScratch0 = amd64.XReg(0)
	xScratch1 = amd64.XReg(1)
	xScratch2 = amd64.XReg(15)
)

// compiler is amd64's code generator over the shared core.
type compiler struct {
	*core
	a *amd64.Asm
	// label of each block's code.
	labels []amd64.Label // by block ID
	// stubs: one exit per frame state and kind.
	stubs   map[stubKey]amd64.Label
	stubFor []stub
	// cold code, emitted after everything else.
	cold []func()
}

type stub struct {
	key   stubKey
	label amd64.Label
}

// CompileAMD64 compiles f for amd64, given the VM's value encoding.
func CompileAMD64(f *ssa.Func, enc abi.Encoding) (*Code, error) { return compileAMD64(nil, f, enc) }

func compileAMD64(w *Workspace, f *ssa.Func, enc abi.Encoding) (code *Code, err error) {
	defer func() {
		if v := recover(); v != nil {
			code, err = nil, fmt.Errorf("%w: %v", ErrUnsupported, v)
		}
	}()
	pools := func(regs ...int) []int { return regs }
	var xmms []int
	for r := amd64.XReg(2); r < xScratch2; r++ {
		xmms = append(xmms, int(r))
	}
	k, err := prepare(w, f, enc, pools(int(amd64.RBX), int(amd64.R8), int(amd64.R9), int(amd64.R10),
		int(amd64.R12), int(amd64.R13), int(amd64.R15)), xmms)
	if err != nil {
		return nil, err
	}
	c := &compiler{core: k}
	if w != nil {
		g := &w.amd64
		if g.stubs == nil {
			g.stubs = map[stubKey]amd64.Label{}
		}
		g.a.Reset()
		clear(g.stubs)
		c.a, c.stubs, c.stubFor, c.cold = &g.a, g.stubs, g.stubFor[:0], g.cold[:0]
		c.labels = slices.Grow(g.labels[:0], numBlocks(f))[:numBlocks(f)]
		defer func() { g.labels, g.stubFor, g.cold = c.labels[:0], c.stubFor[:0], clearFuncs(c.cold) }()
	} else {
		c.a, c.stubs = &amd64.Asm{}, map[stubKey]amd64.Label{}
		c.labels = make([]amd64.Label, numBlocks(f))
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
	bytes, err := c.a.Finish()
	if err != nil {
		return nil, err
	}
	return &Code{Bytes: bytes, Entries: entries, core: c.core}, nil
}

// slotAddr is the base register and displacement of a frame slot's word.
// A captured binding's value is found through its cell, into scratch.
func (c *compiler) slotAddr(slot int, ref bool, scratch amd64.Reg) (amd64.Reg, int32) {
	off := c.enc.NumOffset
	if ref {
		off = c.enc.RefOffset
	}
	switch {
	case slot < c.f.FrameLocals:
		return regLocals, int32(slot)*c.enc.ValueSize + off
	case slot == c.f.ThisSlot:
		return regCtx, abi.OffThis + off
	case slot < c.f.Locals:
		c.a.Load(scratch, regCtx, abi.OffUpvalues)
		c.a.Load(scratch, scratch, int32(slot-c.f.FrameLocals)*8)
		c.a.Load(scratch, scratch, c.enc.UpvalueSlot)
		return scratch, off
	}
	return regStack, int32(slot-c.f.Locals)*c.enc.ValueSize + off
}

// gpr returns a register holding v's word, loading a spilled or lazy value
// into scratch.
func (c *compiler) gpr(v *ssa.Value, scratch amd64.Reg) amd64.Reg {
	if c.isLazy(v) {
		c.materialize(v, scratch)
		return scratch
	}
	l, ok := c.loc(v)
	if !ok {
		panic(fmt.Sprintf("no location for %v (%v)", v, v.Op))
	}
	if l.reg >= 0 {
		return amd64.Reg(l.reg)
	}
	c.a.Load(scratch, regCtx, c.spillDisp(l.spill))
	return scratch
}

// xmm returns an SSE register holding v.
func (c *compiler) xmm(v *ssa.Value, scratch amd64.XReg) amd64.XReg {
	l, ok := c.loc(v)
	if !ok {
		panic(fmt.Sprintf("no location for %v (%v)", v, v.Op))
	}
	if l.reg >= 0 {
		return amd64.XReg(l.reg)
	}
	c.a.LoadSD(scratch, regCtx, c.spillDisp(l.spill))
	return scratch
}

// setG stores src into v's location.
func (c *compiler) setG(v *ssa.Value, src amd64.Reg) {
	l := c.locAt(v)
	if l.reg >= 0 {
		if amd64.Reg(l.reg) != src {
			c.a.MovRR(amd64.Reg(l.reg), src)
		}
		return
	}
	c.a.Store(regCtx, c.spillDisp(l.spill), src)
}

// setX stores src into v's location.
func (c *compiler) setX(v *ssa.Value, src amd64.XReg) {
	l := c.locAt(v)
	if l.reg >= 0 {
		if amd64.XReg(l.reg) != src {
			c.a.SSEOp(amd64.MovAPD, amd64.XReg(l.reg), src)
		}
		return
	}
	c.a.StoreSD(regCtx, c.spillDisp(l.spill), src)
}

// materialize computes a lazy value's word into dst.
func (c *compiler) materialize(v *ssa.Value, dst amd64.Reg) {
	switch v.Op {
	case ssa.OpConst:
		c.a.MovImm(dst, c.constWord(v.Const))
	case ssa.OpConstSource:
		c.a.MovImm(dst, uint64(int64(v.Aux)))
	case ssa.OpBoxF64:
		c.boxF64(c.xmm(v.Args[0], xScratch1), dst)
	case ssa.OpBoxBool:
		c.boxBool(c.gpr(v.Args[0], scratchB), dst)
	default:
		panic("materialize " + v.Op.String())
	}
}

// boxF64 puts x's bits in dst, with NaN canonical.
func (c *compiler) boxF64(x amd64.XReg, dst amd64.Reg) {
	done := c.a.NewLabel()
	c.a.MovQFromX(dst, x)
	c.a.SSEOp(amd64.UcomiSD, x, x)
	c.a.Jcc(amd64.CondNP, done)
	c.a.MovImm(dst, c.enc.CanonicalNaN)
	c.a.Bind(done)
}

// boxBool puts the word for b (0 or 1) in dst. dst may be b.
func (c *compiler) boxBool(b, dst amd64.Reg) {
	tr, done := c.a.NewLabel(), c.a.NewLabel()
	c.a.Op(amd64.Test, b, b, false)
	c.a.Jcc(amd64.CondNE, tr)
	c.a.MovImm(dst, c.enc.False)
	c.a.Jmp(done)
	c.a.Bind(tr)
	c.a.MovImm(dst, c.enc.True)
	c.a.Bind(done)
}

// stubLabel returns the exit stub for a state and kind.
func (c *compiler) stubLabel(s *ssa.FrameState, kind uint64) amd64.Label {
	k := stubKey{s, kind}
	if l, ok := c.stubs[k]; ok {
		return l
	}
	l := c.a.NewLabel()
	c.stubs[k] = l
	c.stubFor = append(c.stubFor, stub{k, l})
	return l
}

// exitTo writes a frame state into the frame and returns with an exit
// record. A slot holding what was loaded from it at this entry is left as
// it is. Every other slot gets its value's word, boxes and constants
// recomputed from their operands -- unless the value is the reference its
// origin slot held, which Go copies, or the slot holds a reference, whose
// pointer word Go clears: those are records (abi.Record). The frame is as
// it was at entry until here, so the origin slot still holds the
// reference, and a slot this stub has written held none. So a value whose
// origin is known only at run time (a shadow) needs no record when it
// turns out to be a primitive, stored where no reference is, or the
// reference the slot itself held.
func (c *compiler) exitTo(s *ssa.FrameState, kind uint64) {
	c.a.MovImm(scratchC, 0)
	c.a.Store(regCtx, abi.OffRecords, scratchC)
	for i, v := range s.Slots {
		if v.Op == ssa.OpLoadSlot && v.Aux == i || c.captured(i) {
			// Unchanged: a captured binding's own value (capturedUnchanged).
			continue
		}
		var w amd64.Reg
		if remat(v) {
			c.materialize(v, scratchA)
			w = scratchA
		} else {
			w = c.gpr(v, scratchA)
		}
		next := c.a.NewLabel()
		if v.Shadow != nil {
			// Which slot it came from is known only at run time: Go looks,
			// unless it is none or this one.
			scalar := c.a.NewLabel()
			from := c.gpr(v.Shadow, scratchB)
			c.a.OpImm(amd64.Cmp, from, -1, true)
			c.a.Jcc(amd64.CondE, scalar)
			c.a.OpImm(amd64.Cmp, from, int32(i), true)
			c.a.Jcc(amd64.CondE, next)
			c.appendRecord(uint64(i)|abi.RecordMaybe, c.gprAfter(v.Shadow), 0, false, &w)
			c.a.Jmp(next)
			c.a.Bind(scalar)
		} else if o, ok := c.origin.Of(v); ok && o >= 0 {
			scalar := c.a.NewLabel()
			c.isReference(v, w, o, scalar)
			if o != i {
				c.appendRecord(uint64(i), nil, uint64(o), true, nil)
			}
			c.a.Jmp(next)
			c.a.Bind(scalar)
		}
		record := c.a.NewLabel()
		base, disp := c.slotAddr(i, true, scratchC)
		c.a.Load(scratchC, base, disp)
		c.a.Op(amd64.Test, scratchC, scratchC, true)
		c.a.Jcc(amd64.CondNE, record)
		base, disp = c.slotAddr(i, false, scratchC)
		c.a.Store(base, disp, w)
		c.a.Jmp(next)
		c.a.Bind(record)
		c.appendRecord(uint64(i)|abi.RecordScalar, nil, 0, true, &w)
		c.a.Bind(next)
	}
	c.record(kind, uint64(s.PC), uint64(s.Depth), uint64(int64(s.Site)))
}

// isReference falls through when v, whose word is in w, is the reference
// slot o held at entry, and jumps to primitive when it is not: when the
// slot held none, or when v is not a load of it and its word is not the
// slot's. Native code makes no word of a reference's kind, so an equal
// word is the slot's value. It uses scratchC.
func (c *compiler) isReference(v *ssa.Value, w amd64.Reg, o int, primitive amd64.Label) {
	base, disp := c.slotAddr(o, true, scratchC)
	c.a.Load(scratchC, base, disp)
	c.a.Op(amd64.Test, scratchC, scratchC, true)
	c.a.Jcc(amd64.CondE, primitive)
	if v.Op != ssa.OpLoadSlot {
		base, disp = c.slotAddr(o, false, scratchC)
		c.a.Load(scratchC, base, disp)
		c.a.Op(amd64.Cmp, w, scratchC, true)
		c.a.Jcc(amd64.CondNE, primitive)
	}
}

// appendRecord adds an abi.Record for slot: its Arg from arg, which
// yields a register once the record's address is in scratchC, or imm when
// useImm; and its Word from word, if not nil, which must not be scratchB
// or scratchC. It uses scratchB and scratchC.
func (c *compiler) appendRecord(slot uint64, arg func() amd64.Reg, imm uint64, useImm bool, word *amd64.Reg) {
	c.a.Load(scratchC, regCtx, abi.OffRecords)
	c.a.ShiftImm(amd64.Shl, scratchC, 5, true)
	c.a.Op(amd64.Add, scratchC, regCtx, true)
	if word != nil {
		c.a.Store(scratchC, abi.OffRecord+16, *word)
	}
	c.a.MovImm(scratchB, slot)
	c.a.Store(scratchC, abi.OffRecord, scratchB)
	if useImm {
		c.a.MovImm(scratchB, imm)
		c.a.Store(scratchC, abi.OffRecord+8, scratchB)
	} else {
		c.a.Store(scratchC, abi.OffRecord+8, arg())
	}
	c.a.Load(scratchC, regCtx, abi.OffRecords)
	c.a.OpImm(amd64.Add, scratchC, 1, true)
	c.a.Store(regCtx, abi.OffRecords, scratchC)
}

// gprAfter returns a function yielding a register holding v, loading it
// into scratchB if it has none; for appendRecord, which has freed scratchB
// by then.
func (c *compiler) gprAfter(v *ssa.Value) func() amd64.Reg {
	return func() amd64.Reg { return c.gpr(v, scratchB) }
}

// record fills the exit record and returns to Go.
func (c *compiler) record(kind, pc, depth, site uint64) {
	c.a.MovImm(scratchA, kind)
	c.a.Store(regCtx, abi.OffExitKind, scratchA)
	c.a.MovImm(scratchA, pc)
	c.a.Store(regCtx, abi.OffExitPC, scratchA)
	c.a.MovImm(scratchA, depth)
	c.a.Store(regCtx, abi.OffExitDepth, scratchA)
	c.a.MovImm(scratchA, site)
	c.a.Store(regCtx, abi.OffExitSite, scratchA)
	c.a.Ret()
}

// block emits one block. next is the block laid out after it.
func (c *compiler) block(b *ssa.Block, next *ssa.Block) {
	if b.PC < 0 {
		c.a.Load(regLocals, regCtx, abi.OffLocals)
		c.a.Load(regStack, regCtx, abi.OffStack)
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
		r := c.gpr(b.Control, scratchA)
		c.a.Store(regCtx, abi.OffRet, r)
		c.a.MovImm(scratchC, 0)
		c.a.Store(regCtx, abi.OffRetFrom, scratchC)
		if s := b.Control.Shadow; s != nil {
			// RetFrom is the source plus one, and 0 for a primitive's -1;
			// Go checks that the slot or cell holds a reference.
			c.a.MovRR(scratchC, c.gpr(s, scratchC))
			c.a.OpImm(amd64.Add, scratchC, 1, true)
			c.a.Store(regCtx, abi.OffRetFrom, scratchC)
		} else if o, ok := c.origin.Of(b.Control); ok && o >= 0 {
			done := c.a.NewLabel()
			c.isReference(b.Control, r, o, done)
			c.a.MovImm(scratchC, uint64(o)+1)
			c.a.Store(regCtx, abi.OffRetFrom, scratchC)
			c.a.Bind(done)
		}
		c.a.MovImm(scratchA, abi.ExitReturn)
		c.a.Store(regCtx, abi.OffExitKind, scratchA)
		c.a.Ret()
	case ssa.BlockExit:
		c.exitTo(b.State, exitKind(int(b.ExitKind)))
	}
}

// edge emits the moves for an edge's phis, a poll on a back-edge, and a jump
// unless the target is laid out next.
func (c *compiler) edge(from, to, next *ssa.Block) {
	idx := -1
	for i, p := range to.Preds {
		if p == from {
			idx = i
		}
	}
	c.phiMoves(to, idx)
	if to.LoopHeader && to.Backedge[idx] {
		poll := c.a.NewLabel()
		c.a.Load(scratchB, regCtx, abi.OffBackEdges)
		c.a.DecMem(scratchB, 0)
		c.a.Jcc(amd64.CondLE, poll)
		header := to.Header
		c.cold = append(c.cold, func() {
			c.a.Bind(poll)
			// The phis hold the header's values now; the state names them,
			// or values live across the loop.
			c.exitTo(header, abi.ExitPoll)
		})
	}
	if to != next {
		c.a.Jmp(c.labels[to.ID])
	}
}

// phiMoves performs the parallel move of an edge's phi arguments into the
// phis' locations, as the core schedules it.
func (c *compiler) phiMoves(to *ssa.Block, idx int) {
	for _, st := range c.phiSchedule(to, idx) {
		switch {
		case st.park == nil:
			c.moveValue(st.dst, st.src, st.parked)
		case isFloat(st.park):
			c.a.SSEOp(amd64.MovAPD, xScratch0, c.xmm(st.park, xScratch0))
		default:
			c.a.MovRR(scratchC, c.gpr(st.park, scratchC))
		}
	}
}

// moveValue copies src into dst's location; parked sources are in scratch.
func (c *compiler) moveValue(dst, src *ssa.Value, parked bool) {
	if isFloat(dst) {
		// A spilled source loads through xScratch1: xScratch0 may hold a
		// parked value still to be moved.
		x := xScratch0
		if !parked {
			x = c.xmm(src, xScratch1)
		}
		c.setX(dst, x)
		return
	}
	r := scratchC
	if !parked {
		r = c.gpr(src, scratchA)
	}
	c.setG(dst, r)
}

// branch emits an If block's end, fusing a float comparison into the jump.
func (c *compiler) branch(b *ssa.Block, next *ssa.Block) {
	yes, no := c.a.NewLabel(), c.a.NewLabel()
	ctl := b.Control
	if ctl.Op == ssa.OpCmpF64 && ctl.Uses == 1 && ctl.Block == b {
		x, y := c.xmm(ctl.Args[0], xScratch0), c.xmm(ctl.Args[1], xScratch1)
		switch ir.Operator(ctl.Aux) {
		case ir.Lt:
			c.a.SSEOp(amd64.UcomiSD, y, x)
			c.a.Jcc(amd64.CondA, yes)
		case ir.Le:
			c.a.SSEOp(amd64.UcomiSD, y, x)
			c.a.Jcc(amd64.CondAE, yes)
		case ir.Gt:
			c.a.SSEOp(amd64.UcomiSD, x, y)
			c.a.Jcc(amd64.CondA, yes)
		case ir.Ge:
			c.a.SSEOp(amd64.UcomiSD, x, y)
			c.a.Jcc(amd64.CondAE, yes)
		case ir.Eq:
			c.a.SSEOp(amd64.UcomiSD, x, y)
			c.a.Jcc(amd64.CondNE, no)
			c.a.Jcc(amd64.CondNP, yes)
		case ir.Ne:
			c.a.SSEOp(amd64.UcomiSD, x, y)
			c.a.Jcc(amd64.CondNE, yes)
			c.a.Jcc(amd64.CondP, yes)
		}
	} else {
		r := c.gpr(ctl, scratchA)
		c.a.Op(amd64.Test, r, r, false)
		c.a.Jcc(amd64.CondNE, yes)
	}
	// The false edge follows.
	c.a.Bind(no)
	c.edge(b, b.Succs[1], nil)
	c.a.Bind(yes)
	c.edge(b, b.Succs[0], next)
}

// arrayOf finds the array a value is, as objectOf finds an object, and
// checks its class.
func (c *compiler) arrayOf(v *ssa.Value, guard func(amd64.Cond)) {
	if !c.objectOf(v, guard) {
		return
	}
	c.a.LoadU8(scratchB, scratchC, c.enc.ObjectClass)
	c.a.OpImm(amd64.Cmp, scratchB, int32(c.enc.ClassArray), false)
	guard(amd64.CondNE)
	c.setG(v, scratchC)
}

// objectOf finds the object a value is, into scratchC: the pointer word of
// the slot it came from, which still holds it (origin.go). A value with an
// object's word is its origin's object, since native code makes no such
// word. It reports false when the value can never be one, having emitted
// the jump to the exit.
func (c *compiler) objectOf(v *ssa.Value, guard func(amd64.Cond)) bool {
	return c.reference(v, v.Args[0], c.enc.Object, guard)
}

// reference finds the reference a is, of the kind whose word is word, into
// scratchC, as objectOf finds an object, exiting to v's state when a is not
// one. It uses scratchA and scratchB.
func (c *compiler) reference(v, a *ssa.Value, word uint64, guard func(amd64.Cond)) bool {
	o := c.origin.At(a)
	if a.Shadow == nil && o < 0 {
		// A primitive is never a reference.
		c.a.Jmp(c.stubLabel(v.State, exitKind(v.Aux)))
		return false
	}
	w := c.gpr(a, scratchA)
	c.a.MovImm(scratchB, word)
	c.a.Op(amd64.Cmp, w, scratchB, true)
	guard(amd64.CondNE)
	if s := a.Shadow; s != nil {
		// The source is known at run time: a local, a captured binding, the
		// receiver, an operand, or a heap cell (origin.go). scratchC becomes
		// its value's address.
		captured, stack, found := c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel()
		c.a.MovRR(scratchC, c.gpr(s, scratchC))
		c.a.Op(amd64.Test, scratchC, scratchC, true)
		guard(amd64.CondS)
		c.a.OpImm(amd64.Cmp, scratchC, abi.MaxRecords, true)
		c.a.Jcc(amd64.CondAE, found)
		c.a.OpImm(amd64.Cmp, scratchC, int32(c.f.FrameLocals), false)
		c.a.Jcc(amd64.CondGE, captured)
		c.a.ShiftImm(amd64.Shl, scratchC, 4, true)
		c.a.Op(amd64.Add, scratchC, regLocals, true)
		c.a.Jmp(found)
		c.a.Bind(captured)
		c.a.OpImm(amd64.Cmp, scratchC, int32(c.f.Locals), false)
		c.a.Jcc(amd64.CondGE, stack)
		if t := c.f.ThisSlot; t >= 0 {
			notThis := c.a.NewLabel()
			c.a.OpImm(amd64.Cmp, scratchC, int32(t), false)
			c.a.Jcc(amd64.CondNE, notThis)
			c.a.MovRR(scratchC, regCtx)
			c.a.OpImm(amd64.Add, scratchC, abi.OffThis, true)
			c.a.Jmp(found)
			c.a.Bind(notThis)
		}
		c.a.OpImm(amd64.Sub, scratchC, int32(c.f.FrameLocals), false)
		c.a.ShiftImm(amd64.Shl, scratchC, 3, true)
		c.a.Load(scratchB, regCtx, abi.OffUpvalues)
		c.a.Op(amd64.Add, scratchC, scratchB, true)
		c.a.Load(scratchC, scratchC, 0)
		c.a.Load(scratchC, scratchC, c.enc.UpvalueSlot)
		c.a.Jmp(found)
		c.a.Bind(stack)
		c.a.OpImm(amd64.Sub, scratchC, int32(c.f.Locals), false)
		c.a.ShiftImm(amd64.Shl, scratchC, 4, true)
		c.a.Op(amd64.Add, scratchC, regStack, true)
		c.a.Bind(found)
		c.a.Load(scratchC, scratchC, c.enc.RefOffset)
	} else {
		base, disp := c.slotAddr(o, true, scratchC)
		c.a.Load(scratchC, base, disp)
	}
	c.a.Op(amd64.Test, scratchC, scratchC, true)
	guard(amd64.CondE)
	return true
}

// property finds the property a property operation names and leaves the
// address of its value in scratchA. An object of the shape the site's cache
// knows has it at the cached index: the shape settles where it is and what
// it is. Any other ordinary object with a small table is searched for the
// key, unrolled, as the VM's own small objects are; the entry must be plain
// data, and writable for a write. It uses scratchB and scratchC.
func (c *compiler) property(v *ssa.Value, guard func(amd64.Cond)) {
	if v.Holders != nil {
		c.holder(v, guard)
		return
	}
	found, scan := c.a.NewLabel(), c.a.NewLabel()
	if v.Const.Bits != 0 {
		p := c.gpr(v.Args[0], scratchA)
		c.a.Load(scratchB, p, c.enc.ObjectShape)
		c.a.MovImm(scratchA, v.Const.Bits)
		c.a.Op(amd64.Cmp, scratchB, scratchA, true)
		c.a.Jcc(amd64.CondNE, scan)
		p = c.gpr(v.Args[0], scratchA)
		c.a.Load(scratchA, p, c.enc.ObjectProps)
		c.a.OpImm(amd64.Add, scratchA, int32(v.Index)*c.enc.PropertySize+c.enc.PropertyValue, true)
		c.a.Jmp(found)
	}
	c.a.Bind(scan)
	p := c.gpr(v.Args[0], scratchA)
	c.a.LoadU8(scratchB, p, c.enc.ObjectClass)
	c.a.OpImm(amd64.Cmp, scratchB, int32(c.enc.ClassObject), false)
	guard(amd64.CondNE)
	c.a.Load(scratchB, p, c.enc.ObjectProps+8)
	c.a.OpImm(amd64.Cmp, scratchB, abi.MaxScan, true)
	guard(amd64.CondA)
	c.a.Load(scratchA, p, c.enc.ObjectProps)
	mask, want := int32(c.enc.PropNotData), int32(0)
	if v.Op == ssa.OpPropWrite {
		mask, want = int32(c.enc.PropNotWritable), int32(c.enc.PropWritable)
	}
	for k := int32(0); k < abi.MaxScan; k++ {
		next := c.a.NewLabel()
		entry := k * c.enc.PropertySize
		c.a.OpImm(amd64.Cmp, scratchB, k, true)
		guard(amd64.CondBE)
		c.a.LoadU32(scratchC, scratchA, entry+c.enc.PropertyKey)
		c.a.OpImm(amd64.Cmp, scratchC, int32(v.Key), false)
		c.a.Jcc(amd64.CondNE, next)
		c.a.LoadU8(scratchC, scratchA, entry+c.enc.PropertyFlags)
		c.a.OpImm(amd64.And, scratchC, mask, false)
		c.a.OpImm(amd64.Cmp, scratchC, want, false)
		guard(amd64.CondNE)
		c.a.OpImm(amd64.Add, scratchA, entry+c.enc.PropertyValue, true)
		c.a.Jmp(found)
		c.a.Bind(next)
	}
	c.a.Jmp(c.stubLabel(v.State, exitKind(v.Aux)))
	c.a.Bind(found)
}

// holder is property for a read the receiver's prototypes answered
// (ssa.PropertySite's Holders): the receiver of the shape the site knows,
// which says it has no such property of its own, each prototype the one
// the site met, of the shape it had, and the property at the index in the
// last one's table; anything else exits. The prototypes' addresses are
// constants, which the VM keeps alive.
func (c *compiler) holder(v *ssa.Value, guard func(amd64.Cond)) {
	p := c.gpr(v.Args[0], scratchA)
	c.a.Load(scratchB, p, c.enc.ObjectShape)
	c.a.MovImm(scratchC, v.Const.Bits)
	c.a.Op(amd64.Cmp, scratchB, scratchC, true)
	guard(amd64.CondNE)
	c.a.Load(scratchB, p, c.enc.ObjectProto)
	for _, h := range v.Holders {
		if h.Object == 0 {
			break
		}
		c.a.MovImm(scratchA, uint64(h.Object))
		c.a.Op(amd64.Cmp, scratchB, scratchA, true)
		guard(amd64.CondNE)
		c.a.Load(scratchB, scratchA, c.enc.ObjectShape)
		c.a.MovImm(scratchC, uint64(h.Shape))
		c.a.Op(amd64.Cmp, scratchB, scratchC, true)
		guard(amd64.CondNE)
		c.a.Load(scratchB, scratchA, c.enc.ObjectProto)
	}
	c.a.Load(scratchA, scratchA, c.enc.ObjectProps)
	c.a.OpImm(amd64.Add, scratchA, int32(v.Index)*c.enc.PropertySize+c.enc.PropertyValue, true)
}

// remainder is JavaScript's % of two integers, by integer division: both
// exact integers below 2**63 in magnitude, the divisor not zero. A zero
// remainder takes the dividend's sign, -0 included, as fmod's does; a
// divisor of -1 leaves exactly that without dividing, which could fault.
// Anything else exits to Go, which computes it as math.Mod does: SSE has
// no remainder, and Go computes one in software.
func (c *compiler) remainder(v *ssa.Value, guard func(amd64.Cond)) {
	if x := c.xmm(v.Args[0], xScratch0); x != xScratch0 {
		c.a.SSEOp(amd64.MovAPD, xScratch0, x)
	}
	if y := c.xmm(v.Args[1], xScratch1); y != xScratch1 {
		c.a.SSEOp(amd64.MovAPD, xScratch1, y)
	}
	for _, p := range []struct {
		r amd64.Reg
		x amd64.XReg
	}{{scratchA, xScratch0}, {scratchC, xScratch1}} {
		c.a.Cvttsd2si(p.r, p.x)
		c.a.Cvtsi2sd(xScratch2, p.r, true)
		c.a.SSEOp(amd64.UcomiSD, xScratch2, p.x)
		guard(amd64.CondNE)
		guard(amd64.CondP)
	}
	c.a.Op(amd64.Test, scratchC, scratchC, true)
	guard(amd64.CondE)
	zero, done := c.a.NewLabel(), c.a.NewLabel()
	c.a.OpImm(amd64.Cmp, scratchC, -1, true)
	c.a.Jcc(amd64.CondE, zero)
	// IDIV takes RDX, the operands' base, which is reloaded after it.
	c.a.Cqo()
	c.a.Idiv(scratchC)
	c.a.MovRR(scratchB, regStack)
	c.a.Load(regStack, regCtx, abi.OffStack)
	c.a.Op(amd64.Test, scratchB, scratchB, true)
	c.a.Jcc(amd64.CondE, zero)
	c.a.Cvtsi2sd(xScratch0, scratchB, true)
	c.a.Jmp(done)
	c.a.Bind(zero)
	c.a.MovQFromX(scratchA, xScratch0)
	c.a.MovImm(scratchB, 1<<63)
	c.a.Op(amd64.And, scratchA, scratchB, true)
	c.a.MovQToX(xScratch0, scratchA)
	c.a.Bind(done)
	c.setX(v, xScratch0)
}

// length is x.length: a string's, which it keeps rope or not, or an
// array's, the dense count or a sparse array's length when that is larger,
// as Object.arrayLength has it.
func (c *compiler) length(v *ssa.Value, guard func(amd64.Cond)) {
	a := v.Args[0]
	if a.Shadow == nil && c.origin.At(a) < 0 {
		c.a.Jmp(c.stubLabel(v.State, exitKind(v.Aux)))
		return
	}
	array, have := c.a.NewLabel(), c.a.NewLabel()
	c.a.MovImm(scratchB, c.enc.String)
	c.a.Op(amd64.Cmp, c.gpr(a, scratchA), scratchB, true)
	c.a.Jcc(amd64.CondNE, array)
	c.reference(v, a, c.enc.String, guard)
	c.a.Load(scratchC, scratchC, c.enc.StringLength)
	c.a.Jmp(have)
	c.a.Bind(array)
	c.reference(v, a, c.enc.Object, guard)
	c.a.LoadU8(scratchB, scratchC, c.enc.ObjectClass)
	c.a.OpImm(amd64.Cmp, scratchB, int32(c.enc.ClassArray), false)
	guard(amd64.CondNE)
	p := scratchA
	c.a.MovRR(p, scratchC)
	c.a.Load(scratchC, p, c.enc.ObjectElems+8)
	c.a.LoadU8(scratchB, p, c.enc.ObjectFlags)
	c.a.OpImm(amd64.And, scratchB, int32(c.enc.FlagSparse), false)
	c.a.Jcc(amd64.CondE, have)
	c.a.LoadU32(scratchB, p, c.enc.ObjectArrayLen)
	c.a.Op(amd64.Cmp, scratchB, scratchC, true)
	c.a.Jcc(amd64.CondBE, have)
	c.a.MovRR(scratchC, scratchB)
	c.a.Bind(have)
	c.a.Cvtsi2sd(xScratch0, scratchC, true)
	c.setX(v, xScratch0)
}

// stringCode is charCodeAt called on a string: the callee must be the
// intrinsic, the string flat, and the index an integer below its length;
// the code unit is a byte of an ASCII string's UTF-8, or one of the code
// units a string caches, and anything else exits.
func (c *compiler) stringCode(v *ssa.Value, guard func(amd64.Cond)) {
	if !c.reference(v, v.Args[0], c.enc.Object, guard) {
		return
	}
	c.a.Load(scratchB, regCtx, abi.OffCharCode+c.enc.RefOffset)
	c.a.Op(amd64.Cmp, scratchC, scratchB, true)
	guard(amd64.CondNE)
	if !c.reference(v, v.Args[1], c.enc.String, guard) {
		return
	}
	c.a.Load(scratchB, scratchC, c.enc.StringLeft)
	c.a.Op(amd64.Test, scratchB, scratchB, true)
	guard(amd64.CondNE)
	c.index(v.Args[2], guard)
	c.a.Load(scratchB, scratchC, c.enc.StringLength)
	c.a.Op(amd64.Cmp, scratchA, scratchB, true)
	guard(amd64.CondAE)
	units, done := c.a.NewLabel(), c.a.NewLabel()
	c.a.LoadU8(scratchB, scratchC, c.enc.StringASCII)
	c.a.Op(amd64.Test, scratchB, scratchB, false)
	c.a.Jcc(amd64.CondE, units)
	c.a.Load(scratchB, scratchC, c.enc.StringData)
	c.a.Op(amd64.Add, scratchB, scratchA, true)
	c.a.LoadU8(scratchA, scratchB, 0)
	c.a.Jmp(done)
	c.a.Bind(units)
	c.a.Load(scratchB, scratchC, c.enc.StringU16)
	c.a.Op(amd64.Test, scratchB, scratchB, true)
	guard(amd64.CondE)
	c.a.Op(amd64.Add, scratchA, scratchA, true)
	c.a.Op(amd64.Add, scratchB, scratchA, true)
	c.a.LoadU16(scratchA, scratchB, 0)
	c.a.Bind(done)
	c.a.Cvtsi2sd(xScratch0, scratchA, true)
	c.setX(v, xScratch0)
}

// index converts an element's key, a double, to an index in scratchA,
// failing unless it is an integer in [0, 2**32): truncation and back must
// give the key (NaN and anything past 2**63 do not), and the top half must
// be clear. Negative zero is index 0. It uses scratchB and xScratch1.
func (c *compiler) index(key *ssa.Value, fail func(amd64.Cond)) {
	k := c.xmm(key, xScratch0)
	c.a.Cvttsd2si(scratchA, k)
	c.a.Cvtsi2sd(xScratch1, scratchA, true)
	c.a.SSEOp(amd64.UcomiSD, xScratch1, k)
	fail(amd64.CondNE)
	fail(amd64.CondP)
	c.a.MovRR(scratchB, scratchA)
	c.a.ShiftImm(amd64.Shr, scratchB, 32, true)
	fail(amd64.CondNE)
}

// element turns the index in scratchA into the address of an array's
// element there, in scratchA, and its number word, in scratchB, failing
// unless the element is present and holds a number. It uses scratchC.
func (c *compiler) element(array *ssa.Value, fail func(amd64.Cond)) {
	p := c.gpr(array, scratchC)
	c.a.Load(scratchB, p, c.enc.ObjectElems+8)
	c.a.Op(amd64.Cmp, scratchA, scratchB, true)
	fail(amd64.CondAE)
	c.a.Load(scratchB, p, c.enc.ObjectElems)
	c.a.ShiftImm(amd64.Shl, scratchA, 4, true)
	c.a.Op(amd64.Add, scratchA, scratchB, true)
	c.a.Load(scratchB, scratchA, c.enc.NumOffset)
	c.a.MovImm(scratchC, abi.NumberLimit)
	c.a.Op(amd64.Cmp, scratchB, scratchC, true)
	fail(amd64.CondAE)
}

// elementCell turns the index in scratchA into the address of an array's
// element there, in scratchA, failing unless it is within the dense
// elements and not a hole (whose word is the uninitialized marker's). It
// uses scratchB and scratchC.
func (c *compiler) elementCell(array *ssa.Value, fail func(amd64.Cond)) {
	p := c.gpr(array, scratchC)
	c.a.Load(scratchB, p, c.enc.ObjectElems+8)
	c.a.Op(amd64.Cmp, scratchA, scratchB, true)
	fail(amd64.CondAE)
	c.a.Load(scratchB, p, c.enc.ObjectElems)
	c.a.ShiftImm(amd64.Shl, scratchA, 4, true)
	c.a.Op(amd64.Add, scratchA, scratchB, true)
	c.a.Load(scratchB, scratchA, c.enc.NumOffset)
	c.a.MovImm(scratchC, c.enc.Uninitialized)
	c.a.Op(amd64.Cmp, scratchB, scratchC, true)
	fail(amd64.CondE)
}

// value emits one value.
func (c *compiler) value(v *ssa.Value, b *ssa.Block) {
	arg := func(i int) *ssa.Value { return v.Args[i] }
	guard := func(cond amd64.Cond) {
		c.a.Jcc(cond, c.stubLabel(v.State, exitKind(v.Aux)))
	}
	switch v.Op {
	case ssa.OpLoadSlot:
		base, disp := c.slotAddr(v.Aux, false, scratchA)
		c.a.Load(scratchA, base, disp)
		c.setG(v, scratchA)
	case ssa.OpConst:
		c.a.MovImm(scratchA, c.constWord(v.Const))
		c.setG(v, scratchA)
	case ssa.OpConstF64:
		c.a.MovImm(scratchA, v.Const.Bits)
		c.a.MovQToX(xScratch0, scratchA)
		c.setX(v, xScratch0)
	case ssa.OpUnboxF64:
		r := c.gpr(arg(0), scratchB)
		c.a.MovRR(scratchA, r)
		c.a.ShiftImm(amd64.Shr, scratchA, 51, true)
		c.a.OpImm(amd64.Cmp, scratchA, 0x1FFF, false)
		guard(amd64.CondE)
		c.a.MovQToX(xScratch0, r)
		c.setX(v, xScratch0)
	case ssa.OpCheckInit:
		r := c.gpr(arg(0), scratchB)
		c.a.MovImm(scratchA, c.enc.Uninitialized)
		c.a.Op(amd64.Cmp, r, scratchA, true)
		guard(amd64.CondE)
	case ssa.OpTruth:
		c.truth(v, guard)
	case ssa.OpArrayOf:
		c.arrayOf(v, guard)
	case ssa.OpObjectOf:
		if c.objectOf(v, guard) {
			c.setG(v, scratchC)
		}
	case ssa.OpPropRead:
		c.property(v, guard)
		c.a.Load(scratchB, scratchA, c.enc.NumOffset)
		c.a.MovImm(scratchC, abi.NumberLimit)
		c.a.Op(amd64.Cmp, scratchB, scratchC, true)
		guard(amd64.CondAE)
		c.a.MovQToX(xScratch0, scratchB)
		c.setX(v, xScratch0)
	case ssa.OpPropCell:
		c.property(v, guard)
		c.setG(v, scratchA)
	case ssa.OpGlobalCell:
		// No script-level lexical binding of the name, which would shadow
		// it: its bit clear, or past the set's words. Then the binding
		// where it was, the name's, and plain, initialized data.
		unshadowed := c.a.NewLabel()
		word := int32(v.Key >> 6)
		c.a.Load(scratchB, regCtx, abi.OffLexNames)
		c.a.Load(scratchA, scratchB, 8)
		c.a.OpImm(amd64.Cmp, scratchA, word, true)
		c.a.Jcc(amd64.CondBE, unshadowed)
		c.a.Load(scratchB, scratchB, 0)
		c.a.Load(scratchB, scratchB, word*8)
		c.a.MovImm(scratchA, 1<<(v.Key&63))
		c.a.Op(amd64.Test, scratchB, scratchA, true)
		guard(amd64.CondNE)
		c.a.Bind(unshadowed)
		c.a.Load(scratchA, regCtx, abi.OffGlobal)
		c.a.Op(amd64.Test, scratchA, scratchA, true)
		guard(amd64.CondE)
		c.a.Load(scratchB, scratchA, c.enc.ObjectProps+8)
		c.a.OpImm(amd64.Cmp, scratchB, int32(v.Index), true)
		guard(amd64.CondBE)
		c.a.Load(scratchA, scratchA, c.enc.ObjectProps)
		c.a.OpImm(amd64.Add, scratchA, int32(v.Index)*c.enc.PropertySize, true)
		c.a.LoadU32(scratchB, scratchA, c.enc.PropertyKey)
		c.a.OpImm(amd64.Cmp, scratchB, int32(v.Key), false)
		guard(amd64.CondNE)
		c.a.LoadU8(scratchB, scratchA, c.enc.PropertyFlags)
		c.a.OpImm(amd64.And, scratchB, int32(c.enc.PropNotData|c.enc.PropUninit), false)
		guard(amd64.CondNE)
		c.a.OpImm(amd64.Add, scratchA, c.enc.PropertyValue, true)
		c.setG(v, scratchA)
	case ssa.OpStringMethod:
		// A string's charCodeAt is the context's cell, if it holds the
		// intrinsic.
		c.a.MovImm(scratchB, c.enc.String)
		c.a.Op(amd64.Cmp, c.gpr(arg(0), scratchA), scratchB, true)
		guard(amd64.CondNE)
		c.a.Load(scratchB, regCtx, abi.OffCharCode+c.enc.RefOffset)
		c.a.Op(amd64.Test, scratchB, scratchB, true)
		guard(amd64.CondE)
		c.a.MovRR(scratchA, regCtx)
		c.a.OpImm(amd64.Add, scratchA, abi.OffCharCode, true)
		c.setG(v, scratchA)
	case ssa.OpStringCode:
		c.stringCode(v, guard)
	case ssa.OpLoadCell:
		c.a.Load(scratchA, c.gpr(arg(0), scratchA), c.enc.NumOffset)
		c.setG(v, scratchA)
	case ssa.OpPropWrite:
		c.property(v, guard)
		c.a.Load(scratchB, scratchA, c.enc.NumOffset)
		c.a.MovImm(scratchC, abi.NumberLimit)
		c.a.Op(amd64.Cmp, scratchB, scratchC, true)
		guard(amd64.CondAE)
		c.boxF64(c.xmm(arg(1), xScratch0), scratchB)
		c.a.Store(scratchA, c.enc.NumOffset, scratchB)
	case ssa.OpLength:
		c.length(v, guard)
	case ssa.OpElemKey:

		c.index(arg(0), guard)
	case ssa.OpElemRead:
		c.index(arg(1), guard)
		c.element(arg(0), guard)
		c.a.MovQToX(xScratch0, scratchB)
		c.setX(v, xScratch0)
	case ssa.OpElemCell:
		c.index(arg(1), guard)
		c.elementCell(arg(0), guard)
		c.setG(v, scratchA)
	case ssa.OpElemWrite:
		c.index(arg(1), guard)
		c.element(arg(0), guard)
		c.boxF64(c.xmm(arg(2), xScratch0), scratchB)
		c.a.Store(scratchA, c.enc.NumOffset, scratchB)
	case ssa.OpBoxF64:
		c.boxF64(c.xmm(arg(0), xScratch0), scratchA)
		c.setG(v, scratchA)
	case ssa.OpBoxBool:
		c.boxBool(c.gpr(arg(0), scratchB), scratchA)
		c.setG(v, scratchA)
	case ssa.OpAddF64, ssa.OpSubF64, ssa.OpMulF64, ssa.OpDivF64:
		// SSE's form is d op= y: d is v's own register, unless v is spilled
		// or that register is y's, which d's copy of x would overwrite.
		x, y := c.xmm(arg(0), xScratch0), c.xmm(arg(1), xScratch1)
		d := xScratch0
		if l := c.locAt(v); l.reg >= 0 && (amd64.XReg(l.reg) != y || y == x) {
			d = amd64.XReg(l.reg)
		}
		if x != d {
			c.a.SSEOp(amd64.MovAPD, d, x)
		}
		c.a.SSEOp(sseOp(v.Op), d, y)
		c.setX(v, d)
	case ssa.OpModF64:
		c.remainder(v, guard)
	case ssa.OpNegF64:
		x := c.xmm(arg(0), xScratch0)
		if x != xScratch0 {
			c.a.SSEOp(amd64.MovAPD, xScratch0, x)
		}
		c.a.MovImm(scratchA, 1<<63)
		c.a.MovQToX(xScratch1, scratchA)
		c.a.SSEOp(amd64.XorPD, xScratch0, xScratch1)
		c.setX(v, xScratch0)
	case ssa.OpCmpF64:
		if v.Uses == 1 && b.Control == v && b.Kind == ssa.BlockIf {
			return // fused into the branch
		}
		c.compare(v)
	case ssa.OpStrictNullish:
		word := c.enc.Null
		if v.Aux == 1 {
			word = c.enc.Undefined
		}
		w := c.gpr(arg(0), scratchB)
		c.a.MovImm(scratchA, word)
		c.a.Op(amd64.Cmp, w, scratchA, true)
		c.a.Setcc(amd64.CondE, scratchA)
		c.a.MovZX8(scratchA, scratchA)
		c.setG(v, scratchA)
	case ssa.OpLooseNullish:
		c.looseNullish(v, guard)
	case ssa.OpEqTagged:
		c.eqTagged(v, guard)
	case ssa.OpNot:
		c.a.MovRR32(scratchA, c.gpr(arg(0), scratchA))
		c.a.OpImm(amd64.Xor, scratchA, 1, false)
		c.setG(v, scratchA)
	case ssa.OpToInt32:
		c.toInt32(v)
	case ssa.OpAndI32, ssa.OpOrI32, ssa.OpXorI32:
		op := aluOp(v.Op)
		c.a.MovRR32(scratchA, c.gpr(arg(0), scratchA))
		c.a.Op(op, scratchA, c.gpr(arg(1), scratchB), false)
		c.setG(v, scratchA)
	case ssa.OpShlI32, ssa.OpSarI32, ssa.OpShrU32:
		op := shiftOp(v.Op)
		c.a.MovRR32(scratchC, c.gpr(arg(1), scratchC))
		c.a.MovRR32(scratchA, c.gpr(arg(0), scratchA))
		c.a.ShiftCL(op, scratchA, false)
		c.setG(v, scratchA)
	case ssa.OpNotI32:
		c.a.MovRR32(scratchA, c.gpr(arg(0), scratchA))
		c.a.Not32(scratchA)
		c.setG(v, scratchA)
	case ssa.OpI32ToF64:
		c.a.Cvtsi2sd(xScratch0, c.gpr(arg(0), scratchA), false)
		c.setX(v, xScratch0)
	case ssa.OpU32ToF64:
		c.a.MovRR32(scratchA, c.gpr(arg(0), scratchA))
		c.a.Cvtsi2sd(xScratch0, scratchA, true)
		c.setX(v, xScratch0)
	default:
		panic("value " + v.Op.String())
	}
}

// compare materializes a float comparison as 0 or 1.
func (c *compiler) compare(v *ssa.Value) {
	x, y := c.xmm(v.Args[0], xScratch0), c.xmm(v.Args[1], xScratch1)
	switch ir.Operator(v.Aux) {
	case ir.Lt:
		c.a.SSEOp(amd64.UcomiSD, y, x)
		c.a.Setcc(amd64.CondA, scratchA)
	case ir.Le:
		c.a.SSEOp(amd64.UcomiSD, y, x)
		c.a.Setcc(amd64.CondAE, scratchA)
	case ir.Gt:
		c.a.SSEOp(amd64.UcomiSD, x, y)
		c.a.Setcc(amd64.CondA, scratchA)
	case ir.Ge:
		c.a.SSEOp(amd64.UcomiSD, x, y)
		c.a.Setcc(amd64.CondAE, scratchA)
	case ir.Eq:
		c.a.SSEOp(amd64.UcomiSD, x, y)
		c.a.Setcc(amd64.CondE, scratchA)
		c.a.Setcc(amd64.CondNP, scratchC)
		c.a.Op(amd64.And, scratchA, scratchC, false)
	case ir.Ne:
		c.a.SSEOp(amd64.UcomiSD, x, y)
		c.a.Setcc(amd64.CondNE, scratchA)
		c.a.Setcc(amd64.CondP, scratchC)
		c.a.Op(amd64.Or, scratchA, scratchC, false)
	}
	c.a.MovZX8(scratchA, scratchA)
	c.setG(v, scratchA)
}

// looseNullish is x == null: x's word is null's or undefined's, or x is an
// object with [[IsHTMLDDA]], found through its origin as objectOf finds it.
func (c *compiler) looseNullish(v *ssa.Value, guard func(amd64.Cond)) {
	a := v.Args[0]
	yes, no, done := c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel()
	w := c.gpr(a, scratchB)
	for _, word := range []uint64{c.enc.Null, c.enc.Undefined} {
		c.a.MovImm(scratchA, word)
		c.a.Op(amd64.Cmp, w, scratchA, true)
		c.a.Jcc(amd64.CondE, yes)
	}
	c.a.MovImm(scratchA, c.enc.Object)
	c.a.Op(amd64.Cmp, w, scratchA, true)
	c.a.Jcc(amd64.CondNE, no)
	if c.reference(v, a, c.enc.Object, guard) {
		c.a.LoadU8(scratchA, scratchC, c.enc.ObjectFlags)
		c.a.OpImm(amd64.And, scratchA, int32(c.enc.FlagHTMLDDA), false)
		c.a.Jcc(amd64.CondNE, yes)
	}
	c.a.Bind(no)
	c.a.MovImm(scratchA, 0)
	c.a.Jmp(done)
	c.a.Bind(yes)
	c.a.MovImm(scratchA, 1)
	c.a.Bind(done)
	c.setG(v, scratchA)
}

// eqTagged is OpEqTagged: x == y or x === y by the two words, and for two
// objects by their pointers, found as objectOf finds an object's.
func (c *compiler) eqTagged(v *ssa.Value, guard func(amd64.Cond)) {
	x, y := v.Args[0], v.Args[1]
	yes, no, done, differ, number := c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel()
	exit := c.stubLabel(v.State, exitKind(v.Aux))
	wx, wy := c.gpr(x, scratchA), c.gpr(y, scratchB)
	c.a.Op(amd64.Cmp, wx, wy, true)
	c.a.Jcc(amd64.CondNE, differ)
	// One word: a number (NaN unequal to itself), a primitive, or objects.
	c.a.MovImm(scratchC, abi.NumberLimit)
	c.a.Op(amd64.Cmp, wx, scratchC, true)
	c.a.Jcc(amd64.CondB, number)
	for _, w := range []uint64{c.enc.Null, c.enc.Undefined, c.enc.True, c.enc.False} {
		c.a.MovImm(scratchC, w)
		c.a.Op(amd64.Cmp, wx, scratchC, true)
		c.a.Jcc(amd64.CondE, yes)
	}
	strings := c.a.NewLabel()
	c.a.MovImm(scratchC, c.enc.String)
	c.a.Op(amd64.Cmp, wx, scratchC, true)
	c.a.Jcc(amd64.CondE, strings)
	c.a.MovImm(scratchC, c.enc.Object)
	c.a.Op(amd64.Cmp, wx, scratchC, true)
	c.a.Jcc(amd64.CondNE, exit)
	if c.reference(v, x, c.enc.Object, guard) {
		c.a.MovQToX(xScratch1, scratchC)
		if c.reference(v, y, c.enc.Object, guard) {
			c.a.MovQFromX(scratchA, xScratch1)
			c.a.Op(amd64.Cmp, scratchA, scratchC, true)
			c.a.Jcc(amd64.CondE, yes)
			c.a.Jmp(no)
		}
	}
	// Two strings: one string, or of different lengths; Go compares the
	// rest.
	c.a.Bind(strings)
	if c.reference(v, x, c.enc.String, guard) {
		c.a.MovQToX(xScratch1, scratchC)
		if c.reference(v, y, c.enc.String, guard) {
			c.a.MovQFromX(scratchA, xScratch1)
			c.a.Op(amd64.Cmp, scratchA, scratchC, true)
			c.a.Jcc(amd64.CondE, yes)
			c.a.Load(scratchA, scratchA, c.enc.StringLength)
			c.a.Load(scratchB, scratchC, c.enc.StringLength)
			c.a.Op(amd64.Cmp, scratchA, scratchB, true)
			c.a.Jcc(amd64.CondNE, no)
			c.a.Jmp(exit)
		}
	}
	c.a.Bind(number)
	c.a.MovQToX(xScratch0, wx)
	c.a.SSEOp(amd64.UcomiSD, xScratch0, xScratch0)
	c.a.Jcc(amd64.CondP, no)
	c.a.Jmp(yes)
	// Two words: two numbers compare as numbers; anything else is
	// strictly unequal, and loosely Go's.
	c.a.Bind(differ)
	other := no
	if v.Index != 1 {
		other = exit
	}
	c.a.MovImm(scratchC, abi.NumberLimit)
	c.a.Op(amd64.Cmp, wx, scratchC, true)
	c.a.Jcc(amd64.CondAE, other)
	c.a.Op(amd64.Cmp, wy, scratchC, true)
	c.a.Jcc(amd64.CondAE, other)
	c.a.MovQToX(xScratch0, wx)
	c.a.MovQToX(xScratch1, wy)
	c.a.SSEOp(amd64.UcomiSD, xScratch0, xScratch1)
	c.a.Jcc(amd64.CondNE, no)
	c.a.Jcc(amd64.CondP, no)
	c.a.Bind(yes)
	c.a.MovImm(scratchA, 1)
	c.a.Jmp(done)
	c.a.Bind(no)
	c.a.MovImm(scratchA, 0)
	c.a.Bind(done)
	c.setG(v, scratchA)
}

// truth computes JavaScript truthiness for the kinds the slot IR decides it
// for, exiting for the rest.
func (c *compiler) truth(v *ssa.Value, guard func(amd64.Cond)) {
	r := c.gpr(v.Args[0], scratchB)
	number, yes, no, done := c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel()
	c.a.MovRR(scratchA, r)
	c.a.ShiftImm(amd64.Shr, scratchA, 51, true)
	c.a.OpImm(amd64.Cmp, scratchA, 0x1FFF, false)
	c.a.Jcc(amd64.CondNE, number)
	for _, w := range []struct {
		word uint64
		to   amd64.Label
	}{{c.enc.True, yes}, {c.enc.False, no}, {c.enc.Undefined, no}, {c.enc.Null, no}} {
		c.a.MovImm(scratchA, w.word)
		c.a.Op(amd64.Cmp, r, scratchA, true)
		c.a.Jcc(amd64.CondE, w.to)
	}
	// Anything else -- the uninitialized marker -- is not decided here.
	c.a.Op(amd64.Cmp, r, r, true)
	guard(amd64.CondE)
	c.a.Bind(number)
	// A number is true unless zero or NaN: ucomisd sets ZF for zero and for
	// NaN, and PF only for NaN.
	c.a.MovQToX(xScratch0, r)
	c.a.SSEOp(amd64.XorPD, xScratch1, xScratch1)
	c.a.SSEOp(amd64.UcomiSD, xScratch0, xScratch1)
	c.a.Setcc(amd64.CondNE, scratchA)
	c.a.Setcc(amd64.CondNP, scratchC)
	c.a.Op(amd64.And, scratchA, scratchC, false)
	c.a.MovZX8(scratchA, scratchA)
	c.a.Jmp(done)
	c.a.Bind(yes)
	c.a.MovImm(scratchA, 1)
	c.a.Jmp(done)
	c.a.Bind(no)
	c.a.MovImm(scratchA, 0)
	c.a.Bind(done)
	c.setG(v, scratchA)
}

// toInt32 is ToUint32's bits: CVTTSD2SI covers |x| < 2^63; beyond, and for
// NaN and the infinities, a cold path works from the exponent and
// significand.
func (c *compiler) toInt32(v *ssa.Value) {
	xv := v.Args[0]
	slow, back := c.a.NewLabel(), c.a.NewLabel()
	x := c.xmm(xv, xScratch0)
	c.a.Cvttsd2si(scratchA, x)
	c.a.MovImm(scratchB, 1<<63)
	c.a.Op(amd64.Cmp, scratchA, scratchB, true)
	c.a.Jcc(amd64.CondE, slow)
	c.a.Bind(back)
	c.a.MovRR32(scratchA, scratchA)
	c.setG(v, scratchA)
	c.cold = append(c.cold, func() {
		zero, positive := c.a.NewLabel(), c.a.NewLabel()
		c.a.Bind(slow)
		c.a.MovQFromX(scratchA, c.xmm(xv, xScratch0))
		// e = exponent; NaN and the infinities give 0.
		c.a.MovRR(scratchC, scratchA)
		c.a.ShiftImm(amd64.Shr, scratchC, 52, true)
		c.a.OpImm(amd64.And, scratchC, 0x7FF, false)
		c.a.OpImm(amd64.Cmp, scratchC, 0x7FF, false)
		c.a.Jcc(amd64.CondE, zero)
		// |x| >= 2^63 here, so e = exponent - 1075 >= 11, and the low 32
		// bits of the integer are the significand's low bits shifted by e,
		// zero once e >= 32; the implicit leading bit lands past bit 62.
		c.a.OpImm(amd64.Sub, scratchC, 1075, false)
		c.a.OpImm(amd64.Cmp, scratchC, 32, false)
		c.a.Jcc(amd64.CondAE, zero)
		c.a.MovRR(scratchB, scratchA) // the sign is bit 63
		c.a.ShiftImm(amd64.Shl, scratchA, 12, true)
		c.a.ShiftImm(amd64.Shr, scratchA, 12, true)
		c.a.ShiftCL(amd64.Shl, scratchA, true)
		c.a.Op(amd64.Test, scratchB, scratchB, true)
		c.a.Jcc(amd64.CondNS, positive)
		c.a.MovRR32(scratchC, scratchA)
		c.a.Op(amd64.Xor, scratchA, scratchA, false)
		c.a.Op(amd64.Sub, scratchA, scratchC, false)
		c.a.Bind(positive)
		c.a.Jmp(back)
		c.a.Bind(zero)
		c.a.Op(amd64.Xor, scratchA, scratchA, false)
		c.a.Jmp(back)
	})
}

// sseOp, aluOp and shiftOp are the instructions for SSA's arithmetic,
// bitwise and shift operations.
func sseOp(op ssa.Op) amd64.SSE {
	switch op {
	case ssa.OpAddF64:
		return amd64.AddSD
	case ssa.OpSubF64:
		return amd64.SubSD
	case ssa.OpMulF64:
		return amd64.MulSD
	}
	return amd64.DivSD
}

func aluOp(op ssa.Op) amd64.ALU {
	switch op {
	case ssa.OpAndI32:
		return amd64.And
	case ssa.OpOrI32:
		return amd64.Or
	}
	return amd64.Xor
}

func shiftOp(op ssa.Op) amd64.Shift {
	switch op {
	case ssa.OpShlI32:
		return amd64.Shl
	case ssa.OpSarI32:
		return amd64.Sar
	}
	return amd64.Shr
}
