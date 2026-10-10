package mir

import (
	"fmt"
	"slices"
	"unsafe"

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
	// tail is where exits that filled records go (recordsTail), if any
	// does; recorded marks the exit being emitted as one.
	tail               amd64.Label
	tailUsed, recorded bool
	// table is where an exit that leaves its frame to Go goes (tableExit),
	// if any does; exits are those exits' descriptions.
	table     amd64.Label
	tableUsed bool
	exits     []*abi.ExitDescriptor
	// enter is where native calls to callees with no native code go
	// (enterExit), if any does.
	enter     amd64.Label
	enterUsed bool
}

type stub struct {
	key   stubKey
	label amd64.Label
}

// CompileAMD64 compiles f for amd64, given the VM's value encoding.
func CompileAMD64(f *ssa.Func, enc abi.Encoding) (*Code, error) { return compileAMD64(nil, f, enc) }

// amd64GPRs and amd64XMMs are the registers amd64's allocator assigns,
// never changed.
var (
	amd64GPRs = []int{int(amd64.RBX), int(amd64.R8), int(amd64.R9), int(amd64.R10),
		int(amd64.R12), int(amd64.R13), int(amd64.R15)}
	amd64XMMs = func() (xmms []int) {
		for r := amd64.XReg(2); r < xScratch2; r++ {
			xmms = append(xmms, int(r))
		}
		return xmms
	}()
)

func compileAMD64(w *Workspace, f *ssa.Func, enc abi.Encoding) (code *Code, err error) {
	defer func() {
		if v := recover(); v != nil {
			code, err = nil, fmt.Errorf("%w: %v", ErrUnsupported, v)
		}
	}()
	k, err := prepare(w, f, enc, amd64GPRs, amd64XMMs)
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
	stubs := len(c.stubFor)
	for i := 0; i < len(c.cold); i++ {
		c.cold[i]()
	}
	// Cold code's guards' exits.
	for i := stubs; i < len(c.stubFor); i++ {
		s := c.stubFor[i]
		c.a.Bind(s.label)
		c.exitTo(s.key.state, s.key.kind)
	}
	// Last: cold code's exits may come here too.
	if c.tailUsed {
		c.a.Bind(c.tail)
		c.recordsTail()
	}
	if c.tableUsed {
		c.a.Bind(c.table)
		c.tableExit()
	}
	if c.enterUsed {
		c.a.Bind(c.enter)
		c.enterExit()
	}
	bytes, err := c.a.Finish()
	if err != nil {
		return nil, err
	}
	return &Code{Bytes: bytes, Entries: entries, Exits: c.exits, core: c.core}, nil
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

// slotSource puts in dst a slot's source (ssa's origin.go): the address
// of its value -- a local's or an operand's in the frame, the receiver's in
// the context, a captured binding's in its cell.
func (c *compiler) slotSource(slot int, dst amd64.Reg) {
	base, disp := c.slotAddr(slot, false, dst)
	c.a.Lea(dst, base, disp-c.enc.NumOffset)
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

// gdst is the register v's word is made in: its own, or scratchA if it is
// spilled, which setG then stores.
func (c *compiler) gdst(v *ssa.Value) amd64.Reg {
	if l := c.locAt(v); l.reg >= 0 {
		return amd64.Reg(l.reg)
	}
	return scratchA
}

// gprInto puts v's word in dst: moved, loaded from its spill slot, or
// made there if it is lazy.
func (c *compiler) gprInto(v *ssa.Value, dst amd64.Reg) {
	if c.isLazy(v) {
		c.materialize(v, dst)
		return
	}
	if l := c.locAt(v); l.reg >= 0 {
		if amd64.Reg(l.reg) != dst {
			c.a.MovRR(dst, amd64.Reg(l.reg))
		}
		return
	}
	c.a.Load(dst, regCtx, c.spillDisp(c.locAt(v).spill))
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
		if v.Aux < 0 {
			c.a.MovImm(dst, 0)
		} else {
			c.slotSource(v.Aux, dst)
		}
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

// returnNative returns to a native caller, if one called (nativeCall),
// with the value in the context's RetValue, both words: its pointer word
// from where RetFrom says the reference is. A native call was made with the
// collector not marking, and no Go has run since, so the pointer may be
// stored. Otherwise it falls through to the return to Go.
func (c *compiler) returnNative() {
	toGo, word := c.a.NewLabel(), c.a.NewLabel()
	c.a.Load(scratchA, regCtx, abi.OffReturnTo)
	c.a.Op(amd64.Test, scratchA, scratchA, true)
	c.a.Jcc(amd64.CondE, toGo)
	c.a.Load(scratchA, regCtx, abi.OffRet)
	c.a.Store(regCtx, abi.OffRetValue+c.enc.NumOffset, scratchA)
	c.a.MovImm(scratchB, 0)
	c.a.Store(regCtx, abi.OffRetValue+c.enc.RefOffset, scratchB)
	c.a.Load(scratchC, regCtx, abi.OffRetFrom)
	c.a.Op(amd64.Test, scratchC, scratchC, true)
	c.a.Jcc(amd64.CondE, word)
	c.a.Load(scratchC, scratchC, c.enc.RefOffset)
	c.a.Store(regCtx, abi.OffRetValue+c.enc.RefOffset, scratchC)
	c.a.Bind(word)
	c.a.Load(scratchA, regCtx, abi.OffReturnTo)
	c.a.JmpReg(scratchA)
	c.a.Bind(toGo)
}

// call calls natively the function a call calls, if it is one of those it
// was seen to call (ssa's OpCall), as V8's code calls another's, and goes
// on after it with the result: the callee's frame is made in the VM's
// stack and its context is the next one; it returns here (returnNative).
// Nothing is written to this frame: what is used after the call is in
// memory, spilled (allocate), and kept, if read from a cell (ssa's
// keep.go). The call's state is recorded instead, each slot that is not
// the frame's as it is -- its number word, and its pointer word read where
// it came from (abi.RecordDirect) -- for Go to make the frame from if the
// callee leaves native code (abi.Context.Live); the callee's arguments are
// copied from there. Which function it calls is looked for twice: before,
// among what each needs, and after, from the callee's record. Anything
// that does not allow the call fails the guard: Go makes it.
func (c *compiler) call(v *ssa.Value, guard func(amd64.Cond)) {
	sites, s := v.Calls, v.State
	if len(sites) == 0 {
		c.a.Jmp(c.stubLabel(s, exitKind(v.Aux)))
		return
	}
	site := sites[0]
	vs := int32(c.enc.ValueSize)
	sp := len(s.Slots)
	calleeSlot := sp - site.Argc - 1
	callee := v.Args[len(v.Args)-site.Argc-1]
	// The collector is not marking: the records and the callee's context
	// take pointers.
	c.a.MovImm(scratchB, c.enc.WriteBarrier)
	c.a.LoadU8(scratchB, scratchB, 0)
	c.a.Op(amd64.Test, scratchB, scratchB, false)
	guard(amd64.CondNE)
	// A context.
	c.a.Load(scratchB, regCtx, abi.OffLevel)
	c.a.OpImm(amd64.Add, scratchB, 1, true)
	c.a.Load(scratchC, regCtx, abi.OffLevelLimit)
	c.a.Op(amd64.Cmp, scratchB, scratchC, true)
	guard(amd64.CondAE)
	// The callee is one of the functions, which has native code. A call
	// through Function.prototype.call is made of its one function, its
	// operands written where the callee has them (directOperand).
	if callee.Shadow == nil && c.origin.At(callee) < 0 || site.Via != 0 && (len(sites) != 1 || !c.operandsLive(s, calleeSlot-1)) {
		c.a.Jmp(c.stubLabel(s, exitKind(v.Aux)))
		return
	}
	c.a.MovImm(scratchB, c.enc.Object)
	c.a.Op(amd64.Cmp, c.gpr(callee, scratchA), scratchB, true)
	guard(amd64.CondNE)
	c.sourceRef(callee, guard)
	checked := c.a.NewLabel()
	for _, t := range sites {
		next := c.a.NewLabel()
		want := t.Callee
		if t.Via != 0 {
			want = t.Via
		}
		c.a.MovImm(scratchB, uint64(want))
		c.a.Op(amd64.Cmp, scratchC, scratchB, true)
		c.a.Jcc(amd64.CondNE, next)
		if t.Via != 0 {
			c.viaGuards(t, v.Args[0], guard)
		}
		if t.Pop {
			// Array.prototype.pop, its fast path's.
			c.popGuards(t, v.Args[0], guard)
			c.a.Jmp(checked)
			c.a.Bind(next)
			continue
		}
		if t.Push {
			// Array.prototype.push, its fast path's.
			c.pushGuards(t, v.Args[0], guard)
			c.a.Jmp(checked)
			c.a.Bind(next)
			continue
		}
		if t.Alloc || t.Receiver {
			// A built-in's construction with nothing to run: its object;
			// or an inlined construction's receiver.
			c.constructGuards(t, guard)
			c.a.Jmp(checked)
			c.a.Bind(next)
			continue
		}
		// No code native callers may call (abi.Context's nativeEntry 0) is
		// no reason to leave: Go runs the callee once its frame is made
		// (enterExit).
		if t.Coerce {
			// A receiver that is not an object is coerced, which Go does:
			// the method call's, or the first argument of a call through
			// Function.prototype.call.
			recv := v.Args[0]
			if t.Via != 0 {
				recv = v.Args[2]
			}
			c.a.MovImm(scratchB, c.enc.Object)
			c.a.Op(amd64.Cmp, c.gpr(recv, scratchA), scratchB, true)
			guard(amd64.CondNE)
		}
		// Room in the VM's stack.
		c.a.Load(scratchB, regCtx, abi.OffStackTop)
		c.a.Load(scratchB, scratchB, 0)
		c.a.OpImm(amd64.Add, scratchB, int32(t.LocalCount+t.MaxStack), true)
		c.a.Load(scratchA, regCtx, abi.OffStackEnd)
		c.a.Op(amd64.Cmp, scratchB, scratchA, true)
		guard(amd64.CondA)
		if t.Pool != 0 {
			c.constructGuards(t, guard)
		}
		c.a.Jmp(checked)
		c.a.Bind(next)
	}
	c.a.Jmp(c.stubLabel(s, exitKind(v.Aux)))
	c.a.Bind(checked)
	if site.Push {
		c.push(v)
		return
	}
	if site.Pop {
		c.pop(v)
		return
	}
	if site.Alloc || site.Receiver {
		// The pool's last object, its cell cleared, is the result, kept
		// (OpCallCell): nothing runs, so nothing is recorded or saved.
		c.a.MovImm(scratchA, uint64(site.Pool))
		c.a.Load(scratchB, scratchA, abi.OffPoolCount)
		c.a.OpImm(amd64.Sub, scratchB, 1, true)
		c.a.Store(scratchA, abi.OffPoolCount, scratchB)
		c.a.ShiftImm(amd64.Shl, scratchB, 3, true)
		c.a.Op(amd64.Add, scratchB, scratchA, true)
		c.a.Load(scratchC, scratchB, abi.OffPoolObjects)
		c.a.MovImm(scratchA, 0)
		c.a.Store(scratchB, abi.OffPoolObjects, scratchA)
		at := abi.OffKeep + int32(v.Index)*vs
		c.a.Store(regCtx, at+c.enc.RefOffset, scratchC)
		c.a.MovImm(scratchA, c.enc.Object)
		c.a.Store(regCtx, at+c.enc.NumOffset, scratchA)
		c.setG(v, scratchA)
		return
	}
	// A call of one function has its operands -- the function, the
	// receiver, the arguments -- written straight where the callee has
	// them (directOperand), not recorded: Go never reads them, as a callee
	// that leaves native code is finished from its own frame, and its
	// result replaces them (the VM's jitCallResult).
	direct, ops := len(sites) == 1 && (site.ThisSlot < 0 || site.Method || site.Pool != 0), calleeSlot
	if site.Method {
		ops--
	}
	direct = direct && c.operandsLive(s, ops)
	// The state, recorded: every slot the frame does not hold as it is.
	record := map[int]int32{}
	k := int32(0)
	for i, x := range s.Slots {
		if x == nil || x.Op == ssa.OpLoadSlot && x.Aux == i || c.captured(i) || direct && i >= ops {
			continue
		}
		var w amd64.Reg
		if remat(x) {
			c.materialize(x, scratchA)
			w = scratchA
		} else {
			w = c.gpr(x, scratchA)
		}
		at := abi.OffRecord + k*32
		c.a.Store(regCtx, at+16, w)
		c.pointerWord(x, w)
		c.a.Store(regCtx, abi.OffRecordRef+k*8, scratchB)
		c.a.MovImm(scratchB, uint64(i)|abi.RecordDirect)
		c.a.Store(regCtx, at, scratchB)
		record[i] = k
		k++
	}
	high := c.a.NewLabel()
	c.a.MovImm(scratchA, uint64(k))
	c.a.Store(regCtx, abi.OffRecords, scratchA)
	c.a.Load(scratchB, regCtx, abi.OffRecordHi)
	c.a.Op(amd64.Cmp, scratchB, scratchA, true)
	c.a.Jcc(amd64.CondAE, high)
	c.a.Store(regCtx, abi.OffRecordHi, scratchA)
	c.a.Bind(high)
	for _, f := range []struct {
		off int32
		v   uint64
	}{{abi.OffExitKind, abi.ExitHost}, {abi.OffExitPC, uint64(s.PC)}, {abi.OffExitDepth, uint64(s.Depth)}, {abi.OffExitSite, uint64(int64(s.Site))}} {
		c.a.MovImm(scratchA, f.v)
		c.a.Store(regCtx, f.off, scratchA)
	}
	if direct {
		// The callee's frame, past the VM's stack's top: xScratch2 holds
		// its address, which pointerWord does not use.
		c.a.Load(scratchA, regCtx, abi.OffStackTop)
		c.a.Load(scratchA, scratchA, 0)
		c.a.ShiftImm(amd64.Shl, scratchA, 4, true)
		c.a.Load(scratchB, regCtx, abi.OffStackBase)
		c.a.Op(amd64.Add, scratchA, scratchB, true)
		c.a.MovQToX(xScratch2, scratchA)
		args, this := sp-site.Argc, calleeSlot-1
		if site.Via != 0 {
			// Through Function.prototype.call: the first argument is the
			// receiver.
			args, this = args+1, args
		}
		for i := 0; i < site.Params && args+i < sp && i < site.LocalCount; i++ {
			c.directOperand(s.Slots[args+i], true, int32(i)*vs)
		}
		if site.ThisSlot >= 0 && site.Pool == 0 {
			c.directOperand(s.Slots[this], false, abi.ContextSize+abi.OffThis)
		}
	}
	// A slot's words: its record's, or the frame's, which holds it still.
	words := func(slot int) (nb amd64.Reg, nd int32, rb amd64.Reg, rd int32) {
		if k, ok := record[slot]; ok {
			return regCtx, abi.OffRecord + k*32 + 16, regCtx, abi.OffRecordRef + k*8
		}
		nb, nd = c.slotAddr(slot, false, scratchC)
		rb, rd = c.slotAddr(slot, true, scratchC)
		return nb, nd, rb, rd
	}
	// What is live across the call, in registers, is saved; nothing else
	// is in a register from here on but what the call makes.
	for _, sv := range c.saves[v] {
		if sv.float {
			c.a.StoreSD(regCtx, c.spillDisp(sv.slot), amd64.XReg(sv.reg))
		} else {
			c.a.Store(regCtx, c.spillDisp(sv.slot), amd64.Reg(sv.reg))
		}
	}
	const calleeCtx, base, locals, tmp, top, high2 = amd64.R8, amd64.R9, amd64.R10, amd64.RBX, amd64.R12, amd64.R13
	c.a.MovRR(calleeCtx, regCtx)
	c.a.OpImm(amd64.Add, calleeCtx, abi.ContextSize, true)
	c.a.Load(base, regCtx, abi.OffStackTop)
	c.a.Load(base, base, 0)
	c.a.Load(locals, regCtx, abi.OffStackBase)
	c.a.MovRR(tmp, base)
	c.a.ShiftImm(amd64.Shl, tmp, 4, true)
	c.a.Op(amd64.Add, locals, tmp, true)
	c.a.Store(calleeCtx, abi.OffLocals, locals)
	// The callee's context's own; what it shares with this one, and its
	// level, Go wrote (abi.Context).
	c.a.MovImm(tmp, 0)
	c.a.Store(calleeCtx, abi.OffUpvalues, tmp)
	c.a.Store(calleeCtx, abi.OffTailReturn, tmp)
	c.a.Store(calleeCtx, abi.OffBase, base)
	c.a.MovImm(tmp, 1)
	c.a.Store(calleeCtx, abi.OffLive, tmp)
	back := c.a.NewLabel()
	c.a.LeaLabel(tmp, back)
	c.a.Store(calleeCtx, abi.OffReturnTo, tmp)
	// Which one: the callee's pointer, from its record.
	if !direct {
		_, _, rb, rd := words(calleeSlot)
		c.a.Load(tmp, rb, rd)
	}
	for i, t := range sites {
		next := c.a.NewLabel()
		if i < len(sites)-1 {
			c.a.MovImm(top, uint64(t.Callee))
			c.a.Op(amd64.Cmp, tmp, top, true)
			c.a.Jcc(amd64.CondNE, next)
		}
		c.a.MovRR(top, locals)
		c.a.OpImm(amd64.Add, top, int32(t.LocalCount)*vs, true)
		c.a.Store(calleeCtx, abi.OffStack, top)
		// The arguments, its parameters'; undefined in every other local.
		for i := 0; i < t.LocalCount; i++ {
			at := int32(i) * vs
			// The arguments, past the receiver of a call through
			// Function.prototype.call.
			argc, first := t.Argc, sp-t.Argc
			if t.Via != 0 {
				argc, first = argc-1, first+1
			}
			if i < t.Params && i < argc && direct {
				continue
			}
			if i < t.Params && i < argc {
				nb, nd, rb, rd := words(first + i)
				c.a.Load(top, nb, nd)
				c.a.Store(locals, at+c.enc.NumOffset, top)
				c.a.Load(top, rb, rd)
				c.a.Store(locals, at+c.enc.RefOffset, top)
				continue
			}
			c.a.MovImm(top, c.enc.Undefined)
			c.a.Store(locals, at+c.enc.NumOffset, top)
			c.a.MovImm(top, 0)
			c.a.Store(locals, at+c.enc.RefOffset, top)
		}
		if t.Pool != 0 {
			// The receiver: the pool's last object, its cell cleared.
			c.a.MovImm(tmp, uint64(t.Pool))
			c.a.Load(top, tmp, abi.OffPoolCount)
			c.a.OpImm(amd64.Sub, top, 1, true)
			c.a.Store(tmp, abi.OffPoolCount, top)
			c.a.ShiftImm(amd64.Shl, top, 3, true)
			c.a.Op(amd64.Add, top, tmp, true)
			c.a.Load(high2, top, abi.OffPoolObjects)
			c.a.MovImm(tmp, 0)
			c.a.Store(top, abi.OffPoolObjects, tmp)
			c.a.Store(calleeCtx, abi.OffThis+c.enc.RefOffset, high2)
			c.a.MovImm(tmp, c.enc.Object)
			c.a.Store(calleeCtx, abi.OffThis+c.enc.NumOffset, tmp)
		} else if t.ThisSlot >= 0 && !direct {
			nb, nd, rb, rd := words(calleeSlot - 1)
			c.a.Load(top, nb, nd)
			c.a.Store(calleeCtx, abi.OffThis+c.enc.NumOffset, top)
			c.a.Load(top, rb, rd)
			c.a.Store(calleeCtx, abi.OffThis+c.enc.RefOffset, top)
		}
		c.a.MovImm(top, uint64(t.Closure))
		c.a.Store(calleeCtx, abi.OffClosure, top)
		if t.Count != 0 {
			c.a.MovImm(top, uint64(t.Count))
			c.a.Load(high2, top, 0)
			c.a.OpImm(amd64.Add, high2, 1, true)
			c.a.Store(top, 0, high2)
		}
		// The VM's stack's top past the callee's frame.
		same := c.a.NewLabel()
		c.a.MovRR(top, base)
		c.a.OpImm(amd64.Add, top, int32(t.LocalCount+t.MaxStack), true)
		c.a.Load(tmp, regCtx, abi.OffStackTop)
		c.a.Store(tmp, 0, top)
		c.a.Load(tmp, regCtx, abi.OffStackHigh)
		c.a.Load(high2, tmp, 0)
		c.a.Op(amd64.Cmp, top, high2, true)
		c.a.Jcc(amd64.CondBE, same)
		c.a.Store(tmp, 0, top)
		c.a.Bind(same)
		c.a.MovImm(scratchA, uint64(t.Entry))
		c.a.Load(scratchA, scratchA, 0)
		c.a.MovRR(regCtx, calleeCtx)
		enter := c.a.NewLabel()
		c.a.Op(amd64.Test, scratchA, scratchA, true)
		c.a.Jcc(amd64.CondE, enter)
		c.a.JmpReg(scratchA)
		c.a.Bind(enter)
		c.a.MovImm(scratchA, uint64(t.Callee))
		if !c.enterUsed {
			c.enter, c.enterUsed = c.a.NewLabel(), true
		}
		c.a.Jmp(c.enter)
		c.a.Bind(next)
	}
	// The callee returned (returnNative), its context in regCtx: its result
	// is v, its pointer word kept (OpCallCell).
	c.a.Bind(back)
	c.a.MovRR(calleeCtx, regCtx)
	c.a.OpImm(amd64.Sub, regCtx, abi.ContextSize, true)
	c.a.Load(regLocals, regCtx, abi.OffLocals)
	c.a.Load(regStack, regCtx, abi.OffStack)
	at := abi.OffKeep + int32(v.Index)*vs
	result := abi.OffRetValue
	if site.Pool != 0 {
		// A construction's result, if not an object, is its receiver.
		object := c.a.NewLabel()
		c.a.Load(scratchA, calleeCtx, abi.OffRetValue+c.enc.NumOffset)
		c.a.MovImm(scratchB, c.enc.Object)
		c.a.Op(amd64.Cmp, scratchA, scratchB, true)
		c.a.Jcc(amd64.CondE, object)
		c.a.Load(tmp, calleeCtx, abi.OffThis+c.enc.RefOffset)
		c.a.Store(calleeCtx, abi.OffRetValue+c.enc.RefOffset, tmp)
		c.a.Load(tmp, calleeCtx, abi.OffThis+c.enc.NumOffset)
		c.a.Store(calleeCtx, abi.OffRetValue+c.enc.NumOffset, tmp)
		c.a.Bind(object)
	}
	c.a.Load(tmp, calleeCtx, result+c.enc.RefOffset)
	c.a.Store(regCtx, at+c.enc.RefOffset, tmp)
	c.a.Load(scratchA, calleeCtx, result+c.enc.NumOffset)
	c.a.Store(regCtx, at+c.enc.NumOffset, scratchA)
	c.a.Load(tmp, regCtx, abi.OffStackTop)
	c.a.Load(top, calleeCtx, abi.OffBase)
	c.a.Store(tmp, 0, top)
	c.a.MovImm(tmp, 0)
	c.a.Store(calleeCtx, abi.OffLive, tmp)
	c.a.Store(calleeCtx, abi.OffReturnTo, tmp)
	for _, sv := range c.saves[v] {
		if sv.float {
			c.a.LoadSD(amd64.XReg(sv.reg), regCtx, c.spillDisp(sv.slot))
		} else {
			c.a.Load(amd64.Reg(sv.reg), regCtx, c.spillDisp(sv.slot))
		}
	}
	c.setG(v, scratchA)
}

// pushGuards checks a call of Array.prototype.push may append as its fast
// path does (ssa.CallSite's Push): the receiver, recv, a dense array,
// extensible, its length writable, with room, of the first of the
// prototypes, which are as they were, with no elements. It uses every
// scratch register.
func (c *compiler) pushGuards(t *ssa.CallSite, recv *ssa.Value, guard func(amd64.Cond)) {
	if recv.Shadow == nil && c.origin.At(recv) < 0 {
		// A primitive, never an array: scratchC, the callee, is not 0.
		c.a.Op(amd64.Test, scratchC, scratchC, true)
		guard(amd64.CondNE)
		return
	}
	c.a.MovImm(scratchB, c.enc.Object)
	c.a.Op(amd64.Cmp, c.gpr(recv, scratchA), scratchB, true)
	guard(amd64.CondNE)
	c.sourceRef(recv, guard)
	c.a.LoadU8(scratchB, scratchC, c.enc.ObjectClass)
	c.a.OpImm(amd64.Cmp, scratchB, int32(c.enc.ClassArray), false)
	guard(amd64.CondNE)
	c.a.LoadU8(scratchB, scratchC, c.enc.ObjectFlags)
	want := c.enc.FlagExtensible | c.enc.FlagLengthWritable
	c.a.OpImm(amd64.And, scratchB, int32(want|c.enc.FlagSparse), false)
	c.a.OpImm(amd64.Cmp, scratchB, int32(want), false)
	guard(amd64.CondNE)
	c.a.Load(scratchB, scratchC, c.enc.ObjectElems+8)
	c.a.Load(scratchA, scratchC, c.enc.ObjectElems+16)
	c.a.Op(amd64.Cmp, scratchB, scratchA, true)
	guard(amd64.CondAE)
	c.a.Load(scratchB, scratchC, c.enc.ObjectProto)
	c.a.MovImm(scratchA, uint64(t.Protos[0].Object))
	c.a.Op(amd64.Cmp, scratchB, scratchA, true)
	guard(amd64.CondNE)
	for i, h := range t.Protos {
		c.a.MovImm(scratchA, uint64(h.Object))
		c.a.Load(scratchB, scratchA, c.enc.ObjectShape)
		c.a.MovImm(scratchC, uint64(h.Shape))
		c.a.Op(amd64.Cmp, scratchB, scratchC, true)
		guard(amd64.CondNE)
		c.a.Load(scratchB, scratchA, c.enc.ObjectElems+8)
		c.a.Op(amd64.Test, scratchB, scratchB, true)
		guard(amd64.CondNE)
		c.a.LoadU8(scratchB, scratchA, c.enc.ObjectFlags)
		c.a.OpImm(amd64.And, scratchB, int32(c.enc.FlagSparse), false)
		guard(amd64.CondNE)
		// Its prototype: the next, or none.
		c.a.Load(scratchB, scratchA, c.enc.ObjectProto)
		if i == 0 {
			c.a.MovImm(scratchC, uint64(t.Protos[1].Object))
			c.a.Op(amd64.Cmp, scratchB, scratchC, true)
		} else {
			c.a.Op(amd64.Test, scratchB, scratchB, true)
		}
		guard(amd64.CondNE)
	}
}

// popGuards checks a call of Array.prototype.pop may pop as its fast path
// does (ssa.CallSite's Pop): the receiver, recv, a dense array of the
// realm's prototype, its length writable, whose last element is not a
// hole. It uses every scratch register.
func (c *compiler) popGuards(t *ssa.CallSite, recv *ssa.Value, guard func(amd64.Cond)) {
	if recv.Shadow == nil && c.origin.At(recv) < 0 {
		// A primitive, never an array: scratchC, the callee, is not 0.
		c.a.Op(amd64.Test, scratchC, scratchC, true)
		guard(amd64.CondNE)
		return
	}
	c.a.MovImm(scratchB, c.enc.Object)
	c.a.Op(amd64.Cmp, c.gpr(recv, scratchA), scratchB, true)
	guard(amd64.CondNE)
	c.sourceRef(recv, guard)
	c.a.LoadU8(scratchB, scratchC, c.enc.ObjectClass)
	c.a.OpImm(amd64.Cmp, scratchB, int32(c.enc.ClassArray), false)
	guard(amd64.CondNE)
	c.a.LoadU8(scratchB, scratchC, c.enc.ObjectFlags)
	c.a.OpImm(amd64.And, scratchB, int32(c.enc.FlagLengthWritable|c.enc.FlagSparse), false)
	c.a.OpImm(amd64.Cmp, scratchB, int32(c.enc.FlagLengthWritable), false)
	guard(amd64.CondNE)
	c.a.Load(scratchB, scratchC, c.enc.ObjectProto)
	c.a.MovImm(scratchA, uint64(t.Protos[0].Object))
	c.a.Op(amd64.Cmp, scratchB, scratchA, true)
	guard(amd64.CondNE)
	// Not empty, and its last element no hole.
	c.a.Load(scratchB, scratchC, c.enc.ObjectElems+8)
	c.a.Op(amd64.Test, scratchB, scratchB, true)
	guard(amd64.CondE)
	c.a.OpImm(amd64.Sub, scratchB, 1, true)
	c.a.ShiftImm(amd64.Shl, scratchB, 4, true)
	c.a.Load(scratchA, scratchC, c.enc.ObjectElems)
	c.a.Op(amd64.Add, scratchA, scratchB, true)
	c.a.Load(scratchB, scratchA, c.enc.NumOffset)
	c.a.MovImm(scratchC, c.enc.Uninitialized)
	c.a.Op(amd64.Cmp, scratchB, scratchC, true)
	guard(amd64.CondE)
}

// pop pops a call of Array.prototype.pop's receiver's last element,
// checked (popGuards): the length one less, the element's words the
// call's result, kept (OpCallCell), its cell undefined. It uses every
// scratch register.
func (c *compiler) pop(v *ssa.Value) {
	if c.enc.ValueSize != 16 {
		panic("mir: a value that is not 16 bytes")
	}
	c.sourceRef(v.Args[0], func(amd64.Cond) {})
	c.a.Load(scratchB, scratchC, c.enc.ObjectElems+8)
	c.a.OpImm(amd64.Sub, scratchB, 1, true)
	c.a.Store(scratchC, c.enc.ObjectElems+8, scratchB)
	c.a.ShiftImm(amd64.Shl, scratchB, 4, true)
	c.a.Load(scratchA, scratchC, c.enc.ObjectElems)
	c.a.Op(amd64.Add, scratchA, scratchB, true)
	at := abi.OffKeep + int32(v.Index)*c.enc.ValueSize
	c.a.Load(scratchB, scratchA, c.enc.RefOffset)
	c.a.Store(regCtx, at+c.enc.RefOffset, scratchB)
	c.a.Load(scratchB, scratchA, c.enc.NumOffset)
	c.a.Store(regCtx, at+c.enc.NumOffset, scratchB)
	c.a.MovImm(scratchC, c.enc.Undefined)
	c.a.Store(scratchA, c.enc.NumOffset, scratchC)
	c.a.MovImm(scratchC, 0)
	c.a.Store(scratchA, c.enc.RefOffset, scratchC)
	c.setG(v, scratchB)
}

// push appends a call of Array.prototype.push's argument to its receiver,
// checked (pushGuards): the value's words past the last element, the
// length one more, which is the call's result, kept (OpCallCell). It uses
// every scratch register, xScratch0 and xScratch1.
func (c *compiler) push(v *ssa.Value) {
	vs := int32(c.enc.ValueSize)
	recv, x := v.Args[0], v.Args[2]
	if vs != 16 {
		panic("mir: a value that is not 16 bytes")
	}
	// The value's words: its pointer word in xScratch1, its number word
	// in xScratch0.
	var w amd64.Reg
	if remat(x) {
		c.materialize(x, scratchA)
		w = scratchA
	} else {
		w = c.gpr(x, scratchA)
	}
	c.a.MovQToX(xScratch0, w)
	number, have := c.a.NewLabel(), c.a.NewLabel()
	if x.Shadow != nil || c.origin.At(x) >= 0 {
		c.a.MovImm(scratchB, abi.NumberLimit)
		c.a.Op(amd64.Cmp, w, scratchB, true)
		c.a.Jcc(amd64.CondB, number)
	}
	c.pointerWord(x, w)
	c.a.Jmp(have)
	c.a.Bind(number)
	c.a.MovImm(scratchB, 0)
	c.a.Bind(have)
	c.a.MovQToX(xScratch1, scratchB)
	// The array, checked: its length grown, then the element's address.
	c.sourceRef(recv, func(amd64.Cond) {})
	c.a.Load(scratchB, scratchC, c.enc.ObjectElems+8)
	c.a.OpImm(amd64.Add, scratchB, 1, true)
	c.a.Store(scratchC, c.enc.ObjectElems+8, scratchB)
	c.a.Load(scratchA, scratchC, c.enc.ObjectElems)
	c.a.MovRR(scratchC, scratchB)
	c.a.OpImm(amd64.Sub, scratchC, 1, true)
	c.a.ShiftImm(amd64.Shl, scratchC, 4, true)
	c.a.Op(amd64.Add, scratchA, scratchC, true)
	c.a.MovQFromX(scratchC, xScratch0)
	c.a.Store(scratchA, c.enc.NumOffset, scratchC)
	c.a.MovQFromX(scratchC, xScratch1)
	c.a.Store(scratchA, c.enc.RefOffset, scratchC)
	// The result: the length, a number.
	c.a.Cvtsi2sd(xScratch0, scratchB, true)
	c.a.MovQFromX(scratchA, xScratch0)
	at := abi.OffKeep + int32(v.Index)*vs
	c.a.Store(regCtx, at+c.enc.NumOffset, scratchA)
	c.a.MovImm(scratchB, 0)
	c.a.Store(regCtx, at+c.enc.RefOffset, scratchB)
	c.setG(v, scratchA)
}

// viaGuards checks a call through Function.prototype.call, whose callee was
// found to be it, calls its function: the receiver, recv, is that one. It
// uses every scratch register.
func (c *compiler) viaGuards(t *ssa.CallSite, recv *ssa.Value, guard func(amd64.Cond)) {
	if recv.Shadow == nil && c.origin.At(recv) < 0 {
		// A primitive, never the function.
		c.a.Op(amd64.Test, scratchC, scratchC, true)
		guard(amd64.CondNE)
		return
	}
	c.a.MovImm(scratchB, c.enc.Object)
	c.a.Op(amd64.Cmp, c.gpr(recv, scratchA), scratchB, true)
	guard(amd64.CondNE)
	c.sourceRef(recv, guard)
	c.a.MovImm(scratchB, uint64(t.Callee))
	c.a.Op(amd64.Cmp, scratchC, scratchB, true)
	guard(amd64.CondNE)
}

// constructGuards checks a construction can take its receiver from its
// pool (abi.ObjectPool): the pool has one, and the function, whose pointer
// is in scratchC, has the prototype the pool's were made with still, its
// own property where it was. It uses every scratch register.
func (c *compiler) constructGuards(t *ssa.CallSite, guard func(amd64.Cond)) {
	c.a.MovImm(scratchA, uint64(t.Pool))
	c.a.Load(scratchB, scratchA, abi.OffPoolCount)
	c.a.Op(amd64.Test, scratchB, scratchB, true)
	guard(amd64.CondE)
	if t.Alloc {
		// A built-in's, whose prototype the realm fixed.
		return
	}
	c.a.Load(scratchB, scratchC, c.enc.ObjectProps+8)
	c.a.OpImm(amd64.Cmp, scratchB, int32(t.ProtoIndex), true)
	guard(amd64.CondBE)
	at := int32(t.ProtoIndex) * c.enc.PropertySize
	c.a.Load(scratchB, scratchC, c.enc.ObjectProps)
	c.a.LoadU32(scratchC, scratchB, at+c.enc.PropertyKey)
	c.a.OpImm(amd64.Cmp, scratchC, int32(t.ProtoKey), false)
	guard(amd64.CondNE)
	c.a.Load(scratchC, scratchB, at+c.enc.PropertyValue+c.enc.RefOffset)
	c.a.Load(scratchB, scratchA, abi.OffPoolProto)
	c.a.Op(amd64.Cmp, scratchC, scratchB, true)
	guard(amd64.CondNE)
}

// directOperand writes a call's operand x, both words, where its callee
// has it: at disp in the callee's frame, whose address is in xScratch2, if
// local, and otherwise at disp from the context. It uses every scratch
// register.
func (c *compiler) directOperand(x *ssa.Value, local bool, disp int32) {
	base := regCtx
	if local {
		base = scratchC
	}
	at := func() {
		if local {
			c.a.MovQFromX(scratchC, xScratch2)
		}
	}
	var w amd64.Reg
	if remat(x) {
		c.materialize(x, scratchA)
		w = scratchA
	} else {
		w = c.gpr(x, scratchA)
	}
	at()
	c.a.Store(base, disp+c.enc.NumOffset, w)
	c.pointerWord(x, w)
	at()
	c.a.Store(base, disp+c.enc.RefOffset, scratchB)
}

// pointerWord leaves in scratchB the pointer word of a value whose number
// word is in w, read where it came from -- its shadow, or the slot its
// origin is -- or 0 for a primitive's. It uses scratchC.
func (c *compiler) pointerWord(x *ssa.Value, w amd64.Reg) {
	done := c.a.NewLabel()
	c.a.MovImm(scratchB, 0)
	switch o, static := c.origin.Of(x); {
	case x.Shadow != nil && cellSource(x.Shadow):
		c.a.Load(scratchB, c.gpr(x.Shadow, scratchC), c.enc.RefOffset)
	case x.Shadow != nil:
		r := c.gpr(x.Shadow, scratchC)
		c.a.Op(amd64.Test, r, r, true)
		c.a.Jcc(amd64.CondE, done)
		c.a.Load(scratchB, r, c.enc.RefOffset)
	case static && o >= 0:
		c.isReference(x, w, o, done)
		base, disp := c.slotAddr(o, true, scratchC)
		c.a.Load(scratchB, base, disp)
	}
	c.a.Bind(done)
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
	c.exitThen(s, kind, nil)
}

// exitThen is exitTo, going on at then, if it is not nil, instead of
// returning to Go: the records are applied first (recordsTail), which a
// caller of it makes sure they are, the collector not marking.
func (c *compiler) exitThen(s *ssa.FrameState, kind uint64, then *amd64.Label) {
	if then == nil && TableExits {
		// To Go, which writes the frame from the exit's description.
		if d := c.exitDescriptor(s, kind, uint8(regLocals), uint8(regStack)); d != nil {
			c.exits = append(c.exits, d)
			c.a.MovAddr(scratchA, uint64(uintptr(unsafe.Pointer(d))))
			if !c.tableUsed {
				c.table, c.tableUsed = c.a.NewLabel(), true
			}
			c.a.Jmp(c.table)
			return
		}
	}
	c.recorded = false
	c.a.MovImm(scratchC, 0)
	c.a.Store(regCtx, abi.OffRecords, scratchC)
	for i, v := range s.Slots {
		if v == nil || v.Op == ssa.OpLoadSlot && v.Aux == i || c.captured(i) {
			// Unchanged: a captured binding's own value (capturedUnchanged);
			// or between a caller's slots and an inlined callee's.
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
			// Where it came from is known only at run time: Go looks,
			// unless it is nowhere or this slot.
			scalar := c.a.NewLabel()
			from := c.gpr(v.Shadow, scratchB)
			c.a.Op(amd64.Test, from, from, true)
			c.a.Jcc(amd64.CondE, scalar)
			c.slotSource(i, scratchC)
			c.a.Op(amd64.Cmp, from, scratchC, true)
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
	if s.Inline != nil {
		// Inlined callees' frames, written past their callers' operands: a
		// context for each, the outermost's next to this one, says where,
		// for Go to make it from; this one's exit is at the call, as each
		// but the innermost's is.
		const next = scratchB
		c.a.MovRR(next, regCtx)
		for i, in := range inlineLevels(s) {
			c.a.OpImm(amd64.Add, next, abi.ContextSize, true)
			k := uint64(abi.ExitHost)
			if in == s.Inline {
				k = kind
			}
			for _, f := range []struct {
				off int32
				v   uint64
			}{{abi.OffInlineClosure, uint64(in.Closure)}, {abi.OffInlineLocals, uint64(in.Locals)}, {abi.OffInlineThis, uint64(in.ThisSlot + 1)},
				{abi.OffExitKind, k}, {abi.OffExitPC, uint64(in.PC)}, {abi.OffExitDepth, uint64(in.Depth)}, {abi.OffExitSite, uint64(int64(in.Site))},
				{abi.OffLive, abi.LiveInline}} {
				c.a.MovImm(scratchA, f.v)
				c.a.Store(next, f.off, scratchA)
			}
			if i == 0 {
				// The callers' frames' first slot, in the VM's stack.
				c.a.MovRR(scratchC, regStack)
				c.a.Load(scratchA, regCtx, abi.OffStackBase)
				c.a.Op(amd64.Sub, scratchC, scratchA, true)
				c.a.ShiftImm(amd64.Shr, scratchC, 4, true)
				c.a.OpImm(amd64.Sub, scratchC, int32(c.f.Locals), true)
			}
			c.a.MovRR(scratchA, scratchC)
			c.a.OpImm(amd64.Add, scratchA, int32(in.Base), true)
			c.a.Store(next, abi.OffBase, scratchA)
		}
		kind = abi.ExitHost
	}
	c.record(kind, uint64(s.PC), uint64(s.Depth), uint64(int64(s.Site)), then)
}

// tableExit is where every exit that leaves its frame to Go goes, its
// description's address in scratchA: it saves the registers values may be
// in, and the frame's, in the context, and returns to Go (abi.ExitTable),
// which writes the frame (jit.ApplyExit).
func (c *compiler) tableExit() {
	c.a.Store(regCtx, abi.OffExitDesc, scratchA)
	for _, r := range append([]int{int(regLocals), int(regStack)}, c.gprs...) {
		c.a.Store(regCtx, abi.OffRegs+int32(r)*8, amd64.Reg(r))
	}
	for _, x := range c.fprs {
		c.a.StoreSD(regCtx, abi.OffXRegs+int32(x)*8, amd64.XReg(x))
	}
	c.a.MovImm(scratchA, abi.ExitTable)
	c.a.Store(regCtx, abi.OffExitKind, scratchA)
	c.a.Ret()
}

// enterExit is where a native call goes whose callee, its function object
// in scratchA, has no native code native callers may call: in the callee's
// context, its frame made, it returns to Go with abi.ExitEnter, for Go to
// run the callee and resume the caller after the call (ReturnTo) -- not a
// host exit, which writes the caller's frame and has Go make the call.
func (c *compiler) enterExit() {
	c.a.Store(regCtx, abi.OffEnterCallee, scratchA)
	c.a.MovImm(scratchA, 0)
	for _, off := range []int32{abi.OffRecords, abi.OffExitPC, abi.OffExitDepth} {
		c.a.Store(regCtx, off, scratchA)
	}
	c.a.MovImm(scratchA, ^uint64(0))
	c.a.Store(regCtx, abi.OffExitSite, scratchA)
	c.a.MovImm(scratchA, abi.ExitEnter)
	c.a.Store(regCtx, abi.OffExitKind, scratchA)
	c.a.Ret()
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
	c.recorded = true
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
func (c *compiler) record(kind, pc, depth, site uint64, then *amd64.Label) {
	c.a.MovImm(scratchA, kind)
	c.a.Store(regCtx, abi.OffExitKind, scratchA)
	c.a.MovImm(scratchA, pc)
	c.a.Store(regCtx, abi.OffExitPC, scratchA)
	c.a.MovImm(scratchA, depth)
	c.a.Store(regCtx, abi.OffExitDepth, scratchA)
	c.a.MovImm(scratchA, site)
	c.a.Store(regCtx, abi.OffExitSite, scratchA)
	switch {
	case !c.recorded && then == nil:
		c.a.Ret()
		return
	case !c.recorded:
		c.a.Jmp(*then)
		return
	case then != nil:
		c.a.LeaLabel(scratchA, *then)
		c.a.Store(regCtx, abi.OffTailReturn, scratchA)
	}
	if !c.tailUsed {
		c.tail, c.tailUsed = c.a.NewLabel(), true
	}
	c.a.Jmp(c.tail)
}

// recordsTail applies an exit's records natively while the collector is
// not marking (abi.Encoding's WriteBarrier), as the VM's jitApplyRecords
// does in Go, and returns; while it marks, Go applies them. It reads every
// value the records write before writing any, since one may read a slot
// another writes: the first pass leaves each record's value in its Word
// and, for its pointer word, its Arg; the second writes them. The pointers
// are held there only meanwhile, where the collector, which cannot run,
// would not see them.
func (c *compiler) recordsTail() {
	ret, read, write, maybe, scalar, copyValue, next := c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel(),
		c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel()
	c.a.MovImm(scratchB, c.enc.WriteBarrier)
	c.a.LoadU8(scratchB, scratchB, 0)
	c.a.Op(amd64.Test, scratchB, scratchB, false)
	c.a.Jcc(amd64.CondNE, ret)
	c.a.Load(scratchA, regCtx, abi.OffRecords)
	c.a.Op(amd64.Test, scratchA, scratchA, true)
	c.a.Jcc(amd64.CondE, ret)
	c.a.MovQToX(xScratch2, scratchA)
	// From the last record to the first: scratchA its address, xScratch1
	// its index.
	c.a.Bind(read)
	c.a.OpImm(amd64.Sub, scratchA, 1, true)
	c.a.MovQToX(xScratch1, scratchA)
	c.a.ShiftImm(amd64.Shl, scratchA, 5, true)
	c.a.Op(amd64.Add, scratchA, regCtx, true)
	c.a.Load(scratchC, scratchA, abi.OffRecord)
	c.a.MovImm(scratchB, abi.RecordScalar)
	c.a.Op(amd64.Test, scratchC, scratchB, true)
	c.a.Jcc(amd64.CondNE, scalar)
	c.a.MovImm(scratchB, abi.RecordMaybe)
	c.a.Op(amd64.Test, scratchC, scratchB, true)
	c.a.Jcc(amd64.CondNE, maybe)
	// Another slot's value, a reference.
	c.a.Load(scratchC, scratchA, abi.OffRecord+8)
	c.sourceAddr()
	c.a.Bind(copyValue)
	c.a.Load(scratchB, scratchC, c.enc.NumOffset)
	c.a.Store(scratchA, abi.OffRecord+16, scratchB)
	c.a.Load(scratchB, scratchC, c.enc.RefOffset)
	c.a.Store(scratchA, abi.OffRecord+8, scratchB)
	c.a.Jmp(next)
	// Its source's value if that holds a reference, else the primitive.
	c.a.Bind(maybe)
	c.a.Load(scratchC, scratchA, abi.OffRecord+8)
	c.a.Op(amd64.Test, scratchC, scratchC, true)
	c.a.Jcc(amd64.CondE, scalar)
	c.a.Load(scratchB, scratchC, c.enc.RefOffset)
	c.a.Op(amd64.Test, scratchB, scratchB, true)
	c.a.Jcc(amd64.CondNE, copyValue)
	// The primitive Word, with no pointer.
	c.a.Bind(scalar)
	c.a.MovImm(scratchB, 0)
	c.a.Store(scratchA, abi.OffRecord+8, scratchB)
	c.a.Bind(next)
	c.a.MovQFromX(scratchA, xScratch1)
	c.a.Op(amd64.Test, scratchA, scratchA, true)
	c.a.Jcc(amd64.CondNE, read)
	c.a.MovQFromX(scratchA, xScratch2)
	c.a.Bind(write)
	c.a.OpImm(amd64.Sub, scratchA, 1, true)
	c.a.MovQToX(xScratch1, scratchA)
	c.a.ShiftImm(amd64.Shl, scratchA, 5, true)
	c.a.Op(amd64.Add, scratchA, regCtx, true)
	c.a.Load(scratchC, scratchA, abi.OffRecord)
	c.a.MovImm(scratchB, ^uint64(abi.RecordScalar|abi.RecordMaybe))
	c.a.Op(amd64.And, scratchC, scratchB, true)
	c.sourceAddr()
	c.a.Load(scratchB, scratchA, abi.OffRecord+16)
	c.a.Store(scratchC, c.enc.NumOffset, scratchB)
	c.a.Load(scratchB, scratchA, abi.OffRecord+8)
	c.a.Store(scratchC, c.enc.RefOffset, scratchB)
	c.a.MovQFromX(scratchA, xScratch1)
	c.a.Op(amd64.Test, scratchA, scratchA, true)
	c.a.Jcc(amd64.CondNE, write)
	c.a.MovImm(scratchB, 0)
	c.a.Store(regCtx, abi.OffRecords, scratchB)
	c.a.Bind(ret)
	// A native call's writing of its caller's state goes on in the
	// caller's code (exitThen).
	toGo := c.a.NewLabel()
	c.a.Load(scratchB, regCtx, abi.OffTailReturn)
	c.a.Op(amd64.Test, scratchB, scratchB, true)
	c.a.Jcc(amd64.CondE, toGo)
	c.a.MovImm(scratchA, 0)
	c.a.Store(regCtx, abi.OffTailReturn, scratchA)
	c.a.JmpReg(scratchB)
	c.a.Bind(toGo)
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
			// RetFrom is the source, 0 for a primitive's; Go checks that
			// the slot or cell holds a reference.
			c.a.Store(regCtx, abi.OffRetFrom, c.gpr(s, scratchC))
		} else if o, ok := c.origin.Of(b.Control); ok && o >= 0 {
			done := c.a.NewLabel()
			c.isReference(b.Control, r, o, done)
			c.slotSource(o, scratchC)
			c.a.Store(regCtx, abi.OffRetFrom, scratchC)
			c.a.Bind(done)
		}
		c.a.MovImm(scratchA, abi.ExitReturn)
		c.a.Store(regCtx, abi.OffExitKind, scratchA)
		// Where it returned from, roughly -- its block's PC -- for the work
		// the stretch did (the VM's jitSSAProfit).
		c.a.MovImm(scratchA, uint64(b.PC))
		c.a.Store(regCtx, abi.OffExitSite, scratchA)
		c.returnNative()
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
	if l := c.locAt(dst); l.reg >= 0 && !parked {
		// Straight into its register.
		c.gprInto(src, amd64.Reg(l.reg))
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
	c.sourceRef(a, guard)
	c.a.Op(amd64.Test, scratchC, scratchC, true)
	guard(amd64.CondE)
	return true
}

// sourceRef loads the pointer word of a, a value with a source -- an origin
// that is a slot, or a shadow -- from there into scratchC. A shadow of 0,
// a primitive's, exits. It uses scratchA and scratchB.
func (c *compiler) sourceRef(a *ssa.Value, guard func(amd64.Cond)) {
	o := c.origin.At(a)
	if s := a.Shadow; s != nil && cellSource(s) {
		c.a.Load(scratchC, c.gpr(s, scratchC), c.enc.RefOffset)
	} else if s != nil {
		// The source is known at run time: a local's, a captured
		// binding's, the receiver's, an operand's or a heap cell's address
		// (origin.go).
		r := c.gpr(s, scratchC)
		c.a.Op(amd64.Test, r, r, true)
		guard(amd64.CondE)
		c.a.Load(scratchC, r, c.enc.RefOffset)
	} else {
		base, disp := c.slotAddr(o, true, scratchC)
		c.a.Load(scratchC, base, disp)
	}
}

// sourceAddr turns the slot in scratchC, a record's, into the address of
// its value (slotSource); a heap cell's address, at or above
// abi.MaxRecords, it leaves as it is. It uses scratchB.
func (c *compiler) sourceAddr() {
	captured, stack, found := c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel()
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
}

// propStore stores a value in a property, as the VM's setPropCached does
// in an own writable data property: its number word, and its pointer
// word, which it finds where the value came from, none for a primitive
// (pointerWord). A store that changes a pointer word -- the value's, or the
// one it replaces -- is left to Go while the collector marks
// (abi.Encoding's WriteBarrier), as is a store to a cell a live reference
// was loaded from (the operands after the value: ssa's storeChecks).
// Everything that exits comes before the first write. It uses every
// scratch register and xScratch1.
func (c *compiler) propStore(v *ssa.Value, guard func(amd64.Cond)) {
	x := v.Args[1]
	var w amd64.Reg
	if remat(x) {
		c.materialize(x, scratchA)
		w = scratchA
	} else {
		w = c.gpr(x, scratchA)
	}
	// A number, the usual value, has none, which a compare says; anything
	// else's is read where it came from.
	number, have := c.a.NewLabel(), c.a.NewLabel()
	if x.Shadow != nil || c.origin.At(x) >= 0 {
		c.a.MovImm(scratchB, abi.NumberLimit)
		c.a.Op(amd64.Cmp, w, scratchB, true)
		c.a.Jcc(amd64.CondB, number)
	}
	c.pointerWord(x, w)
	c.a.Jmp(have)
	c.a.Bind(number)
	c.a.MovImm(scratchB, 0)
	c.a.Bind(have)
	c.a.MovQToX(xScratch1, scratchB)
	added := c.a.NewLabel()
	if v.Add != nil {
		// A property the cache adds, or else one the object has.
		has := c.a.NewLabel()
		c.addAlong(v, has)
		c.a.Jmp(added)
		c.a.Bind(has)
	}
	c.property(v, guard)
	for _, s := range v.Args[2:] {
		// A live value read from this cell keeps its pointer word there:
		// unless it is none, a primitive's (storeChecks).
		other := c.a.NewLabel()
		c.a.Op(amd64.Cmp, c.gpr(s, scratchB), scratchA, true)
		c.a.Jcc(amd64.CondNE, other)
		c.a.Load(scratchB, scratchA, c.enc.RefOffset)
		c.a.Op(amd64.Test, scratchB, scratchB, true)
		guard(amd64.CondNE)
		c.a.Bind(other)
	}
	scalar := c.a.NewLabel()
	c.a.MovQFromX(scratchC, xScratch1)
	c.a.Load(scratchB, scratchA, c.enc.RefOffset)
	c.a.Op(amd64.Or, scratchB, scratchC, true)
	c.a.Jcc(amd64.CondE, scalar)
	c.a.MovImm(scratchB, c.enc.WriteBarrier)
	c.a.LoadU8(scratchB, scratchB, 0)
	c.a.Op(amd64.Test, scratchB, scratchB, false)
	guard(amd64.CondNE)
	c.a.Bind(scalar)
	c.a.Store(scratchA, c.enc.NumOffset, c.gpr(x, scratchB))
	c.a.Store(scratchA, c.enc.RefOffset, scratchC)
	c.a.Bind(added)
}

// addAlong adds the property a write's cache adds (ssa.PropertyAdd), as
// the VM's appendTransition does, when the object may take it -- of the
// shape it is added to, extensible, its prototypes those the cache found
// nothing on that intercepts the write, room in its table -- and goes to
// miss otherwise, before it writes anything. It writes the length, the
// next shape, a pointer, so the collector must not be marking, and the
// entry past the last, the value's pointer word from xScratch1. It uses
// every scratch register.
func (c *compiler) addAlong(v *ssa.Value, miss amd64.Label) {
	add, x := v.Add, v.Args[1]
	if c.enc.PropertyFlags != c.enc.PropertyKey+4 || c.enc.PropertySize != 24 {
		// The key and flags are written as one word, the offset as a sum.
		c.a.Jmp(miss)
		return
	}
	c.a.MovImm(scratchB, c.enc.WriteBarrier)
	c.a.LoadU8(scratchB, scratchB, 0)
	c.a.Op(amd64.Test, scratchB, scratchB, false)
	c.a.Jcc(amd64.CondNE, miss)
	o := c.gpr(v.Args[0], scratchA)
	c.a.Load(scratchB, o, c.enc.ObjectShape)
	c.a.MovImm(scratchC, uint64(add.From))
	c.a.Op(amd64.Cmp, scratchB, scratchC, true)
	c.a.Jcc(amd64.CondNE, miss)
	c.a.LoadU8(scratchB, o, c.enc.ObjectFlags)
	c.a.OpImm(amd64.And, scratchB, int32(c.enc.FlagExtensible), false)
	c.a.Jcc(amd64.CondE, miss)
	// The prototype chain the cache found, object by object, then none.
	c.a.Load(scratchB, o, c.enc.ObjectProto)
	for _, h := range add.Protos {
		c.a.MovImm(scratchC, uint64(h.Object))
		c.a.Op(amd64.Cmp, scratchB, scratchC, true)
		c.a.Jcc(amd64.CondNE, miss)
		if h.Object == 0 {
			break
		}
		c.a.Load(scratchB, scratchC, c.enc.ObjectShape)
		c.a.MovImm(scratchC, uint64(h.Shape))
		c.a.Op(amd64.Cmp, scratchB, scratchC, true)
		c.a.Jcc(amd64.CondNE, miss)
		c.a.MovImm(scratchC, uint64(h.Object))
		c.a.Load(scratchB, scratchC, c.enc.ObjectProto)
	}
	if add.Protos[1].Object != 0 {
		c.a.Op(amd64.Test, scratchB, scratchB, true)
		c.a.Jcc(amd64.CondNE, miss)
	}
	// Room: the table's length below its capacity.
	c.a.Load(scratchB, o, c.enc.ObjectProps+8)
	c.a.Load(scratchC, o, c.enc.ObjectProps+16)
	c.a.Op(amd64.Cmp, scratchB, scratchC, true)
	c.a.Jcc(amd64.CondAE, miss)
	// The length, then the shape, then the entry, at the old length times
	// the entry's size, 24.
	c.a.MovRR(scratchC, scratchB)
	c.a.OpImm(amd64.Add, scratchB, 1, true)
	c.a.Store(o, c.enc.ObjectProps+8, scratchB)
	c.a.MovImm(scratchB, uint64(add.Next))
	c.a.Store(o, c.enc.ObjectShape, scratchB)
	c.a.MovRR(scratchB, scratchC)
	c.a.ShiftImm(amd64.Shl, scratchB, 1, true)
	c.a.Op(amd64.Add, scratchC, scratchB, true)
	c.a.ShiftImm(amd64.Shl, scratchC, 3, true)
	c.a.Load(scratchB, o, c.enc.ObjectProps)
	c.a.Op(amd64.Add, scratchC, scratchB, true)
	c.a.MovImm(scratchB, uint64(v.Key)|uint64(add.Flags)<<32)
	c.a.Store(scratchC, c.enc.PropertyKey, scratchB)
	// The pointer word first: making a lazy value's number word may use
	// xScratch1.
	c.a.MovQFromX(scratchB, xScratch1)
	c.a.Store(scratchC, c.enc.PropertyValue+c.enc.RefOffset, scratchB)
	c.a.Store(scratchC, c.enc.PropertyValue+c.enc.NumOffset, c.gpr(x, scratchA))
}

// property finds the property a property operation names and leaves the
// address of its value in scratchA. An object of the shape the site's cache
// knows has it at the cached index: the shape settles where it is and what
// it is. Any other ordinary object with a small table is searched for the
// key, unrolled, as the VM's own small objects are; the entry must be plain
// data, and writable for a write. It uses scratchB and scratchC.
func (c *compiler) property(v *ssa.Value, guard func(amd64.Cond)) {
	if v.Holders != nil || v.Cases != nil {
		c.holder(v, guard)
		return
	}
	found := c.a.NewLabel()
	if v.Const.Bits != 0 {
		// The shape the site knows, inline; any other, out of line.
		scan := c.a.NewLabel()
		p := c.gpr(v.Args[0], scratchA)
		c.a.Load(scratchB, p, c.enc.ObjectShape)
		c.a.MovImm(scratchA, v.Const.Bits)
		c.a.Op(amd64.Cmp, scratchB, scratchA, true)
		c.a.Jcc(amd64.CondNE, scan)
		p = c.gpr(v.Args[0], scratchA)
		c.a.Load(scratchA, p, c.enc.ObjectProps)
		c.a.OpImm(amd64.Add, scratchA, int32(v.Index)*c.enc.PropertySize+c.enc.PropertyValue, true)
		c.cold = append(c.cold, func() {
			c.a.Bind(scan)
			c.scan(v, guard, found)
		})
		c.a.Bind(found)
		return
	}
	c.scan(v, guard, found)
	c.a.Bind(found)
}

// scan searches an object's small table for the key a property operation
// names, as the VM's own small objects are searched, unrolled, and goes to
// found with the address of its value in scratchA; the entry must be plain
// data, and writable for a write. Anything else fails.
func (c *compiler) scan(v *ssa.Value, guard func(amd64.Cond), found amd64.Label) {
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
}

// stringBytes compares two strings of one length, in code units, by their
// bytes, as the VM does (String.Equals): their pointers are in xScratch1
// and scratchC, the length in scratchA. A rope, which Go flattens, and a
// string longer than abi.MaxEqualUnits exit; strings of different byte
// lengths differ. The loop borrows regLocals and regStack, as remainder
// does, and reloads them from the context before it leaves.
func (c *compiler) stringBytes(exit, yes, no amd64.Label) {
	c.a.OpImm(amd64.Cmp, scratchA, abi.MaxEqualUnits, true)
	c.a.Jcc(amd64.CondA, exit)
	c.a.MovQFromX(scratchA, xScratch1)
	for _, s := range []amd64.Reg{scratchA, scratchC} {
		c.a.Load(scratchB, s, c.enc.StringLeft)
		c.a.Op(amd64.Test, scratchB, scratchB, true)
		c.a.Jcc(amd64.CondNE, exit)
	}
	c.a.Load(scratchB, scratchC, c.enc.StringData+8)
	c.a.Load(scratchA, scratchA, c.enc.StringData+8)
	c.a.Op(amd64.Cmp, scratchA, scratchB, true)
	c.a.Jcc(amd64.CondNE, no)
	c.a.MovQFromX(scratchA, xScratch1)
	c.a.Load(scratchA, scratchA, c.enc.StringData)
	c.a.Load(scratchC, scratchC, c.enc.StringData)
	words, bytes, same, differ := c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel()
	c.a.Bind(words)
	c.a.OpImm(amd64.Cmp, scratchB, 8, true)
	c.a.Jcc(amd64.CondB, bytes)
	c.a.Load(regLocals, scratchA, 0)
	c.a.Load(regStack, scratchC, 0)
	c.a.Op(amd64.Cmp, regLocals, regStack, true)
	c.a.Jcc(amd64.CondNE, differ)
	c.a.OpImm(amd64.Add, scratchA, 8, true)
	c.a.OpImm(amd64.Add, scratchC, 8, true)
	c.a.OpImm(amd64.Sub, scratchB, 8, true)
	c.a.Jmp(words)
	c.a.Bind(bytes)
	c.a.Op(amd64.Test, scratchB, scratchB, true)
	c.a.Jcc(amd64.CondE, same)
	c.a.LoadU8(regLocals, scratchA, 0)
	c.a.LoadU8(regStack, scratchC, 0)
	c.a.Op(amd64.Cmp, regLocals, regStack, false)
	c.a.Jcc(amd64.CondNE, differ)
	c.a.OpImm(amd64.Add, scratchA, 1, true)
	c.a.OpImm(amd64.Add, scratchC, 1, true)
	c.a.OpImm(amd64.Sub, scratchB, 1, true)
	c.a.Jmp(bytes)
	for _, l := range []struct{ at, to amd64.Label }{{same, yes}, {differ, no}} {
		c.a.Bind(l.at)
		c.a.Load(regLocals, regCtx, abi.OffLocals)
		c.a.Load(regStack, regCtx, abi.OffStack)
		c.a.Jmp(l.to)
	}
}

// keepSource leaves in scratchC where a tagged value came from, as an
// exit's record has it: its shadow; the source of the slot its origin is,
// if the value is that slot's reference at entry; or 0 for a primitive. It
// uses scratchA.
func (c *compiler) keepSource(x *ssa.Value) {
	switch o, static := c.origin.Of(x); {
	case x.Shadow != nil:
		if r := c.gpr(x.Shadow, scratchC); r != scratchC {
			c.a.MovRR(scratchC, r)
		}
	case static && o >= 0:
		scalar, done := c.a.NewLabel(), c.a.NewLabel()
		c.isReference(x, c.gpr(x, scratchA), o, scalar)
		c.slotSource(o, scratchC)
		c.a.Jmp(done)
		c.a.Bind(scalar)
		c.a.MovImm(scratchC, 0)
		c.a.Bind(done)
	default:
		c.a.MovImm(scratchC, 0)
	}
}

// keepRef is a tagged value's pointer word, read where it came from
// (keepSource), or 0 for a primitive's or while the collector marks (ssa's
// OpKeepRef).
func (c *compiler) keepRef(v *ssa.Value) {
	x := v.Args[0]
	scalar := c.a.NewLabel()
	c.keepSource(x)
	c.a.MovImm(scratchA, 0)
	c.a.MovImm(scratchB, c.enc.WriteBarrier)
	c.a.LoadU8(scratchB, scratchB, 0)
	c.a.Op(amd64.Test, scratchB, scratchB, false)
	c.a.Jcc(amd64.CondNE, scalar)
	c.a.Op(amd64.Test, scratchC, scratchC, true)
	c.a.Jcc(amd64.CondE, scalar)
	c.a.Load(scratchA, scratchC, c.enc.RefOffset)
	c.a.Bind(scalar)
	c.setG(v, scratchA)
}

// keep copies a value, its pointer word read before (keepRef), into the
// context's keep cell v.Index, and leaves the cell's address as v (ssa's
// OpKeep); while the collector marks, it writes nothing and v is where the
// value came from (keepSource).
func (c *compiler) keep(v *ssa.Value) {
	x := v.Args[0]
	done := c.a.NewLabel()
	c.keepSource(x)
	c.a.MovRR(scratchA, scratchC)
	c.a.MovImm(scratchB, c.enc.WriteBarrier)
	c.a.LoadU8(scratchB, scratchB, 0)
	c.a.Op(amd64.Test, scratchB, scratchB, false)
	c.a.Jcc(amd64.CondNE, done)
	at := abi.OffKeep + int32(v.Index)*int32(c.enc.ValueSize)
	c.a.Store(regCtx, at+c.enc.RefOffset, c.gpr(v.Args[1], scratchB))
	c.a.Store(regCtx, at+c.enc.NumOffset, c.gpr(x, scratchB))
	c.a.MovRR(scratchA, regCtx)
	c.a.OpImm(amd64.Add, scratchA, at, true)
	c.a.Bind(done)
	c.setG(v, scratchA)
}

// holder finds the property a read whose receiver's shape it knows names
// -- on a prototype, which it checks, or the receiver's own -- into
// scratchA, the address of its value; and, for a read that met objects of
// several shapes, as V8's polymorphic inline caches do, for whichever of
// them the receiver's is (ssa.PropertyCase). Any other shape fails.
func (c *compiler) holder(v *ssa.Value, guard func(amd64.Cond)) {
	var first [2]ssa.Holder
	if v.Holders != nil {
		first = *v.Holders
	}
	cases := append([]ssa.PropertyCase{{Shape: uintptr(v.Const.Bits), Index: int32(v.Index), Holders: first}}, v.Cases...)
	done := c.a.NewLabel()
	for i, k := range cases {
		next := c.a.NewLabel()
		p := c.gpr(v.Args[0], scratchA)
		c.a.Load(scratchB, p, c.enc.ObjectShape)
		c.a.MovImm(scratchC, uint64(k.Shape))
		c.a.Op(amd64.Cmp, scratchB, scratchC, true)
		if i == len(cases)-1 {
			guard(amd64.CondNE)
		} else {
			c.a.Jcc(amd64.CondNE, next)
		}
		if p != scratchA {
			c.a.MovRR(scratchA, p)
		}
		c.a.Load(scratchB, scratchA, c.enc.ObjectProto)
		for _, h := range k.Holders {
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
		c.a.OpImm(amd64.Add, scratchA, k.Index*c.enc.PropertySize+c.enc.PropertyValue, true)
		if i < len(cases)-1 {
			c.a.Jmp(done)
			c.a.Bind(next)
		}
	}
	c.a.Bind(done)
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

// guardFor is a guard of v's for code that may keep it: a jump to the stub
// that exits to v's state.
func (c *compiler) guardFor(v *ssa.Value) func(amd64.Cond) {
	return func(cond amd64.Cond) {
		c.a.Jcc(cond, c.stubLabel(v.State, exitKind(v.Aux)))
	}
}

// value emits one value.
func (c *compiler) value(v *ssa.Value, b *ssa.Block) {
	arg := func(i int) *ssa.Value { return v.Args[i] }
	// guard is for the guards emitted here; one passed on, which may be
	// kept for cold code, is made only for the values that pass one
	// (guardFor), not for every value.
	guard := func(cond amd64.Cond) {
		c.a.Jcc(cond, c.stubLabel(v.State, exitKind(v.Aux)))
	}
	switch v.Op {
	case ssa.OpLoadSlot:
		d := c.gdst(v)
		base, disp := c.slotAddr(v.Aux, false, d)
		c.a.Load(d, base, disp)
		c.setG(v, d)
	case ssa.OpConst:
		d := c.gdst(v)
		c.a.MovImm(d, c.constWord(v.Const))
		c.setG(v, d)
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
		c.truth(v, c.guardFor(v))
	case ssa.OpArrayOf:
		c.arrayOf(v, c.guardFor(v))
	case ssa.OpObjectOf:
		if c.objectOf(v, c.guardFor(v)) {
			c.setG(v, scratchC)
		}
	case ssa.OpSameObject:
		c.a.MovImm(scratchB, v.Const.Bits)
		c.a.Op(amd64.Cmp, c.gpr(arg(0), scratchA), scratchB, true)
		guard(amd64.CondNE)
	case ssa.OpCall:
		c.call(v, c.guardFor(v))
	case ssa.OpCallCell:
		c.a.MovRR(scratchA, regCtx)
		c.a.OpImm(amd64.Add, scratchA, abi.OffKeep+int32(v.Args[0].Index)*int32(c.enc.ValueSize), true)
		c.setG(v, scratchA)
	case ssa.OpKeepRef:
		c.keepRef(v)
	case ssa.OpKeep:
		c.keep(v)
	case ssa.OpKept:
		c.setG(v, c.gpr(arg(1), scratchA))
	case ssa.OpFrameRoom:
		c.a.MovRR(scratchA, regStack)
		c.a.Load(scratchB, regCtx, abi.OffStackBase)
		c.a.Op(amd64.Sub, scratchA, scratchB, true)
		c.a.ShiftImm(amd64.Shr, scratchA, 4, true)
		c.a.OpImm(amd64.Add, scratchA, int32(v.Index), true)
		c.a.Load(scratchB, regCtx, abi.OffStackEnd)
		c.a.Op(amd64.Cmp, scratchA, scratchB, true)
		guard(amd64.CondA)
		c.a.Load(scratchA, regCtx, abi.OffLevel)
		c.a.OpImm(amd64.Add, scratchA, int32(max(v.Const.Bits, 1)), true)
		c.a.Load(scratchB, regCtx, abi.OffLevelLimit)
		c.a.Op(amd64.Cmp, scratchA, scratchB, true)
		guard(amd64.CondAE)
	case ssa.OpPropRead:
		c.property(v, c.guardFor(v))
		c.a.Load(scratchB, scratchA, c.enc.NumOffset)
		c.a.MovImm(scratchC, abi.NumberLimit)
		c.a.Op(amd64.Cmp, scratchB, scratchC, true)
		guard(amd64.CondAE)
		c.a.MovQToX(xScratch0, scratchB)
		c.setX(v, xScratch0)
	case ssa.OpPropCell:
		c.property(v, c.guardFor(v))
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
		c.stringCode(v, c.guardFor(v))
	case ssa.OpLoadCell:
		d := c.gdst(v)
		c.a.Load(d, c.gpr(arg(0), scratchA), c.enc.NumOffset)
		c.setG(v, d)
	case ssa.OpPropWrite:
		c.propStore(v, c.guardFor(v))
	case ssa.OpLength:
		c.length(v, c.guardFor(v))
	case ssa.OpElemKey:

		c.index(arg(0), c.guardFor(v))
	case ssa.OpElemRead:
		c.index(arg(1), c.guardFor(v))
		c.element(arg(0), c.guardFor(v))
		c.a.MovQToX(xScratch0, scratchB)
		c.setX(v, xScratch0)
	case ssa.OpElemCell:
		c.index(arg(1), c.guardFor(v))
		c.elementCell(arg(0), c.guardFor(v))
		c.setG(v, scratchA)
	case ssa.OpElemWrite:
		c.index(arg(1), c.guardFor(v))
		c.element(arg(0), c.guardFor(v))
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
		c.remainder(v, c.guardFor(v))
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
		c.looseNullish(v, c.guardFor(v))
	case ssa.OpEqTagged:
		c.eqTagged(v, c.guardFor(v))
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
			c.stringBytes(exit, yes, no)
		}
	}
	c.a.Bind(number)
	c.a.MovQToX(xScratch0, wx)
	c.a.SSEOp(amd64.UcomiSD, xScratch0, xScratch0)
	c.a.Jcc(amd64.CondP, no)
	c.a.Jmp(yes)
	// Two words: two numbers compare as numbers; anything else is
	// strictly unequal. Loosely, null and undefined equal each other and
	// an object with [[IsHTMLDDA]] alone, with nothing converted; any
	// other two are Go's.
	c.a.Bind(differ)
	other := no
	if v.Index != 1 {
		other = exit
		numbers, xNullish, yNullish := c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel()
		for _, w := range []uint64{c.enc.Null, c.enc.Undefined} {
			c.a.MovImm(scratchC, w)
			c.a.Op(amd64.Cmp, wx, scratchC, true)
			c.a.Jcc(amd64.CondE, xNullish)
			c.a.Op(amd64.Cmp, wy, scratchC, true)
			c.a.Jcc(amd64.CondE, yNullish)
		}
		c.a.Jmp(numbers)
		// One is null or undefined, the other of another word.
		for _, k := range []struct {
			at    amd64.Label
			other *ssa.Value
			w     amd64.Reg
		}{{xNullish, y, wy}, {yNullish, x, wx}} {
			c.a.Bind(k.at)
			for _, w := range []uint64{c.enc.Null, c.enc.Undefined} {
				c.a.MovImm(scratchC, w)
				c.a.Op(amd64.Cmp, k.w, scratchC, true)
				c.a.Jcc(amd64.CondE, yes)
			}
			c.a.MovImm(scratchC, c.enc.Object)
			c.a.Op(amd64.Cmp, k.w, scratchC, true)
			c.a.Jcc(amd64.CondNE, no)
			if c.reference(v, k.other, c.enc.Object, guard) {
				c.a.LoadU8(scratchC, scratchC, c.enc.ObjectFlags)
				c.a.OpImm(amd64.And, scratchC, int32(c.enc.FlagHTMLDDA), false)
				c.a.Jcc(amd64.CondNE, yes)
			}
			c.a.Jmp(no)
		}
		c.a.Bind(numbers)
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
	// An object is true unless it is Annex B's [[IsHTMLDDA]], and a string
	// unless it is empty: each through its pointer, where it came from.
	// Anything else -- the uninitialized marker, a symbol, a BigInt -- is
	// not decided here.
	notObject := c.a.NewLabel()
	c.a.MovImm(scratchA, c.enc.Object)
	c.a.Op(amd64.Cmp, r, scratchA, true)
	c.a.Jcc(amd64.CondNE, notObject)
	if c.reference(v, v.Args[0], c.enc.Object, guard) {
		c.a.LoadU8(scratchA, scratchC, c.enc.ObjectFlags)
		c.a.OpImm(amd64.And, scratchA, int32(c.enc.FlagHTMLDDA), false)
		c.a.Jcc(amd64.CondNE, no)
		c.a.Jmp(yes)
	}
	c.a.Bind(notObject)
	c.a.MovImm(scratchA, c.enc.String)
	c.a.Op(amd64.Cmp, r, scratchA, true)
	guard(amd64.CondNE)
	if c.reference(v, v.Args[0], c.enc.String, guard) {
		c.a.Load(scratchA, scratchC, c.enc.StringLength)
		c.a.Op(amd64.Test, scratchA, scratchA, true)
		c.a.Jcc(amd64.CondNE, yes)
		c.a.Jmp(no)
	}
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
