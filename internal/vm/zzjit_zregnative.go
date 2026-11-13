//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"errors"
	"fmt"
	"maps"
	"math"
	"slices"
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/jit"
	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
	jitcompile "github.com/go-quickjs/go-quickjs/internal/jit/compile"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
	"github.com/go-quickjs/go-quickjs/internal/jit/mir"
	"github.com/go-quickjs/go-quickjs/internal/jit/ssa"
)

// Native code made from register code (docs/register-interpreter.md,
// Stage 3, 3a-3): a function's register code lowered (compile.LowerReg),
// built with its feedback (jitRegFeedback), optimized and compiled by the
// same backend as the stack form's (jitRegFor), and run on the register
// frame's window in place (jitRegRun), its exits going on in the register
// interpreter's frame: a host exit runs the register instruction whole by
// the interpreter's slow paths (regWhole) and native code goes on at the
// next entry; a deoptimization, and an exit after which the code has no
// entry, have runReg run the rest of the invocation. Nothing in the
// register tier enters it yet: only a test's entry does (RegNativeCall).

// jitRegEntry is a function's code compiled from its register code, in
// the JIT's cache (jitState's regCache), or the reason none is.
type jitRegEntry struct {
	p    *ir.Program
	code *jit.SSACode
	// keep is what the code reads by address -- the shapes, holders and
	// constants' cells its feedback answered with -- kept alive with it.
	keep *jitFeedback
	// keeps is how many keep cells the code uses; scratch how many scratch
	// slots its program has (ssa.Func's Scratch), past the window in
	// memory; and frame how many slots past those its runs take: the homes
	// of the objects it does not make and its loops' slots (ssa.Func's
	// HomesEnd, LoopsEnd).
	keeps, scratch, frame int
	// loops are its loops' headers that it has entries at, by register
	// pc, sorted: where a frame in the register interpreter goes on in it
	// by on-stack replacement (loopEntry).
	loops []uint32
	// invalid marks code a guard of which failed: its runs finish in the
	// interpreter, and the next asks for it compile it again from what the
	// interpreter learned meanwhile. runs counts the runs under way, which
	// keep it from being closed.
	invalid bool
	runs    int
	bytes   int
	refusal string
	// retPC and retUndef are where the interpreter ends a frame the code
	// returned from by on-stack replacement, retReg the register retPC's
	// ret returns, and goesOn whether it can (findReturns).
	retPC, retReg, retUndef int
	goesOn                  bool
}

// loopEntry is the entry, an instruction of the program, of the loop
// whose header is register pc rpc, if the code has one there: the OSR
// entry map, each loop header's register pc to the build's entry at the
// first instruction of its lowering, its live registers loaded from the
// window (ir.Program's Live).
func (e *jitRegEntry) loopEntry(rpc int) (int, bool) {
	if _, ok := slices.BinarySearch(e.loops, uint32(rpc)); !ok {
		return 0, false
	}
	return e.p.At(uint32(rpc)), true
}

// jitRegFor is cl's register code c compiled, from the cache or anew; nil
// where the JIT has no memory for it now. An entry with no code holds the
// refusal.
func (r *Runtime) jitRegFor(cl *closure, c *regCode) *jitRegEntry {
	s := r.jitState()
	if s.unavailable {
		return nil
	}
	key := r.jitKeyOf(cl.fn)
	if e := s.regCache[key]; e != nil {
		if !e.invalid {
			return e
		}
		s.dropReg(key, e)
	}
	e := r.jitRegCompile(cl, c)
	if e == nil {
		return nil
	}
	if s.regCache == nil {
		s.regCache = map[jitKey]*jitRegEntry{}
	}
	s.regCache[key] = e
	return e
}

// jitRegCompile is jitRegFor's compile (step A6 of 3a-3): LowerReg, the
// SSA build with jitRegFeedback, Optimize, the backend, the code placed in
// the JIT's arena. A refusal is an entry with no code; nil is no room.
func (r *Runtime) jitRegCompile(cl *closure, c *regCode) *jitRegEntry {
	s := r.jit
	e := &jitRegEntry{}
	p, err := jitcompile.LowerReg(c.rc, cl.fn)
	if err == nil {
		err = p.Validate()
	}
	if err != nil {
		e.refusal = err.Error()
		return e
	}
	e.p = p
	if jitParityLog.on.Load() {
		r.jitParityNote(cl)
	}
	rf := r.jitRegFeedbackFor(cl, p)
	w := jit.GetWorkspace()
	defer jit.PutWorkspace(w)
	s.compileBegin()
	defer s.compileEnd()
	f, err := ssa.BuildIn(&w.SSA, p, rf)
	if err != nil {
		e.refusal = fmt.Sprintf("%v: %v", jitRefused, err)
		return e
	}
	// The register frame: its window's registers are its locals, in place
	// at the context's Locals; its captured bindings past them, in their
	// cells; its scratch slots past those, which are only values. Native
	// callers do not call it yet (A5), so it has no call entry.
	f.FrameLocals = p.Regs
	f.Scratch = p.Locals - (p.UpvalueBase + p.Upvalues)
	e.scratch = f.Scratch
	f.ThisSlot = -1
	f.Params, f.StartOnly = p.ParamBase+p.Params, false
	f.Cells = p.Cells
	f.CallEntry = false
	looped := map[uint32]bool{}
	for _, en := range f.Entries {
		if en.OSR {
			looped[p.Origin[en.PC]] = true
		}
	}
	ssa.Optimize(f)
	mc, err := mir.CompileWith(&w.MIR, f, jitEncoding, mir.Settings())
	if err != nil {
		e.refusal = fmt.Sprintf("%v: %v", jitRefused, err)
		return e
	}
	need := jit.CodeFootprint(mc)
	s.budget = r.jitBudget()
	if s.regBytes()+need > s.budget/2 || !s.makeRoom(need) {
		// Half the budget at most, the stack form's code the rest, while
		// only tests run register code natively.
		return nil
	}
	code, err := jit.NewSSACode(s.arena, mc)
	if err != nil {
		if errors.Is(err, jit.ErrUnavailable) {
			s.unavailable = true
		}
		return nil
	}
	e.code, e.bytes = code, need
	e.keeps = f.Keeps
	s.ssaKeeps = max(s.ssaKeeps, f.Keeps)
	if f.HomesEnd != 0 {
		e.frame = f.HomesEnd - p.Locals
	}
	if f.Loops {
		e.frame = max(e.frame, f.LoopsEnd-p.Locals)
	}
	for rpc := range looped {
		if code.HasEntry(p.At(rpc)) {
			e.loops = append(e.loops, rpc)
		}
	}
	slices.Sort(e.loops)
	e.findReturns(c.rc)
	// What the code reads by address, its feedback's keep; the closure
	// and the function the feedback asked are not.
	e.keep = rf.fb.keep()
	e.keep.cl, e.keep.fn, e.keep.prog = nil, nil, nil
	return e
}

// regBytes is what the register code in the cache takes.
func (s *jitState) regBytes() int {
	n := 0
	for key, e := range s.regCache {
		if key.gone() && e.runs == 0 {
			s.dropReg(key, e)
			continue
		}
		n += e.bytes
	}
	for key := range s.regFailed {
		if key.gone() {
			delete(s.regFailed, key)
		}
	}
	return n
}

// dropReg closes e's code, unless a run is in it, and takes it out of the
// cache.
func (s *jitState) dropReg(key jitKey, e *jitRegEntry) {
	if s.regCache[key] == e {
		delete(s.regCache, key)
	}
	if e.code != nil && e.runs == 0 {
		_ = s.closeSSA(e.code, e.keep)
		e.code = nil
	}
}

// dropRegCode drops every entry of the register code's cache, as the
// runtime releases its JIT.
func (s *jitState) dropRegCode() {
	for key, e := range s.regCache {
		s.dropReg(key, e)
	}
}

// jitRegSlot is the value slot i of p names in f's register frame, its
// window at wb: a register, a captured binding in its cell, or a slot past
// the window, where mir puts them -- a scratch slot first, then what is
// past the program's Locals, an object's home, a loop's slot.
func (r *Runtime) jitRegSlot(f *frame, p *ir.Program, wb, i int) *Value {
	scratch := p.Locals - (p.UpvalueBase + p.Upvalues)
	switch {
	case i < p.Regs:
		return &r.stack[wb+i]
	case i >= p.UpvalueBase && i < p.UpvalueBase+p.Upvalues:
		return f.cl.upvalues[i-p.UpvalueBase].slot
	case i >= p.Locals:
		return &r.stack[wb+p.Regs+scratch+i-p.Locals]
	}
	return &r.stack[wb+p.Regs+i-(p.Locals-scratch)]
}

// jitRegApplyRecords writes the slots ctx's records name into f's register
// frame, every value read before any is written, as jitApplyRecords does
// for the stack form's: an exit's own records native code applies itself
// as it leaves, so these are a native call's caller's state.
func (r *Runtime) jitRegApplyRecords(f *frame, p *ir.Program, wb int, ctx *abi.Context) {
	n := int(ctx.Records)
	if n == 0 {
		return
	}
	r.jit.ssaRecords += uint64(n)
	const flags = abi.RecordScalar | abi.RecordMaybe | abi.RecordDirect
	var buf [8]Value
	src := buf[:0]
	if n > len(buf) {
		src = make([]Value, 0, n)
	}
	src = src[:n]
	for i := range n {
		switch rec := &ctx.Record[i]; {
		case rec.Slot&abi.RecordDirect != 0:
			slot := abi.Slot{Num: rec.Word, Ref: ctx.RecordRef[i]}
			src[i] = *(*Value)(unsafe.Pointer(&slot))
		case rec.Slot&abi.RecordScalar != 0:
			src[i] = Value{num: math.Float64frombits(rec.Word)}
		case rec.Slot&abi.RecordMaybe != 0:
			src[i] = Value{num: math.Float64frombits(rec.Word)}
			if s := jitSource(&rec.Arg); s != nil && s.ref != nil {
				src[i] = *s
			}
		default:
			src[i] = *r.jitRegSlot(f, p, wb, int(rec.Arg))
		}
	}
	for i, rec := range ctx.Record[:n] {
		*r.jitRegSlot(f, p, wb, int(rec.Slot&^flags)) = src[i]
	}
	ctx.Records = 0
}

// jitRegMaterialize makes, in f's register frame, the objects e's code did
// not make (ssa's escape.go), as jitMaterializeObjects does in the stack
// form's: a register that holds the word of one gets the object, made of
// its home past the window, one for each home, the homes cleared after.
func (r *Runtime) jitRegMaterialize(f *frame, e *jitRegEntry, wb int) {
	if e.frame == 0 {
		return
	}
	p := e.p
	end := min(len(r.stack), wb+p.Regs+e.scratch+e.frame)
	if !namesVirtual(r.stack[wb:end]) {
		return
	}
	type made struct {
		home, fields int
		o            *Object
	}
	var objs []made
	at := func(slot int) *Value { return r.jitRegSlot(f, p, wb, slot) }
	var get func(w uint64) *Object
	fix := func(v *Value) {
		if isVirtualWord(v) {
			*v = Obj(get(math.Float64bits(v.num)))
		}
	}
	get = func(w uint64) *Object {
		home, fields, _ := abi.IsVirtualWord(w)
		for _, m := range objs {
			if m.home == home {
				return m.o
			}
		}
		sw := math.Float64bits(at(home).num)
		o := newShapedLiteral(*(**shape)(unsafe.Pointer(&sw)), ClassObject, fields)
		objs = append(objs, made{home, fields, o})
		for k := range fields {
			v := *at(home + abi.VirtualFields + k)
			fix(&v)
			o.appendVal(v)
		}
		return o
	}
	for i := wb; i < end; i++ {
		fix(&r.stack[i])
	}
	for _, m := range objs {
		for k := range abi.VirtualFields + m.fields {
			*at(m.home + k) = Undefined
		}
	}
	r.jit.materialized += uint64(len(objs))
}

// jitRegRun runs f, running c with its window at wb, in e's code from
// register pc rpc, an entry's, until it returns, and goes on in the
// register interpreter where the code leaves for good (step A1 of 3a-3).
func (r *Runtime) jitRegRun(f *frame, c *regCode, e *jitRegEntry, wb, rpc int) (Value, error) {
	pc, v, done, err := r.jitRegGoOn(f, c, e, wb, rpc)
	switch {
	case err != nil:
		return Undefined, err
	case done:
		return v, nil
	}
	return r.runReg(f, c, pc)
}

// jitRegGoOn runs f, running c with its window at wb, in e's code from
// register pc rpc, an entry's, until it returns -- done, with what it
// returns -- or throws what no handler of the frame's takes, err, or
// leaves the frame to the register interpreter, which goes on with it at
// register pc pc: after a deoptimization, at a register pc the code has no
// entry at, or where the code is not to run. A frame the interpreter runs
// goes on so in its own loop (jitRegLoop); a call's, in runReg.
func (r *Runtime) jitRegGoOn(f *frame, c *regCode, e *jitRegEntry, wb, rpc int) (pc int, v Value, done bool, err error) {
	s := r.jit
	for {
		ipc := e.p.At(uint32(rpc))
		if e.invalid || e.code == nil || ipc < 0 || !e.code.HasEntry(ipc) || wb+e.p.Regs+e.scratch+e.frame > len(r.stack) {
			// No entry here, or code a guard of which failed: the
			// interpreter runs the rest of the invocation.
			s.interpreted++
			return rpc, Undefined, false, nil
		}
		ctx, ret, ok := r.jitRegStretch(f, c, e, wb, ipc)
		if !ok {
			s.interpreted++
			return rpc, Undefined, false, nil
		}
		exitPC := int(ctx.ExitPC)
		switch ctx.ExitKind {
		case abi.ExitReturn:
			return 0, ret, true, nil
		case abi.ExitDeopt:
			// A guard failed before its instruction committed anything: the
			// frame holds the state at its start, which the interpreter runs
			// the rest of the invocation from, and the code is compiled
			// again at the next ask, from what the interpreter learns.
			s.guards++
			r.jitRegMaterialize(f, e, wb)
			s.regNoteFailed(r.jitKeyOf(f.cl.fn), exitPC)
			r.jitRegInvalidate(f.cl, e)
			s.interpreted++
			return exitPC, Undefined, false, nil
		case abi.ExitHost:
			// Go runs the register instruction whole, then native code goes
			// on at the next entry.
			s.hosts++
			r.jitRegMaterialize(f, e, wb)
			f.pc = regSrc(c, exitPC) + 1
			var next int
			var ran bool
			if p := s.regPending; p.set && p.pc == exitPC {
				// A call Go made from native code, which left after it
				// (jitRegGoCall): what it answered goes to the frame, as
				// the call's register instruction would have it.
				s.regPending = jitGoPending{}
				next, ran, err = r.regFinishCall(f, c, wb, exitPC, p.v, p.err)
			} else {
				s.noteRegHost(c, exitPC)
				next, ran, err = r.regWhole(f, c, wb, exitPC)
			}
			if !ran {
				s.interpreted++
				return exitPC, Undefined, false, nil
			}
			if err != nil {
				return 0, Undefined, false, err
			}
			if r.stopped != nil {
				return 0, Undefined, false, r.stopped
			}
			rpc = next
		case abi.ExitPoll:
			// The interpreter's back-edge check, at a loop's header.
			s.budgets++
			r.backEdges = s.backEdgeBudget()
			if err := r.checkInterruptNow(); err != nil {
				return 0, Undefined, false, err
			}
			if r.jitStressDeopt(s) {
				s.interpreted++
				return exitPC, Undefined, false, nil
			}
			rpc = exitPC
		default:
			// No other way out of register code's native code yet: what it
			// left, the interpreter goes on with from its start.
			r.jitRegMaterialize(f, e, wb)
			s.interpreted++
			return exitPC, Undefined, false, nil
		}
	}
}

// jitRegStretch runs e's code once, from its entry at instruction ipc, on
// f's window at wb, in the next context, and returns that context with the
// exit's kind, register pc and the frame written -- an exit that left its
// frame to Go (abi.ExitTable) applied here -- and, for a return, the value
// returned, read before what the run kept is let go. ok is false where it
// did not run: no context, or the code closed.
func (r *Runtime) jitRegStretch(f *frame, c *regCode, e *jitRegEntry, wb, ipc int) (ctx *abi.Context, ret Value, ok bool) {
	s := r.jit
	idx := s.ctxTop
	if idx >= len(s.ssaCtxs)-1 && !r.jitGrowContexts() {
		return nil, Undefined, false
	}
	top := s.ctxTop
	s.ctxTop = idx + 1
	e.runs++
	ctx = s.ssaCtxs[idx]
	s.ssaCtx = ctx
	ctx.ReturnTo, ctx.Live, ctx.TailReturn = 0, 0, 0
	r.jitShareContexts(idx, f)
	p := e.p
	ctx.Locals = unsafe.Pointer(&r.stack[wb])
	ctx.Stack = unsafe.Add(ctx.Locals, (p.Regs+e.scratch)*int(jitEncoding.ValueSize))
	ctx.Upvalues = unsafe.Pointer(unsafe.SliceData(f.cl.upvalues))
	ctx.Cells, ctx.Loop = nil, 0
	ctx.Open = uint64(len(f.openUpvalues))
	// No frame for Go's calls from the code to read: the stack form's
	// frame is what they read (jitGoRun), and they leave what needs one
	// to an exit.
	if idx >= len(s.goRuns) {
		s.goRuns = append(s.goRuns, make([]jitGoRun, idx+1-len(s.goRuns))...)
	}
	s.goRuns[idx] = jitGoRun{}
	// The register frame its calls of Go make theirs on (jitRegGoCall).
	if idx >= len(s.regRuns) {
		s.regRuns = append(s.regRuns, make([]jitRegRun, idx+1-len(s.regRuns))...)
	}
	s.regRuns[idx] = jitRegRun{f: f, c: c, e: e, wb: wb}
	s.regPending = jitGoPending{}
	s.entries++
	s.ssaEntries++
	saved, raised := r.stackTop, -1
	if need := wb + p.Regs + e.scratch + e.frame; need > saved && need <= len(r.stack) {
		// The scratch slots, homes and loops' slots past the window,
		// which the stack's top covers while the code runs.
		r.stackTop, r.stackHigh, raised = need, max(r.stackHigh, need), need
	}
	err := e.code.Run(ipc, ctx)
	if r.stackTop == raised {
		r.stackTop = saved
	}
	if len(s.builders) != 0 {
		r.jitFreezeBuilders()
	}
	if s.goRuns[idx].pinned {
		r.jitGoUnpin(idx)
	}
	if err == nil && ctx.ExitKind == abi.ExitTable {
		s.ssaRecords += uint64(s.exitScratch.Apply(ctx))
	}
	if err == nil {
		r.jitRegApplyRecords(f, p, wb, ctx)
		if ctx.ExitKind == abi.ExitReturn {
			ret = Value{num: math.Float64frombits(ctx.Ret)}
			if src := jitSource(&ctx.RetFrom); src != nil && src.ref != nil {
				ret = *src
			}
		}
	}
	// What the run kept, as jitRunDone clears a stack form's run's.
	clear(ctx.Keep[:e.keeps])
	ctx.This, ctx.Upvalues, ctx.Instance = abi.Slot{}, nil, abi.Slot{}
	if n := max(ctx.RecordHigh, s.callRefs); n != 0 {
		clear(ctx.RecordRef[:n])
		ctx.RecordHigh = 0
	}
	s.goRuns[idx] = jitGoRun{}
	s.regRuns[idx] = jitRegRun{}
	s.ctxTop = top
	e.runs--
	if e.code == nil || err != nil {
		return nil, Undefined, false
	}
	return ctx, ret, true
}

// jitRegRun is a register frame native code runs on, in a context
// (jitState's regRuns): f, running c, its window at wb, in e's code.
type jitRegRun struct {
	f  *frame
	c  *regCode
	e  *jitRegEntry
	wb int
}

// jitRegGoCall is a call of Go (ssa.GoCall) native code made from register
// code makes at a typeof, which runs nothing, an instanceof, or a call,
// call_w's, call_method_w's or new_w's, on the
// register frame of the run in context idx, as the interpreter's slow path
// makes it (regCallSlow, regConstruct), not leaving native code: the
// function, the receiver and the arguments the code passed (abi.Context's
// GoArgs) put in a window of their own past the VM's stack's top -- not in
// the frame's, whose registers the code keeps values in as it found them
// -- the frame's pc past the call meanwhile, as regCallW sets it, which a
// stack trace shows. Its result goes to the call's keep cell. A call that
// threw, or that the runtime stopped in, or after which the code was found
// invalid, Go finishes at the exit native code then takes (regPending,
// regFinishCall), the call not made again. A call in tail position, which
// the interpreter makes in constant space, is no call of Go
// (jitRegFeedback's NativeCalls).
func (r *Runtime) jitRegGoCall(ctx *abi.Context, idx int) bool {
	s := r.jit
	run := s.regRuns[idx]
	f, c := run.f, run.c
	rpc := int(ctx.ExitPC)
	if rpc >= len(c.rc.Code) {
		return false
	}
	w := regAt(c.code, rpc)
	op := w.Op()
	argc := int(w.X())
	kind := ctx.GoOp >> abi.GoOpKindShift & 0xff
	if op == bytecode.RTypeof && kind == abi.GoCallTypeOf {
		// typeof's name, as regTypeof makes it, which runs nothing.
		*(*Value)(unsafe.Pointer(&ctx.Keep[ctx.GoKeep])) = Str(r.typeofString(jitGoArg(ctx, 0)))
		return true
	}
	switch {
	case op == bytecode.RCallW && kind == 0, op == bytecode.RCallMW && kind == abi.GoCallMethod, op == bytecode.RNewW && kind == abi.GoCallNew:
	case op == bytecode.RInstanceOf && kind == abi.GoCallInstanceOf:
		// As regSlow makes it, by the site's cache: it may run code.
		aux := int(c.rc.Aux[rpc])
		if aux >= len(f.cl.ic) {
			return false
		}
		f.pc = regSrc(c, rpc) + 1
		last, shared := s.ssaCtx, s.ssaSharedGen
		s.goCalls++
		yes, err := r.instanceOfAt(&f.cl.ic[aux], jitGoArg(ctx, 0), jitGoArg(ctx, 1))
		s.ssaCtx = last
		if s.ssaSharedGen != shared {
			r.jitShareContexts(idx, f)
		}
		if err != nil || r.stopped != nil || run.e.invalid {
			s.regPending = jitGoPending{set: true, pc: rpc, ctx: idx, v: Bool(yes), err: err}
			return false
		}
		*(*Value)(unsafe.Pointer(&ctx.Keep[ctx.GoKeep])) = Bool(yes)
		return true
	default:
		return false
	}
	if int(ctx.GoOp>>abi.GoOpArgcShift&0xff) != argc {
		return false
	}
	top := r.stackTop
	end := top + 2 + argc
	if end > len(r.stack) {
		return false
	}
	at := 0
	this := Undefined
	if op == bytecode.RCallMW {
		this, at = jitGoArg(ctx, 0), 1
	}
	r.stack[top] = jitGoArg(ctx, at)
	r.stack[top+1] = this
	for i := range argc {
		r.stack[top+2+i] = jitGoArg(ctx, at+1+i)
	}
	r.stackTop, r.stackHigh = end, max(r.stackHigh, end)
	s.goCalls++
	f.pc = regSrc(c, rpc) + 1
	last, shared := s.ssaCtx, s.ssaSharedGen
	goCallDepth := s.goCallDepth
	s.goCallDepth = r.frameDepth + 1
	var v Value
	var err error
	if op == bytecode.RNewW {
		v, err = r.regConstruct(r.stack[top], top, argc)
	} else {
		v, err = r.regCallSlow(f, c, w, top, rpc)
		if err == errTailCall {
			// The callee's frame ended in a tail call.
			v, err = r.regTailRest()
		}
	}
	s.goCallDepth = goCallDepth
	jitClearArgs(r.stack, top, end)
	r.stackTop = top
	s.ssaCtx = last
	if s.ssaSharedGen != shared {
		r.jitShareContexts(idx, f)
	}
	if err != nil || r.stopped != nil || run.e.invalid {
		s.regPending = jitGoPending{set: true, pc: rpc, ctx: idx, v: v, err: err}
		return false
	}
	*(*Value)(unsafe.Pointer(&ctx.Keep[ctx.GoKeep])) = v
	return true
}

// regFinishCall finishes, on f's registers, a window at wb, the call at
// register pc rpc of c that Go made for native code (jitRegGoCall) with v
// or err as its answer, as regCallW finishes one: its value to its
// register, its exception to the frame's handlers (regCatch). It reports
// as regWhole does.
func (r *Runtime) regFinishCall(f *frame, c *regCode, wb, rpc int, v Value, err error) (next int, ok bool, _ error) {
	w := regAt(c.code, rpc)
	k := regKept{pc: rpc + 1}
	if err == nil {
		dest := int(w.B())
		if op := w.Op(); op == bytecode.RInstanceOf || op == bytecode.RTypeof {
			dest = int(w.A())
		}
		r.stack[wb+dest] = v
		return k.pc, true, nil
	}
	h := r.regCatch(f, c, &k, regWindow(r.stack, wb), err)
	if h < 0 {
		return 0, true, err
	}
	return h, true, nil
}

// RegNativeResult is how a call's native run went (RegNativeCall).
type RegNativeResult struct {
	// Value and Err are what the call answered.
	Value Value
	Err   error
	// Skipped says why the call was made as an ordinary call: no register
	// frame, a refusal, no memory for the code.
	Skipped string
	// Compiled says the code was compiled for this call; Entered, at
	// which register pc native code was first entered, -1 for none;
	// OSR that it was a loop's header the evaluator had reached.
	Compiled bool
	Entered  int
	OSR      bool
}

// RegNativeStats are the runtime's counts of what native code made from
// register code did, for tests: its exits by kind, the invocations the
// interpreter finished, the records Go applied, the objects it made.
type RegNativeStats struct {
	Entries, Hosts, Guards, Polls, Interpreted, Records, Materialized uint64
	// GoCalls counts the calls of Go native code made that ran script, a
	// call's on the register frame among them (jitRegGoCall).
	GoCalls uint64
	// HostOps is Hosts by the register instruction Go ran whole.
	HostOps map[string]uint64
}

// RegNativeStats reports r's counts (RegNativeStats); zero before any.
func (r *Runtime) RegNativeStats() RegNativeStats {
	s := r.jit
	if s == nil {
		return RegNativeStats{}
	}
	return RegNativeStats{Entries: s.ssaEntries, Hosts: s.hosts, Guards: s.guards, Polls: s.budgets,
		Interpreted: s.interpreted, Records: s.ssaRecords, Materialized: s.materialized, GoCalls: s.goCalls, HostOps: maps.Clone(s.regHostOps)}
}

// noteRegHost counts a host exit at register pc rpc of c by its register
// instruction (RegNativeStats' HostOps).
func (s *jitState) noteRegHost(c *regCode, rpc int) {
	if rpc < 0 || rpc >= len(c.rc.Code) {
		return
	}
	w := regAt(c.code, rpc)
	k := w.Op().String()
	if s.regHostOps == nil {
		s.regHostOps = map[string]uint64{}
	}
	s.regHostOps[k]++
}

// RegNativeCall calls fn, a function of r's, with args, as a call from Go
// does, its frame run by native code compiled from its register code at
// its first call (jitRegFor) -- from its start, or, if osr, by on-stack
// replacement at the first loop header the evaluator reaches a second
// time, the frame run by the evaluator with the VM until then, as
// RegLowerCall runs it -- its exits going on in the register frame
// (jitRegRun). For tests, which compare it with an ordinary call.
func (r *Runtime) RegNativeCall(fnv Value, args []Value, osr bool) RegNativeResult {
	var o *Object
	var fd *funcData
	if fnv.IsObject() {
		o = fnv.Object()
		fd = o.fn()
	}
	var c *regCode
	if fd != nil && fd.closure != nil && !fd.bound && fd.extra == nil && fd.closure.realm == r.Realm {
		if c = fd.closure.reg; c == nil {
			c = r.regOf(fd.closure)
		}
	}
	plain := func(why string) RegNativeResult {
		v, err := r.Call(fnv, Undefined, args)
		return RegNativeResult{Value: v, Err: err, Skipped: why, Entered: -1}
	}
	if !r.jitEnabled {
		return plain("no JIT")
	}
	if c == nil || !c.wide || c.suspends || fd.arrow || c.leaf != 0 {
		return plain("no register frame")
	}
	cl := fd.closure
	var compiled bool
	if r.jit == nil || r.jit.regCache[r.jitKeyOf(cl.fn)] == nil || r.jit.regCache[r.jitKeyOf(cl.fn)].invalid {
		compiled = true
	}
	e := r.jitRegFor(cl, c)
	switch {
	case e == nil:
		return plain("no room")
	case e.code == nil:
		return plain("refused: " + e.refusal)
	}
	this := Undefined
	if c.coerceThis {
		this = r.globalThis
	}
	base := r.stackTop
	if base+int(c.numRegs)+e.scratch+e.frame > len(r.stack) {
		return plain("no room")
	}
	m := min(int(c.params), len(args))
	window := r.stack[base : base+2+m : base+2+m]
	window[0] = Obj(o)
	window[1] = this
	copy(window[2:], args[:m])
	res := RegNativeResult{Compiled: compiled, Entered: -1}
	res.Value, res.Err = r.regLowerEnter(o, cl, c, base, m, args, func(f *frame) (Value, error) {
		if !osr || len(e.loops) == 0 {
			res.Entered = 0
			return r.jitRegRun(f, c, e, base, 0)
		}
		// The evaluator, the VM making what it does not, up to a loop
		// header the frame reaches a second time.
		p := e.p
		x := &regRunner{r: r, f: f, c: c, p: p, wb: base}
		slots := make([]ir.Value, p.Locals)
		x.load(slots)
		seen := map[uint32]bool{}
		pc := 0
		for {
			exit, err := p.EvaluateHost(slots, nil, x, pc, 1)
			switch {
			case err == errRegThrown:
				return Undefined, x.thrown
			case err != nil:
				return Undefined, err
			case exit.Kind == ir.Returned:
				return x.fromIR(exit.Value), nil
			}
			at := exit.State.PC
			x.store(slots)
			if exit.Kind != ir.BudgetExit {
				// Left for good: runReg from its pc.
				return r.runReg(f, c, int(at))
			}
			if _, ok := e.loopEntry(int(at)); ok {
				if seen[at] {
					res.Entered, res.OSR = int(at), true
					return r.jitRegRun(f, c, e, base, int(at))
				}
				seen[at] = true
			}
			pc = p.At(at)
			x.handles = x.handles[:0]
			x.load(slots)
		}
	})
	return res
}
