//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"unsafe"

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
	s := r.jit
	for {
		ipc := e.p.At(uint32(rpc))
		if e.invalid || e.code == nil || ipc < 0 || !e.code.HasEntry(ipc) {
			// No entry here, or code a guard of which failed: the
			// interpreter runs the rest of the invocation.
			s.interpreted++
			return r.runReg(f, c, rpc)
		}
		ctx, ret, ok := r.jitRegStretch(f, e, wb, ipc)
		if !ok {
			s.interpreted++
			return r.runReg(f, c, rpc)
		}
		exitPC := int(ctx.ExitPC)
		switch ctx.ExitKind {
		case abi.ExitReturn:
			return ret, nil
		case abi.ExitDeopt:
			// A guard failed before its instruction committed anything: the
			// frame holds the state at its start, which the interpreter runs
			// the rest of the invocation from, and the code is compiled
			// again at the next ask, from what the interpreter learns.
			s.guards++
			r.jitRegMaterialize(f, e, wb)
			e.invalid = true
			s.interpreted++
			return r.runReg(f, c, exitPC)
		case abi.ExitHost:
			// Go runs the register instruction whole, then native code goes
			// on at the next entry.
			s.hosts++
			r.jitRegMaterialize(f, e, wb)
			f.pc = regSrc(c, exitPC) + 1
			next, ran, err := r.regWhole(f, c, wb, exitPC)
			if !ran {
				s.interpreted++
				return r.runReg(f, c, exitPC)
			}
			if err != nil {
				return Undefined, err
			}
			if r.stopped != nil {
				return Undefined, r.stopped
			}
			rpc = next
		case abi.ExitPoll:
			// The interpreter's back-edge check, at a loop's header.
			s.budgets++
			r.backEdges = s.backEdgeBudget()
			if err := r.checkInterruptNow(); err != nil {
				return Undefined, err
			}
			if r.jitStressDeopt(s) {
				s.interpreted++
				return r.runReg(f, c, exitPC)
			}
			rpc = exitPC
		default:
			// No other way out of register code's native code yet: what it
			// left, the interpreter goes on with from its start.
			r.jitRegMaterialize(f, e, wb)
			s.interpreted++
			return r.runReg(f, c, exitPC)
		}
	}
}

// jitRegStretch runs e's code once, from its entry at instruction ipc, on
// f's window at wb, in the next context, and returns that context with the
// exit's kind, register pc and the frame written -- an exit that left its
// frame to Go (abi.ExitTable) applied here -- and, for a return, the value
// returned, read before what the run kept is let go. ok is false where it
// did not run: no context, or the code closed.
func (r *Runtime) jitRegStretch(f *frame, e *jitRegEntry, wb, ipc int) (ctx *abi.Context, ret Value, ok bool) {
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
	s.ctxTop = top
	e.runs--
	if e.code == nil || err != nil {
		return nil, Undefined, false
	}
	return ctx, ret, true
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
}

// RegNativeStats reports r's counts (RegNativeStats); zero before any.
func (r *Runtime) RegNativeStats() RegNativeStats {
	s := r.jit
	if s == nil {
		return RegNativeStats{}
	}
	return RegNativeStats{Entries: s.ssaEntries, Hosts: s.hosts, Guards: s.guards, Polls: s.budgets,
		Interpreted: s.interpreted, Records: s.ssaRecords, Materialized: s.materialized}
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
	if fd != nil && fd.closure != nil && !fd.bound && fd.extra == nil && fd.treeCall != nil && fd.closure.realm == r.Realm {
		c = fd.treeCall.reg
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
