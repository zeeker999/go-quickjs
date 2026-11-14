//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

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
	// seen, used, flushSeen and flushAt are what the cache ages it by, as
	// a stack form's entry's (regAge, regFlush).
	seen, used, flushSeen, flushAt uint64
	// invalid marks code a guard of which failed (jitRegInvalidate): its
	// runs finish in the interpreter, and the function is compiled again
	// once the interpreter has spent a fresh budget (jitRegTierUp). runs
	// counts the runs under way, which keep it from being closed; dropped
	// marks code the cache let go of while one was, closed once the last
	// is done (jitRegStretch).
	invalid bool
	runs    int
	dropped bool
	// key is its function's key in the cache, and life the function's
	// state that outlives its compiles (jitRegLifeOf): what its code
	// learned, its budget, its deoptimizations, what its runs did.
	key  jitKey
	life *jitEntry
	// hostOnly marks a refusal of host operations alone (jitRegCompile),
	// which a later ask lowers again.
	hostOnly bool
	// levels counts the native calls under way of the code that Go runs
	// beside (jitRegLevelAt): the cache neither drops it nor compiles it
	// again while any is.
	levels  int
	bytes   int
	refusal string
	// retPC and retUndef are where the interpreter ends a frame the code
	// returned from by on-stack replacement, retReg the register retPC's
	// ret returns, and goesOn whether it can (findReturns).
	retPC, retReg, retUndef int
	goesOn                  bool
	// serial numbers the compile (jitState's regSerial): a property site
	// that deoptimizes for a shape again in code compiled since it first
	// did is megamorphic (jitDeoptSeenOp).
	serial uint32
	// rc is the register code it was compiled from; loopPC and loopLen the
	// loop a frame last entered it at and that loop's length (loopLength).
	rc              *bytecode.RegCode
	loopPC, loopLen int
	// loopMean is its loops' mean length in register instructions, at
	// least one, which a stretch's back edges are counted by
	// (jitRegStretchWork).
	loopMean int
	// unentered are the register pcs of the loops' headers whose entries
	// escape analysis dropped (ssa.Func's Unentered); osr is the code
	// compiled for a frame waiting at one (jitRegLoopCode), kept apart from
	// this, osrOf in it the code it was compiled beside; osrMiss and
	// osrMissPC the frame, by its address, and the header a loop's check
	// last found no entry at, as the stack form's entry has them
	// (jitLearnLoopEntry).
	unentered  []int
	osr, osrOf *jitRegEntry
	osrMiss    uintptr
	osrMissPC  int
	// at is, for each register pc, the program's first instruction of its
	// lowering, -1 for none (ipc).
	at []int32
	// unresumed are the register pcs after calls the code has no entry
	// after, Go not having gone on there (ssa.Func's Unresumed): where Go
	// goes on at one, the code is compiled again with the entry
	// (jitRegResumeSeen).
	unresumed []int
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
	return e.ipc(rpc), true
}

// loopLength is the length of the loop whose header is register pc rpc,
// from it to a backward jump to it: the one a frame entered the loop
// after (jitRegLoop), or else the first backward jump past the header
// (RegCode's BackLive), which is the loop's own or a loop's inside it.
func (e *jitRegEntry) loopLength(rpc int) int {
	if e.loopLen != 0 && e.loopPC == rpc {
		return e.loopLen
	}
	if e.rc != nil {
		for i := 0; i+1 < len(e.rc.BackLive); i += 2 {
			if pc := int(e.rc.BackLive[i]); pc >= rpc {
				return pc - rpc + 1
			}
		}
	}
	return 1
}

// ipc is the first instruction of the program's lowering of register pc
// rpc (ir.Program's At), -1 where there is none, by a table made as the
// code is placed (at): every entry, at a call, a loop or after an exit,
// asks.
func (e *jitRegEntry) ipc(rpc int) int {
	if uint(rpc) < uint(len(e.at)) {
		return int(e.at[rpc])
	}
	return e.p.At(uint32(rpc))
}

// jitRegFor is cl's register code c compiled, from the cache or anew; nil
// where the JIT has no memory for it now. An entry with no code holds the
// refusal.
func (r *Runtime) jitRegFor(cl *closure, c *regCode) *jitRegEntry {
	return r.jitRegForAs(cl, c, false)
}

// jitRegForAs is jitRegFor, for native callers where callee says
// (jitRegNativeCallee): a function of host operations alone, which a
// compile for Go to enter refuses but where its calls are native ones
// (jitRegCompile), is compiled for them, as the stack form compiles one for
// native callers (LowerSSAInline). A refusal for that is not kept: asked
// again, the function is lowered again.
func (r *Runtime) jitRegForAs(cl *closure, c *regCode, callee bool) *jitRegEntry {
	s := r.jitState()
	if s.unavailable {
		return nil
	}
	key := r.jitKeyOf(cl.fn)
	if s.regPendingJobs[key] != nil {
		// Being compiled in the background (jitRegSubmit): installed at
		// a later check.
		return nil
	}
	if e := s.regCache[key]; e != nil {
		if !e.invalid && !e.hostOnly {
			return e
		}
		if e.levels != 0 {
			// Run by native calls under way, which find their code in the
			// cache (jitRegLevelAt): compiled again once they are done.
			return nil
		}
	}
	return r.jitRegCompileFor(cl, c, key, callee)
}

// jitRegCompileFor compiles cl's function's register code c anew, the
// function's key key, and replaces what the cache holds for it with the
// result, as jitReoptimize replaces a stack form's code: runs of the code
// it replaces go on in it until they next leave it (jitRegGoOnAt), and it
// is closed once the last is done. nil where there is no room, or the
// compile went to the background. A refusal replaces no valid code: the
// function keeps it.
func (r *Runtime) jitRegCompileFor(cl *closure, c *regCode, key jitKey, callee bool) *jitRegEntry {
	e := r.jitRegCompile(cl, c, key, callee, 0)
	if e == nil {
		return nil
	}
	if old := r.jit.regCache[key]; e.code == nil && old != nil && old.code != nil && !old.invalid {
		return e
	}
	r.jitRegCached(key, e)
	return e
}

// jitRegCached keeps e, the code of the function whose key is key, in the
// cache, its link the code's, in place of what the cache held for it,
// which is let go of; and counts the compile, the function's first or
// again (JITStats' Compiled and Reoptimized, JITReport's Reopts).
func (r *Runtime) jitRegCached(key jitKey, e *jitRegEntry) {
	s := r.jit
	if s.regCache == nil {
		s.regCache = map[jitKey]*jitRegEntry{}
	}
	if old := s.regCache[key]; old != nil && old != e {
		s.dropReg(key, old)
	}
	s.regCache[key] = e
	s.regHeld += e.bytes
	s.regLinkUpdate(key, e)
	if life := e.life; life != nil && e.code != nil {
		if life.ssaSerial++; life.ssaSerial == 1 {
			s.compiled++
		} else {
			s.reoptimized++
			if life.reopts < ^uint8(0) {
				life.reopts++
			}
		}
		// Valid code: no budget to spend before the next compile, nothing
		// learned pending.
		life.invalid, life.warming, life.reopt, life.learning = false, false, false, false
	}
}

// jitRegCompile is jitRegFor's compile (step A6 of 3a-3): LowerReg, the
// SSA build with jitRegFeedback, Optimize, the backend, the code placed in
// the JIT's arena. A refusal is an entry with no code; nil is no room, or a
// compile handed to the background (jitRegSubmit), installed at a later
// check (jitRegInstall).
//
// frameAt, where not 0, is the program's instruction a loop's header
// lowers to whose entry a frame waits at (ssa.Func's FrameAt): code for
// on-stack replacement there (jitRegLoopCode), compiled on the spot.
func (r *Runtime) jitRegCompile(cl *closure, c *regCode, key jitKey, callee bool, frameAt int) *jitRegEntry {
	s := r.jit
	if !jitCompiles(cl.fn, r.jitBudget()) {
		// Too large for the JIT's memory, as the stack form's compile
		// finds it (jitCompiles): no room, at every ask.
		return nil
	}
	background := frameAt == 0 && r.jitInBackground(cl.fn)
	if background && s.jobsFull() {
		// As many compiles in the background as there may be: asked again
		// at a later sample, as the stack form's compile waits.
		return nil
	}
	e := &jitRegEntry{key: key, life: s.regLearnedOf(key)}
	if e.life.deferred && e.life.generation == s.generation {
		// No room for it at its last compile, and the cache has let no
		// code go since (jitRegPlace).
		s.refusedRoom++
		return nil
	}
	p, err := jitcompile.LowerReg(c.rc, cl.fn)
	if err != nil && jitcompile.HostOnly(err) {
		// Host operations alone, no loop, no element's work: compiled for
		// its native callers (LowerRegInline), or, hot, where a call it
		// makes is native (LowerRegHot), as the stack form compiles one
		// (jitNativeCallee, jitHotEntry); refused for now otherwise.
		switch {
		case callee:
			p, err = jitcompile.LowerRegInline(c.rc, cl.fn)
		case r.jitRegCallsNatively(cl, c):
			p, err = jitcompile.LowerRegHot(c.rc, cl.fn)
		default:
			e.hostOnly = true
		}
	}
	if err == nil {
		err = p.Validate()
	}
	if err != nil {
		e.refusal = err.Error()
		return e
	}
	e.p = p
	s.regSerial++
	e.serial = s.regSerial
	if jitParityLog.on.Load() {
		r.jitParityNote(cl)
	}
	r.jitRegSeenFeedback(cl, c)
	rf := r.jitRegFeedbackFor(cl, p)
	if background {
		// Asked ahead here, built on a worker (zzjit_zregbg.go).
		snap := jitRegSnapshotOf(rf, p)
		if !snap.s.bad {
			j := &jitRegJob{cl: cl, c: c, key: key, realm: r.Realm, callee: callee, e: e, p: p, rf: rf, snap: snap, deopts: e.life.deopts,
				w: jit.GetWorkspace(), enc: jitEncoding, opts: mir.Settings()}
			r.jitRegSubmit(j)
			return nil
		}
		// What the questions took back cannot be put back: built here,
		// from the feedback as it is now, its lists as the asking left
		// them.
	}
	w := jit.GetWorkspace()
	defer jit.PutWorkspace(w)
	s.compileBegin()
	defer s.compileEnd()
	f, err := ssa.BuildIn(&w.SSA, p, rf)
	if err != nil {
		e.refusal = fmt.Sprintf("%v: %v", jitRefused, err)
		return e
	}
	f.FrameAt = frameAt
	looped := jitRegShape(f, p)
	ssa.Optimize(f)
	mc, err := mir.CompileWith(&w.MIR, f, jitEncoding, mir.Settings())
	if err != nil {
		e.refusal = fmt.Sprintf("%v: %v", jitRefused, err)
		return e
	}
	if !r.jitRegPlace(e, c, rf, f, mc, looped) {
		return nil
	}
	return e
}

// jitRegPlace places mc, e's code, the backend's of f, built from c's
// register code lowered with its feedback rf, in the JIT's arena, and
// fills e from what the build found; it reports false where the JIT has
// no room for it now. Code no placement takes is a refusal, e with no
// code.
func (r *Runtime) jitRegPlace(e *jitRegEntry, c *regCode, rf *jitRegFeedback, f *ssa.Func, mc *mir.Code, looped map[uint32]bool) bool {
	s := r.jit
	p := e.p
	need := jit.CodeFootprint(mc)
	s.budget = r.jitBudget()
	if !s.makeRoom(need) {
		// No room, all the cache holds young or in use: not compiled again
		// for want of it until the cache has let some code go
		// (generation), as the stack form's compile is deferred.
		e.life.deferred, e.life.generation = true, s.generation
		return false
	}
	e.life.deferred = false
	code, err := jit.NewSSACode(s.arena, mc)
	if err != nil {
		if errors.Is(err, jit.ErrUnavailable) {
			s.unavailable = true
			return false
		}
		// Code no placement takes -- past the size one may have -- a
		// refusal, as the stack form's compile makes it: asked again as
		// no room, a callee of Mandreel's was compiled again at every
		// call of it Go made, a second and a half of each run.
		e.refusal = fmt.Sprintf("%v: %v", jitRefused, err)
		return true
	}
	s.inlined += uint64(len(f.Inlined))
	e.code, e.bytes = code, need
	e.scratch = f.Scratch
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
	e.rc = c.rc
	e.loopMean = 1
	if len(e.loops) != 0 {
		sum := 0
		for _, rpc := range e.loops {
			sum += e.loopLength(int(rpc))
		}
		e.loopMean = max(sum/len(e.loops), 1)
	}
	e.at = make([]int32, len(c.rc.Code))
	for rpc := range e.at {
		e.at[rpc] = int32(p.At(uint32(rpc)))
	}
	e.unresumed = slices.Clone(f.Unresumed)
	for _, pc := range f.Unentered {
		if pc >= 0 && pc < len(p.Origin) {
			e.unentered = append(e.unentered, int(p.Origin[pc]))
		}
	}
	// The pools' roots and prototypes escape analysis took, which the
	// objects it does not make are made with at an exit.
	rf.fb.keepVirtuals(f.VirtualAsked)
	// What the code reads by address, its feedback's keep; the closure
	// and the function the feedback asked are not.
	e.keep = rf.fb.keep()
	e.keep.cl, e.keep.fn, e.keep.prog = nil, nil, nil
	if JITCodeHook != nil {
		// -jitcode's record of the code, as the stack form's compiles
		// make it (jitInstall).
		JITCodeHook(jitFunctionOf(c.fn), code.Text(), mc)
	}
	return true
}

// Register code in the JIT's cache is held, aged, reclaimed and flushed as
// the stack form's entries are (makeRoom, flush): its bytes count in what
// the JIT holds (held, regHeld), against the one budget; makeRoom reclaims
// it as it does theirs, refusals first, then code unused this epoch and
// the one before, the least recently used first (regVictims); and flush
// drops code unused between two of its looks (regFlush). What the code
// learned is the function's (regLearned), and outlives it, as a ghost's
// does the stack form's: compiled again once hot again, it is compiled
// with all of it. Before, register code had half the budget, counted by a
// scan of the cache at every compile, and was never let go of but with its
// function.

// regUses is how often e's code was used: Go's entries at its function's
// start and into its loops, the runs Go went on in it after it left, and
// native callers' calls through its link (jitEntry's uses).
func (s *jitState) regUses(e *jitRegEntry) uint64 {
	l := e.life
	if l == nil {
		return 0
	}
	n := l.starts + l.ssaStats.entries + l.goOns
	if k := s.regLinks[e.key]; k != nil {
		n += k.in
	}
	return n
}

// regInUse reports whether e's code runs, or may go on, below Go now: a run
// of it, a native call of it, or a compile of its function in the
// background, which replaces it.
func (s *jitState) regInUse(e *jitRegEntry) bool {
	return e.runs != 0 || e.levels != 0 || s.regPendingJobs[e.key] != nil
}

// regAge notes, at epoch now, which register code was used since the last
// look (jitRegEntry's used), and lets go of what belonged to functions
// that are gone (age).
func (s *jitState) regAge(now uint64) {
	for key, e := range s.regCache {
		if key.gone() {
			if !s.regInUse(e) {
				s.dropReg(key, e)
			}
			continue
		}
		if u := s.regUses(e); u != e.seen {
			e.seen, e.used = u, now
		}
	}
	for key := range s.regLearned {
		if key.gone() {
			delete(s.regLearned, key)
		}
	}
	for key := range s.regLinks {
		if key.gone() {
			delete(s.regLinks, key)
		}
	}
}

// regVictims appends to v what of the register code makeRoom may reclaim
// at epoch now (listVictims): refusals, and code not used this epoch or the
// one before (jitCodeAge), none in use.
func (s *jitState) regVictims(v []jitVictim, now uint64) []jitVictim {
	for key, e := range s.regCache {
		if s.regInUse(e) {
			continue
		}
		class := uint8(jitVictimRefusal)
		if e.code != nil {
			if e.used+jitCodeAge > now {
				continue
			}
			class = jitVictimCode
		}
		v = append(v, jitVictim{key: key, re: e, class: class, used: e.used, serial: uint64(e.serial), bytes: e.bytes})
	}
	return v
}

// regReclaim drops x, register code makeRoom listed, if it still is a
// victim at epoch now (reclaim).
func (s *jitState) regReclaim(x *jitVictim, now uint64) {
	e := x.re
	if s.regCache[x.key] != e || s.regInUse(e) || (e.code != nil) != (x.class == jitVictimCode) || e.code != nil && e.used+jitCodeAge > now {
		return
	}
	s.regLetGo(x.key, e)
	s.evicted++
}

// regFlush drops, at the flush's tick, the register code unused since the
// flush's last look, not young, not in use (flush).
func (s *jitState) regFlush(tick, now uint64) {
	for key, e := range s.regCache {
		if e.code == nil {
			continue
		}
		if u := s.regUses(e); u != e.flushSeen || e.flushAt == 0 {
			e.flushSeen, e.flushAt = u, tick+1
			continue
		}
		if e.used+jitCodeAge > now || s.regInUse(e) {
			continue
		}
		s.regLetGo(key, e)
		s.flushed++
	}
}

// regLetGo drops e, the code of the function whose key is key, from the
// cache for room or for going unused: the function warms up anew, and
// compiled again once it is hot is compiled with what it had learned.
func (s *jitState) regLetGo(key jitKey, e *jitRegEntry) {
	s.dropReg(key, e)
	if l := e.life; l != nil {
		l.invalid, l.warming, l.reopt, l.learning = false, false, false, false
	}
}

// dropReg closes e's code, or, while a run or a native call is in it, has
// the last of them close it (dropped), and takes it out of the cache.
func (s *jitState) dropReg(key jitKey, e *jitRegEntry) {
	if s.regCache[key] == e {
		delete(s.regCache, key)
		s.regLinkUpdate(key, nil)
		s.regHeld -= e.bytes
		if o := e.osr; o != nil {
			// Its code for on-stack replacement goes with it.
			e.osr = nil
			s.regHeld -= o.bytes
			o.dropped = true
			s.regClose(o)
		}
		if e.code != nil {
			// Room made: compiles deferred for want of it are asked again
			// (jitRegPlace), as the stack form's dropEntry has them.
			s.generation++
		}
	}
	e.dropped = true
	s.regClose(e)
}

// regClose closes the code of e, which the cache let go of (dropped), once
// no run and no native call is in it.
func (s *jitState) regClose(e *jitRegEntry) {
	if e.dropped && e.code != nil && e.runs == 0 && e.levels == 0 {
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
	r.jitRegMaterializeSpans(f, e, wb, nil)
}

// jitRegMaterializeSpans is jitRegMaterialize, the windows of the callees
// e's code inlined that an exit inside them wrote too -- spans, each its
// first slot and its end in the VM's stack -- whose registers name the
// same objects as the frame's: one made for each home, which every slot
// naming it, the frame's and the callees', is given.
func (r *Runtime) jitRegMaterializeSpans(f *frame, e *jitRegEntry, wb int, spans [][2]int) {
	if e.frame == 0 {
		return
	}
	p := e.p
	end := min(len(r.stack), wb+p.Regs+e.scratch+e.frame)
	named := namesVirtual(r.stack[wb:end])
	for _, sp := range spans {
		named = named || sp[0] < sp[1] && sp[1] <= len(r.stack) && namesVirtual(r.stack[sp[0]:sp[1]])
	}
	if !named {
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
	for _, sp := range spans {
		for i := sp[0]; i < min(sp[1], len(r.stack)); i++ {
			fix(&r.stack[i])
		}
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
	return r.jitRegGoOnAt(f, c, e, wb, rpc, -1, 0)
}

// jitRegGoOnAt is jitRegGoOn, but where resume is not 0 begun by going on
// in e's code where a native call it made returns to, in context at, the
// frame's level of a native call Go finished the callee of (jitRegUnwind),
// the call's result in the next context's RetValue.
func (r *Runtime) jitRegGoOnAt(f *frame, c *regCode, e *jitRegEntry, wb, rpc, at int, resume uintptr) (pc int, v Value, done bool, err error) {
	for {
		ipc := 0
		if resume == 0 && f.cl.reg == c {
			if ne := f.cl.closureCode.jitReg; ne != nil && ne != e && ne != e.osrOf && !ne.invalid && ne.code != nil {
				// Replaced meanwhile (jitRegCached): on in the function's
				// code now, as the stack form's run goes on in its entry's.
				e = ne
			} else if e.life != nil && e.life.reopt && r.jitRegLearnedDue(f.cl, e) {
				// Due to be compiled again for what it learned: compiled
				// now, at the exit, and gone on in that code, as the stack
				// form's run's code is compiled again at its exits
				// (jitRecompileInRun).
				if ne := r.jitRegReoptimize(f.cl, c, e); ne != nil {
					e = ne
				}
			}
		}
		if resume == 0 {
			ipc = e.ipc(rpc)
			if e.invalid || e.code == nil || ipc < 0 || !e.code.HasEntry(ipc) || wb+e.p.Regs+e.scratch+e.frame > len(r.stack) {
				if ne := r.jitRegResumeSeen(f, c, e, rpc); ne != nil {
					// Compiled again with the entry, gone on in from here.
					e = ne
					continue
				}
				return r.jitRegLeaveTo(f, e, rpc), Undefined, false, nil
			}
		}
		ctx, ret, ok := r.jitRegStretch(f, c, e, wb, ipc, at, resume)
		at, resume = -1, 0
		if !ok {
			return r.jitRegLeaveTo(f, e, rpc), Undefined, false, nil
		}
		next, pc, v, done, err, native := r.jitRegLeft(f, c, e, wb, ctx, ret)
		if !native {
			if !done && err == nil {
				pc = r.jitRegLeaveTo(f, e, pc)
			}
			return pc, v, done, err
		}
		rpc = next
	}
}

// jitRegLeaveTo leaves f, whose run of e's code left it, to the
// interpreter at register pc rpc, and returns rpc. Valid code the frame
// goes on in at the next back edge of the interpreter's, whose check is
// made due at once (jitRegLoop), as the stack form's jitGoOn has it: an
// iteration that left, as V8's code deoptimizes, leaves the rest of that
// iteration to the interpreter, not the rest of the invocation. Before,
// the interpreter ran it all, native code entered again only at a sample
// a full back-edge budget later. Code invalidated, a guard of which
// failed, leaves the interpreter the rest of the invocation, as a
// deoptimized invocation is run.
func (r *Runtime) jitRegLeaveTo(f *frame, e *jitRegEntry, rpc int) int {
	s := r.jit
	s.interpreted++
	if life := e.life; life != nil {
		life.goOns++
		life.ssaStats.ended++
	}
	if cc := f.cl.closureCode; !e.invalid && cc != nil && cc.jitReg != nil && cc.jitReg.code != nil && !cc.jitReg.invalid && len(cc.jitReg.loops) != 0 &&
		!cc.jitReg.life.loopSlow {
		// Code with a loop to go on at: one with none, the frame's
		// invocation ends in the interpreter, and the check made due at
		// once would be its caller's loop's, for nothing -- each of AWFY
		// Richards' runTask's 13 million exits had paid one.
		s.regGoOn, s.regForced = f, true
		r.backEdges = 1
	}
	return rpc
}

// jitRegLeft is what Go makes of the way e's code, run on f's window at wb,
// left in ctx, ret what a return returned: native code goes on at register
// pc next, native; or the frame is done, with v; or it threw err, which no
// handler of the frame's takes; or the interpreter goes on with it at
// register pc pc.
func (r *Runtime) jitRegLeft(f *frame, c *regCode, e *jitRegEntry, wb int, ctx *abi.Context, ret Value) (next, pc int, v Value, done bool, err error, native bool) {
	s := r.jit
	exitPC := int(ctx.ExitPC)
	if ctx.Loop != 0 && ctx.ExitKind != abi.ExitReturn && exitPC < len(c.rc.Code) && regAt(c.code, exitPC).Op() == bytecode.RCallMW {
		// Inside a loop of a built-in's call: Go goes on with it.
		return r.jitRegLoopLeft(f, c, e, wb, ctx)
	}
	switch ctx.ExitKind {
	case abi.ExitReturn:
		return 0, 0, ret, true, nil, false
	case abi.ExitDeopt:
		// A guard failed before its instruction committed anything: the
		// frame holds the state at its start, which the interpreter runs
		// the rest of the invocation from, and the code is invalidated,
		// compiled again from what the interpreter learns once it has
		// spent a fresh budget (jitRegInvalidate).
		s.guards++
		e.life.ssaStats.guards++
		r.jitRegMaterialize(f, e, wb)
		if int64(ctx.ExitSite) < 0 {
			// An entry's speculation, at a loop's header or the start: the
			// entry's, as the stack form's code learns it (EntryGeneric).
			r.jitRegEntrySeen(f, exitPC)
		} else if !r.jitRegBigIntSeen(f, e, wb, int(int64(ctx.ExitSite))) {
			// An operator of BigInts it learned is not generic: the code
			// compiled again makes it of them (jitBigIntNote).
			r.jitRegDeoptSeen(f, c, e, wb, exitPC)
		}
		r.jitRegInvalidate(f.cl, e)
		return 0, exitPC, Undefined, false, nil, false
	case abi.ExitHost:
		// Go runs the register instruction whole, then native code goes
		// on at the next entry.
		s.hosts++
		e.life.ssaStats.hosts++
		r.jitRegMaterialize(f, e, wb)
		f.pc = regSrc(c, exitPC) + 1
		var ran bool
		if p := s.regPending; p.set && p.pc == exitPC && p.ctx == int(ctx.Level) {
			// A call Go made from native code, which left after it
			// (jitRegGoCall), or a native call whose callee Go finished
			// (jitRegUnwind): what it answered goes to the frame, as the
			// call's register instruction would have it.
			s.regPending = jitGoPending{}
			if p.whole {
				next, ran, err = r.regFinishWhole(wb, p)
			} else {
				next, ran, err = r.regFinishCall(f, c, wb, exitPC, p.v, p.err)
			}
		} else {
			s.noteRegHost(c, exitPC)
			seen := r.jitRegHostSeen(f, c, wb, exitPC)
			r.jitRegPrivateSeen(f, c, e, wb, exitPC)
			if r.jitRegBigIntExit(f, e, wb, int(int64(ctx.ExitSite))) {
				// A generic operator that met BigInts, which the code
				// compiled again calls Go for (ssa.GoArith).
				r.jitRegReopt(f.cl, e)
			}
			var callee, recv, cb Value
			w := regAt(c.code, exitPC)
			if a := wb + int(w.A()); (w.Op() == bytecode.RNewW || w.Op() == bytecode.RCallMW) && a+2 < len(r.stack) {
				callee, recv, cb = r.stack[a], r.stack[a+1], r.stack[a+2]
			}
			next, ran, err = r.regWhole(f, c, wb, exitPC)
			if seen.write && err == nil && ran {
				r.jitRegWriteSeen(f, c, e, exitPC, seen)
			}
			if err == nil && ran {
				// A call native code did not make, which it may once
				// compiled again; or a loop of a built-in's call it makes
				// whose checks failed before it began -- its callback with
				// no code native callers may call -- which asks for that
				// code, as the call made by Go asks (jitRegGoCallOn).
				r.jitRegCallSeen(f, c, e, exitPC, callee)
				if w.Op() == bytecode.RCallMW {
					r.jitRegForEachSeen(f, e, exitPC, callee, recv, cb, int(w.X()))
				}
			}
			if l := e.life; l.polyReopt {
				// The site met what the code was not compiled for and
				// its cache cannot hold: compiled again for it once that
				// is due, as the stack form's code is (reoptPending,
				// jitRecompileInRun), its code valid meanwhile.
				l.polyReopt = false
				r.jitRegReopt(f.cl, e)
			}
		}
		if !ran {
			return 0, exitPC, Undefined, false, nil, false
		}
		if err != nil {
			return 0, 0, Undefined, false, err, false
		}
		if r.stopped != nil {
			return 0, 0, Undefined, false, r.stopped, false
		}
		return next, 0, Undefined, false, nil, true
	case abi.ExitPoll:
		// The interpreter's back-edge check, at a loop's header, the
		// stretch's work counted (jitRegStretchWork). Code due to be compiled
		// again for what it learned is, and the loop goes on in that code
		// (jitRegGoOnAt), as the stack form's run does at its polls
		// (jitRecompileInRun); before, it went on in the interpreter until
		// a sample a budget later.
		s.budgets++
		e.life.ssaStats.polls++
		r.backEdges = s.backEdgeBudget()
		if exitPC == 0 {
			// The poll at the code's start (ssa.OpPoll), which the run
			// going on there passes again: counted once. Under a stress
			// budget of one back edge, a function entered at its start
			// had polled and left there for ever.
			r.backEdges++
		}
		if err := r.checkInterruptPoll(); err != nil {
			return 0, 0, Undefined, false, err, false
		}
		if r.jitStressDeopt(s) {
			return 0, exitPC, Undefined, false, nil, false
		}
		return exitPC, 0, Undefined, false, nil, true
	}
	// No other way out of register code's native code yet: what it left,
	// the interpreter goes on with from its start.
	r.jitRegMaterialize(f, e, wb)
	return 0, exitPC, Undefined, false, nil, false
}

// jitRegStretch runs e's code once, from its entry at instruction ipc, on
// f's window at wb, in the next context -- or, resume not 0, going on where
// a native call it made returns to (jitRegGoOnAt), in context at -- and
// returns that context with the exit's kind, register pc and the frame
// written -- an exit that left its frame to Go (abi.ExitTable) applied here
// -- and, for a return, the value returned, read before what the run kept
// is let go. A native call whose callee left native code Go finishes
// (jitRegUnwind), and the code goes on after it where it can, as V8's lazy
// deoptimization leaves the caller's frame alone; where it cannot -- code
// a guard of which failed meanwhile, a throw -- the call's state is
// written to the frame, its exit a host exit at the call, which Go
// finishes with the callee's answer (regPending). ok is false where it did
// not run: no context, or the code closed.
func (r *Runtime) jitRegStretch(f *frame, c *regCode, e *jitRegEntry, wb, ipc, at int, resume uintptr) (ctx *abi.Context, ret Value, ok bool) {
	s := r.jit
	idx := s.ctxTop
	if at >= 0 {
		idx = at
	}
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
	if resume == 0 {
		ctx.Cells, ctx.Loop = nil, 0
	}
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
	// A run begun at a loop's header, by on-stack replacement, whose
	// calls of Go are judged as they come (jitRegRunGoProfit).
	osr := resume == 0 && ipc < len(p.Origin) && e.isLoop(int(p.Origin[ipc]))
	s.regRuns[idx] = jitRegRun{f: f, c: c, e: e, wb: wb, osr: osr, edges: r.backEdges - s.pollCredit, rpc: -1}
	s.regPending = jitGoPending{}
	s.entries++
	s.ssaEntries++
	saved, raised := r.stackTop, -1
	if need := wb + p.Regs + e.scratch + e.frame; need > saved && need <= len(r.stack) {
		// The scratch slots, homes and loops' slots past the window,
		// which the stack's top covers while the code runs.
		r.stackTop, r.stackHigh, raised = need, max(r.stackHigh, need), need
	}
	var err error
	edges := r.backEdges - s.pollCredit
	if resume != 0 {
		err = e.code.Resume(resume, s.ssaCtxs[idx+1])
	} else {
		err = e.code.Run(ipc, ctx)
	}
	for err == nil && s.ssaCtxs[idx+1].Live == abi.LiveCall {
		// A native call's callee left native code: Go finishes it, and
		// every native call between, then the code goes on where the
		// call returns to, its state as it left it, the call's result in
		// the callee's context (RetValue), written with Go's barrier -- the
		// one pointer native code stores before it tests the collector's
		// flag again, into a keep cell nothing native reads the old value
		// of (mir's resumeCall) -- the collector marking or not. Not into
		// code that is invalid now, or closed: what Go ran may have run
		// the function again and found its code wanting.
		callee := s.ssaCtxs[idx+1]
		back, base := callee.ReturnTo, callee.Base
		rpc, _ := jit.CallPC(ctx)
		f.pc = regSrc(c, int(rpc)) + 1
		v, uerr := r.jitRegUnwind(idx)
		if uerr == nil && r.stopped == nil && back != 0 && !e.invalid && e.code != nil && e.code.Contains(back) {
			callee.Base, callee.Live, callee.ReturnTo = base, 0, 0
			*(*Value)(unsafe.Pointer(&callee.RetValue)) = v
			s.resumed++
			s.regPending = jitGoPending{}
			err = e.code.Resume(back, callee)
			continue
		}
		// The call's state written to the frame below, the call's exit a
		// host exit at it, which Go finishes with what the callee answered
		// (jitRegLeft).
		jit.ExpandCall(ctx)
		s.regPending = jitGoPending{set: true, pc: int(rpc), ctx: idx, v: v, err: uerr}
		break
	}
	if r.stackTop == raised {
		r.stackTop = saved
	}
	if err == nil && ctx.Cells != nil {
		// Its cells, as the frame has them, for whatever runs it now.
		r.jitRegReopenCells(f, e, wb, ctx, true)
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
		r.jitRegStretchWork(e, ctx, ipc, edges, osr)
		r.jitRegApplyRecords(f, p, wb, ctx)
		if ctx.ExitKind == abi.ExitReturn {
			ret = Value{num: math.Float64frombits(ctx.Ret)}
			if src := jitSource(&ctx.RetFrom); src != nil && src.ref != nil {
				ret = *src
			}
		}
	}
	if err == nil && ctx.ExitKind == abi.ExitHost {
		// An exit inside a callee the code inlined at the call it left at:
		// Go finishes the callee, and the call goes on with what it
		// answered, as a native callee's (regPending).
		rpc := int(ctx.ExitPC)
		f.pc = regSrc(c, rpc) + 1
		if v, uerr, ok := r.jitRegInlined(idx, f.cl, e, rpc, wb); ok {
			s.regPending = jitGoPending{set: true, pc: rpc, ctx: idx, v: v, err: uerr}
		}
	}
	// What the run kept, as jitRunDone clears a stack form's run's, and
	// what its native calls' callees ran with in the contexts past it --
	// closure, keeps, records, result -- each a frame gone once it
	// returned, up to the first context no call ran in.
	clear(ctx.Keep[:e.keeps])
	ctx.This, ctx.Upvalues, ctx.Instance = abi.Slot{}, nil, abi.Slot{}
	if n := max(ctx.RecordHigh, s.callRefs); n != 0 {
		clear(ctx.RecordRef[:n])
		ctx.RecordHigh = 0
	}
	for i := idx + 1; i < len(s.ssaCtxs); i++ {
		c := s.ssaCtxs[i]
		if c.Closure == nil {
			break
		}
		if n := max(c.RecordHigh, s.callRefs); n != 0 {
			clear(c.RecordRef[:n])
			c.RecordHigh = 0
		}
		clear(c.Keep[:s.ssaKeeps])
		c.Closure, c.Upvalues, c.Cells = nil, nil, nil
		c.This, c.RetValue, c.Instance = abi.Slot{}, abi.Slot{}, abi.Slot{}
	}
	s.goRuns[idx] = jitGoRun{}
	s.regRuns[idx] = jitRegRun{}
	s.ctxTop = top
	e.runs--
	closed := e.code == nil
	// Code the cache let go of while this run was in it, closed now if no
	// other is: what goes on goes on in its replacement (jitRegGoOnAt).
	s.regClose(e)
	if closed || err != nil {
		return nil, Undefined, false
	}
	return ctx, ret, true
}

// isLoop reports whether register pc rpc is the header of a loop e's code
// has an entry at.
func (e *jitRegEntry) isLoop(rpc int) bool {
	_, ok := slices.BinarySearch(e.loops, uint32(rpc))
	return ok
}

// jitRegStretchWork counts the work a stretch of e's code did natively,
// begun at instruction ipc with the back-edge counter, less the polls'
// credit, at edges, and left as ctx says, as jitSSAProfit estimates the
// stack form's: the back edges it took, each its loops' mean length, and
// the instructions from its start to where it left. The code spends it
// toward its next compile (jitRegSpent): a loop that leaves at the same
// exit each turn, for what it learned there, had spent nothing between
// its function's calls, and was compiled again only at the next one.
//
// The stretch judges the code, as jitSSAProfit judges the stack form's:
// each stretch -- an entry, a call of Go -- costs what an exit and an
// entry would (jitSSAMinWork); every jitSSAProbe of them, code whose work
// does not make up for them Go no longer enters (entrySlow), at its start
// nor at its loops' headers, its native callers calling it still, until it
// is retried (jitRegRetrySlow); and stretches begun by on-stack
// replacement are judged apart (loopSlow), Go no longer entering the code
// at its loops.
func (r *Runtime) jitRegStretchWork(e *jitRegEntry, ctx *abi.Context, ipc, edges int, osr bool) {
	life := e.life
	st := &life.ssaStats
	work := uint64(0)
	if back := edges - (r.backEdges - r.jit.pollCredit); back > 0 {
		work = uint64(back) * uint64(e.loopMean)
	}
	if ctx.ExitKind == abi.ExitHost || ctx.ExitKind == abi.ExitDeopt || ctx.ExitKind == abi.ExitReturn {
		if x := e.ipc(int(ctx.ExitPC)); x > ipc {
			work += uint64(x - ipc)
		}
	}
	st.work += work
	if osr {
		st.osrStretches++
		st.osrWork += work
		if st.osrStretches%jitSSAProbe == 0 && st.osrWork < st.osrStretches*jitSSAMinWork {
			life.entrySlow, life.loopSlow = true, true
		}
	}
	stretches := life.starts + st.entries + st.goCalls
	if stretches-st.judged < jitSSAProbe {
		return
	}
	st.judged = stretches
	if st.work-st.workBase < (stretches-st.stretchBase)*jitSSAMinWork {
		life.entrySlow = true
	}
}

// jitRegRunGoProfit judges a call of Go e's code makes at register pc rpc
// in run, a run begun at a loop's header, as jitRunGoProfit does the stack
// form's: each costs native code what an exit and an entry would
// (jitSSAMinWork), against the work done natively since the call before
// -- the back edges taken, at the loops' mean length, or the distance from
// it -- and every jitSSAProbe such calls, code whose calls do not pay Go
// no longer enters (entrySlow, loopSlow), unless it calls functions
// natively, whose work the back edges count only in part.
func (r *Runtime) jitRegRunGoProfit(e *jitRegEntry, run *jitRegRun, rpc int) {
	life := e.life
	st := &life.ssaStats
	edges := r.backEdges - r.jit.pollCredit
	if back := run.edges - edges; back > 0 {
		st.runWork += uint64(back) * uint64(e.loopMean)
	} else if run.rpc >= 0 && rpc > run.rpc {
		st.runWork += uint64(e.ipc(rpc) - e.ipc(run.rpc))
	}
	run.edges, run.rpc = edges, rpc
	st.runGoCalls++
	if st.runGoCalls%jitSSAProbe == 0 && st.runWork < st.runGoCalls*jitSSAMinWork && len(life.nativeCalls) == 0 {
		life.entrySlow, life.loopSlow = true, true
	}
}

// jitRegRetrySlow counts a call Go made of e's code without entering it
// (entrySlow), and has Go enter it again after jitNativeRetry such calls,
// doubled each time it was found not to pay, judging it afresh, as
// jitRetrySlow does the stack form's: code judged on its first stretches,
// its calls not yet decided, may pay once it has learned.
func (r *Runtime) jitRegRetrySlow(e *jitRegEntry) {
	life := e.life
	if life.slowCalls++; life.slowCalls-life.slowSince < jitNativeRetry<<life.slowBackoff {
		return
	}
	st := &life.ssaStats
	life.entrySlow, life.loopSlow = false, false
	life.slowSince, life.slowBackoff = life.slowCalls, min(life.slowBackoff+1, jitNativeBackoffs)
	st.judged = life.starts + st.entries + st.goCalls
	st.workBase, st.stretchBase = st.work, st.judged
	st.runGoCalls, st.runWork, st.osrStretches, st.osrWork = 0, 0, 0, 0
}

// jitRegRun is a register frame native code runs on, in a context
// (jitState's regRuns): f, running c, its window at wb, in e's code.
type jitRegRun struct {
	f  *frame
	c  *regCode
	e  *jitRegEntry
	wb int
	// osr marks a run begun at a loop's header; edges is the back-edge
	// counter, less the polls' credit, when it began or last called Go,
	// and rpc the register pc of that call, -1 for none
	// (jitRegRunGoProfit).
	osr   bool
	edges int
	rpc   int
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
	run := &r.jit.regRuns[idx]
	if run.osr && run.e != nil && run.e.life != nil && jitRegGoSettled(ctx, run.c, int(ctx.ExitPC)) {
		r.jitRegRunGoProfit(run.e, run, int(ctx.ExitPC))
	}
	return r.jitRegGoCallOn(ctx, idx, *run)
}

// jitRegGoSettled reports whether the call of Go ctx makes at register pc
// rpc of c is one Go makes for good, which jitRegRunGoProfit judges: not
// a call of a function, which the code may make natively once it has
// learned the call (jitRegCallSeen), as the stack form judges none that
// is unsettled (jitCallUnsettled).
func jitRegGoSettled(ctx *abi.Context, c *regCode, rpc int) bool {
	if rpc >= len(c.rc.Code) {
		return false
	}
	w := regAt(c.code, rpc)
	switch op := w.Op(); op {
	case bytecode.RCallW, bytecode.RCallMW, bytecode.RNewW, bytecode.RCallWT, bytecode.RCallMWT:
		if ctx.GoOp>>abi.GoOpKindShift&0xff == abi.GoCallWhole {
			return true
		}
		at := 0
		if op.MethodCall() {
			at = 1
		}
		callee := jitGoArg(ctx, at)
		if !callee.IsObject() {
			return true
		}
		fd := callee.Object().fn()
		return fd == nil || fd.native != nil
	}
	return true
}

// jitRegGoCallOn is jitRegGoCall on run's frame, the run's own or a level's
// of its native calls (jitRegGoCallNested).
func (r *Runtime) jitRegGoCallOn(ctx *abi.Context, idx int, run jitRegRun) bool {
	s := r.jit
	f, c := run.f, run.c
	rpc := int(ctx.ExitPC)
	if rpc >= len(c.rc.Code) {
		return false
	}
	if run.e != nil && run.e.life != nil {
		run.e.life.ssaStats.goCalls++
	}
	w := regAt(c.code, rpc)
	op := w.Op()
	argc := int(w.X())
	kind := ctx.GoOp >> abi.GoOpKindShift & 0xff
	if kind == abi.GoCallWhole || (kind == abi.GoCallRest || kind == abi.GoCallStep) && (op == bytecode.RIterRest || op == bytecode.RIterStep) {
		// Or a destructuring's step or rest, which native code had no
		// step of its own for: run whole too.
		return r.jitRegWhole(ctx, idx, run, rpc)
	}
	if op == bytecode.RTypeof && kind == abi.GoCallTypeOf {
		// typeof's name, as regTypeof makes it, which runs nothing.
		*(*Value)(unsafe.Pointer(&ctx.Keep[ctx.GoKeep])) = Str(r.typeofString(jitGoArg(ctx, 0)))
		return true
	}
	switch {
	case op == bytecode.RCallW && kind == 0, op == bytecode.RCallMW && kind == abi.GoCallMethod, op == bytecode.RNewW && kind == abi.GoCallNew:
	case op == bytecode.RCallWT && kind == 0, op == bytecode.RCallMWT && kind == abi.GoCallMethod:
		// A call in tail position native code makes while the stack is
		// shallow, as rhCallWT makes the ordinary call there.
		if r.frameDepth >= tailCallDepth {
			return false
		}
	case op == bytecode.RLit && kind == abi.GoCallMethod && w.X()&bytecode.RLitCall != 0:
		// regexp_lit's call, of the method its expansion read, on its
		// literal's RegExp, of its one argument (compile's regexpLit).
		argc = 1
	case op == bytecode.RGetProp && kind == abi.GoCallGet, op == bytecode.RSetProp && kind == abi.GoCallSet,
		(op == bytecode.RGetMethod || op == bytecode.RGetMethodW) && kind == abi.GoCallGet:
		// get_method's and get_method_w's too, a megamorphic site's miss
		// (ssa's megamorphic read): its value the expansion's, which puts it
		// and the receiver in their registers after the call.
		// A property's read or write whose site found an accessor the code
		// does not inline nor call natively (ssa.AccessorSite's call of
		// Go): made as the interpreter makes it, through the site's cache,
		// the getter or the setter called on the frame.
		f.pc = regSrc(c, rpc) + 1
		last, shared := s.ssaCtx, s.ssaSharedGen
		s.goCalls++
		r.noteRegGoCall(w, false, Value{})
		v, err := r.jitRegPropertyMade(f, w, Atom(ctx.GoOp>>abi.GoOpIndexShift), jitGoArg(ctx, 0), jitGoArg(ctx, 1))
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
	if op.MethodCall() {
		this, at = jitGoArg(ctx, 0), 1
	}
	callee := jitGoArg(ctx, at)
	r.noteRegGoCall(w, false, callee)
	r.stack[top] = callee
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
	} else if op == bytecode.RLit {
		// As regLit calls it: the RegExp given back where the built-in
		// called keeps nothing of it (RMarkLitRecv).
		m := uint8(bytecode.RMarkLitRecv)
		if w.X()&bytecode.RLitDead != 0 {
			m |= bytecode.RMarkDead
		}
		v, err = r.regMarkedCall(f, c, top, 1, rpc, m, uint32(w.C()))
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
	if err == nil && r.stopped == nil {
		// What the call called, which the code may call natively once
		// compiled again (jitRegCallSeen), or make a loop of its own
		// (jitRegForEachSeen).
		r.jitRegCallSeen(f, c, run.e, rpc, callee)
		if op == bytecode.RCallMW {
			r.jitRegForEachSeen(f, run.e, rpc, callee, this, jitGoArg(ctx, at+1), argc)
		}
	}
	if err != nil || r.stopped != nil || run.e.invalid {
		s.regPending = jitGoPending{set: true, pc: rpc, ctx: idx, v: v, err: err}
		return false
	}
	*(*Value)(unsafe.Pointer(&ctx.Keep[ctx.GoKeep])) = v
	return true
}

// jitRegWhole runs, for native code, register code's instruction at
// register pc rpc whole on run's frame (abi.GoCallWhole, ir's Whole), as
// an exit there would have Go run it (regWhole), not leaving native code:
// the registers it reads given the values the code passed (abi.Context's
// GoArgs), its result, the register it writes, to the call's keep cell.
// The code keeps values in the frame's registers as it found them, so
// what the call wrote there is put back -- but a captured register of the
// run's own frame, which is where its cell holds it. What does not go on
// at the next instruction -- a throw, a handler that took one, the
// runtime stopped, the code found invalid -- Go finishes at the exit the
// code then takes (regPending, regFinishWhole): the register the
// instruction wrote, or the handler's, written over the exit's records,
// the instruction not run again. On a native callee's frame (a level),
// what reads the frame's this is left to the exit.
func (r *Runtime) jitRegWhole(ctx *abi.Context, idx int, run jitRegRun, rpc int) bool {
	s := r.jit
	f, c, e, wb := run.f, run.c, run.e, run.wb
	if e == nil || e.p == nil {
		return false
	}
	ipc := e.ipc(rpc)
	if ipc < 0 || ipc+1 >= len(e.p.Code) {
		return false
	}
	if e.p.Origin[ipc+1] == e.p.Origin[ipc] || ipc > 0 && e.p.Origin[ipc-1] == e.p.Origin[ipc] {
		return false
	}
	ops, argc, dest, ok := e.p.Code[ipc].WholeOperands(e.p.Regs)
	whole := ctx.GoOp>>abi.GoOpKindShift&0xff == abi.GoCallWhole
	if !ok || whole && int(ctx.GoOp>>abi.GoOpArgcShift&0xff) != argc || wb+e.p.Regs > len(r.stack) {
		return false
	}
	own := ctx.Live == 0
	switch op := regAt(c.code, rpc).Op(); {
	case !own && (op == bytecode.RLoadThis || op == bytecode.RThis):
		return false
	case op == bytecode.RCloseRegs && (!own || ctx.Cells != nil):
		// A scope's end where the code's captured registers are in cells
		// of their own, which it takes fresh ones for (jitRegCellsClosure):
		// left to the exit, as a native callee's.
		return false
	}
	after := int(e.p.Origin[ipc+1])
	h := c.rc.HandlerAt(rpc)
	if h != nil && int(h.To) == after {
		// A handler at the next instruction, where what threw and what did
		// not would go on alike.
		return false
	}
	var regs [4]int
	var saved [4]Value
	n := 0
	for i := 0; i <= argc; i++ {
		reg := dest
		if i < argc {
			reg = ops[i]
		}
		if reg < 0 || own && e.p.IsCaptured(reg) || slices.Contains(regs[:n], reg) {
			continue
		}
		regs[n], saved[n] = reg, r.stack[wb+reg]
		n++
	}
	for i := range argc {
		if reg := ops[i]; !own || !e.p.IsCaptured(reg) {
			r.stack[wb+reg] = jitGoArg(ctx, i)
		}
	}
	s.goCalls++
	r.noteRegGoCall(regAt(c.code, rpc), true, Value{})
	last, shared := s.ssaCtx, s.ssaSharedGen
	goCallDepth := s.goCallDepth
	s.goCallDepth = r.frameDepth + 1
	next, _, err := r.regWhole(f, c, wb, rpc)
	s.goCallDepth = goCallDepth
	s.ssaCtx = last
	if s.ssaSharedGen != shared {
		r.jitShareContexts(idx, f)
	}
	var v Value
	if dest >= 0 {
		v = r.stack[wb+dest]
	}
	if err != nil || next != after || r.stopped != nil || e.invalid {
		// Left at the call's exit, whose records write the frame as it was
		// before the instruction: then the register it wrote, or the
		// handler's that took what it threw.
		p := jitGoPending{set: true, pc: rpc, ctx: idx, whole: true, next: next, err: err, reg: -1}
		switch {
		case err != nil:
		case h != nil && next == int(h.To):
			p.reg, p.v = int(h.Reg), r.stack[wb+int(h.Reg)]
		default:
			p.reg, p.v = dest, v
		}
		s.regPending = p
		s.wholeLeft++
		return false
	}
	for i := range n {
		r.stack[wb+regs[i]] = saved[i]
	}
	*(*Value)(unsafe.Pointer(&ctx.Keep[ctx.GoKeep])) = v
	return true
}

// regFinishWhole finishes, on the registers of a window at wb, an
// instruction Go ran whole for native code (jitRegWhole) that did not go
// on natively, as p says: the register it wrote written over what the
// exit's records wrote, or its exception, which no handler of the frame's
// took, thrown. It reports as regWhole does.
func (r *Runtime) regFinishWhole(wb int, p jitGoPending) (next int, ok bool, _ error) {
	if p.err != nil {
		return 0, true, p.err
	}
	if p.reg >= 0 {
		r.stack[wb+p.reg] = p.v
	}
	return p.next, true, nil
}

// jitRegPropertyMade makes, for native code, the property read or write w
// of f's register code, of key, on obj -- of val, a write -- as the
// interpreter makes it, through the site's cache: a getter or a setter is
// called, with obj as its this. A write's result is undefined.
func (r *Runtime) jitRegPropertyMade(f *frame, w bytecode.RInstr, key Atom, obj, val Value) (Value, error) {
	var c *propCache
	if site := int(w.C()); site < len(f.cl.ic) {
		c = &f.cl.ic[site]
	}
	if w.Op() == bytecode.RSetProp {
		if obj.IsObject() && c != nil {
			return Undefined, r.setPropCached(c, obj.Object(), key, val, f.cl.fn.Strict)
		}
		return Undefined, r.setValueProp(obj, key, val, f.cl.fn.Strict)
	}
	if obj.IsObject() && c != nil {
		o := obj.Object()
		if v, ok := plainOwn(o, key); ok {
			return v, nil
		}
		if v, ok, err := r.cachedGet(c, o, key); ok {
			return v, err
		}
	}
	return r.getValueProp(obj, key)
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
		switch w.Op() {
		case bytecode.RInstanceOf, bytecode.RTypeof, bytecode.RGetProp, bytecode.RGetMethod:
			// R[A]'s: a getter's, called for get_prop (an accessor
			// inlined, jitRegInlined), as typeof's and instanceof's.
			dest = int(w.A())
		case bytecode.RGetMethodW:
			// R[A]'s, and the receiver to R[A+1], as get_method_w puts
			// them.
			dest = int(w.A())
			r.stack[wb+dest+1] = r.stack[wb+int(w.B())]
		case bytecode.RSetProp:
			// A setter's result, dropped.
			return k.pc, true, nil
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
	// Unwound counts the native calls' callees Go finished after one of
	// them left native code (jitRegUnwind), and Resumed the native callers
	// that went on natively after such a callee.
	Unwound, Resumed uint64
	// HostOps is Hosts by the register instruction Go ran whole.
	HostOps map[string]uint64
	// GoCallOps is the calls of Go native code made that may run script,
	// by the register instruction and what it called -- a built-in by its
	// name, a function as "function" or, where it had code native callers
	// may call, "function with code" -- counted while the runtime reports
	// (SetJITReport).
	GoCallOps map[string]uint64
	// WholeLeft counts the instructions Go ran whole for native code,
	// called from it, that it finished at an exit (jitRegWhole).
	WholeLeft uint64
}

// RegNativeStats reports r's counts (RegNativeStats); zero before any.
func (r *Runtime) RegNativeStats() RegNativeStats {
	s := r.jit
	if s == nil {
		return RegNativeStats{}
	}
	return RegNativeStats{Entries: s.ssaEntries, Hosts: s.hosts, Guards: s.guards, Polls: s.budgets,
		Interpreted: s.interpreted, Records: s.ssaRecords, Materialized: s.materialized, GoCalls: s.goCalls,
		Unwound: s.unwound, Resumed: s.resumed, HostOps: maps.Clone(s.regHostOps), WholeLeft: s.wholeLeft,
		GoCallOps: s.regGoOpsByName()}
}

// jitRegGoKey is what a call of Go register code made made
// (RegNativeStats' GoCallOps): the register instruction and, for a call,
// what it called.
type jitRegGoKey struct {
	op     bytecode.ROp
	whole  bool
	callee string
}

// noteRegGoCall counts, while s reports, a call of Go of register
// instruction w that made what k says, callee the function a call called.
func (r *Runtime) noteRegGoCall(w bytecode.RInstr, whole bool, callee Value) {
	s := r.jit
	if !s.reporting {
		return
	}
	k := jitRegGoKey{op: w.Op(), whole: whole}
	if callee.IsObject() {
		switch fd := callee.Object().fn(); {
		case fd == nil:
			k.callee = "object"
		case fd.native != nil:
			k.callee = fd.name
			if k.callee == "" {
				k.callee = "built-in"
			}
		case fd.closure != nil && !fd.bound && r.jitRegHas(fd.closure):
			k.callee = "function with code"
		case fd.closure != nil && !fd.bound:
			k.callee = "function"
		default:
			k.callee = "bound"
		}
	}
	if s.regGoOps == nil {
		s.regGoOps = map[jitRegGoKey]uint64{}
	}
	s.regGoOps[k]++
}

// regGoOpsByName is regGoOps by a name of each key.
func (s *jitState) regGoOpsByName() map[string]uint64 {
	if len(s.regGoOps) == 0 {
		return nil
	}
	m := make(map[string]uint64, len(s.regGoOps))
	for k, n := range s.regGoOps {
		name := k.op.String()
		if k.whole {
			name = "whole " + name
		}
		if k.callee != "" {
			name += "->" + k.callee
		}
		m[name] += n
	}
	return m
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
	// Compiled here, on the spot: the call runs the code it asks for,
	// whatever the runtime's compiles in the background.
	bg, stressBg := r.jitBackground, r.jitStress.background
	r.jitBackground, r.jitStress.background = false, false
	e := r.jitRegFor(cl, c)
	r.jitBackground, r.jitStress.background = bg, stressBg
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
