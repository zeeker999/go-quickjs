//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
	"weak"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/jit"
	jitcompile "github.com/go-quickjs/go-quickjs/internal/jit/compile"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
	"github.com/go-quickjs/go-quickjs/internal/jit/mir"
	"github.com/go-quickjs/go-quickjs/internal/jit/ssa"
)

// Compiling in the background, as V8's concurrent recompilation does
// (docs/performance-plan.md, D8). A compile is a jitJob in three steps: the
// runtime's goroutine prepares it, building the function from what the
// runtime knows (compileSSAPrepare); a worker of the process's optimizes it
// and runs the backend (jitJob.run), touching nothing of the runtime's; and
// the runtime's goroutine installs it at a later check (jitDrain), placing
// its code, as the compile made on the spot would have once compiled
// (jitCompiled, jitReoptimized). Meanwhile the function runs in the
// existing tiers, or in its code, still valid, that the compile replaces.
//
// The runtime asks it with JITBackground, or QJS_JIT_STRESS's background
// does, which a test that sets its own stress (jitStress) leaves out, as
// it leaves out the budgets and deoptimizations; otherwise it waits for each
// compile, and its tiering is reproducible, which the tests counting
// compiles, the differential fuzzer and the placements need.

// jitJobsMax is how many compiles a runtime has in the background at once,
// as V8 queues at most eight of an isolate's: past it, a function due to be
// compiled waits in the existing tiers, and is asked again at its next
// invocation.
const jitJobsMax = 4

// A jitJob's state, which hands it between its worker and the runtime's
// goroutine: queued for a worker, being worked, done, or abandoned by its
// runtime, whose worker then gives its workspace back (jitJob.run).
const (
	jitJobQueued int32 = iota
	jitJobWorking
	jitJobDone
	jitJobAbandoned
)

// jitInBackground reports whether r compiles fn in the background: a
// runtime that asks it, for a function's own code -- top-level code, which
// runs once, compiled at once -- and not inside another compile, nor while
// a host times the compiles (mir.Timing) or repeats them
// (JITCompileRepeat), which they do on the compiling goroutine, nor while a
// test looks at each compile's SSA (jitSSAOptimized). A compile there that
// finds the runtime with as many in the background as it may (jobsFull)
// waits, asked again at a later check, rather than be made on the spot:
// the callees a compile seeds (jitSeedCalls) are called through their
// links meanwhile, and those it wants (jitWantedCallees) have it compiled
// again once it has spent a budget.
func (r *Runtime) jitInBackground(fn *bytecode.Function) bool {
	return (r.jitBackground || r.jitStress.background) && !fn.TopLevel && r.jit != nil && r.jit.compiling == 0 &&
		mir.Timing == nil && JITCompileRepeat == 0 && jitSSAOptimized == nil
}

// jitPending reports whether fn's entry is being compiled in the
// background.
func (s *jitState) jitPending(fn *bytecode.Function) bool {
	e := s.entryFor(fn)
	return e != nil && e.pending != nil
}

// jobsFull reports whether the runtime has as many compiles in the
// background as it may.
func (s *jitState) jobsFull() bool {
	return len(s.jobs)+len(s.lowerings)+len(s.regJobs) >= jitJobsMax
}

// inFlight reports whether the runtime has compiles or lowerings in the
// background, which a check installs (jitDrain).
func (s *jitState) inFlight() bool {
	return len(s.jobs) != 0 || len(s.lowerings) != 0 || len(s.heldJobs) != 0 || len(s.regJobs) != 0
}

// A compile on the spot finds its callees' code there, a callee's compile
// being made within its caller's, and inlines those it may; in the
// background it would find them being compiled, call them natively, and
// be compiled again only for what an exit teaches it, which a call
// made natively, from a caller made natively, never does. So a function's
// first compile that met a callee small enough to inline, being compiled,
// is held (jitHold): it asks nothing yet, the function running in the
// existing tiers, as it does while it compiles, and is prepared once its
// callees have their code (jitDrain), its calls decided then as a compile on
// the spot decides them. A held compile holds no place in the background
// (jitJobsMax) and no workspace.
//
// Compiles may wait for each other, of functions calling each other: one
// that would wait for a compile that waits for it, however many others are
// between them, does not (jitCompileWaits), so that those held wait for no
// cycle and each is prepared in the end. jitHoldRounds bounds how often one
// is held, its callees' callees each new to it.
//
// A compile a frame waits for at a loop does not wait for a call the frame
// has left behind (jitFrameLeft): the wait would be the loop's, in the
// existing tiers, for a call it does not make.
const jitHoldRounds = 3

// jitAwaits has e's compile, held if any call has it wait (jitHold), wait for
// cl's function's, if that is what the decision of fn's call at pc hangs on,
// and reports whether it does: a callee being compiled, or to be, short
// enough to be inlined once it is, which does not wait for e's function in
// turn.
func (r *Runtime) jitAwaits(e *jitEntry, fn *bytecode.Function, pc int, cl *closure) bool {
	if !e.holds || len(cl.fn.Code) > ssa.MaxInline || r.jitFrameLeft(fn, pc) || !r.jitCalleeAwaited(cl) || r.jit.jitCompileWaits(cl.fn, fn) {
		return false
	}
	if !slices.Contains(e.awaited, cl.fn) {
		e.awaited = append(e.awaited, cl.fn)
	}
	return true
}

// jitFrameLeft reports whether fn's compile is asked for by a frame waiting
// at one of its loops, and the call at pc is behind that frame for good:
// before the loop, in no loop around both. A function run once for its loop,
// as each of microbench's is, had that loop wait in the existing tiers for
// the compile of what set it up.
func (r *Runtime) jitFrameLeft(fn *bytecode.Function, pc int) bool {
	return r.jitFrameFn == fn && pc < r.jitFramePC && jitLoopAround(fn, pc) == 0
}

// jitLoopAround is the header of the innermost of fn's loops pc is in: where
// the first backward jump from pc on to pc or before it goes. 0 if there is
// none, which no compile takes for a header.
func jitLoopAround(fn *bytecode.Function, pc int) int {
	for q := pc; q < len(fn.Code); q++ {
		switch in := fn.Code[q]; in.Op {
		case bytecode.OpJump, bytecode.OpJumpIfFalse, bytecode.OpJumpIfTrue, bytecode.OpJumpIfCmpFalse:
			if int(in.A) <= pc {
				return int(in.A)
			}
		}
	}
	return 0
}

// jitCompileWaits reports whether fn is target, or fn's compile is held for
// target's, through those it waits for, whatever the number between them:
// left out, those past it would wait in a cycle for good, their functions
// compiled never.
func (s *jitState) jitCompileWaits(fn, target *bytecode.Function) bool {
	seen := []*bytecode.Function{fn}
	for i := 0; i < len(seen); i++ {
		if seen[i] == target {
			return true
		}
		if e := s.entryFor(seen[i]); e != nil && e.pending != nil && e.pending.held {
			for _, a := range e.awaited {
				if !slices.Contains(seen, a) {
					seen = append(seen, a)
				}
			}
		}
	}
	return false
}

// jitHold holds e's compile of fn's program p, for its closure cl, until the
// compiles in e.awaited are done: busy and pending as a compile handed over
// is (jitSubmit), installed by install once it is. A frame waiting at a loop
// for it is told of as it would have been (jitFrameAt).
func (r *Runtime) jitHold(e *jitEntry, fn *bytecode.Function, cl *closure, p *ir.Program, install func(*Runtime, *jitJob) bool) {
	s := r.jit
	h := &jitJob{fn: fn, cl: cl, p: p, e: e, realm: r.Realm, install: install, held: true, holdRound: 1}
	if r.jitFrameFn == fn {
		h.frameAt = r.jitFramePC
	}
	e.pending = h
	e.busy++
	s.heldJobs = append(s.heldJobs, h)
	s.bgHeld++
}

// jitHeldReady reports whether h's compile may be prepared: the compiles it
// waited for are done, and there is a place for it.
func (r *Runtime) jitHeldReady(h *jitJob) bool {
	s := r.jit
	if s.jobsFull() {
		return false
	}
	for _, fn := range h.e.awaited {
		if s.jitPending(fn) {
			return false
		}
	}
	return true
}


// jitDrain installs the compiles in the background that are done, in the
// order they were handed over, at a check of the runtime's goroutine: a
// function's entry, a loop's back edge (jitRegDrainJobs). An install
// may find it is to wait (a native call of the code it replaces, under
// way), and is asked again at the next check. Under QJS_JIT_STRESS's
// background=K, each compile is installed at the Kth check after it was
// handed over, waited for if need be, and no other: when code is installed
// does not depend on how fast the workers are.
func (r *Runtime) jitDrain() {
	s := r.jit
	if s == nil || s.draining || !s.inFlight() {
		return
	}
	// An install compiles callees, and may hand over compiles of its own,
	// which this drain takes in turn, not one within it.
	s.draining = true
	defer func() { s.draining = false }()
	stress, k := r.jitStress.background, int(r.jitStress.checks)
	r.jitRegDrainJobs(stress, k)
}

// abandonJobs lets the runtime's compiles in the background go, as it
// releases its JIT: their entries no longer wait for them, and each
// workspace goes back to the process's pool, by the worker if it still
// has it.
func (s *jitState) abandonJobs() {
	for _, j := range s.jobs {
		for {
			st := j.state.Load()
			if st == jitJobDone {
				jit.PutWorkspace(j.w)
				j.w = nil
				break
			}
			if j.state.CompareAndSwap(st, jitJobAbandoned) {
				break
			}
		}
		j.e.pending = nil
		j.e.busy--
	}
	s.jobs = nil
	for _, h := range s.heldJobs {
		h.e.pending, h.e.awaited = nil, nil
		h.e.busy--
	}
	s.heldJobs = nil
	for _, t := range s.lowerings {
		t.state.Store(jitJobAbandoned)
	}
	s.lowerings = nil
	s.jitRegAbandonJobs()
}

// jitWorkers are the process's compile workers, shared by its runtimes:
// at most a quarter of GOMAXPROCS, at least one, each started when a
// compile is handed over and none is free, each ending once none is
// queued, so that an idle process runs none.
var jitWorkers jitWorkerPool

type jitWorkerPool struct {
	mu      sync.Mutex
	queue   []jitTask
	running int
}

// jitTask is what a worker runs: a compile's work (jitJob), a lowering
// (jitLowerTask).
type jitTask interface{ run() }

// jitWorkerLimit is how many workers compile at once.
func jitWorkerLimit() int { return max(1, runtime.GOMAXPROCS(0)/4) }

// submit queues j, starting a worker if fewer than the limit run.
func (p *jitWorkerPool) submit(j jitTask) {
	p.mu.Lock()
	p.queue = append(p.queue, j)
	start := p.running < jitWorkerLimit()
	if start {
		p.running++
	}
	p.mu.Unlock()
	if start {
		go p.work()
	}
}

// work runs the queued compiles, first queued first, until none is left.
func (p *jitWorkerPool) work() {
	for {
		p.mu.Lock()
		if len(p.queue) == 0 {
			p.running--
			p.mu.Unlock()
			return
		}
		j := p.queue[0]
		p.queue[0] = nil
		p.queue = p.queue[1:]
		p.mu.Unlock()
		j.run()
	}
}


// jitWorkerPanics counts the panics the process's workers made refusals
// of, whatever runtime's: one whose runtime never installs it is counted
// too, which the tests look for (TestMain).
var jitWorkerPanics atomic.Int64

// jitLowerTask is a function's lowering in the background, for its first
// compile there (jitLowerLater): the function's own program, of its
// bytecode, which runtimes share, and a copy of its hints, which a worker
// lowers, and the runtime's goroutine takes into the function's lowering
// (jitLowered) at a later check, as jitLower would have.
type jitLowerTask struct {
	fn    *bytecode.Function
	kind  jitLowerKind
	hints jitcompile.Hints
	// l is the lowering it is for, at serial: one dropped since, or
	// relowered for other hints, does not take it.
	l      *jitLowering
	serial uint32
	p      *ir.Program
	err    error
	state  atomic.Int32
	done   chan struct{}
	checks int
}

// run is t's lowering on a worker; the lowering recovers a panic of its
// own into a refusal (jitcompile's lowerRecovered).
func (t *jitLowerTask) run() {
	defer close(t.done)
	if !t.state.CompareAndSwap(jitJobQueued, jitJobWorking) {
		return
	}
	t.p, t.err = jitLowerWith(t.fn, t.kind, t.hints)
	t.state.CompareAndSwap(jitJobWorking, jitJobDone)
}

// jitLowered takes t's program into its function's lowering, as jitLower
// would have lowered it there -- unless the lowering went, was relowered,
// or has the program already.
func (r *Runtime) jitLowered(t *jitLowerTask) {
	s := r.jit
	l := t.l
	if s.lowered[weak.Make(t.fn)] != l || l.serial != t.serial || l.done[t.kind] {
		s.bgDiscarded++
		return
	}
	l.progs[t.kind], l.errs[t.kind], l.done[t.kind] = t.p, t.err, true
	n := t.p.Footprint()
	l.bytes += n
	s.lowerBytes += n
	// In the room made for its compile when it was handed over, which
	// other compiles may have taken since.
	s.settle()
}
