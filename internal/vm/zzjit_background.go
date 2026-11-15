//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/jit/mir"
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

// jobsFull reports whether the runtime has as many compiles in the
// background as it may.
func (s *jitState) jobsFull() bool {
	return len(s.regJobs) >= jitJobsMax
}

// inFlight reports whether the runtime has compiles or lowerings in the
// background, which a check installs (jitDrain).
func (s *jitState) inFlight() bool {
	return len(s.regJobs) != 0
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



