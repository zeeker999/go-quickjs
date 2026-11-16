//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

// regStressForTest gives r what CI's stress step gives every runtime
// (QJS_JIT_STRESS=threshold,budget=3,deopt=7): a function compiled once
// it has run, a poll every three back edges, and every seventh poll
// finishing the invocation in the interpreter. The
// register tests' runtimes run without it (regNativeRuntimes), their counts
// not written for it; their answers are checked under it here.
func regStressForTest(r *Runtime, c jitStressConfig) {
	r.jitCallThreshold, r.jitICThreshold = 1, 1
	c.threshold = true
	r.jitStress = c
	r.jitState().stressBudget = int(c.budget)
}
