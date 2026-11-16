//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/jit"
)

// jitRunFirstForTest runs the script fn in r, as Run does, its functions
// compiled at their first call where r runs register code: the stack
// form's JIT compiles a function at its first call (jitCallThreshold 1,
// as jitRuntimeForTest sets it), before its sites have run, where the
// register form's first call runs in the interpreter and begins its
// budget, its next call compiling it (jitRegCallTier). Each template the
// script prepares -- its functions', and theirs within them -- has its
// first call sent to Go's tier-up (closureCode's jitEnters), which, its
// budget one invocation, compiles it there, as the stack form's does: so
// that what a test asks of native code is asked of the register form's at
// the same call. Under the stack form it is Run.
func jitRunFirstForTest(r *Runtime, fn *bytecode.Function) (Value, error) {
	if !r.regOn() || !r.jitEnabled {
		return r.Run(fn)
	}
	_, v, err := jitRunScriptForTest(r, fn, true)
	return v, err
}

// jitRunScriptForTest runs the script fn in r as Run does, and returns
// the closure it prepared for it, whose constants hold the templates of
// the functions it declares and makes (jitEnterFirstForTest). first has
// them compiled at their first call (jitRunFirstForTest).
func jitRunScriptForTest(r *Runtime, fn *bytecode.Function, first bool) (*closure, Value, error) {
	cl := prepareRun[struct{}](r, fn)
	if first {
		jitEnterFirstForTest(cl)
	}
	this := r.globalThis
	if fn.IsModule {
		this = Undefined
	}
	v, err := r.run(cl, this, nil, Undefined, nil)
	r.endTurn()
	return cl, v, err
}

// jitEnterFirstForTest has the next call of each template within cl's
// constants, and within theirs, go to Go's tier-up (jitRunFirstForTest),
// where r runs register code.
func jitEnterFirstForTest(cl *closure) {
	for _, c := range cl.consts {
		if tmpl := c.template(); tmpl != nil && tmpl.closureCode != nil {
			tmpl.closureCode.jitEnters = true
			jitEnterFirstForTest(tmpl)
		}
	}
}

// jitRunFirstSourceForTest is jitRunFirstForTest of source compiled.
func jitRunFirstSourceForTest(t *testing.T, r *Runtime, source string) (Value, error) {
	t.Helper()
	return jitRunFirstForTest(r, compileForTest(t, source))
}

// jitWarmForTest runs source in r, a runtime with the JIT, with its JIT
// off, as tests that fill their sites' caches before the JIT compiles do,
// and returns a function that turns the JIT on again. The warm-up runs on
// the tiers the JIT's code is compiled from: a runtime whose JIT is off
// runs register code (regOn), so one whose JIT is the stack form's keeps
// the old tiers meanwhile (oldTiers), whose caches its compiles read.
// Where r runs register code, the functions source made, whose calls with
// the JIT off had their budgets stop counting (jitRegCallTier,
// regEdgesOff), as the stack form's closures never stop, have their next
// call go to Go's tier-up, which compiles them there
// (jitEnterFirstForTest), as the stack form compiles a warmed function at
// its next call.
func jitWarmForTest(t *testing.T, r *Runtime, source string) func() {
	t.Helper()
	reg, old := r.regOn(), r.oldTiers
	r.oldTiers = old || !reg
	r.jitEnabled = false
	cl, _, err := jitRunScriptForTest(r, compileForTest(t, source), false)
	r.oldTiers = old
	if err != nil {
		t.Fatal(err)
	}
	return func() {
		r.jitEnabled = true
		if reg {
			jitEnterFirstForTest(cl)
		}
	}
}

// jitClosureCodeForTest is the state of cl's native code, nil where it
// has none: the stack form's entry with code (entryOf), or, where cl runs
// register code, its register code's that outlives its compiles (life),
// whose counts and judgments -- entrySlow among them -- its runs keep.
func jitClosureCodeForTest(r *Runtime, cl *closure) *jitEntry {
	if e := jitClosureRegForTest(r, cl); e != nil {
		return e.life
	}
	if r.jit == nil || cl.reg != nil && r.regOn() {
		return nil
	}
	if e := r.entryOf(cl); e != nil && e.ssa != nil {
		return e
	}
	return nil
}

// jitClosureRegForTest is the cache's register code of cl's function
// with code, nil where it has none or cl does not run register code.
func jitClosureRegForTest(r *Runtime, cl *closure) *jitRegEntry {
	if r.jit == nil || cl.reg == nil || !r.regOn() || cl.closureCode == nil {
		return nil
	}
	if e := r.jit.regCache[r.jitRegKeyOf(cl)]; e != nil && e.code != nil {
		return e
	}
	return nil
}

// jitClosureSSAForTest is cl's native code, nil where it has none, of
// either form.
func jitClosureSSAForTest(r *Runtime, cl *closure) *jit.SSACode {
	if e := jitClosureRegForTest(r, cl); e != nil {
		return e.code
	}
	if e := jitClosureCodeForTest(r, cl); e != nil {
		return e.ssa
	}
	return nil
}

// jitClosureRefusedForTest reports whether the JIT refused cl's function
// for good: its stack form's, or its register code's.
func jitClosureRefusedForTest(r *Runtime, cl *closure) bool {
	if cl.reg != nil && r.regOn() {
		return cl.closureCode.jitRegRefused
	}
	return cl.jitRefused
}

// jitCodeBytesForTest is the size of the native code r holds, of either
// form: the stack form's entries' (jitCodeBytes), and the register form's
// code in its cache.
func jitCodeBytesForTest(r *Runtime) int64 {
	n := r.jitCodeBytes()
	if r.jit != nil {
		for _, e := range r.jit.regCache {
			if e.code != nil {
				n += int64(e.code.Size())
			}
		}
	}
	return n
}
