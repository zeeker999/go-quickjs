//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"slices"
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
)

// frequencySetup's caller makes, at each invocation, three calls of g in a
// loop, one method call, none of g where k is negative, one construction,
// one call of a pure body, which the call path answers without a frame,
// one of g through Function.prototype.call, whose frame is the built-in's,
// and one call in tail position; tail's call of g is in tail position too.
const frequencySetup = `
	function C(x) { let t = x + 1; this.v = t }
	function g(o) { let t = o.n + 1; o.n = t & 1023; return t }
	var obj = { m(o) { let t = o.n + 2; o.n = t & 1023; return t } }
	function pure(o) { return o.n * 2 }
	function tail(o) { return g(o) }
	function caller(o, k) {
		let s = 0
		for (let i = 0; i < 3; i++) s += g(o)
		s += obj.m(o)
		if (k < 0) s += g(o)
		const c = new C(s)
		s += c.v + pure(o)
		s += g.call(null, o)
		return tail(o) + s
	}
	var o = {n: 0}`


// frequencyCounts is name's function's counts: its runs, and each call's
// count, in the order of their PCs.
func frequencyCounts(t *testing.T, r *Runtime, name string) (uint32, []uint32) {
	t.Helper()
	e := jitCallbackEntry(r, name)
	if e == nil || e.calls == nil {
		t.Fatalf("%s counts nothing: %+v", name, e)
	}
	return e.calls.runs, slices.Clone(e.calls.calls)
}





// jitFnForTest is the function of the global name.
func jitFnForTest(t *testing.T, r *Runtime, name string) *bytecode.Function {
	t.Helper()
	return r.global.getOwn(r.atoms.intern(name)).value().Object().fn().closure.fn
}

