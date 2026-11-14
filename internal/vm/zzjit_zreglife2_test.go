//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"strings"
	"testing"
)

// regBothMarking runs f with the collector idle and then, native code
// seeing it mark, marking (jitMarkingForTest), as the JIT's marking rule
// asks of a test of a native path.
func regBothMarking(t *testing.T, f func(t *testing.T)) {
	t.Run("idle", f)
	t.Run("marking", func(t *testing.T) {
		jitMarkingForTest(t)
		f(t)
	})
}

// TestJITRegNoContextDeclines runs a recursion of native calls deeper than
// the JIT has contexts (jitContextsMax), whose deepest levels call a
// compiled leaf, and checks that Go runs the frames it calls past the last
// context in the interpreter without entering their code, as the stack
// form's runSSA declines them: no run ends in the interpreter for want of
// one. Before, each such call entered its callee's code, found no context
// for its run and finished in the interpreter, counted as an entry that did
// no work: AWFY's Havlak, whose
// depth-first search recurses past the contexts, finished 2.7 million
// runs so, each judged against its code.
func TestJITRegNoContextDeclines(t *testing.T) {
	const src = `
function leaf(o) { return o.v + 1 }
function rec(d, o) { if (d == 0) return 0; var s = leaf(o); return s + rec(d - 1, o) }
function run() { var o = { v: 1 }, t = 0; for (var i = 0; i < 20; i++) t += rec(300, o); return t }
`
	regBothMarking(t, func(t *testing.T) {
		regNoInline(t)
		rts := regNativeRuntimes(t, src)
		r := rts[0]
		r.jitStress = jitStressConfig{}
		r.SetJITWarmUp(1, 1)
		r.SetJITReport(true)
		var got [2]string
		for i, rt := range rts {
			fn, _ := rt.GetProp(Obj(rt.Global()), rt.Intern("run"))
			for range 6 {
				v, err := rt.Call(fn, Undefined, nil)
				got[i] = regNativeShow(rt, v, err)
			}
		}
		if got[0] != got[1] || got[0] != "12000" {
			t.Fatalf("with the JIT %q, without %q, want 12000", got[0], got[1])
		}
		if len(r.jit.ssaCtxs) < jitContextsMax {
			t.Skipf("the recursion took %d contexts, not all of them", len(r.jit.ssaCtxs))
		}
		ended := r.RegNativeStats().Ended
		t.Logf("contexts %d; ended %v", len(r.jit.ssaCtxs), ended)
		for k, n := range ended {
			if strings.HasPrefix(k, "no context") {
				t.Errorf("%d runs entered past the last context: %s", n, k)
			}
		}
	})
}
