//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"testing"
)

// jitShapeRunForTest runs setup in an interpreter and in a runtime that
// compiles at the first call, then each round in both, failing at the
// first answer that differs; for each round it reports how many map
// checks deoptimized (jitState's shapeDeopts), how many exits left an
// operation to Go, and how many times the function name was compiled
// again.
func jitShapeRunForTest(t *testing.T, setup, name string, rounds ...string) (*Runtime, []jitShapeRound) {
	t.Helper()
	want := New(Config{})
	t.Cleanup(func() { want.Close(); want.ReleaseClosed() })
	r := jitRuntimeForTest(t, Config{JIT: true})
	for _, rt := range []*Runtime{want, r} {
		if _, err := rt.Run(compileForTest(t, setup)); err != nil {
			t.Fatal(err)
		}
	}
	cl := r.global.getOwn(r.atoms.intern(name)).value().Object().fn().closure
	var out []jitShapeRound
	for i, src := range rounds {
		wv, err := want.Run(compileForTest(t, "String("+src+")"))
		if err != nil {
			t.Fatal(err)
		}
		var deopts, hosts, reopts uint64
		if r.jit != nil {
			deopts, hosts, reopts = r.jit.shapeDeopts, r.jit.hosts, r.jit.reoptimized
		}
		gv, err := r.Run(compileForTest(t, "String("+src+")"))
		if err != nil {
			t.Fatal(err)
		}
		if got, w := gv.String().Go(), wv.String().Go(); got != w {
			t.Fatalf("round %d, %s: got %s, interpreter %s", i, src, got, w)
		}
		e := r.jit.hint(cl)
		if e == nil || e.ssa == nil {
			t.Fatalf("round %d, %s: %s has no code", i, src, name)
		}
		out = append(out, jitShapeRound{r.jit.shapeDeopts - deopts, r.jit.hosts - hosts, r.jit.reoptimized - reopts})
	}
	return r, out
}

type jitShapeRound struct{ deopts, hosts, compiles uint64 }
