//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"testing"
)

// jitLazyFramesAgree runs lazyFramesRounds in the interpreter and with the
// JIT, recompile the host's in both, and checks each answer is the
// interpreter's.
func jitLazyFramesAgree(t *testing.T) *Runtime {
	t.Helper()
	install := func(rt *Runtime) {
		rt.global.setOwnRaw(rt.atoms.intern("recompile"), rt.NewFunction("recompile", 0,
			func(rt *Runtime, _ Value, _ []Value) (Value, error) {
				if rt.jit == nil {
					return Int(0), nil
				}
				cl := rt.global.getOwn(rt.atoms.intern("helper")).value().Object().fn().closure
				if jitRegClosureForTest(rt, cl) {
					// The register cache lets go of helper's code, which is
					// compiled again at its next calls: closed once no frame
					// put off is in it (regClose, closeSSA).
					if e := rt.jit.regCache[rt.jitRegKeyOf(cl)]; e != nil && e.code != nil {
						rt.jit.dropReg(e.key, e)
					}
					return Int(0), nil
				}
				if e := rt.jit.cache[rt.jitKeyOf(cl.fn)]; e != nil && e.ssa != nil {
					e.reopt = true
					rt.jitReoptimize(cl, e)
				}
				return Int(0), nil
			}), propDefault)
	}
	want := New(Config{})
	t.Cleanup(func() { want.Close(); want.ReleaseClosed() })
	r := jitRuntimeForTest(t, Config{JIT: true})
	for _, rt := range []*Runtime{want, r} {
		install(rt)
		if _, err := rt.Run(compileForTest(t, lazyFramesSetup)); err != nil {
			t.Fatal(err)
		}
	}
	for i, src := range lazyFramesRounds() {
		src = "''+(" + src + ")"
		wv, werr := want.Run(compileForTest(t, src))
		gv, gerr := r.Run(compileForTest(t, src))
		switch {
		case (werr == nil) != (gerr == nil):
			t.Fatalf("round %d (%s): error %v, the interpreter's %v", i, src, gerr, werr)
		case werr != nil:
			if gerr.Error() != werr.Error() {
				t.Fatalf("round %d (%s): error %q, the interpreter's %q", i, src, gerr, werr)
			}
		case gv.String().Go() != wv.String().Go():
			t.Fatalf("round %d (%s): got %s, the interpreter's %s", i, src, gv.String().Go(), wv.String().Go())
		}
	}
	return r
}
