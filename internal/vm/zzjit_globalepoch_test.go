//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"fmt"
	"runtime/debug"
	"testing"
)

// Native code reads a global's binding at its position in the global
// object's table while the runtime's count of changes to the bindings
// other than appends, and of script-level lexical bindings declared, is
// what it was compiled with (shapeTree's globalEpoch, ssa.GlobalSite's
// Epoch): a value assigned is read as it is; a binding deleted and added
// again, one turned into an accessor, one a later script's let shadows,
// and globals added meanwhile, are each read as the interpreter reads
// them, with the collector idle and marking.
func TestJITGlobalEpoch(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	gc := debug.SetGCPercent(-1)
	t.Cleanup(func() { debug.SetGCPercent(gc) })
	const setup = `
		// k configurable, as an assignment makes it, which a delete takes and
		// a later script's let may shadow.
		globalThis.k = 3; var step = 1;
		function add(x) { return x + step; }
		function hot(n) { var s = 0; for (var i = 0; i < n; i++) s = (s + add(i) * k) | 0; return s; }
		function addGlobals() { for (var g = 0; g < 50; g++) globalThis['extra' + g] = g; }`
	rounds := []string{
		`hot(300)`, `hot(300)`, `hot(300)`,
		`k = 5, hot(300)`,
		`step = 2, add = function (x) { return x - step; }, hot(300)`,
		`addGlobals(), hot(300)`,
		`delete globalThis.k, globalThis.k = 7, hot(300)`,
		`Object.defineProperty(globalThis, 'step', { get() { return 4; }, configurable: true }), hot(300)`,
		`Object.defineProperty(globalThis, 'step', { value: 6, writable: true, configurable: true }), hot(300)`,
		`hot(300)`,
	}
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprint(marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			both := jitBothForTest(t, setup)
			for range 3 {
				both.agree(t, `hot(300)`)
			}
			r := both.jit
			e := r.jit.hint(r.global.getOwn(r.atoms.intern("hot")).value().Object().fn().closure)
			if e == nil || e.ssa == nil {
				t.Fatalf("hot is not native: %+v", e)
			}
			both.agree(t, rounds...)
			// A later script's lexical binding of a name the code reads as a
			// global, in a runtime whose code Go still enters natively: the code
			// reads the let's.
			lex := jitBothForTest(t, setup)
			for range 3 {
				lex.agree(t, `hot(300)`)
			}
			if e := lex.jit.jit.hint(lex.jit.global.getOwn(lex.jit.atoms.intern("hot")).value().Object().fn().closure); e == nil || e.ssa == nil || e.entrySlow {
				t.Fatalf("hot is not entered natively: %+v", e)
			}
			lex.script(t, `let k = 11;`)
			lex.agree(t, `hot(300)`, `hot(300)`, `k = 13, hot(300)`)
		})
	}
}
