//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"runtime/debug"
	"slices"
	"testing"
)

// A small callee whose loop calls nothing is inlined (ssa's
// maxInlineLoop): EarleyBoyer's sc_assq, a search of a short list, in
// the caller's code, which then makes no call of it. Its returns from
// inside the loop and after it, and a list it walks to the end, are each
// the interpreter's, with the collector idle and marking.
func TestJITInlinesSmallLoop(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	gc := debug.SetGCPercent(-1)
	t.Cleanup(func() { debug.SetGCPercent(gc) })
	const setup = `
		function P(car, cdr) { this.car = car; this.cdr = cdr; }
		function assq(o, al) { while (al !== null) { if (al.car.car === o) return al.car; al = al.cdr; } return false; }
		var al = null;
		for (var k = 0; k < 6; k++) al = new P(new P('k' + k, k * 10), al);
		function unify(n) {
			var s = 0;
			for (var i = 0; i < n; i++) {
				var hit = assq('k' + (i % 8), al);
				s = (s + (hit === false ? -1 : hit.cdr)) | 0;
			}
			return s;
		}`
	for _, marking := range []bool{false, true} {
		if marking {
			jitMarkingForTest(t)
		}
		both := jitBothForTest(t, setup)
		for range 6 {
			both.agree(t, `unify(400)`, `assq('k5', al).cdr`, `assq('none', al)`, `assq('k0', null)`)
		}
		r := both.jit
		callee := r.global.getOwn(r.atoms.intern("assq")).value().Object().fn().closure
		caller := r.global.getOwn(r.atoms.intern("unify")).value().Object().fn().closure
		e := r.jit.hint(caller)
		if e == nil || e.ssa == nil || !slices.Contains(e.ssaInlined, callee) {
			t.Fatalf("unify's code does not inline assq: %+v", e)
		}
	}
}
