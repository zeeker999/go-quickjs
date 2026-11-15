//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"runtime/debug"
	"slices"
	"testing"
)


// An inlined callee is the code itself: an exit inside it leaves the code
// there, as the code's own exits do, and Go goes on after the call, so the
// call keeps its entry from the first compile. AWFY's Queens: placeQueen
// built with none after its inlined getRowColumn and setRowColumn held its
// column boxed through its loop, unboxed at each use, and spilled twice
// as many values. Each answer is the interpreter's.
func TestJITEntriesAfterInlinedCalls(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	both := jitBothForTest(t, `
		var Q = class {
			constructor() { this.rows = [true, true, true, true, true, true, true, true]; this.maxs = []; this.mins = []; for (let i = 0; i < 16; i++) { this.maxs[i] = true; this.mins[i] = true } }
			get(r, c) { return this.rows[r] && this.maxs[c + r] && this.mins[c - r + 7] }
			set(r, c, v) { this.rows[r] = v; this.maxs[c + r] = v; this.mins[c - r + 7] = v }
			place(c) {
				for (let r = 0; r < 8; r += 1) {
					if (this.get(r, c)) {
						this.set(r, c, false)
						if (c === 7 || this.place(c + 1)) return true
						this.set(r, c, true)
					}
				}
				return false
			}
		}
		function run() { let n = 0; for (let i = 0; i < 4; i++) if (new Q().place(0)) n++; return n }`)
	for range 20 {
		both.agree(t, `run()`)
	}
	r := both.jit
	cl := r.global.getOwn(r.atoms.intern("Q")).value().Object().getOwn(r.atoms.intern("prototype")).value().Object().getOwn(r.atoms.intern("place")).value().Object().fn().closure
	e := r.jit.hint(cl)
	if e == nil || e.ssa == nil || len(e.ssaInlinedAt) == 0 {
		t.Fatalf("place inlines none of its calls: %+v", e)
	}
	for _, pc := range e.ssaInlinedAt {
		if !e.ssa.HasEntry(int(pc)+1) || slices.Contains(e.ssaUnresumed, pc+1) {
			t.Errorf("place's code has no entry after its inlined call at %d", pc)
		}
	}
}
