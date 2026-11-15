//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"testing"
)

// lazyFramesSetup recurses natively, each level a native call, and at the
// bottom calls through Go what the JIT does not compile -- a function with
// a with statement -- which reads the frames of the levels above it: a
// stack trace's, a thrown error's, a sloppy function's caller chain. It
// also has code compiled again while the levels are in theirs --
// recompile, the host's, compiles helper again, which no level is in:
// its old code is not closed until they have left (closeSSA), as no code
// is while a call has put their frames off. The levels' own code, which
// they return into, is not compiled again while they are in it
// (otherwiseInUse).
const lazyFramesSetup = `
	function trace() { with ({}) {} return new Error("e").stack.split("\n").slice(1, 12).join("|") }
	function thrower() { with ({}) {} throw new Error("t") }
	function callers() { with ({}) {} var n = 0; for (var f = callers.caller; f && n < 30; f = f.caller) n++; return n }
	function P() {} P.prototype.k = 1
	var objs = [new P(), new P()]
	function helper(x) { return x + 1 }
	function change() { with ({}) {} P.prototype.k = P.prototype.k + 1; return recompile() }
	function down(n, what) {
		if (n === 0) {
			if (what === 0) return trace()
			if (what === 1) return callers()
			if (what === 2) return change() + objs[n & 1].k
			return thrower()
		}
		return helper(down(n - 1, what)) - 1 + (what === 2 ? objs[n & 1].k : 0)
	}
	function sloppyDown(n) { return n === 0 ? callers() : sloppyDown(n - 1) }
	function caught(n) { try { return down(n, 3) } catch (e) { return e.stack.split("\n").slice(1, 12).join("|") } }`

func lazyFramesRounds() []string {
	var rounds []string
	for range 30 {
		rounds = append(rounds, `down(20, 0)`, `sloppyDown(20)`, `down(20, 2)`, `caught(20)`)
	}
	return rounds
}

func jitLazyFramesRan(t *testing.T, r *Runtime) {
	t.Helper()
	if s := r.jit; s.lazyCalls == 0 || s.lazyMade == 0 || s.lazyHeld == 0 || len(s.lazy) != 0 || s.lazyTop != 0 {
		t.Fatalf("frames put off by %d calls, made %d times, %d codes held for them, %d still put off (top %d); want each, none left",
			s.lazyCalls, s.lazyMade, s.lazyHeld, len(s.lazy), s.lazyTop)
	}
	if len(r.jit.goPinned) != 0 || len(r.jit.retired) != 0 {
		t.Fatalf("calls of Go left %d pinned, %d retired", len(r.jit.goPinned), len(r.jit.retired))
	}
}
