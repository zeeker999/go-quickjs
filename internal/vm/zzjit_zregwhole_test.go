//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"fmt"
	"testing"
)

// regWholeSrc runs register instructions Go runs whole for native code
// (jitRegWhole): private members' reads and writes, a private getter that
// assigns a captured binding and one that throws now and then, caught in
// the same frame or not; a for...in's and a for...of's cursor; object and
// array spreads, a spread call and a computed key's definition; a private
// read into a binding a closure captures; a native callee's private read;
// and a getter that changes a global the code was compiled to read as a
// constant, which finds the code invalid after the call.
const regWholeSrc = `
var seen = 0;
var K = 3;
class P {
	#x = 1; #y = 2;
	get #g() { seen++; return this.#x + 1 }
	get #t() { if (this.#x % 37 == 36) throw new RangeError("t " + this.#x); return this.#x }
	get #k() { if (this.#x == 400) K = 4; return this.#x }
	step(i) { this.#x = this.#x + i % 5; this.#y = this.#x ^ i; return this.#y }
	read() { return this.#x }
	g() { return this.#g }
	t() { return this.#t }
	k() { return this.#k }
}
function privates(n) {
	var p = new P(), s = 0;
	for (var i = 0; i < n; i++) s = (s + p.step(i) + p.read() + p.g()) % 1000003;
	return s + ":" + seen;
}
function caught(n) {
	var p = new P(), s = 0, e = "";
	for (var i = 0; i < n; i++) {
		p.step(1);
		try { s += p.t() } catch (err) { e = err.message }
	}
	return s + e;
}
function uncaught(n) {
	var p = new P(), s = 0;
	for (var i = 0; i < n; i++) { p.step(1); s += p.t() }
	return s;
}
function captured(n) {
	var p = new P(), v = 0, f = function () { return v * 2 }, s = 0;
	for (var i = 0; i < n; i++) { p.step(i); v = p.read(); s += f() }
	return s;
}
function invalid(n) {
	var p = new P(), s = 0;
	for (var i = 0; i < n; i++) { p.step(1); s += p.k() + K }
	return s;
}
function forin(n) {
	var o = { a: 1, b: 2, c: 3 }, s = 0, k;
	for (var i = 0; i < n; i++) for (k in o) s += o[k];
	return s;
}
function spreads(n) {
	var o = { a: 1, b: 2 }, a = [1, 2, 3], s = 0;
	function f(x, y, z) { return x + y + z }
	for (var i = 0; i < n; i++) {
		var c = { ...o, d: i }, b = [...a, i];
		s += c.a + c.d + b[3] + f(...a) + { ["k" + (i % 3)]: i }["k" + (i % 3)];
	}
	return s;
}
`

// TestJITRegWholeGoCalls runs regWholeSrc's functions in a runtime whose
// register code calls Go for the instructions it runs whole (jitRegWhole)
// and in one with no JIT, the collector idle and marking, calls of Go
// clobbering what they may: the answers equal, and the instructions left
// to Go at exits only where they did not go on natively.
func TestJITRegWholeGoCalls(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, inline := range []bool{false, true} {
		for _, marking := range []bool{false, true} {
			t.Run(fmt.Sprintf("inline=%v/marking=%v", inline, marking), func(t *testing.T) {
				if marking {
					jitMarkingForTest(t)
				}
				if !inline {
					regNoInline(t)
				}
				jitClobberCallsForTest(t)
				rts := regNativeRuntimes(t, regWholeSrc)
				rts[0].SetJITWarmUp(1, 1)
				for _, name := range []string{"privates", "caught", "uncaught", "captured", "invalid", "forin", "spreads"} {
					var got [2]string
					for i, r := range rts {
						fn, _ := r.GetProp(Obj(r.Global()), r.Intern(name))
						for range 6 {
							v, err := r.Call(fn, Undefined, []Value{Float(500)})
							got[i] = regNativeShow(r, v, err)
						}
					}
					if got[0] != got[1] {
						t.Errorf("%s: with the JIT %q, without %q", name, got[0], got[1])
					}
				}
				st := rts[0].RegNativeStats()
				t.Logf("calls of Go %d, hosts %v, finished at exits %d", st.GoCalls, st.HostOps, st.WholeLeft)
				if st.WholeLeft == 0 {
					t.Errorf("no instruction Go ran whole was finished at an exit")
				}
				for _, op := range []string{"get_private", "set_private", "for_in_start", "copy_data_props", "array_spread", "call_spread"} {
					// Each left only where it did not go on natively: a throw,
					// the code found invalid -- some tens over the runs.
					if n := st.HostOps[op]; n > 100 {
						t.Errorf("%s left to Go %d times", op, n)
					}
				}
			})
		}
	}
}
