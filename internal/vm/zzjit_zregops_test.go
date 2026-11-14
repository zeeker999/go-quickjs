//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"fmt"
	"testing"
)

// regOpsSrc's instanceofs are of constructors the code reads before them,
// a global and a captured binding, in a loop and in a callee inlined in
// it, which native code makes along the value's prototypes
// (jitRegFeedback's InstanceOf); swap rebinds the global to another
// function and gives one a Symbol.hasInstance, which the code was not
// compiled for. hyp's callee calls Math's functions and charCodeAt, inlined
// in its caller's loop (the builder's callSlotsOf in an inlined frame).
const regOpsSrc = `
function Pair(a, b) { this.car = a; this.cdr = b }
class Node { constructor(v) { this.v = v } }
var items = [];
for (var i = 0; i < 64; i++) items.push(i % 3 == 0 ? new Pair(i, null) : i % 3 == 1 ? new Node(i) : { v: i });
var Cap = (function () { function C() {} items[5] = new C(); return function (x) { return x instanceof C } })();
function count(n) { var s = 0; for (var i = 0; i < n; i++) { var x = items[i & 63]; if (x instanceof Pair) s += 1; else if (x instanceof Node) s += 2; if (Cap(x)) s += 7 } return s }
function isPair(x) { return x instanceof Pair }
function viaCall(n) { var s = 0; for (var i = 0; i < n; i++) if (isPair(items[i & 63])) s++; return s }
function swap() { Pair = function () {}; Object.defineProperty(Node, Symbol.hasInstance, { value: function (x) { return typeof x.v === "number" } }) }
function dist(x, y, s) { return Math.sqrt(x * x + y * y) + Math.floor(x / 3) + s.charCodeAt(x & 3) }
function hyp(n) { var t = 0; for (var i = 0; i < n; i++) t += dist(i, i + 1, "abcd"); return t }
var words = ["alpha", "beta", "gamma", "delta"];
function searches(n) { var s = 0, b = 12345678901234567890n; for (var i = 0; i < n; i++) { var w = words[i & 3]; s += w.indexOf("a") + words.indexOf(w) + parseInt(i / 7) + (words.includes(w) ? 1 : 0); if ((i & 255) == 0) s += b.toString().length + (b + BigInt(i)).toString(16).length } return s }
function main(n) { return count(n) + "," + viaCall(n) + "," + hyp(n) + "," + searches(n) }
`

// TestJITRegInstanceOfNative runs regOpsSrc's main natively and in a
// runtime without the JIT, before and after swap, and compares their
// answers: the instanceofs made natively -- no call of Go -- and Math's
// calls in an inlined callee too.
func TestJITRegInstanceOfNative(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			rts := regNativeRuntimes(t, regOpsSrc)
			run := func(name string, n float64) [2]string {
				var got [2]string
				for i, r := range rts {
					fn, _ := r.GetProp(Obj(r.Global()), r.Intern(name))
					v, err := r.Call(fn, Undefined, []Value{Float(n)})
					got[i] = regNativeShow(r, v, err)
				}
				if got[0] != got[1] {
					t.Errorf("%s: with the JIT %q, without %q", name, got[0], got[1])
				}
				return got
			}
			for range 20 {
				run("main", 3000)
			}
			if got := run("main", 3000); got[1] != "3330,1031,8157962.728070096,655290" {
				t.Errorf("main: %q", got[1])
			}
			st := rts[0].RegNativeStats()
			t.Logf("hosts %v, calls of Go %d %v", st.HostOps, st.GoCalls, st.GoCallOps)
			if st.GoCalls > 100 {
				t.Errorf("%d calls of Go: instanceof or Math's calls not made natively", st.GoCalls)
			}
			run("swap", 0)
			for range 5 {
				run("main", 3000)
			}
		})
	}
}

// regPlainSetSrc's w is called natively, not inlined: its write of s calls
// a setter, a call of Go its native code makes at register pc 5, where w's
// stack form has o.q = t. Made without a frame by the stack form's
// instruction there (jitGoCallPlain), the write had stored q and never
// called the setter.
const regPlainSetSrc = `
function w(o, v) { var t = v + 1; o.p = 1; o.q = t; o.r = t; o.s = t; o.t = v; return t }
var log = 0;
function mk(k) { var o = { set s(x) { log += x } }; o["k" + k] = k; return o }
var objs = []; for (var i = 0; i < 3; i++) objs.push(mk(i));
function loop(n) { var s = 0; for (var i = 0; i < n; i++) s += w(objs[i % 3], i); return s }
function main() { log = 0; loop(3000); return JSON.stringify(objs) + log }
`

// TestJITRegPlainSetterCall checks a register native callee's setter call,
// a call of Go at a register pc, writes the property its register
// instruction names (jitRegPlainSet), not the stack form's at that pc.
func TestJITRegPlainSetterCall(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	regNoInline(t)
	rts := regNativeRuntimes(t, regPlainSetSrc)
	const want = `[{"k0":0,"p":1,"q":2998,"r":2998,"t":2997},{"k1":1,"p":1,"q":2999,"r":2999,"t":2998},{"k2":2,"p":1,"q":3000,"r":3000,"t":2999}]4501500`
	for i, r := range rts {
		fn, _ := r.GetProp(Obj(r.Global()), r.Intern("main"))
		for k := range 10 {
			v, err := r.Call(fn, Undefined, nil)
			if got := regNativeShow(r, v, err); got != want {
				t.Fatalf("runtime %d, run %d: %q, want %q", i, k, got, want)
			}
		}
	}
	if st := rts[0].RegNativeStats(); st.GoCalls == 0 {
		t.Error("the setter never called by Go from native code")
	}
}
