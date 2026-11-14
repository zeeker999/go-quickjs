//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"fmt"
	"testing"
)

// regClosSrc's loops make closures of a per-turn let, each closed at the
// turn's end (close_regs, natively from the function's pool of cells):
// perTurn's closures each keep their turn's binding, assigned after they
// were made; nested's inner loop closes its own bindings, not the outer
// turn's, which closures of both share; after's closure sees what its
// binding is given after an inner block closed another; arrows's arrows
// made by an arrow read the method's this; appends's closures append to
// a captured string, once and twice a call.
const regClosSrc = `
function perTurn(n) {
	var fs = [];
	for (let i = 0; i < n; i++) {
		let k = i * 2;
		fs.push(() => k + i);
		k++;
	}
	var s = 0;
	for (var j = 0; j < fs.length; j++) s += fs[j]();
	return s;
}
function nested(n) {
	var s = 0, keep = [];
	for (var i = 0; i < n; i++) {
		let outer = i;
		const fo = () => outer;
		for (let j = 0; j < 3; j++) {
			let inner = j;
			const fi = () => inner * 10 + outer;
			outer++;
			s += fi();
			if (i == n - 1) keep.push(fi);
		}
		s += fo();
	}
	for (var k = 0; k < keep.length; k++) s += keep[k]();
	return s;
}
function after(n) {
	var s = 0, last = null, lastB = null;
	for (var i = 0; i < n; i++) {
		let a = i;
		const get = () => a;
		{
			let b = a + 1;
			lastB = () => b;
			s += lastB();
		}
		a = a * 3;
		s += get();
		let c = a;
		const getC = () => c;
		c++;
		s += getC();
		last = get;
	}
	return s + last() + lastB();
}
function Box(v) { this.v = v }
Box.prototype.sum = function (n) {
	var s = 0;
	var outer = (k) => { var t = 0; for (var j = 0; j < k; j++) { var inner = () => this.v + j; t += inner() } return t };
	for (var i = 0; i < n; i++) s += outer(8);
	return s;
};
function arrows(n) {
	var s = 0;
	for (var i = 0; i < 4; i++) s += new Box(i + 1).sum(n);
	return s;
}
function appendOnce(a, x) { var s = a; var add = function (y) { s += y }; add(x); return s }
function appendTwice(a, x) { var s = a; var add = function (y) { s += y }; add(x); add(x); return s }
function appends(n) {
	var c = 0, last = "";
	for (var i = 0; i < n; i++) {
		c += appendOnce("ab", "<" + (i & 7) + ">").length;
		last = appendTwice("cd", "[" + (i & 3) + "]");
		c += last.length;
	}
	var t = "", sink = { push: function (x) { return x !== null && (t += x), true } };
	for (var j = 0; j < n; j++) sink.push(j & 1 ? "x" : "yz");
	return c + "," + last + "," + t.length + "," + t.slice(0, 7);
}
function main(n) { return perTurn(n) + "," + nested(n) + "," + after(n) + "," + arrows(n) + "," + appends(n) }
`

// TestJITRegClosures runs regClosSrc's main natively and in a runtime
// without the JIT, with the collector idle and marking, and checks both
// give its answer: each turn's closures keep their own bindings, an
// inner scope's end closes its registers alone, and arrows an arrow makes
// read its this. No scope's end leaves native code.
func TestJITRegClosures(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	const want = "1499500,2040036,3502497,192000,13000,cd[3][3],1500,yzxyzxy"
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			rts := regNativeRuntimes(t, regClosSrc)
			var got [2]string
			for i, r := range rts {
				fn, _ := r.GetProp(Obj(r.Global()), r.Intern("main"))
				for range 30 {
					v, err := r.Call(fn, Undefined, []Value{Float(1000)})
					got[i] = regNativeShow(r, v, err)
					if got[i] != want {
						t.Fatalf("runtime %d (JIT %v): %q, want %q", i, i == 0, got[i], want)
					}
				}
			}
			st := rts[0].RegNativeStats()
			t.Logf("entries %d, hosts %v, calls of Go %d", st.Entries, st.HostOps, st.GoCalls)
			if n := st.HostOps["close_regs"]; n > 50 {
				t.Errorf("close_regs left to Go at %d host exits", n)
			}
			if n := st.HostOps["closure_w"]; n > 50 {
				t.Errorf("closure_w left to Go at %d host exits", n)
			}
		})
	}
}

// regDerivedArrowsSrc's constructor makes, by an arrow mk, arrows that
// share its this binding: before super() binds it -- mk run while it is
// not, whose arrows throw when called then and read the binding once
// bound -- and after, mk hot by then, whose arrows read this.
const regDerivedArrowsSrc = `
class Base { constructor() { this.b = 1 } }
class Derived extends Base {
	constructor(n) {
		const mk = (k) => { let g = null; for (let j = 0; j < k; j++) g = () => this; return g };
		let early = 0, gs = [];
		for (let i = 0; i < n; i++) {
			try { mk(20)() } catch (e) { if (e instanceof ReferenceError) early++ }
			gs.push(mk(20));
		}
		super();
		let same = 0;
		for (let i = 0; i < n; i++) {
			if (mk(20)() === this) same++;
			if (gs[i]() === this) same++;
		}
		this.r = early + "," + same;
	}
}
function main(n) { var s = ""; for (var i = 0; i < 3; i++) s += new Derived(n).r + ";"; return s }
`

// TestJITRegArrowsOfArrowsThis runs regDerivedArrowsSrc natively and
// without the JIT, with the collector idle and marking: arrows an arrow
// makes natively from its pool capture the this its derived
// constructor's super() bound, never the binding's state before.
func TestJITRegArrowsOfArrowsThis(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	const want = "2000,4000;2000,4000;2000,4000;"
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			rts := regNativeRuntimes(t, regDerivedArrowsSrc)
			for i, r := range rts {
				fn, _ := r.GetProp(Obj(r.Global()), r.Intern("main"))
				for range 10 {
					v, err := r.Call(fn, Undefined, []Value{Float(2000)})
					if got := regNativeShow(r, v, err); got != want {
						t.Fatalf("runtime %d (JIT %v): %q, want %q", i, i == 0, got, want)
					}
				}
			}
			st := rts[0].RegNativeStats()
			t.Logf("entries %d, hosts %v, calls of Go %d %v", st.Entries, st.HostOps, st.GoCalls, st.GoCallOps)
		})
	}
}
