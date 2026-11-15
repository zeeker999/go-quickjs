//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"testing"
)

// regWalkSrc is Octane Typescript's AstWalker: a recursive walk whose
// every level calls the visitors its walker holds, a walker for each pass
// with visitors of its own, and the children's walker, which calls the
// walk again, natively.
const regWalkSrc = `
function Walker(kids, pre, post) { this.kids = kids; this.pre = pre; this.post = post }
Walker.prototype.walk = function (n) {
	var p = this.pre(n, this);
	this.kids[n.t](n, this);
	return this.post(p, this);
};
var kids = [
	function (n, w) { },
	function (n, w) { w.walk(n.a) },
	function (n, w) { w.walk(n.a); w.walk(n.b) },
];
function tree(d) {
	if (d == 0) return { t: 0, v: 1 };
	if (d % 3 == 0) return { t: 1, v: d, a: tree(d - 1) };
	return { t: 2, v: d, a: tree(d - 1), b: tree(d - 2 < 0 ? 0 : d - 2) };
}
var root = tree(9);
var pres = [
	function (n, w) { return n.v }, function (n, w) { return n.v + 1 }, function (n, w) { return n.v + 2 },
	function (n, w) { return n.v + 3 }, function (n, w) { return n.v + 4 }, function (n, w) { return n.v + 5 },
	function (n, w) { return n.v + 6 }, function (n, w) { return n.v + 7 },
];
var posts = [
	function (p, w) { return p * 1 % 1000 }, function (p, w) { return p * 2 % 1000 }, function (p, w) { return p * 3 % 1000 },
	function (p, w) { return p * 4 % 1000 }, function (p, w) { return p * 5 % 1000 }, function (p, w) { return p * 6 % 1000 },
	function (p, w) { return p * 7 % 1000 }, function (p, w) { return p * 8 % 1000 },
];
function walks() {
	var s = 0;
	for (var k = 0; k < pres.length; k++) s += new Walker(kids, pres[k], posts[k]).walk(root);
	return s;
}
`

// TestJITRegRecursiveRecompiled runs a recursive walk whose levels are
// native calls of its own code, which learns at its exits the visitors
// each new walker brings, and checks that the walk is compiled again for
// them while its outer levels are in its code, as the stack form's code is
// (otherwiseInUse): its exits stop once it has learned them. Go finishes
// those levels, each in the code it holds; only a native call of it that
// has not returned, whose code Go would find by its function's, keeps it
// from being compiled (regCalled). Counted as levels instead, every level
// of the recursion Go was finishing kept it from being compiled again, and
// it left at each new visitor's call for good: Octane Typescript's
// AstWalker.walk, 285,000 times a run.
func TestJITRegRecursiveRecompiled(t *testing.T) {
	regBothMarking(t, func(t *testing.T) {
		regNoInline(t)
		rts := regNativeRuntimes(t, regWalkSrc)
		r := rts[0]
		r.jitStress = jitStressConfig{}
		r.SetJITWarmUp(2, 2)
		var got [2]string
		for i, rt := range rts {
			fn, _ := rt.GetProp(Obj(rt.Global()), rt.Intern("walks"))
			for range 60 {
				v, err := rt.Call(fn, Undefined, nil)
				got[i] = regNativeShow(rt, v, err)
			}
		}
		if got[0] != got[1] || got[0] != "492" {
			t.Fatalf("with the JIT %q, without %q, want 492", got[0], got[1])
		}
		w, _ := r.GetProp(Obj(r.Global()), r.Intern("Walker"))
		proto, _ := r.GetProp(w, r.Intern("prototype"))
		walk, _ := r.GetProp(proto, r.Intern("walk"))
		cl := walk.Object().fn().closure
		e := r.jit.regCache[r.jitRegKeyOf(cl)]
		if e == nil || e.code == nil || e.life == nil {
			t.Fatal("the walk has no register code")
		}
		hosts := e.life.ssaStats.hosts
		t.Logf("walk: %d compiles, %d host exits, %d native calls of it", e.life.ssaSerial, hosts, r.jit.regLinks[e.key].in)
		if e.life.reopt {
			t.Errorf("the walk is still to be compiled for what it learned")
		}
		// Each run has its eight walkers' visitors called at 3 of the walk's
		// instructions, at some 100 levels each: before, every one of them
		// past the first compile left the code.
		if hosts > 100 {
			t.Errorf("the walk left its code %d times; compiled again for its visitors, it leaves at a few of their first calls", hosts)
		}
	})
}
