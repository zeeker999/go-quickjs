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

// regLoopsSrc's for...of and for...in loops step their cursors by calls of
// Go native code makes on the frame (jitRegWhole's step): arrays, a
// string, a Map, a generator, an iterator that throws, a loop left by
// break, a loop variable a closure captures; and a callee made anew at
// each call of its maker, NBody's advance's arrow, given to forEach in a
// native callee (jitRegCellsClosure, jitRegForEachSeen at a host exit).
const regLoopsSrc = `
function* gen(n) { for (var i = 0; i < n; i++) yield i * 2 }
var bad = { [Symbol.iterator]() { var k = 0; return { next() { if (++k > 3) throw new Error("bad" + k); return { value: k, done: false } } } } };
function loops(n) {
	var arr = [1, 2, 3, 4, 5], m = new Map([["a", 1], ["b", 2]]), s = 0, t = "", fs = [];
	for (var i = 0; i < n; i++) {
		for (const x of arr) s += x;
		for (const c of "hey") t = c;
		for (const [k, v] of m) s += v + k.length;
		for (const g of gen(3)) s += g;
		for (const x of arr) { if (x == 3) break; s += x }
		for (const k in { p: 1, q: 2 }) t += k;
		if (i % 50 == 0) for (const x of arr) fs.push(() => x);
		try { for (const b of bad) s += b } catch (e) { t += e.message }
	}
	var u = 0; for (const f of fs) u += f();
	return s + ":" + t + ":" + u;
}
class Body { constructor(x) { this.x = x; this.vx = 1.5 } }
class Sys {
	constructor() { this.bodies = [new Body(1), new Body(2), new Body(3)] }
	advance(dt) { this.bodies.forEach((b) => { b.x += dt * b.vx }) }
}
function advances(n) { var s = new Sys(); for (var i = 0; i < n; i++) s.advance(i % 3 == 0 ? 0.5 : 0.25); return s.bodies[0].x + s.bodies[2].x }
class Ball { constructor(i) { this.x = i } bounce() { this.x = (this.x * 7 + 3) % 101; return this.x < 20 } }
function bounces() { const balls = new Array(100); for (let i = 0; i < 100; i++) balls[i] = new Ball(i); let b = 0; for (let k = 0; k < 50; k++) for (const ball of balls) if (ball.bounce()) b += 1; return b }
function main(n) { return loops(n) + "," + advances(n * 10) + "," + bounces() }
`

// TestJITRegLoopSteps runs regLoopsSrc's main natively and in a runtime
// without the JIT and compares their answers; no step, no closure made by
// a native callee, no forEach given a new arrow each call, leaves native
// code at every turn.
func TestJITRegLoopSteps(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			rts := regNativeRuntimes(t, regLoopsSrc)
			var got [2]string
			for i, r := range rts {
				fn, _ := r.GetProp(Obj(r.Global()), r.Intern("main"))
				for range 20 {
					v, err := r.Call(fn, Undefined, []Value{Float(300)})
					got[i] = regNativeShow(r, v, err)
				}
			}
			if got[0] != got[1] {
				t.Errorf("with the JIT %q, without %q", got[0], got[1])
			}
			st := rts[0].RegNativeStats()
			t.Logf("hosts %v, calls of Go %d", st.HostOps, st.GoCalls)
			for _, op := range []string{"iter_next", "closure_w", "call_method_w"} {
				if n := st.HostOps[op]; n > 200 {
					t.Errorf("%s left to Go at %d host exits", op, n)
				}
			}
		})
	}
}

// regAllocSrc constructs arrays with the realm's Array, of no argument and
// of a length, which register code takes from a pool (jitRegAllocSeen,
// ssa.CallSite's Alloc), in a recursive native callee too; one past what a
// pool's arrays have room for, a negative length, and Array replaced.
const regAllocSrc = `
function tree(d) { var a = new Array(4); if (d == 0) return a.length + (1 in a ? 1 : 0); for (var i = 0; i < 4; i++) a[i] = tree(d - 1); return a[0] + a[3] }
function lists(n) {
	var s = 0, kept = [];
	for (var i = 0; i < n; i++) { var x = new Array(); x.push(i); var y = new Array(i & 7); y[0] = i; s += x.length + y.length + y[0]; if (i % 100 == 0) kept.push(y) }
	var big = new Array(5000), bad = "";
	try { new Array(-1) } catch (e) { bad = e.name }
	return s + big.length + bad + kept.length + kept[3].length + JSON.stringify(kept[1]);
}
function swap() { var A = Array; Array = function (n) { this.n = n } ; return A }
function main() { return tree(5) + "," + lists(3000) }
`

// TestJITRegArrayAlloc runs regAllocSrc's main natively and in a runtime
// without the JIT, before and after Array is replaced, and compares their
// answers: the constructions made from pools, not by Go at each.
func TestJITRegArrayAlloc(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			rts := regNativeRuntimes(t, regAllocSrc)
			rts[0].SetJITReport(true)
			call := func(name string) [2]string {
				var got [2]string
				for i, r := range rts {
					fn, _ := r.GetProp(Obj(r.Global()), r.Intern(name))
					v, err := r.Call(fn, Undefined, nil)
					got[i] = regNativeShow(r, v, err)
				}
				if got[0] != got[1] {
					t.Errorf("%s: with the JIT %q, without %q", name, got[0], got[1])
				}
				return got
			}
			for range 20 {
				call("main")
			}
			if got := call("main"); got[1] != `128,4517375RangeError304[100,null,null,null]` {
				t.Errorf("main: %q", got[1])
			}
			st := rts[0].RegNativeStats()
			t.Logf("calls of Go %v", st.GoCallOps)
			if n := st.GoCallOps["new_w->Array"]; n > 3000 {
				t.Errorf("%d constructions of Array made by Go", n)
			}
			call("swap")
			for range 3 {
				call("main")
			}
		})
	}
}

// regClosureSrc's functions make closures in loops natively from pools
// (jitRegFeedback's Closure, CellsPool): of a loop's let, which each
// iteration's closures capture anew, of a parameter, of a binding they
// assign, by value, an arrow whose nested arrow reads its maker's this
// (Go's), in native callees, kept and called later.
const regClosureSrc = `
function lets(n) { var fs = []; for (let i = 0; i < n; i++) { fs.push(function () { return i }) } var s = 0; for (var j = 0; j < fs.length; j++) s += fs[j](); return s }
function counter(k) { var c = k; var inc = function () { c++; return c }; inc(); return inc() }
function counters(n) { var s = 0; for (var i = 0; i < n; i++) s += counter(i); return s }
function adder(a) { return function (b) { return a + b } }
function adders(n) { var s = 0; for (var i = 0; i < n; i++) s += adder(i)(1); return s }
var obj = { v: 7, nested(n) { var s = 0; for (var i = 0; i < n; i++) { var f = () => () => this.v + i; s += f()() } return s } };
function mixed(n) { var t = 0, fs = []; for (var i = 0; i < n; i++) { const k = i * 2; var g = () => k + t; t += g(); if (i % 64 == 0) fs.push(g) } var u = 0; for (var h of fs) u += h(); return t + ":" + u }
function main(n) { return lets(n) + "," + counters(n) + "," + adders(n) + "," + obj.nested(n) + "," + mixed(n) }
`

// TestJITRegClosurePools runs regClosureSrc's main natively and in a
// runtime without the JIT and compares their answers, the closures made
// from pools and the collector idle and marking.
func TestJITRegClosurePools(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			rts := regNativeRuntimes(t, regClosureSrc)
			rts[0].SetJITReport(true)
			var got [2]string
			for i, r := range rts {
				fn, _ := r.GetProp(Obj(r.Global()), r.Intern("main"))
				for range 20 {
					v, err := r.Call(fn, Undefined, []Value{Float(500)})
					got[i] = regNativeShow(r, v, err)
				}
			}
			if got[0] != got[1] {
				t.Errorf("with the JIT %q, without %q", got[0], got[1])
			}
			st := rts[0].RegNativeStats()
			t.Logf("hosts %v, calls of Go %d %v", st.HostOps, st.GoCalls, st.GoCallOps)
		})
	}
}

// regClassSrc's methods read and write through super, ask #p in o, and a
// derived constructor with a loop -- entered natively by on-stack
// replacement -- returns its this or an object: each Go's whole on the
// frame, no longer an exit (compile's RegAccess).
const regClassSrc = `
class A { constructor(x) { this.x = x } get g() { return this.x * 2 } m(k) { return this.x + k } }
class B extends A {
	#p = 1;
	constructor(x) { super(x); this.y = x + 1 }
	m(k) { return super.m(k) + super.g + this.#p + super["m"](1) }
	static has(o) { return #p in o }
	get g() { return super.g + 1 }
	set s(v) { super.x = v }
}
class D extends A { constructor(n) { super(n); var s = 0; for (var i = 0; i < n; i++) s += super.m(i); this.s = s; if (n == 77) return { neg: n } } }
function run(n) {
	var s = 0;
	for (var i = 0; i < n; i++) { var b = new B(i & 15); s += b.m(i) + b.g + (B.has(b) ? 1 : 0) + (B.has({}) ? 100 : 0); b.s = 3; s += b.x }
	var d = new D(500), e = new D(77);
	try { B.has(1) } catch (err) { s += err.name }
	return s + ":" + d.s + ":" + e.neg + ":" + (e instanceof D);
}
function main() { return run(3000) }
`

// TestJITRegClassOps runs regClassSrc's main natively and in a runtime
// without the JIT and compares their answers: super's reads and writes,
// #p in o and a derived constructor's return made on the frame, not at
// exits.
func TestJITRegClassOps(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			rts := regNativeRuntimes(t, regClassSrc)
			var got [2]string
			for i, r := range rts {
				fn, _ := r.GetProp(Obj(r.Global()), r.Intern("main"))
				for range 10 {
					v, err := r.Call(fn, Undefined, nil)
					got[i] = regNativeShow(r, v, err)
				}
			}
			if got[0] != got[1] {
				t.Errorf("with the JIT %q, without %q", got[0], got[1])
			}
			if got[1] != "4654308TypeError:374750:77:false" {
				t.Errorf("main: %q", got[1])
			}
			st := rts[0].RegNativeStats()
			t.Logf("hosts %v", st.HostOps)
			for _, op := range []string{"super_get", "super_get_index", "private_in", "super_base", "derived_ret"} {
				if n := st.HostOps[op]; n > 20 {
					t.Errorf("%s left to Go at %d host exits", op, n)
				}
			}
		})
	}
}

// regSpreadSrc spreads objects and arrays and defines literals' fields
// after a spread, which native code makes itself first where it can --
// by the layouts Go copied, a dense array's elements, an element
// appended, a field along its shape's transition (regWholeSite) -- or
// Go without the frame (jitRegFrameless), or on it: a getter on the
// source, holes, a string, null, an iterator replaced, a frozen target.
const regSpreadSrc = `
function sp(n) {
	var s = 0, base = { a: 1, b: 2 }, arr = [1, 2, 3], log = 0;
	var getter = { get a() { log++; return 5 }, z: 1 };
	for (var i = 0; i < n; i++) {
		var o = { ...base, c: i }, g = { ...getter, k: i }, t = { ..."hi", ...null, d: 1 };
		var a = [...arr, i, ...arr], b = [0, , 2], c = [..."xy", ...b];
		s += o.a + o.c + g.a + g.k + t[1].length + t.d + a.length + a[3] + b.length + c.length + (1 in b ? 1 : 0) + (3 in c ? 1 : 0);
	}
	return s + ":" + log + ":" + JSON.stringify(c) + Object.keys(o);
}
function swap() { Array.prototype[Symbol.iterator] = function* () { yield 9 } }
function main() { return sp(3000) }
`

// TestJITRegSpreads runs regSpreadSrc's main natively and in a runtime
// without the JIT and compares their answers, before and after arrays'
// iterator is replaced, the collector idle and marking: no spread, push
// or field definition a call of Go each.
func TestJITRegSpreads(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			rts := regNativeRuntimes(t, regSpreadSrc)
			rts[0].SetJITReport(true)
			call := func(name string) [2]string {
				var got [2]string
				for i, r := range rts {
					fn, _ := r.GetProp(Obj(r.Global()), r.Intern(name))
					v, err := r.Call(fn, Undefined, nil)
					got[i] = regNativeShow(r, v, err)
				}
				if got[0] != got[1] {
					t.Errorf("%s: with the JIT %q, without %q", name, got[0], got[1])
				}
				return got
			}
			for range 10 {
				call("main")
			}
			if got := call("main"); got[1] != `13567500:3000:["x","y",0,null,2]a,b,c` {
				t.Errorf("main: %q", got[1])
			}
			st := rts[0].RegNativeStats()
			t.Logf("calls of Go %v", st.GoCallOps)
			// Of 33,000 turns: a push and a field natively; of three object
			// spreads and four array spreads each, those of a string, of a
			// getter's object and of an array with a hole Go's.
			for op, most := range map[string]uint64{"whole array_push": 1000, "whole define_field": 1000, "whole array_spread": 66000, "whole copy_data_props": 66000} {
				if n := st.GoCallOps[op]; n > most {
					t.Errorf("%s: %d calls of Go", op, n)
				}
			}
			call("swap")
			for range 3 {
				call("main")
			}
		})
	}
}

// TestJITRegOpsCompileForARM64 compiles again, with arm64's backend, on
// any host, the register code this file's sources were compiled to: the
// native instanceof, the element reads and writes learned, the steps, the
// pools of arrays and closures, the spreads' fast paths.
func TestJITRegOpsCompileForARM64(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	RecordJITCompiles(true)
	defer RecordJITCompiles(false)
	for _, src := range []string{regElemSrc, regOpsSrc, regLoopsSrc, regAllocSrc, regClosureSrc, regClassSrc, regSpreadSrc, regArrowThisSrc, regStorageSrc} {
		rts := regNativeRuntimes(t, src)
		r := rts[0]
		fn, _ := r.GetProp(Obj(r.Global()), r.Intern("main"))
		for range 10 {
			r.Call(fn, Undefined, []Value{Float(300)})
		}
		fns, _, _ := regCompileARM64(t, r)
		if fns == 0 {
			t.Error("nothing compiled for arm64")
		}
	}
}

// regStorageSrc is AWFY Storage's tree: new Array(n) of a length native
// code computes, in a recursive native callee, whose construction Go makes
// from native code until it is learned -- by its operand, which the
// frame's register need not hold (jitRegAllocLength).
const regStorageSrc = `
class Random { constructor() { this.seed = 74755 } next() { this.seed = ((this.seed * 1309) + 13849) & 65535; return this.seed } }
class Storage {
	benchmark() { const random = new Random(); this.count = 0; this.buildTreeDepth(7, random); return this.count }
	buildTreeDepth(depth, random) { this.count += 1; if (depth === 1) return new Array((random.next() % 10) + 1); const arr = new Array(4); for (let i = 0; i < 4; i += 1) arr[i] = this.buildTreeDepth(depth - 1, random); return arr }
}
function main() { return new Storage().benchmark() }
`

// TestJITRegArrayAllocComputedLength checks Storage's constructions of
// Array are made from pools, not by Go at each.
func TestJITRegArrayAllocComputedLength(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	rts := regNativeRuntimes(t, regStorageSrc)
	rts[0].SetJITReport(true)
	for i, r := range rts {
		fn, _ := r.GetProp(Obj(r.Global()), r.Intern("main"))
		for range 20 {
			v, err := r.Call(fn, Undefined, nil)
			if got := regNativeShow(r, v, err); got != "5461" {
				t.Fatalf("runtime %d: %q, want 5461", i, got)
			}
		}
	}
	if n := rts[0].RegNativeStats().GoCallOps["new_w->Array"]; n > 100 {
		t.Errorf("%d constructions of Array made by Go", n)
	}
}

// regArrowThisSrc's methods and functions make arrows that read their
// this, from closure pools (jitRegFeedback's Closure, ssa.ClosureSite's
// LexThisReg): the maker's register 1, a method's receiver, a sloppy
// function's global object, a strict one's undefined; kept and called
// later.
const regArrowThisSrc = `
class Vec { constructor(x) { this.x = x; this.items = [1, 2, 3] }
	sum() { var s = 0; this.items.forEach((v) => { s += v * this.x }); return s }
	mk() { return () => this.x }
}
function sloppy() { var f = () => this === globalThis; return f() }
function strict() { "use strict"; var f = () => this; return f() }
function main() {
	var s = 0, fs = [];
	for (var i = 0; i < 3000; i++) { var v = new Vec(i & 7); s += v.sum(); if (i % 100 == 0) fs.push(v.mk()); s += sloppy() ? 1 : 0; s += strict() === undefined ? 1 : 0 }
	var t = 0; for (var f of fs) t += f();
	return s + ":" + t;
}
`

// TestJITRegArrowThisPools runs regArrowThisSrc's main natively and in a
// runtime without the JIT: arrows reading this taken from pools, the
// collector idle and marking.
func TestJITRegArrowThisPools(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			rts := regNativeRuntimes(t, regArrowThisSrc)
			for i, r := range rts {
				fn, _ := r.GetProp(Obj(r.Global()), r.Intern("main"))
				for range 10 {
					v, err := r.Call(fn, Undefined, nil)
					if got := regNativeShow(r, v, err); got != "69000:60" {
						t.Fatalf("runtime %d: %q, want 69000:60", i, got)
					}
				}
			}
		})
	}
}
