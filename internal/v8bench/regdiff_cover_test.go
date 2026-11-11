package v8bench

import (
	"fmt"
	"strings"
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/compiler/regcomp"
)

// regDiffAll is regDiff, failing where regcomp refuses any function the
// cases compile, eval code's included: the forms here are register code.
func regDiffAll(t *testing.T, src string, calls []string) {
	t.Helper()
	var refused []string
	regcomp.OnRefuse = func(fn *bytecode.Function, why string) {
		refused = append(refused, fn.Name+": "+why)
	}
	defer func() { regcomp.OnRefuse = nil }()
	regDiff(t, src, calls)
	for _, r := range refused {
		t.Errorf("refused: %s", r)
	}
}

// The last refusals' forms (docs/register-interpreter.md, "Stage 2: zero
// refusals"), compared with the register tier off and on.

// TestRegDiffCallTargets compares sloppy code's assignments to a call --
// =, a compound assignment, an update and a for-in or for-of head -- each
// made and then a ReferenceError at the target, the value never made;
// Annex B's for (var a = init in o); and a let's own initializer
// assigning it in its dead zone.
func TestRegDiffCallTargets(t *testing.T) {
	src := `
var frames = function (e) { return e.name + ": " + e.message + " | " + e.stack.split("\n").slice(1, 3).map(function (s) { return s.trim() }).join(" | ") };
var log = [];
function f() { log.push("f"); return { valueOf() { log.push("valueOf"); return 1 } } }
function g() { log.push("g"); return 1 }
function t1() { f() = g() }
function t2() { f() += g() }
function t3() { f()++ }
function t4() { --f() }
function t5() { for (f() in { a: 1 }) ; }
function t6() { for (f() of [1, 2]) ; }
function t7() { for (f() in {}) ; return "none" }
function t8() { var r = []; for (var a = (log.push("init"), 7) in (r.push(a), { p: 1, q: 2 })) r.push(a); return r.join() + "/" + a }
function t9() { for (var a = 0 in {}) ; return a }
function t10() { let y = [y] = []; }
function t11() { let y = { y } = {}; }
function t12() { const c = [c] = [1]; }
function t13() { let z = (() => [z] = [])(); }
for (var ga = 3 in {}) ;
function t14() { return ga }
function run(fn) { log = []; try { return fn() + "|" + log.join() } catch (e) { return frames(e) + "|" + log.join() } }
`
	var calls []string
	for i := 1; i <= 14; i++ {
		calls = append(calls, fmt.Sprintf("run(t%d)", i))
	}
	regDiffAll(t, src, calls)
}

// TestRegDiffFinallyCopies compares finally clauses that declare
// bindings, compiled once for each way out: a return, a break, a
// continue, a throw and the normal end, closures made inside keeping
// their own, catch clauses inside, at top level and in functions.
func TestRegDiffFinallyCopies(t *testing.T) {
	src := `
var kept = [];
function f1(n) {
	// let i: with var i, a continue leaves the body's captured let
	// open in the stack form, which the next turn overwrites (0,2,2 for
	// 0,1,2), where the register form closes it.
	for (let i = 0; i < 4; i++) {
		try {
			let inner = i * 10;
			kept.push(() => inner);
			if (i == n) return "ret" + i;
			if (i == 1) continue;
			if (i == 2) break;
		} finally {
			const fin = i + 100;
			kept.push(() => fin);
			try { throw fin } catch (fe) { kept.push(() => fe + 1) }
		}
	}
	return "end";
}
function f2() {
	try { throw new Error("x") } catch (er1) {
		try { throw "ex2" } catch (er1) { if (er1 !== "ex2") return "bad" }
		return er1.message;
	} finally {
		try { throw "ex3" } catch (er1) { kept.push(er1) }
	}
}
function f3() {
	var out = [];
	try {
		for (const p of ["a", "b"]) { out.push(p); if (p == "b") return out.join() }
	} finally {
		for (const q of ["c"]) out.push(q);
		function inside() { return "fd" }
		out.push(inside());
	}
}
function f4(v) { try { let a = v; return a } finally { const a2 = v + 1; v = a2 } }
var top = [];
try { top.push(1) } finally { for (const tp of [2, 3]) top.push(() => tp) }
try { throw 4 } catch (er1) { top.push(er1) } finally { try { throw 5 } catch (er1) { top.push(er1) } }
function show(a) { return a.map(function (x) { return typeof x == "function" ? x() : x }).join() }
`
	calls := []string{
		`kept = []; f1(0) + show(kept)`, `kept = []; f1(3) + show(kept)`, `kept = []; f1(9) + show(kept)`,
		`kept = []; f2() + show(kept)`, `f3()`, `f4(1)`, `show(top)`,
	}
	regDiffAll(t, src, calls)
}

// TestRegDiffWideForms compares what an instruction's operands cannot
// number, made as wide forms: calls of more than 255 arguments -- plain,
// a method's, an optional chain's, new, super(), a tagged template's, a
// direct eval's --, spread calls through an optional chain, array
// literals past one instruction's elements, and functions with more
// cache sites than 16 bits number, their sites shared.
func TestRegDiffWideForms(t *testing.T) {
	var args, holes, consts strings.Builder
	for i := 0; i < 300; i++ {
		if i > 0 {
			args.WriteString(", ")
		}
		fmt.Fprintf(&args, "i + %d", i)
	}
	for i := 0; i < 3000; i++ {
		if i%7 == 3 {
			holes.WriteString(", ")
			continue
		}
		fmt.Fprintf(&holes, "i * %d, ", i)
	}
	for i := 0; i < 70000; i++ {
		fmt.Fprintf(&consts, "%d,", i%97)
	}
	var sites, callSites strings.Builder
	// 20,000 reads and as many writes: the stack form's 40,000 sites and
	// the register form's past 16 bits, shared. 70,000 calls of a local:
	// call sites past 16 bits, and no property site.
	for i := 0; i < 20000; i++ {
		fmt.Fprintf(&sites, "s += o.p%d; o.q%d = s; ", i%5, i%3)
	}
	for i := 0; i < 70000; i++ {
		fmt.Fprintf(&callSites, "s += hl(%d); ", i%11)
	}
	a := args.String()
	tmpl := strings.Repeat("${i}-", 300)
	src := `
var frames = function (e) { return e.name + ": " + e.message + " | " + e.stack.split("\n").slice(1, 3).map(function (s) { return s.trim() }).join(" | ") };
function count() { return arguments.length + ":" + arguments[0] + ":" + arguments[arguments.length - 1] }
var obj = { m: function () { return (this === obj) + count.apply(null, arguments) } };
function w1(i) { return count(` + a + `) }
function w2(i) { return obj.m(` + a + `) }
function w3(i, o) { return o?.m(` + a + `) }
function w4(i) { function C() { this.n = arguments.length + arguments[299] } return new C(` + a + `).n }
class B { constructor() { this.n = arguments.length + ":" + arguments[299] } }
class D extends B { constructor(i) { super(` + a + `) } }
function w5(i) { return new D(i).n }
function tag(s) { return s.length + ":" + arguments.length + ":" + arguments[300] }
function w6(i) { return tag` + "`" + tmpl + "`" + ` }
function w7(i) { return eval("i + 1", ` + a + `) }
function w8(o, xs) { return o?.m(...xs) }
function w9(o, xs) { return (o?.m)(...xs) }
function w10(i) { var a = [` + holes.String() + `]; return a.length + ":" + a[2999] + ":" + (3 in a) + ":" + a[2996] }
function w11() { var a = [` + consts.String() + `]; a[5] = "w"; return a.length + ":" + a[69999] + ":" + a[5] }
function w12(o) { var s = 0; ` + sites.String() + ` return s + ":" + o.q2 }
function h(k) { return k }
function w13() { var s = 0, hl = h; ` + callSites.String() + ` return s }
function w14(i) { return count(` + a + `, oops.x) }
var wo = { v: 9, m(x) { return (this === wo) + ":" + this.v + ":" + x + ":" + arguments.length } };
function plain() { return this === globalThis }
function w15(i, o) { with (o) { return m(...[i, 2]) + "/" + m(` + a + `) + "/" + plain(...[]) } }
function w16(o) { with (o) { return m(...[1]) } }
`
	calls := []string{
		`w1(1)`, `w2(1)`, `w3(1, obj)`, `w3(1, null)`, `w4(2)`, `w5(3)`, `w6(4)`, `w7(5)`,
		`w8(obj, [1, 2])`, `w8(undefined, [1])`, `w9(obj, [3])`, `try { w9(null, []) } catch (e) { frames(e) }`,
		`w10(2)`, `w11() + w11()`,
		`w12({ p0: 1, p1: 2, p2: 3, p3: 4, p4: 5 })`, `w12({ get p0() { return 1 }, p1: 1 })`,
		`w13()`,
		`try { w14(1) } catch (e) { frames(e) }`,
		`w15(1, wo)`, `try { w16({}) } catch (e) { frames(e) }`,
	}
	regDiffAll(t, src, calls)
}

// TestRegDiffStackSitesPast16Bits compares functions whose stack form alone
// has more cache sites than 16 bits number, which the register form numbers
// from 1 with the stack form's moved past them: property reads and writes,
// method calls, global reads and writes, strict code's global store,
// instanceof, object literals and an array literal of constants, read
// polymorphically, and a throw's position past them all.
func TestRegDiffStackSitesPast16Bits(t *testing.T) {
	var sloppy, strict strings.Builder
	for i := 0; i < 36000; i++ {
		fmt.Fprintf(&sloppy, "s += o.p%d; o.q%d = s;\n", i%5, i%3)
	}
	for i := 0; i < 34000; i++ {
		fmt.Fprintf(&strict, "gs = gs + o.p%d;\n", i%4)
	}
	src := `
var frames = function (e) { return e.name + ": " + e.message + " | " + e.stack.split("\n").slice(1, 3).map(function (s) { return s.trim() }).join(" | ") };
var gv = 1, gs = 0;
function C() {}
function w1(o) {
var s = gv;
` + sloppy.String() + `
var lit = { a: s, b: o.p0 }, arr = [1, 2, 3], m = o.m ? o.m(2) : 0;
gv = gv + 1;
return s + ":" + o.q2 + ":" + lit.a + lit.b + ":" + arr.join() + ":" + (o instanceof C) + ":" + m + ":" + gv + ":" + o.r.x;
}
function w2(o) {
"use strict";
` + strict.String() + `
gs = gs + 1; undeclared = gs;
return gs;
}
`
	calls := []string{
		`try { w1({ p0: 1, p1: 2, p2: 3, p3: 4, p4: 5, r: { x: 7 } }) } catch (e) { frames(e) }`,
		`try { var c = new C(); c.p0 = 2; c.m = function (x) { return x * this.p0 }; c.r = { x: 8 }; w1(c) } catch (e) { frames(e) }`,
		`try { w1({ get p0() { return 1 }, p1: 1 }) } catch (e) { frames(e) }`,
		`try { w2({ p0: 1, p1: 1, p2: 1, p3: 1 }) } catch (e) { frames(e) }`,
		`try { undeclared = 0; w2({ p0: 1, p1: 1, p2: 1, p3: 1 }) } catch (e) { frames(e) }`,
	}
	regDiffAll(t, src, calls)
}

// TestRegDiffEvalCode compares eval code run as register code: direct
// evals' var and function declarations hoisted into the caller's
// variables or the global object, strict eval's own, indirect evals, a
// with object's names, a class's private names, super, new.target and
// arguments in the code, the Function constructor, block functions,
// the completion value, closures over the eval's bindings, and a throw's
// stack trace.
func TestRegDiffEvalCode(t *testing.T) {
	src := `
var frames = function (e) { return e.name + ": " + e.message + " | " + e.stack.split("\n").slice(1, 3).map(function (s) { return s.trim() }).join(" | ") };
function f1() { var a = 1; eval("var b = a + 1; function g() { return b * 10 }"); return b + g() }
function f2() { "use strict"; var a = 2; eval("var a = 5"); return a }
function f3() { let x = 3; return eval("x + 1") }
function f4() { var r = []; for (var i = 0; i < 3; i++) eval("var c" + i + " = i; r.push(c" + i + ")"); return r.join() + typeof c2 }
function f5(o) { with (o) { return eval("k + 1") } }
class C { #x = 5; m() { return eval("this.#x") } }
class D { constructor() { this.v = 1 } }
class E extends D { constructor() { super(); this.w = eval("super.constructor === D") } m() { return eval("super.toString === Object.prototype.toString") } }
function f6() { return eval("new.target") === f6 }
function f7() { return eval("arguments.length") }
function f8() { eval("let q = 1; var q2 = q"); return typeof q + q2 }
function f9() { eval("\n  throw new Error('e9')") }
function f10() { eval("var fn = function () { return w }; var w = 'W'"); return fn() }
function f11() { var x = 1; (function () { eval("x = 2") })(); return x }
function f12() { eval("function h() { return 'h' }"); return h() }
function f13() { eval("{ function bf() { return 'bf' } }"); return typeof bf }
function f14() { var r = []; eval("for (let i = 0; i < 3; i++) r.push(() => i)"); return r.map(function (g) { return g() }).join() }
function f15() { return eval("var u = 1; u++; u") + typeof u }
function f16() { eval("'use strict'; var s = 1"); return typeof s }
function f17() { var ev = eval; ev("var iv2 = 3"); return typeof iv2 + iv2 }
function f18() { var f = 88, r = typeof f; eval("r += typeof f; function f() { return 33 }"); return r + f() }
function f19() { var wf = 1; with ({}) { eval("function wf() { return 2 }") } return typeof wf }
`
	calls := []string{
		`f1()`, `f2()`, `f3()`, `f4()`, `f5({ k: 4 })`, `new C().m()`,
		`new E().w + "," + new E().m()`, `new f6()`, `f6()`, `f7(1, 2, 3)`,
		`new Function("a", "b", "return a + b")(2, 3)`, `new Function("return this")() === globalThis`,
		`f8()`, `try { f9() } catch (e) { frames(e) }`, `f10()`, `f11()`, `f12()`, `f13()`, `f14()`,
		`f15()`, `f16()`, `f17()`, `f18()`, `f19()`,
		`eval("1; with ({}) { }") + "," + eval("2; with ({}) { 3 }")`,
		`eval("var gv = 7; gv * 2") + gv`, `(0, eval)("var gi = 8; typeof gi") + gi`,
		`(0, eval)("'use strict'; var sv = 9; typeof sv") + typeof sv`,
		`eval("1; try { 2 } finally { 3 }")`, `eval("for (var z = 0; z < 3; z++) {} z")`,
		`eval("let gl = 1; gl") + typeof gl`,
		`try { eval("\n\n  null.x") } catch (e) { frames(e) }`,
		`try { (0, eval)("undefinedName") } catch (e) { frames(e) }`,
	}
	regDiffAll(t, src, calls)
}
