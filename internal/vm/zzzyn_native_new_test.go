//go:build amd64 || arm64

package vm

import (
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
)

// nativeNewSrc constructs with functions and classes of every kind a
// site meets, each enough times for its site's cache to hold it
// (regCallConstruct, past slackTracking objects): bodies returning an
// object, a primitive and nothing, one storing its parameters alone
// (leafCall), one reading new.target and arguments past its parameters,
// one capturing its this and a local in closures, one throwing, one
// constructing itself deeper than the native frame stack goes, one whose
// call in tail position returns what the construction must not, a
// class's and one with fields, a prototype replaced, made a primitive and
// given an accessor the body's store meets, a bound constructor, and a
// stack trace taken in a constructor.
const nativeNewSrc = `
	var out = []
	function P(x, y) { this.x = x; this.y = y; this.z = x + y }
	function RO(x) { this.x = x; return { o: x } }
	function RP(x) { this.x = x; return x }
	function RU(x) { this.x = x; return }
	function L(a, b) { this.a = a; this.b = b }
	function NT(a) { this.t = new.target === NT; this.n = arguments.length; this.l = arguments[arguments.length - 1]; this.a = a }
	function CL(x) { var k = x * 2; this.get = function () { return k + this.x }; this.x = x; this.inc = () => { k++; return this } }
	function TH(x) { if (x % 5 == 3) throw new RangeError("t" + x); this.x = x }
	function DEEP(n) { this.n = n; this.next = n > 0 ? new DEEP(n - 1) : null }
	function TC(x) { "use strict"; this.x = x; return tcv(x) }
	function tcv(x) { return x & 1 ? { odd: x } : x }
	class C { constructor(a) { this.a = a; this.k = a + 1 } m() { return this.a * 10 } }
	class F { f = 7; constructor(a) { this.a = a } }
	function PR(x) { this.x = x }
	function ST(x) { this.s = new Error("e").stack.split("\n").length; this.x = x }
	function depth(o) { var d = 0; while (o) { d++; o = o.next } return d }
	var BP = P.bind(null, 100)
	for (var j = 0; j < 40; j++) {
		var p = new P(j, 2), ro = new RO(j), rp = new RP(j), ru = new RU(j), l = new L(j, "b")
		out.push(p.z, p instanceof P, ro.o, ro instanceof RO, rp.x, ru.x, l.a + l.b)
		var nt = new NT(j, 1, 2, "last"), nt0 = new NT()
		out.push(nt.t, nt.n, nt.l, nt.a, nt0.n, nt0.a, nt0.l)
		var cl = new CL(j); out.push(cl.get(), cl.inc() === cl, cl.get())
		try { out.push(new TH(j).x) } catch (e) { out.push(e.name + e.message) }
		var tc = new TC(j); out.push(tc.x, tc.odd)
		var c = new C(j); out.push(c.k, c.m(), c instanceof C)
		var f = new F(j); out.push(f.f + f.a)
		if (j == 12) PR.prototype = { tag: "second" }
		if (j == 20) PR.prototype = 5
		if (j == 28) { PR.prototype = { set x(v) { this.got = v * 3 } } }
		var pr = new PR(j); out.push(pr.tag, pr.x, pr.got, Object.getPrototypeOf(pr) === Object.prototype)
		out.push(new BP(j).z)
		if (j % 13 == 0) out.push(depth(new DEEP(300)))
		var st = new ST(j); out.push(st.s > 1)
	}
	out.join()`

// TestRegNativeNew checks new_w's native handler against the Go table:
// the same answers, marking or not, and a construction of a function or a
// class its site holds made with no exit once its objects' room is
// settled.
func TestRegNativeNew(t *testing.T) {
	was := nativeWarmExits
	nativeWarmExits = 16
	defer func() { nativeWarmExits = was }()
	nativeCheck(t, nativeNewSrc, `
		function P(x) { this.x = x; this.y = x + 1; if (x < 0) throw x }
		class K { constructor(a) { this.a = a; this.b = [a] } }
		var s = 0
		for (var i = 0; i < 1000; i++) s += new P(i).y + new K(i).b[0]
		String(s)`, bytecode.RNewW)
}

// nativeInstanceOfSrc tests instanceof of functions and classes against
// objects of their chains and of others, primitives, objects made with no
// prototype or given one after (objPolyProto), a proxy in the chain, a
// class with a Symbol.hasInstance of its own, a bound function, an arrow,
// a constructor whose prototype is replaced by a primitive and one that is
// no object, every answer and error at one site.
const nativeInstanceOfSrc = `
	var out = []
	function A() {} function B() {} B.prototype = Object.create(A.prototype)
	class K {} class L extends K {} class M extends L {}
	class H { static [Symbol.hasInstance](v) { return v === 7 } }
	var BA = A.bind(null), arrow = () => 1, P = function () {}
	var poly = Object.setPrototypeOf({}, M.prototype), bare = Object.create(null)
	var px = new Proxy(new L(), {}), pxIn = Object.create(new Proxy(new K(), {}))
	function at(v, c) { try { return v instanceof c } catch (e) { return e.name } }
	var vals = [new A(), new B(), new K(), new L(), new M(), poly, bare, px, pxIn, 7, "s", null, undefined, {}, [], P]
	var ctors = [A, B, K, L, M, Object, Array, H, BA, Function, P]
	for (var r = 0; r < 30; r++) {
		if (r == 10) P.prototype = 3
		if (r == 20) P.prototype = new M()
		for (var i = 0; i < vals.length; i++)
			for (var j = 0; j < ctors.length; j++) out.push(at(vals[i], ctors[j]))
		out.push(at(vals[r % vals.length], arrow), at(1, 2))
	}
	out.join()`

// TestRegNativeInstanceOf checks instanceof's native handler against the
// Go table: the same answers and errors, marking or not, and a site of
// one constructor answered with no exit once its cache holds it.
func TestRegNativeInstanceOf(t *testing.T) {
	nativeCheck(t, nativeInstanceOfSrc, `
		class K {} class L extends K {}
		function F() {}
		var o = new L(), f = new F(), s = 0
		for (var i = 0; i < 1000; i++) if (o instanceof K && !(f instanceof K) && f instanceof F && !(i instanceof F)) s++
		String(s)`, bytecode.RInstanceOf)
}

// nativeDerivedSrc constructs derived classes of every kind their
// constructions and super() calls meet: a super() of arguments and one
// handing on the frame's (a class with no constructor), chains two and
// three deep, a parent returning an object of its own, one throwing, one
// reading new.target, a derived constructor returning an object, a
// primitive, nothing before super(), super() twice, super() in an arrow,
// this captured by a closure before and after, fields of the derived
// class and of the parent, a parent replaced by changing the class's
// prototype, arguments read, and a stack trace taken in the parent.
const nativeDerivedSrc = `
	var out = []
	class A { constructor(x, y) { this.x = x; this.y = y; this.t = new.target.name } sum() { return this.x + (this.y | 0) } }
	class B extends A { constructor(x) { super(x, x * 2); this.b = 1 } }
	class C extends B { constructor(x, z) { super(x); this.c = z } }
	class D extends C {}
	class R { constructor(v) { if (v % 4 == 1) return { own: v }; this.r = v } }
	class RS extends R { constructor(v) { super(v); this.s = 2 } }
	class T { constructor(v) { if (v % 5 == 2) throw new Error("t" + v); this.v = v } }
	class TS extends T { constructor(v) { super(v); this.w = 1 } }
	class Obj extends A { constructor(v) { super(v); return { replaced: v } } }
	class Prim extends A { constructor(v) { super(v); return v } }
	class NoSuper extends A { constructor(v) { if (v > 100) super(v) } }
	class Twice extends A { constructor(v) { super(v); super(v) } }
	class Arrow extends A { constructor(v) { var f = () => super(v, 1); f(); this.k = this.x } }
	class Cap extends A { constructor(v) { var g = () => this; super(v, 3); this.g = g } }
	class F extends A { f = 5; constructor(v) { super(v) } }
	class PF { p = 7; constructor(v) { this.v = v } }
	class GF extends PF { constructor(v) { super(v); this.q = this.p + 1 } }
	class Args extends A { constructor() { super(arguments.length, arguments[1]) } }
	class St extends A { constructor(v) { super(v); this.n = new Error().stack.split("\n").length > 2 } }
	class Swap extends A { constructor(v) { super(v) } }
	function at(f) { try { return f() } catch (e) { return e.name + ":" + e.message } }
	for (var j = 0; j < 40; j++) {
		var b = new B(j), c = new C(j, "z"), d = new D(j, "d")
		out.push(b.sum(), b.b, b.t, c.c, c.t, d.c, d.t, d instanceof A, Object.getPrototypeOf(d) === D.prototype)
		var rs = new RS(j); out.push(rs.own, rs.r, rs.s, rs instanceof RS)
		out.push(at(() => new TS(j).w))
		out.push(at(() => new Obj(j).replaced), at(() => new Prim(j)), at(() => new NoSuper(j)), at(() => new Twice(j).x))
		out.push(at(() => new Arrow(j).k), at(() => { var o = new Cap(j); return o.g() === o && o.y }))
		out.push(new F(j).f, new GF(j).q, new Args(j, "a", 3).y, new Args().x, new St(j).n)
		if (j == 20) Object.setPrototypeOf(Swap, R)
		out.push(at(() => { var o = new Swap(j); return [o.x, o.r, o.own].join("/") }))
	}
	out.join()`

// TestRegNativeDerived checks constructions of derived classes and their
// super() calls, which the native handlers enter with their this
// bindings, against the Go table: the same answers and errors, marking
// or not, and new_w and super_call made with no exit once their caches
// hold them.
func TestRegNativeDerived(t *testing.T) {
	was := nativeWarmExits
	nativeWarmExits = 16
	defer func() { nativeWarmExits = was }()
	nativeCheck(t, nativeDerivedSrc, `
		class A { constructor(x) { this.x = x; this.y = x + 1 } }
		class B extends A { constructor(x) { super(x); this.z = 2 } }
		class C extends B {}
		var s = 0
		for (var i = 0; i < 1000; i++) s += new B(i).z + new C(i).y
		String(s)`, bytecode.RNewW, bytecode.RSuperCall, bytecode.RSuperForward, bytecode.RDerivedRet)
}
