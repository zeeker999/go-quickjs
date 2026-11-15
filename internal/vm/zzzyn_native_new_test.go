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
