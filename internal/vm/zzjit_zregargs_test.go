//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

// regArgsInlineSrc's callees keep their arguments: they read them by
// index and count them -- sloppy and strict, past the call's count, by a
// key the build does not know, with a property read in the reader whose
// shape changes, which leaves the inlined frame -- or hand them on,
// f.apply(this, arguments), to a function of none, two and six
// parameters, to a sloppy one reading its this, to a forwarder in turn,
// to one that throws now and then, as prototype.js's Class.create's
// constructors hand them to initialize, and as Array.apply(null,
// arguments) makes an array of them; and a sloppy function's arguments
// object aliasing its parameters, a rest parameter and an arguments
// object made in a native callee; and a reader and a forwarder inlined
// whose call seldom made leaves native code inside them, the frame Go
// makes of each given the call's arguments (jitRegInlineArgs); and
// functions called through Function.prototype.call, G.call(x, ...),
// TypeScript's _super.call(this, ...) among them, natively (ssa.CallSite's
// Via), a sloppy one's null receiver coerced by Go.
const regArgsInlineSrc = `
function rd() { return arguments[0] + (arguments[1] | 0) * 2 + arguments.length }
function rdStrict() { "use strict"; return arguments[0] - arguments[2] + arguments.length }
function rdPast() { return (arguments[3] === undefined ? 100 : arguments[3]) + arguments[0] }
function rdKey(k) { return arguments[k] }
function rdProp() { return arguments[0].v + arguments.length }
function t0() { return this === undefined ? 0 : 1 }
function t2(a, b) { return a + b }
function t6(a, b, c, d, e, f) { return a + b + (c | 0) + (d | 0) + (e | 0) + (f === undefined ? -1 : f) }
function tThis(a) { return (this === globalThis ? 7 : 3) + a }
function tThrow(a) { if (a % 1000 === 999) throw new RangeError("t" + a); return a }
function f0() { return t0.apply(this, arguments) }
function f2() { return t2.apply(this, arguments) }
function f6() { return t6.apply(this, arguments) }
function fThis() { return tThis.apply(this, arguments) }
function fFwd() { return f2.apply(this, arguments) }
function fThrow() { return tThrow.apply(this, arguments) }
function fArr() { return Array.apply(null, arguments) }
var Class = { create: function () { return function () { this.initialize.apply(this, arguments) } } };
var Pt = Class.create();
Pt.prototype.initialize = function (x, y) { this.x = x; this.y = y === undefined ? -5 : y };
function mapped(a, b) { return arguments[0] + b + arguments.length }
function mappedObj() { var o = arguments; return o.length + o[0] }
function rest(a, ...more) { return a + more.length + (more[0] | 0) }
function argsObj() { "use strict"; var o = arguments; return o.length * 10 + o[1] }
var Base = (function () { function Base(x, y) { this.x = x; this.y = y } return Base })();
var Derived = (function (_super) {
	function Derived(x, y, z) { _super.call(this, x, y); this.z = z }
	Derived.prototype = Object.create(_super.prototype);
	return Derived;
})(Base);
function viaSloppy(a) { return (this === globalThis ? 1000 : 0) + a }
function viaStrict(a, b) { "use strict"; return (this === undefined ? 50 : this) + a + (b | 0) }
var coldN = 0;
function cold() { coldN++ }
function rdCold() { if (arguments[0] % 5000 === 4999) cold(); return arguments[1] + arguments.length + arguments[0] }
function fCold() { if (arguments[0] % 5000 === 4998) cold(); return t2.apply(this, arguments) }
function main(n) {
	var s = 0, m = "", o1 = { v: 1 }, o2 = { w: 0, v: 2 };
	coldN = 0;
	for (var i = 0; i < n; i++) {
		s += rd(i, 1) + rd(i) + rdStrict(i, 2, 3) + rdPast(i, 1) + rdPast(i, 1, 2, 3);
		s += rdKey(i & 1, 5, 6) + rdProp(i & 3 ? o1 : o2, 0);
		s += f0() + f2(i, 1) + f6(i, 1, 2, 3, 4, 5) + f6(i, 1) + fThis(i) + fFwd(i, 2);
		try { s += fThrow(i) } catch (e) { m = e.message }
		var p = new Pt(i, 2), q = new Pt(i);
		s += p.x + p.y + q.y + fArr(i, 1, 2).length + fArr().length + fArr(3).length;
		s += mapped(i, 2) + mappedObj(i) + rest(i, 1, 2) + rest(i) + argsObj(i, 4);
		s += rdCold(i, 3) + fCold(i, 4);
		var d = new Derived(i, 1, 2);
		s += d.x + d.y + d.z + viaSloppy.call(null, i) + viaStrict.call(7, i) + viaStrict.call(undefined, i, 2);
		s = s % 1000000007;
	}
	return s + ":" + m + ":" + coldN;
}
`

// regTailMutualSrc's two functions call each other in tail position, a
// million deep, from a loop that calls them natively: past tailCallDepth
// each call is a jump to the other's code in the frame given up (mir's
// windowTailJump), the stack no deeper than at a thousand.
const regTailMutualSrc = `
function ev(n, acc) { "use strict"; return n === 0 ? acc : od(n - 1, acc + 1) }
function od(n, acc) { "use strict"; return n === 0 ? -acc : ev(n - 1, acc + 2) }
function main(n) { var s = 0; for (var i = 0; i < n; i++) s = (s + ev(1024 + (i & 1), 0)) | 0; return s + ":" + ev(1000000, 0) + ":" + od(1000001, 0) }
`
