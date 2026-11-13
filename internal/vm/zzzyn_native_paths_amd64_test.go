package vm

import (
	"sync/atomic"
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
)

// The native paths added after the first set, each run by the native
// handlers against the Go table's loop (QJS_NONATIVE's), with the
// collector idle and as while it marks: the same answers, the same exits
// marking or not, and none of the operations a path is for where the
// program meets only what the path answers.

// nativeWarmExits is how often a fast case's operation may exit while
// the caches it reads fill, in a run of a thousand of it.
const nativeWarmExits = 8

// nativeCheck runs src by the Go table and by the native handlers,
// marking and not, and fails on a different answer or different exits;
// where fast is non-empty, it runs fastSrc natively too and fails where
// any of those operations exited past their caches' filling.
func nativeCheck(t *testing.T, src, fastSrc string, fast ...bytecode.ROp) {
	t.Helper()
	nativeCounting(t)
	want := nativeRunSrc(t, src, false)
	var exits [2][256]int64
	for i, marking := range []bool{false, true} {
		nativeMarking(t, marking)
		before := nativeExitOps()
		if got := nativeRunSrc(t, src, true); got != want {
			t.Errorf("native (marking %v):\n%s\nwant\n%s", marking, got, want)
		}
		after := nativeExitOps()
		for op := range after {
			exits[i][op] = after[op] - before[op]
		}
	}
	nativeMarking(t, false)
	if exits[0] != exits[1] {
		for op := range exits[0] {
			if exits[0][op] != exits[1][op] {
				t.Errorf("%s: %d exits, %d while marking", bytecode.ROp(op), exits[0][op], exits[1][op])
			}
		}
	}
	if len(fast) == 0 {
		return
	}
	want = nativeRunSrc(t, fastSrc, false)
	before := nativeExitOps()
	if got := nativeRunSrc(t, fastSrc, true); got != want {
		t.Errorf("native fast case: %s, want %s", got, want)
	}
	after := nativeExitOps()
	for _, op := range fast {
		// A few while the caches fill, none after.
		if n := after[op] - before[op]; n > nativeWarmExits {
			t.Errorf("%s exited %d times", op, n)
		}
	}
	if atomic.LoadInt64(&nativeEnters) == 0 {
		t.Error("nothing ran natively")
	}
}

// TestRegNativeExitCalls runs instructions the code leaves to Go by its
// call of their Go handlers (nativeCallH): handlers that go on, that give
// back ^pc and finish in their slow function, that throw to a handler of
// the frame's or of a caller's, and that end a frame (a generator's
// yield), in the base frame and in frames the code entered itself.
func TestRegNativeExitCalls(t *testing.T) {
	nativeCheck(t, `
		var out = []
		function strs(n) { var s = "abcabc", t = 0; for (var i = 0; i < n; i++) t += s.indexOf("c", i % 6) + String(i).length; return t }
		function thrower(o) { try { return o.x.y } catch (e) { return e.constructor.name } }
		function deep(n) { return n == 0 ? thrower(null) : deep(n - 1) + "" }
		function getter(n) { var o = { get g() { if (n-- == 2) throw new RangeError("r"); return n } }, r = []; for (var i = 0; i < 4; i++) { try { r.push(o.g) } catch (e) { r.push(e.message) } } return r.join() }
		function* gen(n) { for (var i = 0; i < n; i++) yield i * 2 }
		function gens(n) { var t = 0; for (var v of gen(n)) t += v; var g = gen(3); t += g.next().value + g.next().value; return t }
		function stack() { function f(n) { return n ? f(n - 1) : new Error("e").stack } function g(n) { var s = "x".repeat(n); return n ? g(n - 1) + s.length : h() } function h() { return new Error("h").stack } return f(5) + g(3) }
		out.push(strs(50), thrower({x: {y: 1}}), thrower({}), deep(20), getter(5), gens(10), stack())
		var calls = 0
		function inner(x) { calls++; return [x].concat([1]).length + Math.max(x, 1) }
		function outer(n) { var t = 0; for (var i = 0; i < n; i++) t += inner(i); return t }
		out.push(outer(100), calls)
		out.join("\n")`, "")
}

// TestRegNativePostDec runs x-- of numbers natively and of anything else
// by Go.
func TestRegNativePostDec(t *testing.T) {
	nativeCheck(t, `
		var vals = [3, 0, -0, 1.5, NaN, Infinity, -Infinity, 2 ** 53, -(2 ** 31), "5", "", "x", null, undefined, true, {valueOf() { return 7 }}, 10n, -1n], out = []
		function dec(v) { var a = v, b = a--; return [String(a), String(b), typeof a, typeof b].join() }
		for (var v of vals) out.push(dec(v))
		function loop(n) { var j, k = 0; for (j = n; j-- > 0;) k += j; return k + ":" + j }
		out.push(loop(100), loop(0), loop(2.5))
		var o = {n: 3}
		function prop() { var r = []; while (o.n-- > 0) r.push(o.n); return r.join() }
		out.push(prop())
		out.join("\n")`, `
		function loop(n) { var j, k = 0; for (j = n; j-- > 0;) k += j; return k + ":" + j }
		loop(1000)`, bytecode.RPostDec)
}

// TestRegNativePrimMethods runs get_method_w of strings' and numbers'
// methods natively, from the realm's cache of their prototypes' data
// properties, with the prototype changed under the sites -- a method
// replaced, one made a getter, one added -- and receivers of other kinds,
// which Go reads.
func TestRegNativePrimMethods(t *testing.T) {
	nativeCheck(t, `
		var out = []
		function calls(n) { var s = "hello", t = ""; for (var i = 0; i < n; i++) t += s.charAt(i % 5) + (i).toString(16) + s.indexOf("l") + (1.5).toFixed(1); return t }
		out.push(calls(20))
		String.prototype.charAt = function (i) { return "<" + this + i + ">" }
		out.push(calls(3))
		Object.defineProperty(Number.prototype, "toFixed", { get: function () { var self = this; return function () { return "g" + self } }, configurable: true })
		out.push(calls(3))
		Number.prototype.extra = function () { return this * 2 }
		function ex(n) { var t = 0; for (var i = 0; i < n; i++) t += (i).extra(); return t }
		out.push(ex(10))
		function kinds() { var r = []; for (var v of ["s", 1, true, Symbol("q"), 1n]) { try { r.push(v.toString()) } catch (e) { r.push(e.name) } } return r.join() }
		out.push(kinds())
		function miss() { try { return "x".nope() } catch (e) { return e.constructor.name } }
		out.push(miss(), miss())
		out.join("\n")`, `
		function calls(n) { var s = "hello", t = 0; for (var i = 0; i < n; i++) t += s.indexOf("l", i % 3) + s.charCodeAt(1) + (i).toString().length; return t }
		String(calls(1000))`, bytecode.RGetMethodW)
}

// TestRegNativeArrayForOf runs for-of over dense arrays natively: holes,
// an array grown, shrunk and made sparse under the loop, a break, the
// iterator replaced, and cursors of other kinds, which Go steps.
func TestRegNativeArrayForOf(t *testing.T) {
	nativeCheck(t, `
		var out = []
		function sum(a) { var s = 0; for (var v of a) s += v; return s }
		out.push(sum([1, 2, 3]), sum([]), String(sum([1, , 3])), sum([1.5, -0, NaN].concat([4])))
		function grow() { var a = [1, 2], r = []; for (var v of a) { r.push(v); if (a.length < 6) a.push(v * 10) } return r.join() }
		function shrink() { var a = [1, 2, 3, 4, 5], r = []; for (var v of a) { r.push(v); a.length = 3 } return r.join() }
		function sparse() { var a = [1, 2, 3], r = []; for (var v of a) { r.push(v); a[100] = 9 } return r.length }
		function brk() { var r = []; for (var v of [{x: 1}, "s", null, undefined]) { r.push(String(v && v.x)); if (v === null) break } return r.join() }
		out.push(grow(), shrink(), sparse(), brk())
		var saved = Array.prototype[Symbol.iterator]
		Array.prototype[Symbol.iterator] = function* () { yield "own" }
		out.push(sum([1, 2]))
		Array.prototype[Symbol.iterator] = saved
		function others() { var r = []; for (var v of "ab") r.push(v); for (var v of new Set([3, 4])) r.push(v); for (var [k, v] of new Map([[5, 6]])) r.push(k + v); return r.join() }
		out.push(others(), sum([7, 8]))
		out.join("\n")`, `
		function sum(a, n) { var s = 0; for (var i = 0; i < n; i++) for (var v of a) s += v; return s }
		String(sum([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 100))`, bytecode.RIterNext)
}

// TestRegNativeCharCodeAt runs s.charCodeAt(i) natively over flat
// strings, ASCII and not, at positions in and out of range and of every
// kind, with ropes, the method replaced and bound, and receivers that are
// no string, which Go answers.
func TestRegNativeCharCodeAt(t *testing.T) {
	nativeCheck(t, `
		var out = []
		function codes(s, ps) { var r = []; for (var p of ps) r.push(s.charCodeAt(p)); return r.join() }
		var ps = [0, 1, 2.9, -0.5, -1, 3, 4, 100, NaN, Infinity, -Infinity, "1", null, undefined, true, 1e300, -0, 2 ** 53]
		out.push(codes("abcd", ps), codes("añ€😀", ps), codes("", ps))
		var rope = "ab"; for (var i = 0; i < 20; i++) rope += String.fromCharCode(65 + i)
		out.push(codes(rope, [0, 5, 21]), codes(rope, [21, 22]))
		var wide = "é"; for (var i = 0; i < 5; i++) wide += "ü"
		out.push(codes(wide, [0, 5, 6]))
		function sum(s) { var t = 0; for (var i = 0; i < s.length; i++) t = (t * 31 + s.charCodeAt(i)) | 0; return t }
		out.push(sum("hello world"), sum("ünïcödé"))
		var saved = String.prototype.charCodeAt
		String.prototype.charCodeAt = function (i) { return -i }
		out.push(sum("abc"))
		String.prototype.charCodeAt = saved.bind("zzz")
		out.push(sum("abc"))
		String.prototype.charCodeAt = saved
		function other(v) { try { return v.charCodeAt(0) } catch (e) { return e.name } }
		out.push(other(5), other(null), other({charCodeAt: function () { return "o" }}), other(new String("q")), sum("abc"))
		out.join("\n")`, `
		function sum(s) { var t = 0; for (var i = 0; i < s.length; i++) t = (t * 31 + s.charCodeAt(i)) | 0; return t }
		var s = "the quick brown fox jumps over the lazy dog", u = "ünïcödé ünïcödé", t = 0
		for (var j = 0; j < 30; j++) t += sum(s) + sum(u)
		String(t)`, bytecode.RCharCodeAt)
}
