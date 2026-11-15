//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

// regMadeArgsSrc's functions make closures that read their arguments by
// index and count them -- sloppy and strict, a captured binding read --
// and call them, the closure made anew each turn: inlined where called and
// never made (jitRegFeedback's madeInline, InlineSite's ReadsArguments);
// one reads past the call's count and one by a key the build does not
// know, Go's, which stops the closure being inlined at its call once it
// has left native code there too often (jitInlineLeft); one throws
// once from inside, where Go makes the closure and its frame is given the
// call's arguments (jitRegInlineArgs); and one with a rest parameter,
// called, not inlined.
const regMadeArgsSrc = `
function reads(n) {
	var s = 0, k = 3;
	for (var i = 0; i < n; i++) {
		var f = function () { return arguments[0] * k + (arguments[1] | 0) + arguments.length };
		s += f(i, 2);
		var g = function () { "use strict"; return arguments[1] - arguments[0] * arguments.length };
		s += g(i, 1, 2);
		s = s % 1000000007;
	}
	return s;
}
function past(n) {
	var s = 0;
	for (var i = 0; i < n; i++) {
		var g = function () { "use strict"; return (arguments[2] === undefined ? 7 : arguments[2]) - arguments[0] };
		s += g(i, 1);
	}
	return s;
}
function keyed(n) {
	var s = 0;
	for (var i = 0; i < n; i++) {
		var f = function (j) { "use strict"; return arguments[j] + arguments.length };
		s += f(i & 1, 5, 6);
	}
	return s;
}
function throws(n) {
	var s = 0, m = "";
	for (var i = 0; i < n; i++) {
		var f = function () { if (arguments[0] === 9000) throw new RangeError("at " + arguments[0]); return arguments[0] + arguments.length };
		try { s += f(i, 1) } catch (e) { m = e.message }
	}
	return s + ":" + m;
}
function rests(n) {
	var s = 0;
	for (var i = 0; i < n; i++) {
		var f = function (a, ...more) { return a + more.length + (more[0] | 0) };
		s += f(i, 4, 5) + f(i);
	}
	return s;
}
function main(n) { return reads(n) + ";" + past(n) + ";" + keyed(n) + ";" + throws(n) + ";" + rests(n) }
`
