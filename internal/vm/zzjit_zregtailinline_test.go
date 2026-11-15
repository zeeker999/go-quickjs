//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

// regTailInlineSrc's callers call register functions whose calls are in
// tail position (tail_call_w, tail_call_method_w), which register code
// inlines as the stack form's does, a call in tail position as the
// ordinary call: a method returning one of two methods' results, as
// Richards' tasks return their scheduler's, and an arrow's, which reads
// the this it captured; a recursion whose calls of inc in tail position
// are inlined, made deep in the stack too, where the interpreter gives
// the frame up to the call; a strict pair calling each other in tail
// position, run deeper than any stack of frames would go; and calls whose
// results go to registers a closure captures, a construction's and a
// call's.
const regTailInlineSrc = `
class Task {
	constructor(n) { this.n = n; this.held = false }
	hold() { this.held = true; return this }
	release() { this.held = false; return this }
	step(i) { this.n = this.n + 1; return (i & 1) ? this.hold() : this.release() }
}
function tails(n) {
	var t = new Task(0), h = 0;
	for (var i = 0; i < n; i++) if (t.step(i).held) h++;
	return t.n * 1000 + h;
}
class Sched {
	constructor() { this.count = 0; this.f = (k) => this.bump(k) }
	bump(k) { this.count += k; return this.count }
}
var sched = new Sched(), arrow = sched.f;
function arrows(n) {
	var s = new Sched(), last = 0;
	for (var i = 0; i < n; i++) last = s.f(i & 3) + arrow(0);
	return last - sched.count;
}
function ping(n) { "use strict"; return n === 0 ? "done" : pong(n - 1) }
function pong(n) { "use strict"; return ping(n) }
function deep(n) { return ping(n) + n }
function inc(v) { "use strict"; return v + 1 }
function rec(n) { "use strict"; if (n === 0) return 0; var v = rec(n - 1); return inc(v) }
function Box(v) { this.v = v }
function sq(a) { return a * a }
function captured(n) {
	var s = 0;
	for (var i = 0; i < n; i++) {
		var b = new Box(i);
		var q = sq(i);
		s += get();
	}
	function get() { return b.v + q }
	return s;
}
`

