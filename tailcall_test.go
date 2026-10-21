package quickjs_test

import (
	"context"
	"strings"
	"testing"
	"time"

	quickjs "github.com/go-quickjs/go-quickjs"
)

// TestTailCalls pins that a strict function's calls in tail position do not
// keep its frame: a chain of them far deeper than the call stack allows runs
// to the end.
func TestTailCalls(t *testing.T) {
	cases := []struct{ src, want string }{
		{`"use strict"; function f(n) { return n === 0 ? "done" : f(n - 1) } f(100000)`, "done"},
		{`"use strict"; function even(n) { return n === 0 || odd(n - 1) } function odd(n) { return n !== 0 && even(n - 1) } String(even(100001))`, "false"},
		{`"use strict"; const o = { m(n) { return n === 0 ? this.tag : this.m(n - 1) }, tag: "method" }; o.m(100000)`, "method"},
		{`"use strict"; function f(n, acc) { if (n === 0) return acc; return (0, f)(n - 1, acc + 1) } String(f(100000, 0))`, "100000"},
		{`"use strict"; function f(n) { return n === 0 ? "bound" : g(n - 1) } const g = f.bind(null); g(100000)`, "bound"},
		{`"use strict"; const f = n => n === 0 ? "arrow" : f(n - 1); f(100000)`, "arrow"},
	}
	for _, tc := range cases {
		rt := quickjs.New(quickjs.WithMaxCallDepth(400))
		v, err := rt.Eval(tc.src)
		if err != nil {
			t.Errorf("%s: %v", tc.src, err)
		} else if v.String() != tc.want {
			t.Errorf("%s = %q, want %q", tc.src, v.String(), tc.want)
		}
		rt.Close()
	}
	// Sloppy code, a try block and a for-of body have no tail calls: each of
	// these still runs out of stack.
	for _, src := range []string{
		`function f(n) { return n === 0 ? 0 : f(n - 1) } f(100000)`,
		`"use strict"; function f(n) { try { return n === 0 ? 0 : f(n - 1) } finally {} } f(100000)`,
		`"use strict"; function f(n) { for (const x of [1]) return n === 0 ? 0 : f(n - 1) } f(100000)`,
	} {
		rt := quickjs.New(quickjs.WithMaxCallDepth(400))
		_, err := rt.Eval(src)
		if err == nil || !strings.Contains(err.Error(), "RangeError") {
			t.Errorf("%s: got %v, want a RangeError", src, err)
		}
		rt.Close()
	}
}

// TestTailCallRefusedCallees pins that a call in tail position past the
// depth where the frame is given up reaches its callee as the ordinary call
// would, where the tree tier's frame settles the callee itself
// (tailArgsOf): each callee warmed first, so that it has a tree, and each
// caller run as one. A proxy is called through, a class constructor is
// refused, a sloppy function given no receiver sees the global object, a
// bound function gets its target, receiver and arguments, and an arrow its
// own this. Node's answers.
func TestTailCallRefusedCallees(t *testing.T) {
	// f and h call each other in tail position, which a call of a function
	// of its own does not make: past the depth, each call's frame is given
	// up, so that CALL is made where its frame is too.
	const run = "; function f(n) { return n === 0 ? CALL : h(n - 1) } function h(n) { return f(n) }" +
		" var r; for (var i = 0; i < 40; i++) r = f(100); r"
	cases := []struct{ setup, call, want string }{
		{`function g() { return "proxied" } for (var i = 0; i < 300; i++) g(); const p = new Proxy(g, {})`, "p()", "proxied"},
		{`class C {} for (var i = 0; i < 300; i++) new C()`, "C()", "TypeError"},
		{`var g = Function("return this === globalThis"); for (var i = 0; i < 300; i++) g()`, "g()", "true"},
		{`function g(a, b) { return a + b + this.x } for (var i = 0; i < 300; i++) g.call({ x: 0 }, 1, 2);
		  const b = g.bind({ x: 1 }, 2)`, "b(3)", "6"},
		{`const o = { m() { return () => this.v }, v: 7 }; const a = o.m(); for (var i = 0; i < 300; i++) a()`, "a()", "7"},
	}
	for _, c := range cases {
		src := `"use strict"; ` + c.setup + strings.ReplaceAll(run, "CALL", c.call)
		rt := quickjs.New(quickjs.WithMaxCallDepth(400))
		v, err := rt.Eval(src)
		if c.want == "TypeError" {
			if err == nil || !strings.Contains(err.Error(), "TypeError") {
				t.Errorf("%s: got %v, want a TypeError", src, err)
			}
		} else if err != nil {
			t.Errorf("%s: %v", src, err)
		} else if v.String() != c.want {
			t.Errorf("%s = %q, want %q", src, v.String(), c.want)
		}
		rt.Close()
	}
	// Mutual tail calls far past the call stack's limit, each callee settled
	// at the frame that ends in its call: Node's answer at 1,001.
	rt := quickjs.New(quickjs.WithMaxCallDepth(400))
	defer rt.Close()
	v, err := rt.Eval(`"use strict"; function ev(n) { return n === 0 ? true : od(n - 1) }
		function od(n) { return n === 0 ? false : ev(n - 1) }
		var r; for (var i = 0; i < 20; i++) r = ev(100001); r`)
	if err != nil || v.String() != "false" {
		t.Errorf("ev(100001) = %v, %v; want false", v, err)
	}
	// Methods: the settled call keeps its receiver (Node's answer at 1,001).
	v, err = rt.Eval(`"use strict"; const o = { k: 5, f(n) { return n === 0 ? this.k : this.g(n - 1) }, g(n) { return this.f(n) } };
		var r; for (var i = 0; i < 20; i++) r = o.f(100001); r`)
	if err != nil || v.String() != "5" {
		t.Errorf("o.f(100001) = %v, %v; want 5", v, err)
	}
	// A chain of them that never ends is interrupted, each hop a tick.
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	if _, err := rt.EvalContext(ctx, `"use strict"; function p(n) { return q(n + 1) } function q(n) { return p(n) } p(0)`); err == nil {
		t.Error("an endless chain of tail calls was not interrupted")
	}
}
