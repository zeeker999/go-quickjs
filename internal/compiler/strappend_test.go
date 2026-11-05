package compiler

import (
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/ast"
	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/parser"
)

// marked counts, over fn and what it nests, the stores marked as appends and
// the str_freezes.
func marked(fn *bytecode.Function) (stores, freezes int) {
	for _, in := range fn.Code {
		switch {
		case in.Op == bytecode.OpSetUpvalue && in.B != 0:
			stores++
		case in.Op == bytecode.OpStrFreeze:
			freezes++
		}
	}
	for _, k := range fn.Constants {
		if k.Kind == bytecode.ConstFunction && k.Fn != nil {
			s, n := marked(k.Fn)
			stores, freezes = stores+s, freezes+n
		}
	}
	return stores, freezes
}

func compileStrict(t *testing.T, src string, opts Options) *bytecode.Function {
	t.Helper()
	var prog *ast.Program
	var err error
	if prog, err = parser.Parse(src, parser.Options{}); err != nil {
		t.Fatalf("%s: %v", src, err)
	}
	fn, err := Compile(prog, opts)
	if err != nil {
		t.Fatalf("%s: %v", src, err)
	}
	return fn
}

// What is an owned append and what is not, and which reads are told so. The
// rule is that of strappend.go, and each case is what would be unsound, or
// costly, were it taken.
func TestOwnedAppends(t *testing.T) {
	for _, tc := range []struct {
		src             string
		stores, freezes int
	}{
		{`function f(xs) { var g = ""; var h = { push(a) { null !== a && (g += a); return true } }; return g }`, 1, 1},
		{`function f() { var s = ""; const add = x => { s += x }; return s + s }`, 1, 2},
		{`function f() { let s = ""; const add = x => { s += "lit" }; add(); return s }`, 1, 1},
		{`function f() { var s = ""; const a = x => { s += x }, b = y => { s += y }; return s }`, 2, 1},
		{`function f() { var s = ""; const add = x => { x || (s += x) } }`, 1, 0},
		{`function f() { var s = ""; const add = x => { (x, s += x, 1) } }`, 1, 0},
		// A read compiled first: nothing is owned, and nothing frozen.
		{`function f() { var s = ""; const r = s; const add = x => { s += x } }`, 0, 0},
		{`function f() { var s = ""; function get() { return s } const add = x => { s += x }; return s }`, 0, 0},
		{`function f() { var s = ""; const get = () => s; const add = x => { s += x } }`, 0, 0},
		// The value is used.
		{`function f() { var s = ""; const add = x => s += x }`, 0, 0},
		{`function f() { var s = ""; const add = x => { return s += x } }`, 0, 0},
		{`function f() { var s = ""; const add = x => { return x && (s += x) } }`, 0, 0},
		// The binding is not a closure's upvalue.
		{`function f(x) { var s = ""; s += x; return s }`, 0, 0},
		{`var s = ""; function f(x) { s += x }`, 0, 0},
		{`let s = ""; { const add = x => { s += x } }`, 0, 0},
		// What it is.
		{`function f(s) { const add = x => { s += x } }`, 0, 0},
		{`function f() { const s = ""; const add = x => { s += x } }`, 0, 0},
		{`function f() { for (let s = ""; ; ) { const add = x => { s += x } } }`, 0, 0},
		{`function f() { try {} catch (s) { const add = x => { s += x } } }`, 0, 0},
		{`function f() { const add = x => { s += x }; let s = "" }`, 0, 0},
		{`var g = function s() { const add = x => { s += x } }`, 0, 0},
		// A right side that could run code.
		{`function f() { var s = ""; const add = x => { s += x() } }`, 0, 0},
		{`function f() { var s = ""; const add = x => { s += x.y } }`, 0, 0},
		{`function f() { var s = ""; const add = x => { s += x + "" } }`, 0, 0},
		{`function f() { var s = ""; const add = x => { s += global } }`, 0, 0},
		{`function f() { var s = ""; const add = x => { s += 1 } }`, 0, 0},
		// Other operators.
		{`function f() { var s = 1; const add = x => { s -= x } }`, 0, 0},
		{`function f() { var s = 1; const add = x => { s = s + x } }`, 0, 0},
		// eval, anywhere in the function that declares it.
		{`function f() { var s = ""; const add = x => { s += x }; return eval("s") }`, 0, 0},
		{`function f() { var s = ""; const add = x => { s += x }; const g = () => () => eval("s") }`, 0, 0},
		// with.
		{`function f(o) { var s = ""; with (o) { const add = x => { s += x } } }`, 0, 0},
		// Counters are left alone: a local's reads have no str_freeze.
		{`function f(n) { var t = 0; for (var i = 0; i < n; i += 2) t += i; return t + i }`, 0, 0},
		{`function f() { var t = 0; const add = () => { t += 1 }; return t }`, 0, 0},
	} {
		fn := compileStrict(t, tc.src, Options{})
		if s, n := marked(fn); s != tc.stores || n != tc.freezes {
			t.Errorf("%s:\n  %d marked stores, %d str_freezes; want %d, %d", tc.src, s, n, tc.stores, tc.freezes)
		}
	}
	// Code for a debugger, which reads any binding, has none.
	if s, n := marked(compileStrict(t, `function f() { var s = ""; const add = x => { s += x }; return s }`, Options{Debug: true})); s != 0 || n != 0 {
		t.Errorf("code for a debugger: %d marked stores, %d str_freezes", s, n)
	}
}
