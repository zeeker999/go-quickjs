package vm

import "testing"

// TestCaptureByValue pins what a closure sees of a binding it may take by
// value (bytecode.UpvalueDesc's ByValue) -- a const initialized when it is
// made, a strict function's parameter nothing stores to -- and of every
// binding that looks like one and is not: one assigned after the closure
// is made, by its maker, by another closure or one a closure makes, by an
// update, by an eval, through a mapped arguments object; one still in its
// dead zone, as a hoisted function or a switch's other case sees it, or a
// parameter list's later parameter. Each on both tiers.
func TestCaptureByValue(t *testing.T) {
	defer SetTreeTier(true)
	for _, tt := range []struct {
		name, src, want string
	}{
		{name: "param", src: `'use strict'; function f(x) { const g = () => x; return g() + g() } f(21)`, want: "42"},
		{name: "params three", src: `'use strict'; function f(a, b, c) { return () => a + b * c } f(1, 2, 3)()`, want: "7"},
		{name: "const", src: `'use strict'; function f() { const y = 5; return () => y } f()()`, want: "5"},
		{name: "loop consts", src: `'use strict'; function f() { const fs = []; for (const v of [1, 2, 3]) { const w = v * 10; fs.push(() => v + w) } return fs.map((g) => g()).join() } f()`, want: "11,22,33"},
		{name: "after return", src: `'use strict'; function mk(x) { return () => x } const p = mk(1), q = mk(2); p() + ',' + q()`, want: "1,2"},
		{name: "mixed", src: `'use strict'; function f(a, b) { const g = function () { return a + b }; b = 10; return g() } f(1, 2)`, want: "11"},
		{name: "nested", src: `'use strict'; function f(x) { return () => () => x } f(9)()()`, want: "9"},
		{name: "assigned after", src: `'use strict'; function f(x) { const g = () => x; x = 2; return g() } f(1)`, want: "2"},
		{name: "assigned before", src: `'use strict'; function f(x) { x = 3; const g = () => x; return g() } f(1)`, want: "3"},
		{name: "updated", src: `'use strict'; function f(x) { const g = () => x; x++; return g() } f(1)`, want: "2"},
		{name: "updated as a key", src: `'use strict'; function f(x, a) { const g = () => x; a[x++]; return g() } f(0, [7])`, want: "1"},
		{name: "compound", src: `'use strict'; function f(x) { const g = () => x; x += 5; return g() } f(1)`, want: "6"},
		{name: "destructured", src: `'use strict'; function f(x) { const g = () => x; [x] = [8]; return g() } f(1)`, want: "8"},
		{name: "loop target", src: `'use strict'; function f(x) { const g = () => x; for (x of [4]) {} return g() } f(1)`, want: "4"},
		{name: "sibling assigns", src: `'use strict'; function f(x) { const g = () => x; const h = () => { x = 3 }; h(); return g() } f(1)`, want: "3"},
		{name: "grandchild assigns", src: `'use strict'; function f(x) { const g = () => x; const h = () => () => { x = 4 }; h()(); return g() } f(1)`, want: "4"},
		{name: "itself assigns", src: `'use strict'; function f(x) { const g = () => { x++; return x }; return g() + g() } f(1)`, want: "5"},

		{name: "sloppy arguments", src: `function f(x) { const g = () => x; arguments[0] = 5; return g() } f(1)`, want: "5"},
		{name: "hoisted function", src: `'use strict'; function f() { function g() { return y } const y = 7; return g() } f()`, want: "7"},
		{name: "switch case", src: `'use strict'; function f(k) { switch (k) { case 0: const y = 1; case 1: return (() => typeof y)() } } try { f(1) } catch (e) { e.name }`, want: "ReferenceError"},
		{name: "later parameter", src: `function f(g = () => b, b = 8) { return g() } f()`, want: "8"},
		{name: "class name", src: `'use strict'; class C { static m() { return C } } C.m() === C`, want: "true"},
		{name: "generator", src: `'use strict'; function* f(x) { const g = () => x; yield 1; yield g() } const it = f(3); it.next(); it.next().value`, want: "3"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			for _, on := range []bool{false, true} {
				SetTreeTier(on)
				r := New(Config{})
				v, err := r.Run(compileForTest(t, tt.src))
				if err != nil {
					r.Close()
					t.Fatalf("tree %v: %v", on, err)
				}
				got, err := r.ToString(v)
				if err != nil || got.Go() != tt.want {
					t.Errorf("tree %v: got %v, %v; want %s", on, got, err, tt.want)
				}
				r.Close()
			}
		})
	}
}

// TestCaptureByValueAllocs pins what a closure over its strict maker's
// parameters costs: an arrow over three it takes by value is the function
// object alone, its copies in the same allocation; a function taking one
// by value and sharing another, the object and one room for both cells,
// as when it shared both.
func TestCaptureByValueAllocs(t *testing.T) {
	skipUnlessAllocsOrdinary(t)
	defer SetTreeTier(true)
	for _, tt := range []struct {
		name, src string
		want      float64
	}{
		{name: "by value", src: `'use strict'; function f(a, b, c) { return () => a + b + c }; for (let i = 0; i < 10; i++) f(1, 2, 3); f`, want: 1},
		{name: "mixed", src: `'use strict'; function f(a, b, c) { b++; return function () { return a + b + c } }; for (let i = 0; i < 10; i++) f(1, 2, 3); f`, want: 2},
	} {
		for _, on := range []bool{false, true} {
			SetTreeTier(on)
			r := New(Config{})
			v, err := r.Run(compileForTest(t, tt.src))
			if err != nil {
				t.Fatal(err)
			}
			args := []Value{Int(1), Int(2), Int(3)}
			if n := testing.AllocsPerRun(100, func() {
				if _, err := r.Call(v, Undefined, args); err != nil {
					t.Fatal(err)
				}
			}); n != tt.want {
				t.Errorf("%s, tree %v: %v allocations a call, want %v", tt.name, on, n, tt.want)
			}
			r.Close()
		}
	}
}
