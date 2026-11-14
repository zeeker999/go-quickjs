//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"fmt"
	"strings"
	"testing"
)

// regTailSrc makes calls in tail position from register code's native
// code: tasks' functions, called by a method in tail position, as AWFY's
// Richards calls its tasks', the method compiled before it met the
// second; and recursions through calls in tail position past
// tailCallDepth, of one function and of two by turns, whose frames are
// given up.
const regTailSrc = `
"use strict";
function next(x) { return x + 1 }
class Task {
	constructor(fn) { this.fn = fn; this.h = 1 }
	run(i) { return this.fn(i, this.h) }
}
function f1(w, d) { return next(w + d) }
function f2(w, d) { return next(w - d) }
var t1 = new Task(f1), t2 = new Task(f2), runs = 0;
function tasks(n) {
	// f2 first called at the second run, once run is compiled for f1.
	var b = runs++ ? t2 : t1;
	var acc = 0;
	for (var i = 0; i < n; i++) acc += (i & 1 ? t1 : b).run(i);
	return acc;
}
function mk(k) { var o = { n: 0 }; return function (w, d) { o.n += k; return next(w + d + o.n) } }
function fresh(n) {
	// Two functions anew at each run, which the code learns up to
	// jitCallTargets of, and calls the rest of through their links.
	var a = new Task(mk(1)), b = new Task(mk(2)), acc = 0;
	for (var i = 0; i < n; i++) acc += (i & 1 ? a : b).run(i);
	return acc;
}
// Seven functions, past the four a call calls natively: the rest called
// through their links (ssa.CallSite's Generic), an arrow reading its this
// among them, and one called with more arguments than it has parameters.
function g0(x) { return x + 1 }
function g1(x, y) { return x * 2 + (y === undefined) }
function g2(x) { return x - (this === undefined) }
function g3(x) { return -x }
function g4(x) { return x * x }
var g5 = { v: 7, mk() { return (x) => x + this.v } }.mk();
function g6(x, y, z) { return x + (z === undefined ? 100 : z) }
var shapes = [g0, g1, g2, g3, g4, g5, g6];
function poly(n) { var s = 0; for (var i = 0; i < n; i++) s += shapes[i % 7](i); return s }
function even(n, acc) { if (n === 0) return acc; return odd(n - 1, acc + 1) }
function odd(n, acc) { if (n === 0) return -acc; return even(n - 1, acc + 2) }
function self(n, acc) { if (n === 0) return acc; return self(n - 1, acc + n) }
function deep(n) { var s = 0; for (var i = 0; i < n; i++) s += even(40 + (i & 7), 0) + self(40 + (i & 7), 0); return s }
`

// TestJITRegTailCalls runs regTailSrc's callers in a runtime whose register
// code runs natively and in one with no JIT, and compares their answers. A
// call in tail position that met a function its code did not call natively
// leaves native code, and Go makes it as the interpreter would -- the
// ordinary call while the stack is shallow, whose callee the code learns --
// so that tasks leaves native code at none of its calls once its code
// learned their functions -- where the interpreter had made the rest of
// the invocation, which learned nothing, and the call left native code at
// every call after; and a recursion through calls in tail position, past
// tailCallDepth, runs each frame given to a callee in its native code
// (regRestartRun).
func TestJITRegTailCalls(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			RecordJITCompiles(true)
			defer RecordJITCompiles(false)
			rts := regNativeRuntimes(t, regTailSrc)
			rts[0].SetJITWarmUp(1, 1)
			run := func(name string, n float64) {
				t.Helper()
				var got [2]string
				for i, r := range rts {
					fn, _ := r.GetProp(Obj(r.Global()), r.Intern(name))
					v, err := r.Call(fn, Undefined, []Value{Float(n)})
					got[i] = regNativeShow(r, v, err)
				}
				if got[0] != got[1] {
					t.Errorf("%s: with the JIT %q, without %q", name, got[0], got[1])
				}
			}
			for range 4 {
				run("tasks", 2000)
			}
			r := rts[0]
			before := r.RegNativeStats().HostOps
			for range 4 {
				run("tasks", 2000)
			}
			after := r.RegNativeStats().HostOps
			for _, op := range []string{"tail_call_w", "tail_call_method_w"} {
				if n := after[op] - before[op]; n != 0 {
					t.Errorf("%d exits at %s once the code learned its callees", n, op)
				}
			}
			for _, name := range []string{"fresh", "poly"} {
				for range 4 {
					run(name, 2000)
				}
				before := r.RegNativeStats()
				for range 4 {
					run(name, 2000)
				}
				// Past the functions it calls natively, the others called
				// through their links: none of Go's, nor an exit.
				after := r.RegNativeStats()
				if n := after.Hosts - before.Hosts; n != 0 {
					t.Errorf("%s: %d exits once its calls were learned: %v", name, n, after.HostOps)
				}
				if n := after.GoCalls - before.GoCalls; n != 0 {
					t.Errorf("%s: %d calls of Go once its calls were learned", name, n)
				}
			}
			for range 2 {
				run("deep", 2000)
			}
			entries := r.RegNativeStats().Entries
			run("deep", 2000)
			// The frames given to a callee in place past tailCallDepth: 8 or
			// more of self's, for each of the 2,000 calls, each entered in
			// its native code.
			if n := r.RegNativeStats().Entries - entries; n < 2000*8 {
				t.Errorf("%d entries of native code: the frames given to callees ran in the interpreter", n)
			}
			// arm64's backend compiles the same code, its generic calls of
			// register code among it, on any host.
			if fns, _, _ := regCompileARM64(t, r); fns == 0 {
				t.Error("nothing compiled for arm64")
			}
		})
	}
}

// regClassSrc constructs classes' objects, of base classes with no field to
// initialize, from register code's native code: natively, as the stack
// form's code constructs them (jitConstructs), where Go had constructed
// each; and a construction that throws, and a class's constructor called,
// which throws.
const regClassSrc = `
class Vec { constructor(x, y) { this.x = x; this.y = y } add(o) { return new Vec(this.x + o.x, this.y + o.y) } }
class Cell { constructor(v) { if (v < 0) throw new RangeError("neg " + v); this.v = v; this.next = null } }
function classes(n) {
	var a = new Vec(0, 0), s = 0, head = null;
	for (var i = 0; i < n; i++) {
		a = a.add(new Vec(i, 1));
		var c = new Cell(i);
		c.next = head;
		head = (i & 15) ? c : null;
		s += c.v;
	}
	return a.x + a.y + s;
}
function checks(n) {
	var m = "", head = null;
	for (var i = 0; i < n; i++) {
		try { head = new Cell(i % 100 - 1) } catch (e) { m = e.message }
		try { Vec(1, 2) } catch (e) { m += e instanceof TypeError }
	}
	return m + (head instanceof Cell) + Object.keys(new Vec(1, 2));
}
`

// TestJITRegClassConstructions runs regClassSrc's functions natively and in
// a runtime with no JIT and compares them: once compiled, classes'
// constructions are native, none Go's.
func TestJITRegClassConstructions(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			rts := regNativeRuntimes(t, regClassSrc)
			r := rts[0]
			r.SetJITWarmUp(1, 1)
			r.jitState().reporting = true
			news := func() (n uint64) {
				for k, v := range r.RegNativeStats().GoCallOps {
					if strings.HasPrefix(k, "new_w") {
						n += v
					}
				}
				return n
			}
			var made uint64
			for k := range 8 {
				if k == 4 {
					made = news()
				}
				for _, name := range []string{"classes", "checks"} {
					var got [2]string
					for i, r := range rts {
						fn, _ := r.GetProp(Obj(r.Global()), r.Intern(name))
						v, err := r.Call(fn, Undefined, []Value{Float(3000)})
						got[i] = regNativeShow(r, v, err)
					}
					if got[0] != got[1] {
						t.Fatalf("%s: with the JIT %q, without %q", name, got[0], got[1])
					}
				}
			}
			// None of Vec's and Cell's, but the RangeErrors' at most.
			if n := news() - made; n > 4*30 {
				t.Errorf("%d constructions Go made in 4 runs", n)
			}
		})
	}
}
