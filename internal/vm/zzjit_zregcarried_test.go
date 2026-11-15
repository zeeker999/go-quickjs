//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"fmt"
	"testing"
)

// regCarriedExitSrc is the stack form's osrSetup's loops, for register code: a
// loop a call leaves native code in at turn k, at a write of a number's
// property with no entry after it, the way a frame comes to a loop whose
// entry the function's code does not have, its object s carried unmade.
const regCarriedExitSrc = `
function P(v) { this.v = v }
P.prototype.add = function (o) { return new P(this.v + o.v) }
var obj = {a: 1}
function early(n, k) {
	const one = new P(1)
	let s = new P(0)
	for (let i = 0; i < n; i++) {
		const o = i === k ? 1 : obj
		o.a = 1
		s = s.add(one)
	}
	return s.v
}
function calls(c, n, k) {
	let t = 0
	for (let j = 0; j < c; j++) t += early(n, k)
	return t
}
function mixed(c) {
	let t = 0
	for (let j = 0; j < c; j++) t += early(100, j % 2 === 0 ? 5 : -1)
	return t
}
`

// regCarriedCall calls name with args in both runtimes, the first the
// JIT's, and fails unless they agree on want.
func regCarriedCall(t *testing.T, rts [2]*Runtime, want string, name string, args ...float64) {
	t.Helper()
	var got [2]string
	for i, r := range rts {
		fn, _ := r.GetProp(Obj(r.Global()), r.Intern(name))
		vs := make([]Value, len(args))
		for k, a := range args {
			vs[k] = Float(a)
		}
		v, err := r.Call(fn, Undefined, vs)
		got[i] = regNativeShow(r, v, err)
	}
	if got[0] != got[1] || got[0] != want {
		t.Fatalf("%s%v: with the JIT %q, without %q, want %q", name, args, got[0], got[1], want)
	}
}

// regCodeNamed is the register code the cache holds for the global
// function name.
func regCodeNamed(t *testing.T, r *Runtime, name string) *jitRegEntry {
	t.Helper()
	fn, _ := r.GetProp(Obj(r.Global()), r.Intern(name))
	cl := fn.Object().fn().closure
	return r.jit.regCache[r.jitKeyOf(cl.fn)]
}

// TestJITRegCarriedExitsGoOn has every call of a loop leave native code
// early, with no entry after the exit, its object carried unmade: the
// frames go on natively at the loop's next back edge, in the code kept
// for the loop (jitRegLoopCode), not in the interpreter to the loop's
// end, as the stack form's jitGoOn has it -- whatever entries the
// function's code has. Asked only of code with a loop entry
// (jitRegLeaveTo), it had every call finish in the interpreter.
func TestJITRegCarriedExitsGoOn(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			rts := regNativeRuntimes(t, regCarriedExitSrc)
			rts[0].jitStress = jitStressConfig{}
			for range 2 {
				regCarriedCall(t, rts, "600000", "mixed", 6000)
			}
			r := rts[0]
			e := regCodeNamed(t, r, "early")
			if e == nil || e.code == nil {
				t.Fatalf("early has no register code: %+v", e)
			}
			t.Logf("early: loops %v unentered %v osr %v went on %d", e.loops, e.unentered, e.osr != nil, r.jit.wentOn)
			if len(e.unentered) == 0 {
				t.Fatalf("the function's code does not keep the loop's objects unmade: loops %v", e.loops)
			}
			o := e.osr
			if o == nil || o.code == nil {
				t.Fatalf("no code kept for the loop, after %d frames left native code", e.life.goOns)
			}
			if r.jit.wentOn < 2000 {
				t.Errorf("%d frames went on natively at the loop, of %d that left", r.jit.wentOn, e.life.goOns)
			}
		})
	}
}

// TestJITRegCarriedFramesLeave has three calls in four -- every call --
// leave a loop whose object the function's code keeps unmade: past
// jitFramesLeaving frames gone on in the code kept for the loop, the
// function's code is compiled with the loop's entry, its objects made
// (jitRegFramesLeave), and the code kept goes, as the stack form's
// jitFramesLeave has it: each call had paid the exit's objects and a run
// of the code kept, for the few turns before it left.
func TestJITRegCarriedFramesLeave(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			rts := regNativeRuntimes(t, regCarriedExitSrc)
			rts[0].jitStress = jitStressConfig{}
			for range 2 {
				regCarriedCall(t, rts, "300000", "calls", 3000, 100, 5)
			}
			r := rts[0]
			e := regCodeNamed(t, r, "early")
			if e == nil || e.code == nil {
				t.Fatalf("early has no register code: %+v", e)
			}
			t.Logf("early: loops %v unentered %v osr %v went on %d resumed %v", e.loops, e.unentered, e.osr != nil, r.jit.wentOn, e.life.resumed)
			if len(e.unentered) != 0 || e.osr != nil || len(e.loops) == 0 {
				t.Fatalf("every call left and the code goes on leaving its objects unmade: loops %v, unentered %v, osr %v", e.loops, e.unentered, e.osr != nil)
			}
		})
	}
}
