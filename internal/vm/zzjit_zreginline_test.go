//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"fmt"
	"testing"
)

// regInlineSrc's callers call small register functions that register code
// inlines (jitRegFeedback's Inline): methods, plain calls of strict and
// sloppy functions reading this, a closure's captured binding, missing
// and extra arguments; and callees that leave the inlined code -- a shape
// the callee was not compiled for, a receiver that is no object, a throw
// the caller catches, a call it makes that Go runs -- whose frames Go makes
// from the exit's, finishing them in the interpreter.
const regInlineSrc = `
function P(x) { this.x = x }
P.prototype.get = function () { return this.x };
P.prototype.add = function (k) { return this.x + k };
function sq(a) { return a * a }
function pair(a, b) { return b === undefined ? a : a - b }
function sloppy() { return this === globalThis ? 1 : 0 }
function strict() { "use strict"; return this === undefined ? 2 : 0 }
var withK = (function () { var k = 5; return function (a) { return a + k } })();
function inlines(n) {
	var p = new P(3), s = 0;
	for (var i = 0; i < n; i++) s += p.get() + p.add(i) + sq(i) + pair(i) + pair(i, 1, 2) + sloppy() + strict() + withK(i);
	return s;
}
function rd(o) { return o.x }
function shapes(n) {
	var a = { x: 1 }, b = { y: 0, x: 2.5 }, s = 0;
	for (var i = 0; i < n; i++) s += rd(i % 97 == 96 ? b : a);
	return s;
}
function throws(n) {
	var a = { x: 1 }, s = 0, m = "";
	for (var i = 0; i < n; i++) { try { s += rd(i % 50 == 49 ? undefined : a) } catch (e) { m = e.constructor.name } }
	return s + m;
}
function str(o) { return o.x + "!" }
function callsGo(n) { var s = 0; for (var i = 0; i < n; i++) s += str(i % 3 ? { x: i } : { x: "k" + i }).length; return s }
function mid(o) { return rd(o) + 1 }
function nested(n) { var a = { x: 3 }, b = { z: 1, x: 4 }, s = 0; for (var i = 0; i < n; i++) s += mid(i % 61 == 60 ? b : a); return s }
`

// TestJITRegInline runs regInlineSrc's callers in a runtime whose register
// code inlines its callees and in one with no JIT, and compares their
// answers, the collector idle and marking; the callers inline what they
// call, and their code compiles for arm64 too.
func TestJITRegInline(t *testing.T) {
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
			rts := regNativeRuntimes(t, regInlineSrc)
			rts[0].SetJITWarmUp(1, 1)
			for _, c := range []struct {
				name string
				n    float64
			}{{"inlines", 2000}, {"shapes", 2000}, {"throws", 1000}, {"callsGo", 600}, {"nested", 2000}} {
				var got [2]string
				for i, r := range rts {
					fn, _ := r.GetProp(Obj(r.Global()), r.Intern(c.name))
					for range 8 {
						v, err := r.Call(fn, Undefined, []Value{Float(c.n)})
						got[i] = regNativeShow(r, v, err)
					}
				}
				if got[0] != got[1] {
					t.Errorf("%s: with the JIT %q, without %q", c.name, got[0], got[1])
				}
			}
			r := rts[0]
			st := r.RegNativeStats()
			t.Logf("inlined %d; unwound %d, guards %d, hosts %d, entries %d", r.jit.inlined, st.Unwound, st.Guards, st.Hosts, st.Entries)
			if r.jit.inlined == 0 {
				t.Error("no callee inlined")
			}
			// Left at the exits of the callees inlined, learned from -- the
			// code compiled again -- or the call made natively after
			// jitInlineExits: not at every call that misses (about 1,600
			// in shapes, 10,000 in nested, with the code's runs going on in
			// code compiled stale).
			if st.Unwound > 200 {
				t.Errorf("unwound %d: inlined callees left too often", st.Unwound)
			}
			// The same code, arm64's backend's, compiled on any host.
			if fns, _, inlined := regCompileARM64(t, r); inlined == 0 {
				t.Errorf("arm64: %d functions compiled, none inlining a callee", fns)
			}
		})
	}
}
