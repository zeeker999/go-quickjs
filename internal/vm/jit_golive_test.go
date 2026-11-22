//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"fmt"
	"testing"
)

// goLiveSetup's loop converts strings to int32s (s | 0), which the code
// leaves to Go (ssa.GoNumber) out of line; the loop calls nothing, so it
// counts its budget in a register. oddInts' third element is an object,
// which Go refuses: the conversion leaves for the interpreter, which calls
// its valueOf. callInts' loop calls, so its budget counts in memory.
const goLiveSetup = `
var conv = 0;
function toInts(a, n) { var r = 0; for (var j = 0; j < n; j++) r = (r + (a[j & 3] | 0)) | 0; return r }
function plainInts(n) { return toInts(['12345', '7', ' 42 ', '0x10'], n) }
function oddInts(n) { conv = 0; return toInts(['12345', '7', { valueOf() { conv++; return 5 } }, '3'], n) + ':' + conv }
function noop() { return 0 }
function callInts(a, n) { var r = 0; for (var j = 0; j < n; j++) r = (r + (a[j & 3] | 0) + noop()) | 0; return r }
function oddCallInts(n) { conv = 0; return callInts(['12345', '7', { valueOf() { conv++; return 5 } }, '3'], n) + ':' + conv }
`

// A string's conversion that the code leaves to Go (ssa.GoNumber) goes
// through the VM's routine for calls of Go (GoFast), saving only what
// lives across it (goLive), as a call of Go does, rather than through the
// routine that saves and restores every register and goes by callGo and
// jit.HostCall: string_to_int had taken 1.27 times the interpreter's
// time. Each answer is the interpreter's, a refusal's too, with the
// collector idle and marking, and each turn's conversion is a call
// through the routine.
func TestJITStringConversionGoesLive(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			var rounds []string
			for range 6 {
				rounds = append(rounds, "plainInts(40)")
			}
			r := jitAgreesForTest(t, goLiveSetup, rounds...)
			fast := r.jit.goFast
			for _, rd := range [][2]string{
				{"plainInts(40)", "124100"},
				{"oddInts(40)", "123600:10"},
				{"plainInts(40)", "124100"},
				{"oddCallInts(40)", "123600:10"},
			} {
				v, err := r.Run(compileForTest(t, "''+("+rd[0]+")"))
				if err != nil {
					t.Fatalf("%s: %v", rd[0], err)
				}
				if got := v.String().Go(); got != rd[1] {
					t.Errorf("%s = %q, want %q", rd[0], got, rd[1])
				}
			}
			jitNativeForTest(t, r, "toInts")
			if got := r.jit.goFast - fast; got < 2*40 {
				t.Errorf("%d calls of Go through the routine in 120 turns, want at least one a turn of plainInts'", got)
			}
		})
	}
}

// goLiveEqSetup's loops compare strings Go compares (ssa.GoEqual): a rope
// made at each turn against a flat string, which native code leaves to Go
// at every other turn (the first and third elements are long enough to
// make ropes, ropeThreshold); the comparison its branch's (eqLoop), its value kept
// (keptLoop), and against a constant (constLoop). looseLoop compares
// loosely an object, whose valueOf runs: Go refuses it, and the
// comparison leaves for the interpreter.
const goLiveEqSetup = `
var eqConv = 0, tail = 'q';
var lefts = ['abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz0123456789', 'c', 'zyxwvutsrqponmlkjihgfedcba9876543210zyxwvutsrqponmlkjihgfedcba9876543210', 'g'];
var rights = ['abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz0123456789q', 'cq', 'zyxwvutsrqponmlkjihgfedcba9876543210zyxwvutsrqponmlkjihgfedcba9876543210x', 'gq'];
function eqLoop(a, b, n) { var c = 0; for (var j = 0; j < n; j++) if (a[j & 3] + tail === b[j & 3]) c++; return c }
function keptLoop(a, b, n) { var c = 0, d = 0; for (var j = 0; j < n; j++) { var e = a[j & 3] + tail === b[j & 3]; c = (c + (e ? 3 : 1)) | 0; if (e) d++ } return c + ':' + d }
function constLoop(a, n) { var c = 0; for (var j = 0; j < n; j++) if (a[j & 3] + tail === 'abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz0123456789q') c += j; return c }
function looseLoop(a, b, n) { var c = 0; for (var j = 0; j < n; j++) if (a[j & 3] == b[j & 3]) c++; return c }
function plainEq(n) { return eqLoop(lefts, rights, n) + ',' + keptLoop(lefts, rights, n) + ',' + constLoop(lefts, n) }
function oddEq(n) { eqConv = 0; return looseLoop(['abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz0123456789q', 'cd', 'x', 'gh'], ['abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz0123456789q', { valueOf() { eqConv++; return 'cd' } }, 'x', 'gh'], n) + ':' + eqConv }
`

// A comparison that leaves strings to Go (ssa.GoEqual) calls Go through
// GoFast, saving only what lives across it, as a string's conversion does
// (TestJITStringConversionGoesLive): each answer the interpreter's, a
// comparison Go refuses too, with the collector idle and marking; and
// every call of Go the loops make goes through the routine -- each turn's
// concatenation, and each comparison of a rope, which had been a call
// through the routine that saves every register.
func TestJITStringEqualityGoesLive(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			var rounds []string
			for range 6 {
				rounds = append(rounds, "plainEq(40)", "oddEq(40)")
			}
			r := jitAgreesForTest(t, goLiveEqSetup, rounds...)
			for _, name := range []string{"eqLoop", "keptLoop", "constLoop"} {
				jitNativeForTest(t, r, name)
			}
			// Each call of Go, by either route, comes to jitHostCall
			// while the hook is set, and one through the routine counts
			// in goFast first.
			var calls uint64
			jitTestSwitch(t, func(jc *jitTestConfig) { jc.goCallWork = func() { calls++ } })
			fast := r.jit.goFast
			for _, rd := range [][2]string{
				{"plainEq(40)", "30,100:30,180"},
				{"oddEq(40)", "40:10"},
				{"plainEq(40)", "30,100:30,180"},
			} {
				v, err := r.Run(compileForTest(t, "''+("+rd[0]+")"))
				if err != nil {
					t.Fatalf("%s: %v", rd[0], err)
				}
				if got := v.String().Go(); got != rd[1] {
					t.Errorf("%s = %q, want %q", rd[0], got, rd[1])
				}
			}
			jitTestSwitch(t, func(jc *jitTestConfig) { jc.goCallWork = nil })
			routed := r.jit.goFast - fast
			t.Logf("routed %d, calls %d", routed, calls)
			if routed < 2*3*40 {
				t.Errorf("%d calls of Go through the routine, want at least one a turn of each of plainEq's loops", routed)
			}
			// The two plainEq rounds compare 2*3*20 ropes; a poll or the
			// refused comparison's exit may call Go otherwise.
			if other := calls - routed; other >= 3*20 {
				t.Errorf("%d of %d calls of Go not through the routine, want the ropes' comparisons through it", other, calls)
			}
		})
	}
}
