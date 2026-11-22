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
