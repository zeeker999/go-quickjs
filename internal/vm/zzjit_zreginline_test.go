//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"fmt"
	"testing"
)

// regAccessorSrc reads and writes accessors register code does not inline:
// a class's getter and setter (inlining off), a prototype's getter, a
// bound function's getter, a built-in's (Map's size); one throws now and
// then.
const regAccessorSrc = `
class Pt { constructor(x) { this._x = x } get x() { return this._x } set x(v) { if (v === -1) throw new RangeError("neg"); this._x = v } }
var proto = { get g() { return this.k * 2 } };
var holder = { k: 5 };
var bound = Object.create(Object.defineProperty({}, "b", { get: function () { return this.k + 1 }.bind(holder) }));
function accessorGoCalls(n) {
	var p = new Pt(1), q = Object.create(proto), m = new Map([[1, 2]]), s = 0, e = "";
	q.k = 3;
	for (var i = 0; i < n; i++) {
		p.x = i;
		s += p.x + q.g + bound.b + m.size;
		try { p.x = i % 50 == 49 ? -1 : i } catch (err) { e = err.message }
	}
	return s + e;
}
`

// TestJITRegAccessorGoCalls runs regAccessorSrc's accessorGoCalls in a
// runtime whose register code calls Go for the accessors it does not
// inline (jitRegGoCallOn's GoCallGet, GoCallSet), and in one with no JIT,
// the collector idle and marking: the answers equal, and its reads and
// writes left to Go at exits only where a setter threw.
func TestJITRegAccessorGoCalls(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			regNoInline(t)
			rts := regNativeRuntimes(t, regAccessorSrc)
			rts[0].SetJITWarmUp(1, 1)
			var got [2]string
			for i, r := range rts {
				fn, _ := r.GetProp(Obj(r.Global()), r.Intern("accessorGoCalls"))
				for range 8 {
					v, err := r.Call(fn, Undefined, []Value{Float(1000)})
					got[i] = regNativeShow(r, v, err)
				}
			}
			if got[0] != got[1] {
				t.Errorf("with the JIT %q, without %q", got[0], got[1])
			}
			st := rts[0].RegNativeStats()
			t.Logf("calls of Go %d, hosts %v", st.GoCalls, st.HostOps)
			// 8 runs of 1,000 turns, a setter's throw every 50: 160 host
			// exits at set_prop at most; the reads, none but a first's.
			if n := st.HostOps["get_prop"]; n > 8 {
				t.Errorf("get_prop left to Go %d times", n)
			}
			if n := st.HostOps["set_prop"]; n > 200 {
				t.Errorf("set_prop left to Go %d times", n)
			}
		})
	}
}
