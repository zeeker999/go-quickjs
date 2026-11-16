//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"fmt"
	"strings"
	"testing"
)

// regClassConstructSrc constructs classes' objects, of base classes with no
// field to initialize, from register code's native code: natively, where Go
// had constructed each; and a construction that throws, and a class's
// constructor called, which throws.
const regClassConstructSrc = `
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

// TestJITRegClassConstructions runs regClassConstructSrc's functions
// natively and in a runtime with no JIT and compares them: once compiled,
// classes' constructions are native, none Go's.
func TestJITRegClassConstructions(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			rts := regNativeRuntimes(t, regClassConstructSrc)
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

// regInlineExitSrc's callees throw on a path seldom taken, as AWFY's
// Towers' pushDisk does: inlined all the same, the throw an exit of the
// inlined frame (jitRegFeedback's InlineExit), which leaves it to Go once
// taken.
const regInlineExitSrc = `
function check(d) { if (d < 0) throw new RangeError("bad " + d); return d * 2 }
function guard(o) { if (o === null) throw o; return o.v + 1 }
function towers(n) {
	var s = 0, o = { v: 3 }, m = "";
	for (var i = 0; i < n; i++) {
		try { s += check(i % 1000 == 999 ? -i : i) + guard(o) } catch (e) { m = e.message }
	}
	return s + m;
}
`

// TestJITRegInlineExits runs regInlineExitSrc's towers natively and in a
// runtime with no JIT and compares them: its callees are inlined, their
// throws taken now and then.
func TestJITRegInlineExits(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			rts := regNativeRuntimes(t, regInlineExitSrc)
			r := rts[0]
			r.SetJITWarmUp(1, 1)
			for range 6 {
				var got [2]string
				for i, r := range rts {
					fn, _ := r.GetProp(Obj(r.Global()), r.Intern("towers"))
					v, err := r.Call(fn, Undefined, []Value{Float(5000)})
					got[i] = regNativeShow(r, v, err)
				}
				if got[0] != got[1] {
					t.Fatalf("towers: with the JIT %q, without %q", got[0], got[1])
				}
			}
			if st := r.JITStats(); st.Inlined == 0 {
				t.Errorf("no callee inlined: %+v", st)
			}
		})
	}
}
