//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"fmt"
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
)

func TestJITColdReadKeepsCode(t *testing.T) {
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			r, run := jitTierRuntimeForTest(t, 8, `
				var holder = {x: "first"}, o = Object.create(holder);
				function read(o, use) {
					let value;
					for (let i = 0; i < 4; i++) {
						if (use) value = o.x;
						else value = i;
					}
					return value;
				}`)
			for range 32 {
				run(`read(o, false)`)
			}
			e := jitNativeForTest(t, r, "read")
			for _, src := range []string{
				`read(o, true)`,
				`(holder.x = 7, read(o, true))`,
				`(Object.setPrototypeOf(o, {x: null}), read(o, true))`,
				`read({x: true, y: 0}, true)`,
				`(Object.setPrototypeOf(o, {}), read(o, true))`,
				`(Object.setPrototypeOf(o, {get x() { return "getter" }}), read(o, true))`,
				`(() => { try { read({get x() { throw new RangeError("cold") }}, true) }
					catch (e) { return e.name + ":" + e.message } })()`,
			} {
				run(src)
				if e.deopts != 0 || e.invalid {
					t.Fatalf("%s: deopts=%d invalid=%v", src, e.deopts, e.invalid)
				}
			}
		})
	}
}

// The lookup excludes special classes, which need their newly learned shape.
func TestJITColdReadLearnsSpecialClass(t *testing.T) {
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			r, run := jitTierRuntimeForTest(t, 8, `
				var error = new Error("cold");
				function readError(e, use) {
					let value;
					for (let i = 0; i < 4; i++) {
						if (use) value = e.message;
						else value = i;
					}
					return value;
				}`)
			for range 32 {
				run(`readError(error, false)`)
			}
			e := jitNativeForTest(t, r, "readError")
			// Feedback may arrive while a background build still sees an
			// empty site. Fill it after compilation, before the cold path.
			cl := jitGlobalClosureForTest(r, "readError")
			errorObject := r.global.getOwn(r.atoms.intern("error")).value().Object()
			filled := false
			for _, w := range cl.reg.rc.Code {
				if w.Op() == bytecode.RGetProp {
					c := &cl.ic[int(w.C())]
					r.fillPropCache(c, errorObject, r.atoms.intern("message"), false)
					filled = c.fills != 0
				}
			}
			if !filled {
				t.Fatal("did not fill the cold site's cache")
			}
			for range 32 {
				run(`readError(error, true)`)
			}
			hosts := e.ssaStats.hosts
			for range 16 {
				run(`readError(error, true)`)
			}
			if e.deopts != 0 || e.invalid || e.ssaStats.hosts != hosts {
				t.Fatalf("deopts=%d invalid=%v; host exits %d -> %d", e.deopts, e.invalid, hosts, e.ssaStats.hosts)
			}
		})
	}
}
