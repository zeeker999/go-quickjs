//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"fmt"
	"testing"
)

// An element store takes a property's value without numeric speculation.
func TestJITPropertyStoredAsValue(t *testing.T) {
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			r, run := jitTierRuntimeForTest(t, 8, `
				var o = { x: "5" }, a = [0, 0, 0, 0];
				function store(o, a) {
					for (let i = 0; i < 4; i++) a[i] = o.x;
					return a[0] === o.x && a[3] === o.x;
				}`)
			for range 32 {
				run(`store(o, a)`)
			}
			e := jitNativeForTest(t, r, "store")
			if e.deopts != 0 {
				t.Fatalf("value store deoptimized %d times", e.deopts)
			}
			for _, value := range []string{`"7"`, `null`, `undefined`, `true`, `42`, `{valueOf(){throw new RangeError("coerce")}}`} {
				run(`(o.x = ` + value + `, store(o, a))`)
				if e.deopts != 0 {
					t.Fatalf("stored %s deoptimized %d times", value, e.deopts)
				}
			}
		})
	}
}
