//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"fmt"
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

// A read warmed on two layouts that then deoptimizes retains its next
// native code when another layout arrives. The fallback also checks the
// current prototype chain.
func TestJITPolymorphicReadMissesKeepCode(t *testing.T) {
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			r, run := jitTierRuntimeForTest(t, 8, jitTierSetup)
			for range 16 {
				run(`f(a,1)`)
				run(`f(b,1)`)
			}
			first := jitNativeForTest(t, r, "f")
			before := first.deopts
			run(`f(c,1)`)
			if first.deopts != before+1 {
				t.Fatal("the reader did not deoptimize for its first new layout")
			}
			for range 32 {
				run(`f(c,1)`)
			}
			e := jitNativeForTest(t, r, "f")
			if site := regLearnSite(t, r, "f", ir.PropertyRead); site.Megamorphic || !site.Fallback {
				t.Fatalf("the polymorphic reader lost its separate fallback policy: %+v", site)
			}
			cl := jitGlobalClosureForTest(r, "f")
			code := r.jitRegFor(cl, cl.reg)
			fb := r.jitRegFeedbackFor(cl, code.p)
			snap := jitRegSnapshotOf(fb, code.p)
			for pc, in := range code.p.Code {
				if in.Op == ir.PropertyRead {
					if site, ok := snap.Property(pc); !ok || site.Megamorphic || !site.Fallback {
						t.Fatalf("the background snapshot lost the fallback policy: %+v, %v", site, ok)
					}
				}
			}
			deopts := e.deopts
			for _, src := range []string{
				`[f(c,20),f(d,20)].join()`,
				`(C.prototype.x=7,f(c,20))`,
				`(c.x=11,f(c,20))`,
				`(delete c.x,f(c,20))`,
				`(Object.setPrototypeOf(c,D.prototype),f(c,20))`,
				`(Object.defineProperty(D.prototype,"x",{get(){return 13},configurable:true}),f(c,20))`,
				`(Object.defineProperty(D.prototype,"x",{get(){throw new RangeError("read")}}),(()=>{try{f(c,20)}catch(e){return e.name+":"+e.message}})())`,
			} {
				run(src)
				if e.invalid || e.deopts != deopts {
					t.Fatalf("%s invalidated the polymorphic reader: invalid %v, deopts %d -> %d", src, e.invalid, deopts, e.deopts)
				}
			}
		})
	}
}

// A numeric guard elsewhere in the function must not widen stable reads.
func TestJITPolymorphicReadNeedsItsOwnMiss(t *testing.T) {
	r, run := jitTierRuntimeForTest(t, 8, jitTierSetup+`
function stable(o, x) { for (let i=0;i<2;i++) x+=1; return x+o.x }
`)
	for range 32 {
		run(`stable(a,1)`)
		run(`stable(b,1)`)
	}
	e := jitNativeForTest(t, r, "stable")
	before := e.deopts
	run(`stable(a,"s")`)
	if e.deopts != before+1 {
		t.Fatal("the numeric guard did not deoptimize")
	}
	for range 32 {
		run(`stable(a,"s")`)
		run(`stable(b,"s")`)
	}
	jitNativeForTest(t, r, "stable")
	if site := regLearnSite(t, r, "stable", ir.ReferenceRead); site.Megamorphic || site.Fallback || len(site.Cases) == 0 {
		t.Fatalf("another guard's deoptimization widened the polymorphic read: %+v", site)
	}
}
