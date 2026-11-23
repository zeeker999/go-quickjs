//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"fmt"
	"testing"
)

func TestJITPolymorphicOwnSlot(t *testing.T) {
	for _, marking := range []bool{false, true} {
		for _, body := range []string{"s=(s+o.x)|0", "s=o.x"} {
			for _, miss := range []string{
				`({p:0,y:99})`,
				`({})`,
				`Object.create({x:13})`,
				`({p:0,get x(){return 17}})`,
				`({p:0,get x(){throw new RangeError("slot")}})`,
				`new Proxy({p:0,x:19},{get(t,k){return k==="x"?23:t[k]}})`,
				`(()=>{delete c.x;return c})()`,
			} {
				t.Run(fmt.Sprintf("marking=%v/%s/%s", marking, body, miss), func(t *testing.T) {
					if marking {
						jitMarkingForTest(t)
					}
					r, run := jitTierRuntimeForTest(t, 8, `
var a={p:0,x:3}, b={p:0,x:5,b:0}, c={p:0,x:7,c:0};
function f(o,n){let s=0;for(let i=0;i<n;i++){`+body+`};return s}
`)
					for range 16 {
						run(`f(a,1)`)
						run(`f(b,1)`)
					}
					e := jitNativeForTest(t, r, "f")
					before := e.deopts
					if body == "s=o.x" {
						run(`(c.x={toString(){return "slot"}},f(c,2))`)
					}
					for _, src := range []string{
						`f(c,2)`,
						`(Object.setPrototypeOf(c,{z:1}),f(c,2))`,
						`(Object.defineProperty(c,"x",{value:11,writable:false}),f(c,2))`,
						`(()=>{for(let i=0;i<30;i++)c["q"+i]=i;return f(c,2)})()`,
					} {
						run(src)
						if e.invalid || e.deopts != before {
							t.Fatalf("%s deoptimized a read whose own slot still matches", src)
						}
					}
					run(`(()=>{try{return f(` + miss + `,2)}catch(e){return e.name+":"+e.message}})()`)
					if e.deopts != before+1 {
						t.Fatal("a mismatching slot did not deoptimize")
					}
				})
			}
		}
	}
}
