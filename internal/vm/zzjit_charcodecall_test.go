//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import "testing"

func TestJITCharCodeCallLegacyArguments(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		t.Run(map[bool]string{false: "idle", true: "marking"}[marking], func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			r := jitRuntimeForTest(t, Config{JIT: true, NodeQuirks: true})
			if _, err := r.Run(compileForTest(t, charCodeCallLegacySource)); err != nil {
				t.Fatal(err)
			}
			// Compile and enter the intrinsic before replacing the method;
			// its lookup guards may invalidate before the saved-call guard.
			if _, err := r.Run(compileForTest(t, `var custom=o.charCodeAt;o="abc";
                for(let k=0;k<10;k++)legacy(0);String.prototype.charCodeAt=custom`)); err != nil {
				t.Fatal(err)
			}
			p := compileForTest(t, `legacy(42)`)
			for range 10 {
				v, err := r.Run(p)
				if err != nil || v.String().Go() != "1:42" {
					t.Fatalf("got %v, %v; want 1:42", v, err)
				}
			}
			jitNativeForTest(t, r, "legacy")
		})
	}
}

func TestJITCharCodeCall(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		t.Run(map[bool]string{false: "idle", true: "marking"}[marking], func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			r := jitAgreesForTest(t, `
                var saved = String.prototype.charCodeAt, reads = 0;
                function codes(s,n) {
                    let h=0;
                    for(let j=0;j<n;j++) {let i=j&3;h+=s.charCodeAt(++i);h+=s.charCodeAt(i+1)}
                    return h;
                }
                function once(s,i) {let a=s.charCodeAt(++i);return a}
            `,
				`codes("abcdef",1000)`, `codes("abcdef",1000)`, `codes("abcdef",1000)`,
				`(String.prototype.charCodeAt=function(i){return 900+i},codes("abcdef",1000))`,
				`(String.prototype.charCodeAt=saved,codes("abcdef",1000))`,
				`(Object.defineProperty(String.prototype,"charCodeAt",{configurable:true,get(){reads++;return saved}}), codes("abcdef",100)+":"+reads)`,
				`(Object.defineProperty(String.prototype,"charCodeAt",{configurable:true,writable:true,value:saved}), once("abcdef",{valueOf(){String.prototype.charCodeAt=function(){return 7};return 0}}))`,
			)
			jitNativeForTest(t, r, "codes")
		})
	}
}
