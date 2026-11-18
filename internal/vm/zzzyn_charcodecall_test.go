//go:build amd64 || arm64

package vm

import (
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
)

// Saved-method calls use the same native code-unit kernel as the fused
// local-index form, both with the collector idle and while it marks.
func TestRegNativeCharCodeCall(t *testing.T) {
	nativeCounting(t)
	for _, marking := range []bool{false, true} {
		t.Run(map[bool]string{false: "idle", true: "marking"}[marking], func(t *testing.T) {
			nativeMarking(t, marking)
			before := nativeExitOps()
			got := nativeRunSrc(t, nativeKernelSource(t, "string_charcode_update", "string_charcode_add")+`
                string_charcode_update(1000);
                var a = global_res;
                string_charcode_add(1000);
                a + ":" + global_res`, true)
			if got != "104468:104468" {
				t.Fatalf("got %s", got)
			}
			if n := nativeExitOps()[bytecode.RCharCodeCall] - before[bytecode.RCharCodeCall]; n != 0 {
				t.Fatalf("char_code_call left native code %d times", n)
			}
			before = nativeExitOps()
			got = nativeRunSrc(t, nativeKernelSource(t, "extra_charcode_custom", "string_charcode_custom")+`
                string_charcode_custom(1000); ''+global_res`, true)
			if got != "8468" {
				t.Fatalf("custom method: got %s", got)
			}
			after := nativeExitOps()
			// The first call may materialize the script function's data in
			// Go; the following 999 use the native call-cache continuation.
			if n := after[bytecode.RCharCodeCall] - before[bytecode.RCharCodeCall]; n > 1 {
				t.Fatalf("script override left the specialized handler %d times", n)
			}
			if n := after[bytecode.RCallMW] - before[bytecode.RCallMW]; n > 2 {
				t.Fatalf("script override missed its call cache %d times", n)
			}
			before = nativeExitOps()
			got = nativeRunSrc(t, `
                var o={n:7,charCodeAt:function(x){let h=0;for(let j=0;j<3;j++)h+=j;return this.n+x+h}};
                function customFrame(n){let h=0,i=0;for(let k=0;k<n;k++)h+=o.charCodeAt(++i);return h}
                ''+customFrame(1000)`, true)
			if got != "510500" {
				t.Fatalf("script method frame: got %s", got)
			}
			if n := nativeExitOps()[bytecode.RCharCodeCall] - before[bytecode.RCharCodeCall]; n > 2 {
				t.Fatalf("script method frame left native code %d times", n)
			}
		})
	}
}
