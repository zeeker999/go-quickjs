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
			before = nativeExitOps()
			got = nativeRunSrc(t, `
                function make(n){return function(x){let h=0;for(let j=0;j<3;j++)h+=j;return n+x+h}}
                function templates(){let h=0,s={};for(let k=0;k<1000;k++){s.charCodeAt=make(k);h+=s.charCodeAt(k+0)}return ''+h}
                templates()`, true)
			if got != "1002000" {
				t.Fatalf("script method templates: got %s", got)
			}
			if n := nativeExitOps()[bytecode.RCharCodeCall] - before[bytecode.RCharCodeCall]; n > 2 {
				t.Fatalf("script method templates left native code %d times", n)
			}
		})
	}
}

func TestRegNativeCharCodeCallRecoversIntrinsic(t *testing.T) {
	nativeCounting(t)
	was := regNativeOn
	regNativeOn = true
	t.Cleanup(func() { regNativeOn = was })
	for _, marking := range []bool{false, true} {
		t.Run(map[bool]string{false: "idle", true: "marking"}[marking], func(t *testing.T) {
			nativeMarking(t, marking)
			r := New(Config{})
			defer r.Close()
			r.global.setOwnRaw(r.atoms.intern("charExitCount"), Obj(r.newNativeFunc("charExitCount", 0,
				func(_ *Runtime, _ Value, _ []Value) (Value, error) {
					return Int(int(nativeExitOps()[bytecode.RCharCodeCall])), nil
				})), propWritable|propConfigurable)
			v, err := r.Run(compileForTest(t, `
                var arg={valueOf(){return 0}},saved=String.prototype.charCodeAt;
                function nextarg(){return arg}
                function read(){let v="abc".charCodeAt(nextarg());return v}
                function many(){let h=0;for(let i=0;i<1000;i++)h+=read();return h}
                var out=[],before,h;
                read();arg=0;before=charExitCount();h=many();out.push(h+":"+(charExitCount()-before));
                String.prototype.charCodeAt=function(i){return i+7};read();read();
                before=charExitCount();h=many();out.push(h+":"+(charExitCount()-before));
                String.prototype.charCodeAt=saved;
                before=charExitCount();h=many();out.push(h+":"+(charExitCount()-before));
                out.join("|")`))
			if err != nil || v.String().Go() != "97000:0|7000:0|97000:0" {
				t.Fatalf("got %v, %v; want 97000:0|7000:0|97000:0", v, err)
			}
		})
	}
}
