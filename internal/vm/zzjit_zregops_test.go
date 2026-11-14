//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import "testing"

// regPlainSetSrc's w is called natively, not inlined: its write of s calls
// a setter, a call of Go its native code makes at register pc 5, where w's
// stack form has o.q = t. Made without a frame by the stack form's
// instruction there (jitGoCallPlain), the write had stored q and never
// called the setter.
const regPlainSetSrc = `
function w(o, v) { var t = v + 1; o.p = 1; o.q = t; o.r = t; o.s = t; o.t = v; return t }
var log = 0;
function mk(k) { var o = { set s(x) { log += x } }; o["k" + k] = k; return o }
var objs = []; for (var i = 0; i < 3; i++) objs.push(mk(i));
function loop(n) { var s = 0; for (var i = 0; i < n; i++) s += w(objs[i % 3], i); return s }
function main() { log = 0; loop(3000); return JSON.stringify(objs) + log }
`

// TestJITRegPlainSetterCall checks a register native callee's setter call,
// a call of Go at a register pc, writes the property its register
// instruction names (jitRegPlainSet), not the stack form's at that pc.
func TestJITRegPlainSetterCall(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	regNoInline(t)
	rts := regNativeRuntimes(t, regPlainSetSrc)
	const want = `[{"k0":0,"p":1,"q":2998,"r":2998,"t":2997},{"k1":1,"p":1,"q":2999,"r":2999,"t":2998},{"k2":2,"p":1,"q":3000,"r":3000,"t":2999}]4501500`
	for i, r := range rts {
		fn, _ := r.GetProp(Obj(r.Global()), r.Intern("main"))
		for k := range 10 {
			v, err := r.Call(fn, Undefined, nil)
			if got := regNativeShow(r, v, err); got != want {
				t.Fatalf("runtime %d, run %d: %q, want %q", i, k, got, want)
			}
		}
	}
	if st := rts[0].RegNativeStats(); st.GoCalls == 0 {
		t.Error("the setter never called by Go from native code")
	}
}
