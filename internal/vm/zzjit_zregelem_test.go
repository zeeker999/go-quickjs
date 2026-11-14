//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"fmt"
	"testing"
)

// regElemSrc reads and writes elements native code learns the kinds of at
// its exits (zzjit_zregelem.go): a typed array's, a BigInt64Array's, an
// object's property by its name, a string's character, an object's
// property written by its name, and additions of an array's strings.
const regElemSrc = `
var mem = new Uint8Array(256);
for (var i = 0; i < 256; i++) mem[i] = i;
var big = new BigInt64Array(16);
for (var i = 0; i < 16; i++) big[i] = BigInt(i * 3);
var dict = { add: 1, sub: 2, mul: 3, div: 4 };
var names = ["add", "sub", "mul", "div"];
function readTyped(a, n) { var s = 0; for (var i = 0; i < n; i++) { var v = a[i & 255]; s += v === 7 ? 1 : 0 } return s }
function readBig(a, n) { var s = 0n; for (var i = 0; i < n; i++) s += a[i & 15]; return s }
function keyed(n) { var s = 0; for (var i = 0; i < n; i++) { var k = names[i & 3]; var v = dict[k]; s += v === 3 ? 1 : 0 } return s }
function chars(str, n) { var s = 0; for (var i = 0; i < n; i++) { var c = str[i % str.length]; if (c === "a") s++ } return s }
function writes(n) { var o = {}; for (var i = 0; i < n; i++) o[names[i & 3]] = i; return o.add + o.div }
function adds(n) { var t = "", parts = ["x", "y", "z"]; for (var i = 0; i < n; i++) { t = t + parts[i % 3]; if (t.length > 50) t = "" } return t }
function counts(n) { var c = {}; for (var i = 0; i < n; i++) { var k = names[i & 3]; c[k] = (c[k] | 0) + 1 } return c.add + c.mul }
function main(n) {
	return readTyped(mem, n) + "," + readBig(big, n) + "," + keyed(n) + "," + chars("banana", n) + "," + writes(n) + "," + adds(n) + "," + counts(n);
}
`

// TestJITRegElementKinds runs regElemSrc's main natively and in a runtime
// without the JIT and compares their answers: each element access, which
// had left native code at every turn -- the read of a typed array's, a
// string's or an object's element by its cell, the write of an object's
// property by its name -- is made natively, or by Go from native code,
// once its code is compiled again for what it met (jitRegElemSeen).
func TestJITRegElementKinds(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			rts := regNativeRuntimes(t, regElemSrc)
			var got [2]string
			for i, r := range rts {
				fn, _ := r.GetProp(Obj(r.Global()), r.Intern("main"))
				for range 20 {
					v, err := r.Call(fn, Undefined, []Value{Float(3000)})
					got[i] = regNativeShow(r, v, err)
				}
			}
			if want := "12,67404,750,1500,5995,xyzxyzxyzxyzxyzxyzxyzxyzxyzxyzxyzxyzxyzxyz,1500"; got[0] != got[1] || got[1] != want {
				t.Errorf("with the JIT %q, without %q, want %q", got[0], got[1], want)
			}
			st := rts[0].RegNativeStats()
			t.Logf("hosts %v, entries %d", st.HostOps, st.Entries)
			// Of 20 runs of 21,000 accesses each: left to Go only before the
			// code learned them.
			for _, op := range []string{"get_index", "set_index", "add_index"} {
				if n := st.HostOps[op]; n > 50 {
					t.Errorf("%s left to Go at %d host exits", op, n)
				}
			}
		})
	}
}
