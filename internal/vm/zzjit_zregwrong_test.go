//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"fmt"
	"testing"
)

// Wrong answers the register JIT gave where the vm package's JIT tests,
// which had run on the stack form's JIT alone, first ran on it. Each
// case is checked by regWrongCheck: natively from the start and by
// on-stack replacement, with the collector idle and marking, against its
// exact answer and an ordinary call's.

// regWrongCase is a call of the global function name with args, made
// warm times in each runtime first, so that the caches its code is
// compiled from have met what the call meets, and its exact answer.
type regWrongCase struct {
	name string
	args []float64
	warm int
	want string
}

// regWrongCheck runs each case of src (regWrongCase) natively from its
// register code -- from its start, and by on-stack replacement at a loop
// (RegNativeCall), three times -- and as an ordinary call in a runtime
// without the JIT, with the collector idle and with native code seeing it
// mark (jitMarkingForTest): each answer must be the case's.
func regWrongCheck(t *testing.T, src string, cases []regWrongCase) {
	t.Helper()
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		for _, osr := range []bool{false, true} {
			t.Run(fmt.Sprintf("marking=%v/osr=%v", marking, osr), func(t *testing.T) {
				if marking {
					jitMarkingForTest(t)
				}
				rts := regNativeRuntimes(t, src)
				for _, c := range cases {
					args := make([]Value, len(c.args))
					for i, a := range c.args {
						args[i] = Float(a)
					}
					for k, r := range rts {
						fn, _ := r.GetProp(Obj(r.Global()), r.Intern(c.name))
						for range c.warm {
							if _, err := r.Call(fn, Undefined, args); err != nil {
								t.Fatalf("%s warming: %v", c.name, err)
							}
						}
						// Three calls: what one leaves -- its pools run
						// dry, cells left open -- the next meets.
						for call := range 3 {
							var got string
							if k == 1 {
								v, err := r.Call(fn, Undefined, args)
								got = regNativeShow(r, v, err)
							} else {
								res := r.RegNativeCall(fn, args, osr)
								if res.Skipped != "" {
									t.Fatalf("%s: not native: %s", c.name, res.Skipped)
								}
								if res.Entered < 0 || osr && !res.OSR {
									t.Fatalf("%s: native code not entered (at %d, OSR %v)", c.name, res.Entered, res.OSR)
								}
								got = regNativeShow(r, res.Value, res.Err)
							}
							if got != c.want {
								t.Errorf("%s%v, call %d, JIT %v: got %q, want %q", c.name, c.args, call, k == 0, got, c.want)
							}
						}
					}
				}
			})
		}
	}
}

// A getter or a setter that reads its arguments is not inlined where a
// register read or write calls it: inlined, its arguments.length and
// arguments[k] were read at an exit whose frame had no arguments, and a
// setter's arguments.length was 0 (TestJITAccessors' args(30), 1035 for
// 3735).
func TestJITRegAccessorArguments(t *testing.T) {
	regWrongCheck(t, `function A(){this.k=0}
	Object.defineProperty(A.prototype,"n",{get(){return arguments.length+(arguments[0]===undefined?10:0)}});
	Object.defineProperty(A.prototype,"m",{set(v){this.k=arguments.length*100+v}});
	Object.defineProperty(A.prototype,"f",{set(v){this.k=arguments[0]*2}});
	function args(n){const a=new A();let s=0;for(let i=0;i<n;i++){s+=a.n;a.m=i;s+=a.k}return s}
	function sets(n){const a=new A();let s="";for(let i=0;i<n;i++){a.m=i;s+=a.k+","}return s}
	function firsts(n){const a=new A();let s=0;for(let i=0;i<n;i++){a.f=i;s+=a.k}return s}`, []regWrongCase{
		{"args", []float64{30}, 3, "3735"},
		{"sets", []float64{5}, 3, "100,101,102,103,104,"},
		{"firsts", []float64{30}, 3, "870"},
	})
}

// A closure Go makes on the frame Go entered the code with, for a pool of
// closures run dry, takes the cells native code keeps its captured
// registers in, its by-value captures' too, as the stack form's does
// (jitCellsClosure): made by value from the window's registers, its
// other captures had upvalues opened on them, which went on reading the
// registers native code had left, and one left open there was taken by
// a later entry for its register's cell, two iterations' closures then
// sharing a let (TestJITCellsClosedMarking's loops(200)).
func TestJITRegClosureCellsByValue(t *testing.T) {
	regWrongCheck(t, `function lc(n) {
		const fs = []
		for (let i = 0; i < n; i++) {
			const k = i * 2
			let m = i
			fs.push(() => { m++; return m + k })
			if (i % 3 === 0) fs[fs.length - 1]()
		}
		let s = []
		for (let j = 0; j < fs.length; j++) s.push(fs[j]())
		return s.join()
	}`, []regWrongCase{
		{"lc", []float64{40}, 1, "2,4,7,11,13,16,20,22,25,29,31,34,38,40,43,47,49,52,56,58,61,65,67,70,74,76,79,83,85,88,92,94,97,101,103,106,110,112,115,119"},
		{"lc", []float64{5}, 0, "2,4,7,11,13"},
	})
}
