//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"fmt"
	"strings"
	"testing"
)

// regWarmGenSrc has a generator doing work between yields, which its own
// loop runs, stepped by for...of; a generator whose parameters have a
// default, run past them at its making; an async function of a loop that
// does not await, every call of it a first run; and one that awaits
// within its loop.
const regWarmGenSrc = `
function* busy(n) {
	let s = 0;
	for (let i = 0; i < n; i++) {
		s = (s + i * 3) | 0;
		if ((i & 255) == 0) yield s;
	}
	return s;
}
function drive(n) {
	let x = 0;
	for (const v of busy(n)) x ^= v;
	return x;
}
function* pair(a, b = a + 1) { yield a; yield b; }
function pairs(n) {
	let s = 0;
	for (let j = 0; j < n; j++) for (const v of pair(j)) s += v;
	return s;
}
async function sum(n) {
	let s = 0;
	for (let i = 0; i < n; i++) s += i & 7;
	return s;
}
var out = [];
function firsts(n) {
	for (let j = 0; j < n; j++) sum(j).then(function (v) { if (j == n - 1) out.push(v) });
	return n;
}
async function waits(n) {
	let s = 0;
	for (let i = 0; i < n; i++) {
		s += i;
		if (i % 100 == 0) s += await (i % 7);
	}
	return s;
}
function awaits(n) { waits(n).then(function (v) { out.push(v) }); return n }
`

// TestJITRegSuspendNative runs a generator and async functions whose
// register code suspends at a yield, an await and its parameters' end,
// with the JIT and without, the collector idle and marking: native code
// suspends the frame there itself (jitRegSuspend), as the stack form's
// jitSuspend does, where each suspension had been a run ended in the
// interpreter, the frame handed to runReg only to suspend; and an async
// function's every first run enters its code past gen_entry (regGenRun),
// as the stack form's resumption enters a first run.
func TestJITRegSuspendNative(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			rts := regNativeRuntimes(t, regWarmGenSrc)
			rts[0].SetJITReport(true)
			rts[0].jitStress = jitStressConfig{}
			calls := []struct {
				name string
				n    float64
				want string
			}{
				{"drive", 100000, "1938755200"},
				{"pairs", 2000, "4000000"},
				{"firsts", 300, "300"},
				{"awaits", 3000, "3000"},
			}
			var got [2]string
			for i, r := range rts {
				var b strings.Builder
				for _, c := range calls {
					fn, _ := r.GetProp(Obj(r.Global()), r.Intern(c.name))
					for k := range 20 {
						v, err := r.Call(fn, Undefined, []Value{Float(c.n)})
						if s := regNativeShow(r, v, err); s != c.want {
							t.Fatalf("%s, call %d: %s, want %s", c.name, k, s, c.want)
						}
					}
				}
				if err := r.DrainJobs(); err != nil {
					t.Fatal(err)
				}
				v, _ := r.GetProp(Obj(r.Global()), r.Intern("out"))
				b.WriteString(regNativeShow(r, v, nil))
				got[i] = b.String()
			}
			if got[0] != got[1] {
				t.Fatalf("with the JIT %q, without %q", got[0], got[1])
			}
			if want := strings.Repeat("1039,", 20) + strings.Repeat("4498586,", 19) + "4498586"; got[1] != want {
				t.Fatalf("out %q, want %q", got[1], want)
			}
			r := rts[0]
			st := r.RegNativeStats()
			for why, n := range st.Ended {
				if strings.Contains(why, "at yield") || strings.Contains(why, "at end_params") ||
					strings.Contains(why, "at await in waits") {
					t.Errorf("%d runs ended in the interpreter: %s", n, why)
				}
			}
			fn, _ := r.GetProp(Obj(r.Global()), r.Intern("sum"))
			e := r.jit.regCache[r.jitRegKeyOf(fn.Object().fn().closure)]
			if e == nil || e.code == nil {
				t.Fatal("sum not compiled")
			}
			if r.jit.regResumes < 1000 {
				t.Errorf("%d first runs and resumptions entered native code", r.jit.regResumes)
			}
			t.Logf("ended: %v; resumes %d", st.Ended, r.jit.regResumes)
		})
	}
}

// TestJITRegGenLearnedJudgedAfresh checks that code compiled again for a
// generator's step it learned (jitRegGenSeen) has the loops it was judged
// slow for judged afresh (jitRegCached), as the stack form's genLearned
// has jitReplaceCode judge them: left slow, the consumer's loop, whose
// steps had each been a call of Go, stayed with the interpreter until a
// retry a thousand samples later, generator_next's first round twice
// the stack form's.
func TestJITRegGenLearnedJudgedAfresh(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	rts := regNativeRuntimes(t, regWarmGenSrc)
	r := rts[0]
	r.jitStress = jitStressConfig{}
	fn, _ := r.GetProp(Obj(r.Global()), r.Intern("pairs"))
	for range 4 {
		if v, err := r.Call(fn, Undefined, []Value{Float(2000)}); regNativeShow(r, v, err) != "4000000" {
			t.Fatalf("pairs: %s", regNativeShow(r, v, err))
		}
	}
	cl := fn.Object().fn().closure
	e := r.jit.regCache[r.jitRegKeyOf(cl)]
	if e == nil || e.code == nil {
		t.Fatal("pairs not compiled")
	}
	// Judged slow at its loop, then a generator's step learned.
	e.life.loopSlow, e.life.entrySlow = true, true
	r.jitRegGenLearned(cl, e)
	if !e.life.genLearned || !e.life.reopt {
		t.Fatal("the step learned asked for no compile again")
	}
	ne := r.jitRegReoptimize(cl, cl.reg, e)
	if ne == nil || ne.code == nil {
		t.Fatal("not compiled again")
	}
	if ne.life.loopSlow || ne.life.genLearned {
		t.Errorf("compiled again: loopSlow %v, genLearned %v", ne.life.loopSlow, ne.life.genLearned)
	}
	if v, err := r.Call(fn, Undefined, []Value{Float(2000)}); regNativeShow(r, v, err) != "4000000" {
		t.Fatalf("pairs: %s", regNativeShow(r, v, err))
	}
}
