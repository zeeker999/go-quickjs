//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"context"
	"testing"
	"time"
)

// regReportOf is r's report line for the register code of the function
// named name, and whether it has one.
func regReportOf(r *Runtime, name string) (JITFunction, bool) {
	for _, f := range r.JITReport() {
		if f.Name == name+" [reg]" {
			return f, true
		}
	}
	return JITFunction{}, false
}

// TestJITRegGoesOnAtBackEdge has register code leave a loop's turn where it
// has no entry -- after a loop of a built-in's call it makes natively, the
// call given a Set's forEach every other turn, which Go makes -- and checks
// that the frame goes on natively at the loop's next back edge
// (jitRegLeaveTo, jitRegLoop's forced check), as the stack form's jitGoOn
// has it, the interpreter running that turn's rest only: before, it ran
// the rest of the invocation, and native code was entered again only at a
// sample a back-edge budget later.
func TestJITRegGoesOnAtBackEdge(t *testing.T) {
	const src = `
function loop(n) {
	var a = [1, 2, 3], st = new Set([1, 2, 3]), acc = { v: 0 }, cb = function (x) { acc.v += x };
	for (var i = 0; i < n; i++) (i % 2 ? a : st).forEach(cb);
	return acc.v;
}
`
	rts := regNativeRuntimes(t, src)
	rts[0].SetJITWarmUp(1, 1)
	rts[0].jitStress = jitStressConfig{}
	var got [2]string
	for i, r := range rts {
		fn, _ := r.GetProp(Obj(r.Global()), r.Intern("loop"))
		for range 4 {
			v, err := r.Call(fn, Undefined, []Value{Float(2000)})
			got[i] = regNativeShow(r, v, err)
		}
	}
	if got[0] != got[1] || got[0] != "12000" {
		t.Fatalf("with the JIT %q, without %q, want 12000", got[0], got[1])
	}
	s := rts[0].jit
	f, ok := regReportOf(rts[0], "loop")
	t.Logf("went on %d; loop %+v", s.wentOn, f)
	if !ok || f.Status != "native" && f.Status != "slow" {
		// Slow: its stretches, half of them a turn and a call of Go, may
		// not pay (jitRegStretchWork); it went on natively until then.
		t.Fatalf("loop not compiled: %+v", f)
	}
	if s.wentOn == 0 {
		t.Errorf("no frame went on natively at its next back edge")
	}
	// Each turn that left goes on natively at the next: as many entries
	// as turns left, not one for each invocation.
	if f.Ended != 0 && f.Native < f.Ended {
		t.Errorf("%d entries for %d turns left to the interpreter", f.Native, f.Ended)
	}
}

// TestJITRegLearnedStaysValid has register code learn a call at a call of
// Go -- a callee compiled after it -- and checks that it stays its
// function's code, called by native callers and entered by Go, until it
// has spent a budget since, then is compiled again for it (jitRegReopt,
// jitRegLearnedDue), as the stack form's code is (jitLearnedDue). Before,
// learning dropped it from its closure and its link at once, until two
// more samples.
func TestJITRegLearnedStaysValid(t *testing.T) {
	const src = `
function add(a, b) { return a + b }
var target = null;
function caller(n) { var s = 0; for (var i = 0; i < n; i++) s = target(s, i) | 0; return s }
`
	rts := regNativeRuntimes(t, src)
	r := rts[0]
	r.jitStress = jitStressConfig{}
	r.SetJITWarmUp(1, 1)
	for _, rt := range rts {
		add, _ := rt.GetProp(Obj(rt.Global()), rt.Intern("add"))
		if err := rt.SetProp(Obj(rt.Global()), rt.Intern("target"), add); err != nil {
			t.Fatal(err)
		}
	}
	call := func(rt *Runtime, n int) string {
		fn, _ := rt.GetProp(Obj(rt.Global()), rt.Intern("caller"))
		v, err := rt.Call(fn, Undefined, []Value{Float(float64(n))})
		return regNativeShow(rt, v, err)
	}
	if a, b := call(r, 50), call(rts[1], 50); a != b {
		t.Fatalf("with the JIT %q, without %q", a, b)
	}
	fv, _ := r.GetProp(Obj(r.Global()), r.Intern("caller"))
	cl := fv.Object().fn().closure
	e := cl.closureCode.jitReg
	if e == nil {
		t.Fatalf("caller not compiled")
	}
	if e.invalid {
		t.Fatalf("caller's code invalid after learning a call")
	}
	for range 50 {
		if a, b := call(r, 300), call(rts[1], 300); a != b {
			t.Fatalf("with the JIT %q, without %q", a, b)
		}
	}
	f, _ := regReportOf(r, "caller")
	t.Logf("caller %+v", f)
	if f.Reopts == 0 && e.life.reopt {
		t.Errorf("caller never compiled again for what it learned")
	}
	if ne := cl.closureCode.jitReg; ne == nil || ne.invalid || ne.code == nil {
		t.Errorf("caller has no valid code: %v", ne)
	}
}

// TestJITRegCallSamplesSurvivePolls runs interpreted callers -- a
// recursion, which has no loop to sample -- of a callee whose native code
// loops, and checks the callers are compiled: native code's polls, and
// the interpreter's back-edge checks, leave the calls' counter as it is
// (checkInterruptPoll). Rearmed at each, it had run out only where calls
// outnumbered back edges four to one, and the recursion was sampled about
// once a run, AWFY's Towers' moveDisks never compiled.
func TestJITRegCallSamplesSurvivePolls(t *testing.T) {
	const src = `
var o = { v: 0 };
function leaf(x) { for (var k = 0; k < 4; k++) x.v = (x.v + k) | 0 }
function rec(d) { if (d == 0) { leaf(o); return } rec(d - 1); rec(d - 1) }
function run() { rec(13); return o.v }
`
	rts := regNativeRuntimes(t, src)
	r := rts[0]
	r.jitStress = jitStressConfig{}
	var got [2]string
	for i, rt := range rts {
		fn, _ := rt.GetProp(Obj(rt.Global()), rt.Intern("run"))
		for range 12 {
			v, err := rt.Call(fn, Undefined, nil)
			got[i] = regNativeShow(rt, v, err)
		}
	}
	if got[0] != got[1] {
		t.Fatalf("with the JIT %q, without %q", got[0], got[1])
	}
	for _, name := range []string{"leaf", "rec"} {
		if f, ok := regReportOf(r, name); !ok || f.Status != "native" {
			t.Errorf("%s not compiled: %+v", name, f)
		}
	}
}

// TestJITRegStartPollOnce runs functions entered at their start under a
// back-edge budget of one: the poll at the code's start (ssa.OpPoll)
// leaves at every entry, and the run going on at the start passes it
// again, which is counted once. Counted again, it left there again, for
// ever.
func TestJITRegStartPollOnce(t *testing.T) {
	const src = `
function get(o) { return o.a + 1 }
function drive(n) { var o = { a: 1 }, s = 0; for (var i = 0; i < n; i++) s += get(o); return s }
`
	regNoInline(t)
	rts := regNativeRuntimes(t, src)
	r := rts[0]
	r.SetJITWarmUp(1, 1)
	r.jitStress = jitStressConfig{budget: 1}
	r.jitState().stressBudget = 1
	var got [2]string
	for i, rt := range rts {
		fn, _ := rt.GetProp(Obj(rt.Global()), rt.Intern("drive"))
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		rt.SetContext(ctx)
		stop := watchStopForTest(rt, ctx)
		v, err := rt.Call(fn, Undefined, []Value{Float(300)})
		stop()
		cancel()
		got[i] = regNativeShow(rt, v, err)
	}
	if got[0] != got[1] || got[0] != "600" {
		t.Fatalf("with the JIT %q, without %q, want 600", got[0], got[1])
	}
	if f, ok := regReportOf(r, "get"); !ok || f.Native == 0 {
		t.Errorf("get never entered natively: %+v", f)
	}
}

// TestJITRegInstallsAtCall compiles a function's register code in the
// background and checks that a call of it from the interpreter -- at a
// site the interpreter's caches answer, no sample -- installs the code
// once done and enters it (jitEnters, jitRegEnter), as the stack form's
// calls install theirs (tryJITAt): before, only a sample installed it.
func TestJITRegInstallsAtCall(t *testing.T) {
	const src = `
function f(x) { var s = 0; for (var i = 0; i < x; i++) s += i; return s }
function g(n) { return f(n) }
`
	rts := regNativeRuntimes(t, src)
	r := rts[0]
	r.jitStress = jitStressConfig{}
	r.jitBackground = true
	fv, _ := r.GetProp(Obj(r.Global()), r.Intern("f"))
	cl := fv.Object().fn().closure
	c := cl.reg
	if c == nil {
		c = r.regOf(cl)
	}
	if e := r.jitRegFor(cl, c); e != nil {
		t.Fatalf("compiled on the spot: %+v", e)
	}
	if !cl.jitEnters || cl.closureCode.jitReg != nil {
		t.Fatalf("a compile in the background not marked for its calls")
	}
	for _, j := range r.jit.regJobs {
		<-j.done
	}
	r.interruptCounter = 1 << 30
	gv, _ := r.GetProp(Obj(r.Global()), r.Intern("g"))
	v, err := r.Call(gv, Undefined, []Value{Float(10)})
	if got := regNativeShow(r, v, err); got != "45" {
		t.Fatalf("g(10) = %q, want 45", got)
	}
	e := cl.closureCode.jitReg
	if e == nil || e.code == nil || e.life.starts == 0 {
		t.Fatalf("the call did not install and enter the code: %+v", e)
	}
}

// TestJITRegOSRAtDroppedEntry runs osrSetup's sum -- objects escape
// analysis leaves unmade carried across its loop, which drops the loop's
// entry -- as the stack form's TestJITLearnsLoopEntry does: called often,
// its code is compiled without the entry, and a call that leaves the loop
// at a turn with no entry after must have the frame go on natively all the
// same: found twice at the header with no entry, the function is compiled
// apart with the entry kept (jitRegLoopCode), as jitLearnLoopEntry has it.
// Before, the frame ran the rest of the loop in the interpreter.
func TestJITRegOSRAtDroppedEntry(t *testing.T) {
	rts := regNativeRuntimes(t, osrSetup)
	r := rts[0]
	r.jitStress = jitStressConfig{}
	r.SetJITWarmUp(1, 1)
	for _, round := range osrRounds("sum", "") {
		var got [2]string
		for i, rt := range rts {
			v, err := rt.Run(compileForTest(t, round))
			got[i] = regNativeShow(rt, v, err)
		}
		if got[0] != got[1] {
			t.Fatalf("%s: with the JIT %q, without %q", round, got[0], got[1])
		}
	}
	fv, _ := r.GetProp(Obj(r.Global()), r.Intern("sum"))
	e := fv.Object().fn().closure.closureCode.jitReg
	if e == nil {
		t.Fatalf("sum not compiled")
	}
	t.Logf("unentered %v, loops %v, osr %v, learned %d", e.unentered, e.loops, e.osr != nil, r.jit.osrLearned)
	if len(e.unentered) == 0 {
		t.Skipf("sum's code kept every loop's entry")
	}
	if e.osr == nil || r.jit.osrLearned == 0 {
		t.Errorf("a frame at a loop whose entry was dropped never had code with it")
	}
}
