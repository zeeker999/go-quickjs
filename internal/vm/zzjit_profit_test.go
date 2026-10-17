//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"testing"
	"weak"
)

// jitProfitRuntimes runs setup in a runtime with the JIT, which compiles at
// a function's first call, and one without, and returns a function that
// runs src in both, failing unless the answers agree, and one that finds a
// function's entry.
func jitProfitRuntimes(t *testing.T, setup string) (run func(src string), entry func(name string) *jitEntry) {
	t.Helper()
	want := New(Config{})
	t.Cleanup(func() { want.Close(); want.ReleaseClosed() })
	r := jitRuntimeForTest(t, Config{JIT: true})
	for _, rt := range []*Runtime{want, r} {
		if _, err := rt.Run(compileForTest(t, setup)); err != nil {
			t.Fatal(err)
		}
	}
	run = func(src string) {
		t.Helper()
		wv, err := want.Run(compileForTest(t, src))
		if err != nil {
			t.Fatal(err)
		}
		gv, err := r.Run(compileForTest(t, src))
		if err != nil {
			t.Fatalf("%s: %v", src, err)
		}
		if !jitSameValueForTest(gv, wv) {
			t.Fatalf("%s: got %v, interpreter %v", src, gv, wv)
		}
	}
	entry = func(name string) *jitEntry {
		cl := r.global.getOwn(r.atoms.intern(name)).value().Object().fn().closure
		return r.jit.cache[weak.Make(cl.fn)]
	}
	return run, entry
}

// Code Go enters is judged by what its stretches' ends and its calls of Go
// cost against the tree tier, in instructions of work (jitCostHost and the
// others, TestJITCalibration's), against the work it did natively
// (jitSSAProfit). A loop that leaves for Go at every iteration (typeof),
// with little work besides, is given up after its first stretches; so is
// one with about 40 instructions of work an exit, which a cost of 20 an
// exit had kept native at a loss -- Box2D's functions, leaving about twice
// a run, had made Box2D slower with the JIT than without. One whose work
// makes up for its exits many times over stays native. A loop of calls of
// a built-in, each costing natively about twelve instructions of work more
// than the tree tier's call of it, stays native with half the work around
// each that between has around each exit; one of calls of a function of
// script's, for which Go makes a frame and runs the tree tier, with little
// else, is given up. Each answer is the interpreter's.
func TestJITProfitDecisions(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	setup := `var str='abcdefgh';
		function leaves(n,x){let s=0;for(let i=0;i<n;i++){const t=typeof x;s=(s+t.length+i)|0}return s}
		function between(n,x){let s=0;for(let i=0;i<n;i++){const t=typeof x;s=(s+t.length)|0;
			s=(s*31+i)|0;s^=s>>>7;s=(s+i*i)|0;s^=s<<3;s=(s*7+(i>>1))|0;s^=s>>>5}return s}
		function pays(n,x){let s=0;for(let i=0;i<n;i++){const t=typeof x;s=(s+t.length)|0;
			for(let j=0;j<40;j++){s=(s*31+j)|0;s^=s>>>7;s=(s+j*i)|0}}return s}
		function builtins(n){let s=0;for(let i=0;i<n;i++){s=(s+str.indexOf('c'))|0;s=(s*31+i)|0;s^=s>>>7;s=(s+i*i)|0}return s}
		function m(i){return arguments.length+i}
		function scripts(n){let s=0;for(let i=0;i<n;i++)s=(s+m(i))|0;return s}`
	run, entry := jitProfitRuntimes(t, setup)
	const src = `[leaves(2000,1),between(2000,1),pays(300,1),builtins(3000),scripts(2000)].join()`
	run(src)
	// Judged as each stretch ends: leaves and between after their first
	// stretches, scripts, one stretch, as it returns. The second run does
	// not enter them.
	for _, name := range []string{"leaves", "between"} {
		if e := entry(name); e == nil || e.ssa == nil || !e.entrySlow || e.ssaStats.entries > 2*jitSSAProbe {
			t.Errorf("%s kept running natively, at a loss: %+v", name, e)
		}
	}
	before := map[string]jitSSAStats{}
	for _, name := range []string{"leaves", "between", "scripts"} {
		if e := entry(name); e != nil {
			before[name] = e.ssaStats
		}
	}
	run(src)
	for _, name := range []string{"leaves", "between", "scripts"} {
		if e := entry(name); e == nil || e.ssa == nil || !e.entrySlow || e.ssaStats.entries != before[name].entries || e.ssaStats.goCalls != before[name].goCalls {
			t.Errorf("%s kept running natively, at a loss: %+v", name, e)
		}
	}
	if e := entry("pays"); e == nil || e.ssa == nil || e.entrySlow || e.ssaStats.hosts < 300 {
		t.Errorf("pays, its work many times its exits' cost, was given up: %+v", e)
	}
	if e := entry("builtins"); e == nil || e.ssa == nil || e.entrySlow || e.ssaStats.goCalls < 3000 || e.ssaStats.goScripts != 0 || e.ssaStats.hosts != 0 {
		t.Errorf("builtins, its calls of Go cheap, was given up: %+v", e)
	}
	if e := entry("scripts"); e != nil && (e.ssaStats.goScripts == 0 || e.ssaStats.goScripts != e.ssaStats.goCalls) {
		t.Errorf("scripts' calls of m were not counted as calls of a function of script's: %+v", e.ssaStats)
	}
}

// A native callee is judged for its native callers (jitCalleeJudge): its
// exits, each a frame Go makes and a caller resumed (jitCostCalleeLeft),
// against its work and what calling it natively saves them. A callee that
// leaves at most of its calls, doing little work, stops being called
// natively (notNative), its callers calling it through Go -- while its
// caller, whose stretches its exits ended, stays native: the callee answers
// for its exits, not the caller. One that leaves as often but does much
// work at each call stays called natively. And one that leaves at its
// start, its loop after the exit, which Go goes on with natively, is judged
// by all its work once called through Go, as Go's own entries are: Go
// enters it, natively, and does not give it up -- AWFY's Bounce's
// benchmark had been given up to the tree tier for the little work it did
// before its exit, and Bounce took twice as long. Each answer is the
// interpreter's.
func TestJITProfitNativeCallee(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	setup := `
		function leafLeaves(i,x){let s=i;for(let j=0;j<1;j++)s=(s+j)|0;const t=typeof x;return (s+t.length)|0}
		function leafWorks(i,x){let s=i;for(let j=0;j<40;j++){s=(s*31+j)|0;s^=s>>>7;s=(s+j*j)|0}const t=typeof x;return (s+t.length)|0}
		function outer(n,x){let s=0;for(let i=0;i<n;i++){s=(s+leafLeaves(i,x))|0;
			s=(s*31+i)|0;s^=s>>>7;s=(s+i*i)|0;s^=s<<3;s=(s*7+(i>>1))|0;s^=s>>>5;s=(s*17+i)|0;s^=s>>>3;s=(s+i*5)|0;s^=s<<1}return s}
		function outerWorks(n,x){let s=0;for(let i=0;i<n;i++)s=(s+leafWorks(i,x))|0;return s}
		function leavesFirst(i,x){const t=typeof x;let s=i+t.length;for(let j=0;j<200;j++){s=(s*31+j)|0;s^=s>>>7;s=(s+j*j)|0}return s}
		function outerFirst(n,x){let s=0;for(let i=0;i<n;i++)s=(s+leavesFirst(i,x))|0;return s}`
	run, entry := jitProfitRuntimes(t, setup)
	for range 3 {
		run(`[outer(1000,1),outerWorks(500,1),outerFirst(300,1)].join()`)
	}
	leaf := entry("leafLeaves")
	if leaf == nil || leaf.ssa == nil || !leaf.notNative || leaf.link.entry != 0 {
		t.Errorf("leafLeaves, leaving at most of its native calls, kept being called natively: %+v", leaf)
	}
	if e := entry("outer"); e == nil || e.ssa == nil || e.entrySlow {
		t.Errorf("outer was given up for its callee's exits: %+v", e)
	}
	if e := entry("leafWorks"); e == nil || e.ssa == nil || e.notNative || e.nativeOut == 0 {
		t.Errorf("leafWorks, its work making up for its exits, stopped being called natively: %+v", e)
	}
	if e := entry("outerWorks"); e == nil || e.ssa == nil || e.entrySlow {
		t.Errorf("outerWorks was given up: %+v", e)
	}
	if e := entry("leavesFirst"); e == nil || e.ssa == nil || e.entrySlow || e.ssaStats.work < 300*200 {
		t.Errorf("leavesFirst, its loop native once Go goes on with its call, was given up: %+v", e)
	}
}

// A native callee's exit costs more the deeper it is below the code Go
// entered: Go makes a frame for each level between and resumes each
// (jitCostCalleeLevel). The same callee, leaving at each call with some
// work, pays called natively from the code Go entered, and does not two
// native calls down, where its native callers stop calling it natively --
// SplayTree's splay_, three levels down, had been called so at every node
// it inserted, its exit 3 times a callee's. Each answer is the
// interpreter's.
func TestJITProfitCalleeDepth(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	setup := `
		function leafNear(i,x){let s=i;for(let j=0;j<10;j++){s=(s*31+j)|0;s^=s>>>7;s=(s+j)|0}const t=typeof x;return (s+t.length)|0}
		function leafDeep(i,x){let s=i;for(let j=0;j<10;j++){s=(s*31+j)|0;s^=s>>>7;s=(s+j)|0}const t=typeof x;return (s+t.length)|0}
		function mid(i,x){let s=i;for(let j=0;j<0;j++)s++;return (s+leafDeep(i,x))|0}
		function near(n,x){let s=0;for(let i=0;i<n;i++)s=(s+leafNear(i,x))|0;return s}
		function deep(n,x){let s=0;for(let i=0;i<n;i++)s=(s+mid(i,x))|0;return s}`
	run, entry := jitProfitRuntimes(t, setup)
	for range 3 {
		run(`[near(500,1),deep(500,1)].join()`)
	}
	if e := entry("leafNear"); e == nil || e.ssa == nil || e.notNative || e.nativeOut == 0 {
		t.Errorf("leafNear, its work making up for its exits one level down, stopped being called natively: %+v", e)
	}
	if e := entry("leafDeep"); e == nil || e.ssa == nil || !e.notNative {
		t.Errorf("leafDeep, leaving two native calls down, kept being called natively: %+v", e)
	}
}

// A native callee that leaves on more than one in jitUnwindShare of its
// calls while it learns is compiled again for what its exits taught,
// whatever its exits cost, its native callers calling it still
// (jitUnwindLearn): a learning trigger, apart from whether it pays. Only
// one that does not pay, its compiles spent, stops being called natively.
func TestJITProfitUnwoundLearns(t *testing.T) {
	defer func(was func() bool) { jitMarking = was }(jitMarking)
	jitMarking = func() bool { return false }
	r := jitRuntimeForTest(t, Config{JIT: true})
	if _, err := r.Run(compileForTest(t, `function g(n){let s=0;for(let i=0;i<n;i++)s=(s+i)|0;return s}g(300);g(300)`)); err != nil {
		t.Fatal(err)
	}
	g := r.jit.cache[weak.Make(r.global.getOwn(r.atoms.intern("g")).value().Object().fn().closure.fn)]
	if g == nil || g.ssa == nil {
		t.Fatal("g has no code")
	}
	// Every call leaves, each after 1,000 instructions of work: it pays.
	e := &jitEntry{ssa: g.ssa, link: &jitLink{}}
	leave := func() {
		e.polyPending = true
		for range jitSSAProbe {
			e.link.in++
			r.jitUnwound(e, 1000, 1)
		}
	}
	for i := range jitUnwindReopts {
		leave()
		if !e.unwindReopt || e.notNative || int(e.unwindReopts) != i+1 {
			t.Fatalf("learning, leaving at every call %d times: compiled again %v, notNative %v", i+1, e.unwindReopt, e.notNative)
		}
		e.unwindReopt = false
	}
	leave()
	if e.unwindReopt || e.notNative {
		t.Fatalf("paying, its compiles spent: compiled again %v, notNative %v", e.unwindReopt, e.notNative)
	}
}

// What code's native calls save against the tree tier's calls
// (jitCostNativeCall) counts as its work (jitNativeCallsMade): calls, which
// the work its stretches are estimated by leaves out. A loop of native
// calls of a small callee, and a call of Go of a function of script's,
// pays; with little else and no native calls, it does not -- DeltaBlue's
// addToGraph and incrementalAdd, methods of calls, had been given up, and
// DeltaBlue took 8% longer. Each answer is the interpreter's.
func TestJITProfitNativeCalls(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	setup := `function cal(i){let s=i;for(let j=0;j<0;j++)s++;return s}
		function m(i){return arguments.length+i}
		function calls(n){let s=0;for(let i=0;i<n;i++)s=(s+m(i)+cal(i)+cal(i+1)+cal(i+2))|0;return s}
		function alone(n){let s=0;for(let i=0;i<n;i++)s=(s+m(i)+i)|0;return s}`
	run, entry := jitProfitRuntimes(t, setup)
	for range 3 {
		run(`[calls(2000),alone(2000)].join()`)
	}
	if e := entry("calls"); e == nil || e.ssa == nil || e.entrySlow || e.ssaStats.goScripts == 0 {
		t.Errorf("calls, its native calls saving more than its calls of Go cost, was given up: %+v", e)
	}
	if e := entry("alone"); e == nil || e.ssa == nil || !e.entrySlow {
		t.Errorf("alone, its calls of Go with little else, kept running natively: %+v", e)
	}
}
