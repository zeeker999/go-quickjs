//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"fmt"
	"runtime"
	"runtime/debug"
	"slices"
	"strings"
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
	"github.com/go-quickjs/go-quickjs/internal/jit/ssa"
)

// jitTierSetup has f read x off its receivers' prototypes, which a site that
// knew nothing searches no receiver's own table for: A's, B's, C's and D's
// objects are of four shapes.
const jitTierSetup = `function A(){} A.prototype.x=1; function B(){this.y=0} B.prototype.x=2;
	function C(){this.z=0;this.w=0} C.prototype.x=3; function D(){this.q=0} D.prototype.x=4;
	var a=new A(), b=new B(), c=new C(), d=new D();
	function f(o,n){let s=0;for(let i=0;i<n;i++)s=(s+o.x)|0;return s}`

// jitFewCallsForTest has r compile a function after a few calls, the
// warm-up of tests of what native code does once compiled, rather than
// V8's budgets (jitInvocationsForMaglev), unless it compiles at once.
func jitFewCallsForTest(r *Runtime) {
	if r.jitCallThreshold > jitInvocationsForFeedback {
		r.jitCallThreshold, r.jitICThreshold = jitInvocationsForFeedback, jitInvocationsForFeedback
	}
}

// jitTierRuntimeForTest runs setup in a runtime that compiles a function at
// its threshold's call, and in an interpreter, and returns a run that has
// both run src, failing on an answer that differs.
func jitTierRuntimeForTest(t *testing.T, threshold uint16, setup string) (*Runtime, func(src string)) {
	t.Helper()
	return jitTierRuntimeWithForTest(t, threshold, setup, nil)
}

// jitTierRuntimeWithForTest is jitTierRuntimeForTest, each runtime given
// install's globals, where it is not nil, before setup runs.
func jitTierRuntimeWithForTest(t *testing.T, threshold uint16, setup string, install func(*Runtime)) (*Runtime, func(src string)) {
	t.Helper()
	want := New(Config{})
	t.Cleanup(func() { want.Close(); want.ReleaseClosed() })
	r := jitRuntimeForTest(t, Config{JIT: true})
	r.jitCallThreshold, r.jitICThreshold = threshold, threshold
	for _, rt := range []*Runtime{want, r} {
		if install != nil {
			install(rt)
		}
		if _, err := rt.Run(compileForTest(t, setup)); err != nil {
			t.Fatal(err)
		}
	}
	return r, func(src string) {
		t.Helper()
		wv, err := want.Run(compileForTest(t, "String("+src+")"))
		if err != nil {
			t.Fatal(err)
		}
		gv, err := r.Run(compileForTest(t, "String("+src+")"))
		if err != nil {
			t.Fatal(err)
		}
		if got, w := gv.String().Go(), wv.String().Go(); got != w {
			t.Fatalf("%s: got %s, interpreter %s", src, got, w)
		}
	}
}

// An eager deoptimization of a function's current code invalidates it, as
// V8's deoptimizer does: Go enters it no more, nor do native callers call
// it, its link cleared; the existing tiers run the function, and it is
// compiled again once they have spent a fresh budget -- as many
// invocations as it took to compile it first (jitCallThreshold), each its
// length (jitTierBudget) -- for what its caches learned meanwhile: f,
// compiled for A, meets B, runs in the interpreter for three calls, and is
// compiled again at the fourth, for both, never deoptimizing again. Each
// answer is the interpreter's.
func TestJITDeoptInvalidates(t *testing.T) {
	t.Parallel()
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	r, run := jitTierRuntimeForTest(t, 4, jitTierSetup)
	for range 4 {
		run(`f(a,10)`)
	}
	e := jitNativeForTest(t, r, "f")
	if e.invalid || e.ssaSerial != 1 || e.link.entry == 0 {
		t.Fatalf("f is not compiled at its fourth call: %+v", e)
	}
	run(`f(b,10)`)
	if !e.invalid || e.deopts != 1 || e.link.entry != 0 || e.ssaSerial != 1 {
		t.Fatalf("f's deoptimization did not invalidate its code: invalid %v, deopts %d, link %#x, compiles %d", e.invalid, e.deopts, e.link.entry, e.ssaSerial)
	}
	entries := e.ssaStats.entries
	for k := 1; k <= 3; k++ {
		run(`f(b,10)`)
		if e.ssaSerial != 1 || !e.invalid || e.ssaStats.entries != entries {
			t.Fatalf("call %d after the deoptimization: compiled %d times, invalid %v, entered %d times", k, e.ssaSerial, e.invalid, e.ssaStats.entries-entries)
		}
	}
	guards := r.jit.guards
	run(`f(b,10)`)
	if e.ssaSerial != 2 || e.invalid || e.link.entry == 0 || e.ssaStats.entries == 0 {
		t.Fatalf("f was not compiled again at the budget's end: compiled %d times, invalid %v", e.ssaSerial, e.invalid)
	}
	run(`[f(a,10),f(b,10)].join()`)
	if r.jit.guards != guards || e.invalid {
		t.Fatalf("f compiled again for both deoptimized %d times", r.jit.guards-guards)
	}
}

// A change of a function's caches' feedback while its invalidated code waits
// for the existing tiers' budget delays the compile by a fresh budget, as
// V8's NotifyICChanged does: f, invalidated by B, meets C in the interpreter
// at its second call, and is compiled at the eighth, not the fourth, for
// all three shapes. The caches filled again with what they held -- the
// same entries, in another order -- delay nothing (jitFeedbackMark).
func TestJITFeedbackChangeDelaysTierUp(t *testing.T) {
	t.Parallel()
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	r, run := jitTierRuntimeForTest(t, 4, jitTierSetup)
	for range 4 {
		run(`f(a,10)`)
	}
	e := jitNativeForTest(t, r, "f")
	run(`f(b,10)`)
	if !e.invalid {
		t.Fatal("f's deoptimization did not invalidate its code")
	}
	for k, src := range []string{`f(b,10)`, `f(c,10)`, `f(b,10)`, `f(b,10)`, `f(b,10)`, `f(b,10)`, `f(b,10)`} {
		run(src)
		if e.ssaSerial != 1 || !e.invalid {
			t.Fatalf("call %d after the deoptimization: compiled again, its feedback changed at the second", k+1)
		}
	}
	guards := r.jit.guards
	run(`f(b,10)`)
	if e.ssaSerial != 2 || e.invalid {
		t.Fatalf("f was not compiled again a budget after its feedback changed: compiled %d times, invalid %v", e.ssaSerial, e.invalid)
	}
	run(`[f(a,10),f(b,10),f(c,10)].join()`)
	if r.jit.guards != guards {
		t.Fatalf("f compiled again for its three shapes deoptimized %d times", r.jit.guards-guards)
	}

	// The mark is what the caches hold: filled again with the same, the
	// same mark; a shape they did not hold, another.
	cl := r.global.getOwn(r.atoms.intern("f")).value().Object().fn().closure
	m := jitFeedbackMark(cl)
	run(`[f(a,10),f(b,10),f(c,10),f(a,10)].join()`)
	if got := jitFeedbackMark(cl); got != m {
		t.Fatalf("mark %d, after meeting what the caches held, %d", got, m)
	}
	run(`f(d,10)`)
	if got := jitFeedbackMark(cl); got == m {
		t.Fatal("mark unchanged by a shape the caches did not hold")
	}
}

// A function that keeps deoptimizing is compiled no more past jitMaxDeopts
// eager deoptimizations, as V8 gives up optimizing one: f, at the limit,
// deoptimizes once more, and the interpreter runs it from then on, however
// many budgets it spends.
func TestJITDeoptLimit(t *testing.T) {
	t.Parallel()
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	r, run := jitTierRuntimeForTest(t, 4, jitTierSetup)
	for range 4 {
		run(`f(a,10)`)
	}
	e := jitNativeForTest(t, r, "f")
	e.deopts = jitMaxDeopts
	run(`f(b,10)`)
	if !e.invalid || e.deopts != jitMaxDeopts+1 {
		t.Fatalf("f's deoptimization past the limit: invalid %v, deopts %d", e.invalid, e.deopts)
	}
	tiered := r.jit.tieredUp
	for k := 0; k < 40; k++ {
		run(fmt.Sprintf(`f(%c,10)`, "ab"[k&1]))
	}
	if e.ssaSerial != 1 || !e.invalid || r.jit.tieredUp != tiered {
		t.Fatalf("f compiled again past the deoptimization limit: compiled %d times, invalid %v", e.ssaSerial, e.invalid)
	}
}

// A long invocation whose loop deoptimizes goes on in the interpreter, its
// code invalidated, and is compiled again once its back edges have spent
// the budget with its caches' feedback settled, entered at the loop's back
// edge by on-stack replacement, as V8's code optimized again is: g's
// receivers change at its thousandth iteration, and the rest of its
// twenty thousand run natively again, in the same invocation.
func TestJITDeoptTiersUpByOSR(t *testing.T) {
	t.Parallel()
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	r, run := jitTierRuntimeForTest(t, 1, jitTierSetup+`
		function g(n){let s=0;for(let i=0;i<n;i++){const o=i<1000?a:b;s=(s+o.x)|0}return s}`)
	run(`g(10)`)
	e := jitNativeForTest(t, r, "g")
	tiered, osrs := r.jit.tieredUp, r.jit.osrs
	run(`g(20000)`)
	if e.deopts == 0 || e.invalid || r.jit.tieredUp == tiered || r.jit.osrs == osrs || e.ssaStats.entries == 0 {
		t.Fatalf("g did not go on natively by on-stack replacement: deopts %d, invalid %v, tiered up %d, entered at loops %d", e.deopts, e.invalid, r.jit.tieredUp-tiered, r.jit.osrs-osrs)
	}
	guards := r.jit.guards
	run(`g(20000)`)
	if r.jit.guards != guards || e.invalid {
		t.Fatalf("g compiled again by on-stack replacement deoptimized %d times", r.jit.guards-guards)
	}
}

// Code that learns at its exits what to be compiled again for is not
// invalidated, as V8 never invalidates code for feedback: it is the tier
// that learns it -- the existing tiers' caches note no call's target --
// and runs on, compiled again once it has spent a budget since it began
// to learn (jitLearnedDue): apply, compiled at its fourth call, learns at
// its calls of Go that its callback is inc, keeps its code, Go entering
// it, and is compiled again, inc inlined, only once a budget of its
// invocations has passed, Go making no call of inc after.
func TestJITLearnedCodeRunsOn(t *testing.T) {
	t.Parallel()
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	r, run := jitTierRuntimeForTest(t, 4, `function inc(x){return x+1}
		function apply(fn,n){let s=0;for(let i=0;i<n;i++)s=fn(s);return s}`)
	for range 4 {
		run(`apply(inc,3)`)
	}
	e := jitNativeForTest(t, r, "apply")
	k := 0
	for ; k < 16 && !e.reoptPending(); k++ {
		run(`apply(inc,3)`)
	}
	if !e.reoptPending() || e.ssaSerial != 1 {
		t.Fatalf("apply learned nothing to compile again for in %d calls: compiled %d times", k, e.ssaSerial)
	}
	starts := e.starts
	for k = 0; k < 64 && e.ssaSerial == 1; k++ {
		if e.invalid || e.link.entry == 0 {
			t.Fatalf("call %d after apply learned: its code invalidated", k)
		}
		run(`apply(inc,3)`)
	}
	if e.ssaSerial != 2 || e.invalid {
		t.Fatalf("apply not compiled again for what it learned: compiled %d times, invalid %v", e.ssaSerial, e.invalid)
	}
	if k < 2 || e.starts-starts < 2 {
		t.Fatalf("apply compiled again %d calls after it learned, entered %d times meanwhile: no budget spent", k, e.starts-starts)
	}
	calls := r.jit.goCalls
	run(`apply(inc,100)`)
	if r.jit.goCalls != calls {
		t.Fatalf("apply compiled again made %d calls of Go", r.jit.goCalls-calls)
	}

	// What it learns counts in the mark that delays the compile: a call
	// decided, as a cache's new shape does.
	cl := r.global.getOwn(r.atoms.intern("apply")).value().Object().fn().closure
	m := jitLearnedMark(cl, e)
	e.nativeCalls = append(e.nativeCalls, jitInline{pc: 0})
	if jitLearnedMark(cl, e) == m {
		t.Fatal("mark unchanged by a call learned")
	}
	e.nativeCalls = e.nativeCalls[:len(e.nativeCalls)-1]
}

// v8TierRuntimeForTest is a runtime with V8's budgets (jitInvocationsForMaglev,
// jitInvocationsAfterICUpdate), and an interpreter beside it, as
// jitTierRuntimeForTest's.
func v8TierRuntimeForTest(t *testing.T, setup string) (*Runtime, func(src string)) {
	t.Helper()
	r, run := jitTierRuntimeForTest(t, jitInvocationsForMaglev, setup)
	r.jitICThreshold = jitInvocationsAfterICUpdate
	if r.jitStress.threshold {
		t.Skip("the runtime compiles eagerly (QJS_JIT_STRESS)")
	}
	return r, run
}

// A function is compiled once the existing tiers have spent
// invocation_count_for_maglev invocations' worth of its budget, each its
// bytecode's length, as V8 tiers up to Maglev: k, whose caches meet one
// shape from its first call on, is compiled at its 400th call, not at its
// 399th. A change of its caches' feedback raises what is left to
// minimum_invocations_after_ic_update invocations' worth, as
// NotifyICChanged does: g, the same, meeting a second shape at its 100th
// call, is compiled at its 900th, the change found when its budget was
// spent at its 400th, not before. Neither has a loop, whose back edges
// would spend their budget too.
func TestJITV8Budgets(t *testing.T) {
	t.Parallel()
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	r, run := v8TierRuntimeForTest(t, jitTierSetup+`
		function k(o){const p=[o];return p[0].x}
		function g(o){const p=[o];return p[0].x}`)
	fe := func(name string) *jitEntry {
		cl := r.global.getOwn(r.atoms.intern(name)).value().Object().fn().closure
		return r.jit.entryFor(cl.fn)
	}
	for n := 1; n < jitInvocationsForMaglev; n++ {
		run(`k(a)`)
		if e := fe("k"); e != nil && e.ssa != nil {
			t.Fatalf("k compiled at call %d", n)
		}
	}
	run(`k(a)`)
	if e := fe("k"); e == nil || e.ssa == nil || e.ssaStats.entries == 0 {
		t.Fatalf("k not compiled at call %d", jitInvocationsForMaglev)
	}

	want := jitInvocationsForMaglev + jitInvocationsAfterICUpdate
	for k := 1; k < want; k++ {
		o := "a"
		if k >= 100 && k&1 == 0 {
			o = "b"
		}
		run(`g(` + o + `)`)
		if e := fe("g"); e != nil && e.ssa != nil {
			t.Fatalf("g compiled at call %d, its feedback changed at the 100th", k)
		}
	}
	run(`g(a)`)
	if e := fe("g"); e == nil || e.ssa == nil {
		t.Fatalf("g not compiled at call %d", want)
	}
	guards := r.jit.guards
	run(`[g(a),g(b)].join()`)
	if r.jit.guards != guards {
		t.Fatal("g, compiled for both shapes, deoptimized")
	}
}

// A function past the 4,096 instructions the JIT had compiled is compiled
// as any other, where what its compile holds fits the JIT's budget
// (jitCompiles): zlib's bk, 5,039 instructions and called hundreds of
// times a run, had run in the interpreter. At the runtime's default budget
// too, which the estimate of a word a slot an instruction had kept long
// from.
func TestJITCompilesLongFunctions(t *testing.T) {
	t.Parallel()
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	src := `function long(n){let t=0;for(let i=0;i<n;i++){` + strings.Repeat(`t=(t*3+i)|0;`, 1100) + `}return t}`
	for _, budget := range []int64{32 << 20, 0} {
		r, run := jitTierRuntimeForTest(t, 2, src)
		r.jitMemoryLimit = budget
		cl := r.global.getOwn(r.atoms.intern("long")).value().Object().fn().closure
		if len(cl.fn.Code) <= 4096 {
			t.Fatalf("long is %d instructions", len(cl.fn.Code))
		}
		for range 4 {
			run(`long(3)`)
		}
		e := r.jit.entryFor(cl.fn)
		if compiles := jitCompiles(cl.fn, r.jitBudget()); !compiles || e == nil || e.ssa == nil || e.ssaStats.entries == 0 {
			t.Fatalf("long (%d instructions) not run natively within %d bytes: %+v", len(cl.fn.Code), r.jitBudget(), e != nil && e.ssa != nil)
		}
	}
}

// jitRareCallForTest has h make call, of i, once in twenty-five of its
// calls: less often than jitInliningFrequency.
func jitRareCallForTest(call string) string {
	return `var q=0;function h(n,k){let t=0;for(let i=0;i<n;i++){t=(t+i)|0;if(k&&(++q%25===0))t=(t+` + call + `)|0}return t}`
}

// A callee too long to inline (ssa.MaxInline) that its native caller calls
// seldom warms up as any function does, and is compiled once it has spent
// its own budget -- not at its native caller's first call of it, nor when
// that caller is compiled: it would not be inlined in V8 either
// (jitCalledOften), and Mandreel's per-frame functions, __udivmoddi4's
// 1,233 instructions among them, had been compiled so. g, called by h's
// code once in twenty-five of h's calls, is compiled at about its own 400th call,
// and is called natively by h from then on. Meanwhile h's call waits for
// it, decided no more at each of g's calls (jitCallWaiting).
func TestJITLongCalleeWarmsUp(t *testing.T) {
	t.Parallel()
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	r, run := v8TierRuntimeForTest(t, `function g(x){`+strings.Repeat(`x=(x*3+1)|0;`, 16)+`return x}`+jitRareCallForTest(`g(i)`))
	g := r.global.getOwn(r.atoms.intern("g")).value().Object().fn().closure
	h := r.global.getOwn(r.atoms.intern("h")).value().Object().fn().closure
	if len(g.fn.Code) <= ssa.MaxInline {
		t.Fatalf("g is %d instructions, short enough to inline", len(g.fn.Code))
	}
	for range jitInvocationsForMaglev {
		run(`h(4,0)`)
	}
	if e := r.jit.entryFor(h.fn); e == nil || e.ssa == nil {
		t.Fatal("h has no code")
	}
	decided := r.jit.callsDecided
	for i := range 25 * (jitInvocationsForMaglev - 8) {
		run(`h(1,1)`)
		if e := r.jit.entryFor(g.fn); e != nil && e.ssa != nil {
			t.Fatalf("g compiled at its %d. call", (i+1)/25)
		}
	}
	// h's call waits for g (jitCallWaiting): decided once, not at each of
	// g's calls while it warms.
	if n := r.jit.callsDecided - decided; n > 4 {
		t.Fatalf("h's call of g decided %d times while g warmed", n)
	}
	for range 25 * jitInvocationsForMaglev {
		run(`h(1,1)`)
	}
	e := r.jit.entryFor(g.fn)
	if e == nil || e.ssa == nil {
		t.Fatalf("g not compiled once warm: %+v", e)
	}
	in := e.link.in
	run(`h(250,1)`)
	if e.link.in-in < 10 {
		t.Fatalf("h's code called g natively %d times of 10", e.link.in-in)
	}
}

// A callee too long to inline that its native caller calls often is
// compiled for it once it has been invoked jitInliningFrequency hundredths
// as often as the caller, as V8 would inline it, well before its own
// budget is spent: g, called at each of h's calls once h is compiled, is
// compiled by its 100th, and h calls it natively. One past what a
// function inlines in all (ssa.MaxInlineTotal), which V8 would not inline,
// waits for its budget however often it is called: big. The calls are
// counted as invocations of their caller, not by the work of its loop's
// back edges, which a long loop's would make look seldom: w's loop, run
// once, calls g at every tenth of its iterations, which is compiled when w
// is, before its own budget is spent. Octane's Richards had run
// HandlerTask.run, called 0.37 times a call of its caller, in the tree
// tier for 750 calls.
func TestJITFrequentLongCalleeCompiledForCaller(t *testing.T) {
	t.Parallel()
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	body := func(n int) string { return strings.Repeat(`x=(x*3+1)|0;`, n) }
	r, run := v8TierRuntimeForTest(t, `function g(x){`+body(16)+`return x}
		function big(x){`+body(60)+`return x}
		function h(n,k){let t=0;for(let i=0;i<n;i++){t=(t+i)|0;if(k===1)t=(t+g(i))|0;if(k===2)t=(t+big(i))|0}return t}`)
	closure := func(name string) *closure {
		return r.global.getOwn(r.atoms.intern(name)).value().Object().fn().closure
	}
	g, big, h := closure("g"), closure("big"), closure("h")
	if n := len(g.fn.Code); n <= ssa.MaxInline || n > ssa.MaxInlineTotal || len(big.fn.Code) <= ssa.MaxInlineTotal {
		t.Fatalf("g is %d instructions, big %d", n, len(big.fn.Code))
	}
	for range jitInvocationsForMaglev {
		run(`h(4,0)`)
	}
	he := r.jit.entryFor(h.fn)
	if he == nil || he.ssa == nil {
		t.Fatal("h has no code")
	}
	calls := 0
	for ; calls < jitInvocationsForMaglev/4; calls++ {
		if e := r.jit.entryFor(g.fn); e != nil && e.ssa != nil {
			break
		}
		run(`h(1,1)`)
	}
	if calls < jitInvocationsForFeedback || calls >= jitInvocationsForMaglev/4 {
		t.Fatalf("g compiled at its %d. call", calls)
	}
	// h's code is compiled again for the call once it has spent a budget
	// (jitLearnedDue).
	for range 2 * jitInvocationsForMaglev {
		run(`h(1,1)`)
	}
	ge := r.jit.entryFor(g.fn)
	in := ge.link.in
	run(`h(100,1)`)
	if he = r.jit.entryFor(h.fn); !slices.ContainsFunc(he.nativeCalls, func(x jitInline) bool { return x.cl == g }) || ge.link.in-in < 100 {
		t.Fatalf("h's code called g natively %d times of 100", ge.link.in-in)
	}
	rb, runb := v8TierRuntimeForTest(t, `function big(x){`+body(60)+`return x}
		function h(n,k){let t=0;for(let i=0;i<n;i++){t=(t+i)|0;if(k)t=(t+big(i))|0}return t}`)
	big = rb.global.getOwn(rb.atoms.intern("big")).value().Object().fn().closure
	for range jitInvocationsForMaglev {
		runb(`h(4,0)`)
	}
	for i := range jitInvocationsForMaglev / 2 {
		runb(`h(1,1)`)
		if e := rb.jit.entryFor(big.fn); e != nil && e.ssa != nil {
			t.Fatalf("big compiled at its %d. call", i+1)
		}
	}
	r2, run2 := v8TierRuntimeForTest(t, `function g(x){`+body(16)+`return x}
		function w(n){let t=0;for(let i=0;i<n;i++){t=(t*3+i)|0;if(i%10===0)t=(t+g(i))|0}return t}`)
	g2 := r2.global.getOwn(r2.atoms.intern("g")).value().Object().fn().closure
	run2(fmt.Sprintf(`w(%d)`, 64*backEdgeCheckInterval))
	if e := r2.jit.entryFor(g2.fn); e == nil || e.ssa == nil || e.invoked >= jitInvocationsForMaglev || e.link.in == 0 {
		t.Fatalf("g, called at every tenth iteration of w's loop, not compiled for w: %+v", e)
	}
}

// Code Go does not enter at its start for its short stretches (entrySlow)
// is still entered by on-stack replacement at a loop that has run a full
// budget of back edges in the existing tiers, as V8's code is: the
// judgement is of the calls Go enters it at, short ones, and a long
// invocation, whose loop proves it long, goes on natively. sum, judged
// slow, runs a call of a million iterations natively from its loop.
// DeltaBlue's forEach, judged slow on its many short calls, had run its
// plan's, thousands of constraints long, in the interpreter.
func TestJITSlowCodeEnteredByOSR(t *testing.T) {
	t.Parallel()
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	r, run := jitTierRuntimeForTest(t, 1, `function sum(n){let s=0;for(let i=0;i<n;i++)s=(s+i)|0;return s}`)
	run(`sum(10)`)
	e := jitNativeForTest(t, r, "sum")
	e.entrySlow = true
	starts, osrs, entries := e.starts, r.jit.osrs, e.ssaStats.entries
	run(`sum(3)`)
	if e.starts != starts || r.jit.osrs != osrs {
		t.Fatal("Go entered sum, judged slow, at a short call")
	}
	run(`sum(1000000)`)
	if r.jit.osrs == osrs || e.ssaStats.entries == entries {
		t.Fatalf("sum's long loop was not entered by on-stack replacement: %d tries at loops, %d native entries", r.jit.osrs-osrs, e.ssaStats.entries-entries)
	}
}

// test262's buildString, under V8's budgets: codePoints.length = 0, at a
// megamorphic site, between the chunks it appends to result. A write
// there that Go made on the store's miss path, which native code's
// analyses took for no call, gave a wrong string, and crashed. Each answer
// is the interpreter's.
func TestJITMegamorphicWriteKeeps(t *testing.T) {
	t.Parallel()
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	const src = `
		function buildString(args) {
			const loneCodePoints = args.loneCodePoints
			const ranges = args.ranges
			const CHUNK_SIZE = 10000
			let result = String.fromCodePoint.apply(null, loneCodePoints)
			for (let i = 0; i < ranges.length; i++) {
				let range = ranges[i]
				let start = range[0]
				let end = range[1]
				let codePoints = []
				for (let length = 0, codePoint = start; codePoint <= end; codePoint++) {
					codePoints[length++] = codePoint
					if (length === CHUNK_SIZE) {
						result += String.fromCodePoint.apply(null, codePoints)
						codePoints.length = length = 0
					}
				}
				result += String.fromCodePoint.apply(null, codePoints)
			}
			return result
		}
		const s = buildString({ loneCodePoints: [], ranges: [[0xDC00, 0xDFFF], [0x80, 0xDBFF], [0xE000, 0x10FFFF]] })
		s.length + ':' + s.charCodeAt(s.length >> 1) + ':' + s.charCodeAt(s.length - 1)`
	want := New(Config{})
	t.Cleanup(func() { want.Close(); want.ReleaseClosed() })
	wv, err := want.Run(compileForTest(t, src))
	if err != nil {
		t.Fatal(err)
	}
	r := New(Config{JIT: true})
	t.Cleanup(func() { r.Close(); r.ReleaseClosed() })
	gv, err := r.Run(compileForTest(t, src))
	if err != nil {
		t.Fatal(err)
	}
	if got, w := gv.String().Go(), wv.String().Go(); got != w {
		t.Fatalf("got %s, interpreter %s", got, w)
	}
}

// jitSetPropPC is the PC of fn's one set_prop.
func jitSetPropPC(t *testing.T, fn *bytecode.Function) int {
	t.Helper()
	for pc, in := range fn.Code {
		if in.Op == bytecode.OpSetProp {
			return pc
		}
	}
	t.Fatal("no set_prop")
	return -1
}

// jitGetPropPC is the PC of fn's first get_prop.
func jitGetPropPC(t *testing.T, fn *bytecode.Function) int {
	t.Helper()
	for pc, in := range fn.Code {
		if in.Op == bytecode.OpGetProp {
			return pc
		}
	}
	t.Fatal("no get_prop")
	return -1
}

// An inlined callee's array and object literals are made natively, from
// pools the code inlining it keeps (jitFeedback's Literal), as its own
// are: reset, inlined into each, makes an empty array and an empty object
// at every call, and each, native, leaves only for the pools to be filled
// again. Inlined, they had been Go's, at every one:
// Typescript's errorSink = [] and commentStack = [], fourteen thousand
// times in five runs. Each answer is the interpreter's.
func TestJITInlinedLiterals(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	runtime.GC()
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	r, run := jitTierRuntimeForTest(t, 1, `function Sink(){this.errors=[];this.box={}}
		Sink.prototype.reset=function(){this.errors=[];this.box={};return this.errors};
		var s=new Sink;
		function warm(n){let t=0;for(let i=0;i<n;i++)t+=s.reset().length;return t}
		function each(n){let t=0;for(let i=0;i<n;i++){const a=s.reset();a.push(i);s.box.v=i;t=(t+a.length+s.errors.length+s.box.v)|0}return t}`)
	for range 4 {
		run(`warm(64)`)
		run(`each(64)`)
	}
	e := jitNativeForTest(t, r, "each")
	reset := r.global.getOwn(r.atoms.intern("s")).value().Object().proto().getOwn(r.atoms.intern("reset")).value().Object().fn().closure
	if !slices.Contains(e.ssaInlined, reset) {
		t.Fatal("each does not inline reset: no test of an inlined callee's literals")
	}
	hosts := e.ssaStats.hosts
	run(`each(600)`)
	if n := e.ssaStats.hosts - hosts; n > 2*600/abi.PoolCapacity+4 {
		t.Errorf("each left native code %d times", n)
	}
}
