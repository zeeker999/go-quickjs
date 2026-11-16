//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"fmt"
	"slices"
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
)

// Native code counts nothing, as V8's optimized code counts no feedback:
// once caller and c2 are compiled, from their budgets, their invocations
// and their calls -- natively, inlined, or through Go, as c2's of gw,
// which is not compiled -- are not counted, and the counts the existing
// tiers made while they warmed are their code's entries'. Each answer is
// the interpreter's.
func TestJITCountsNotNative(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	r := frequencyRuntime(t, true)
	want := New(Config{})
	t.Cleanup(func() { want.Close(); want.ReleaseClosed() })
	if _, err := want.Run(compileForTest(t, frequencySetup)); err != nil {
		t.Fatal(err)
	}
	both := jitBoth{jit: r, want: want}
	both.script(t, `
		function gw(o) { with (o) { return n } }
		function c2(o, i) { let s = g(o) + gw(o); if (i < 0) s += gw(o); for (let j = 0; j < 40; j++) s = (s * 31 + j) | 0; return s }
		function drive(n) { let s = 0; for (let i = 0; i < n; i++) s = (s + caller(o, i) + c2(o, i)) | 0; return s }`)
	for range 10 {
		both.agree(t, `drive(100)`)
	}
	// Register code's counts are its closures' (jitRegCounts), which
	// the feedback dispatcher counts and native code does not: the
	// counts the stack tiers make by site are theirs alone (their
	// frequencies TestJITRegFrequency's).
	var counts [][]uint64
	for _, name := range []string{"caller", "c2"} {
		e := regEntryNamedForTest(r, name)
		if e == nil || e.invalid {
			t.Fatalf("%s has no code: %+v", name, e)
		}
		c := jitGlobalClosure(r, name).jitRegCounts
		if c == nil || regNativeOn && c[0] == 0 {
			t.Fatalf("%s counts nothing: %v", name, c)
		}
		counts = append(counts, slices.Clone(c))
	}
	goCalls := r.jit.goCalls
	both.agree(t, `drive(1000)`)
	if r.jit.goCalls-goCalls < 1000 {
		t.Errorf("c2's code made %d calls of Go in 1,000 runs", r.jit.goCalls-goCalls)
	}
	for i, name := range []string{"caller", "c2"} {
		if got := jitGlobalClosure(r, name).jitRegCounts; !slices.Equal(got, counts[i]) {
			t.Errorf("%s's code counted: %v, after %v", name, got, counts[i])
		}
	}
	return
	for _, name := range []string{"caller", "c2"} {
		if e := jitCallbackEntry(r, name); e == nil || e.ssa == nil || e.invalid || e.calls == nil {
			t.Fatalf("%s has no code: %+v", name, e)
		}
	}
	// The last run counted may have gone on in native code from its loop,
	// by on-stack replacement, its calls after it uncounted.
	runs, calls := frequencyCounts(t, r, "caller")
	within := func(n, per uint32) bool { return n <= per*runs && n+per >= per*runs }
	if runs < 43 || runs > 600 || !within(calls[0], 3) || !within(calls[1], 1) || calls[2] != 0 || !within(calls[3], 1) || !within(calls[6], 1) {
		t.Fatalf("caller counts %d runs, calls %v", runs, calls)
	}
	runs2, calls2 := frequencyCounts(t, r, "c2")
	if runs2 < 8 || runs2 > 600 || !slices.Equal(calls2, []uint32{runs2, runs2, 0}) {
		t.Fatalf("c2 counts %d runs, calls %v", runs2, calls2)
	}
	goCalls := r.jit.goCalls
	both.agree(t, `drive(1000)`)
	if r.jit.goCalls-goCalls < 1000 {
		t.Errorf("c2's code made %d calls of Go in 1,000 runs", r.jit.goCalls-goCalls)
	}
	for _, c := range []struct {
		name  string
		runs  uint32
		calls []uint32
	}{{"caller", runs, calls}, {"c2", runs2, calls2}} {
		if n, got := frequencyCounts(t, r, c.name); n != c.runs || !slices.Equal(got, c.calls) {
			t.Errorf("%s's code counted: %d runs, %v, after %d, %v", c.name, n, got, c.runs, c.calls)
		}
	}
}

// The counts outlive the code the cache drops, as V8's feedback outlives
// the code optimized from it (jitGhost): caller's, flushed unused, are its
// next entry's, which goes on counting from them.
func TestJITCountsOutliveCode(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	setEpoch := jitFlushForTest(t)
	r := frequencyRuntime(t, true)
	if _, err := r.Run(compileForTest(t, `for (let i = 0; i < 700; i++) caller(o, i)`)); err != nil {
		t.Fatal(err)
	}
	// Register code's counts are its closure's (jitRegCounts), which
	// outlive its code: the code flushed, they go on counting.
	e, code := jitCachedForTest(r, "caller")
	c := jitGlobalClosure(r, "caller").jitRegCounts
	if !code || c == nil {
		t.Fatalf("caller has no code: %+v", e)
	}
	runs := c[0]
	setEpoch(1000 + jitFlushAge)
	r.jit.flush()
	setEpoch(1000 + 2*jitFlushAge)
	r.jit.flush()
	if e, _ := jitCachedForTest(r, "caller"); e != nil {
		t.Fatalf("caller's unused code was not flushed: %+v", e)
	}
	if _, err := r.Run(compileForTest(t, `for (let i = 0; i < 20; i++) caller(o, i)`)); err != nil {
		t.Fatal(err)
	}
	if now := jitGlobalClosure(r, "caller").jitRegCounts; len(now) == 0 || &now[0] != &c[0] || now[0] < runs {
		t.Fatalf("caller counts %v, not %v from %d runs", now, c, runs)
	}
	return
	e := jitCallbackEntry(r, "caller")
	if e == nil || e.ssa == nil || e.calls == nil {
		t.Fatalf("caller has no code: %+v", e)
	}
	c, runs := e.calls, e.calls.runs
	setEpoch(1000 + jitFlushAge)
	r.jit.flush()
	setEpoch(1000 + 2*jitFlushAge)
	r.jit.flush()
	if e := jitCallbackEntry(r, "caller"); e != nil {
		t.Fatalf("caller's unused code was not flushed: %+v", e)
	}
	if _, err := r.Run(compileForTest(t, `for (let i = 0; i < 20; i++) caller(o, i)`)); err != nil {
		t.Fatal(err)
	}
	e = jitCallbackEntry(r, "caller")
	if e == nil || e.calls != c || c.runs < runs {
		t.Fatalf("caller's entry %+v counts %+v, not %+v from %d runs", e, e.calls, c, runs)
	}
}

// jitFnForTest is the function of the global name.
func jitFnForTest(t *testing.T, r *Runtime, name string) *bytecode.Function {
	t.Helper()
	return r.global.getOwn(r.atoms.intern(name)).value().Object().fn().closure.fn
}
