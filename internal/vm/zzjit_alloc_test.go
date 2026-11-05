//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"fmt"
	"runtime"
	"strings"
	"testing"
)

// The tree tier's object literals (zzz_zliteral.go), additions to
// captured bindings (zzz_zupvstr.go), push and pop (pureElemOp, which
// native code's growth of an array calls too), a for...in's cursors used
// again (zzz_forincursor.go) and arrays grown past 16,384 elements
// (zzz_growelems.go) answer as the interpreter
// does with the JIT on, the tree tier warming code up and running what
// native code leaves -- under every stress setting, with the collector
// idle and while it marks -- and native code runs some of each.
func TestJITLiteralsUpvalueAddsPushes(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		// The string doubled past the longest would pass jitDifferential's
		// memory limit first: doubled twelve times.
		upvalues := strings.Replace(upvalueAddTestSrc, "i < 40", "i < 12", 1)
		for name, src := range map[string]string{"literals": literalTestSrc, "upvalue adds": upvalues, "push and pop": pushTestSrc, "for-in cursors": forInCursorTestSrc, "array growth": arrayGrowthTestSrc} {
			t.Run(name, func(t *testing.T) {
				if marking {
					jitMarkingForTest(t)
				}
				if st := jitDifferential(t, src); st.SSAEntries == 0 {
					t.Errorf("marking %v: no native entry", marking)
				}
			})
		}
	}
}

// Appends that grow a string in place answer as Node does natively, under
// every stress setting, with the collector idle and while it marks, and native
// code makes them. Not by jitDifferential: its runtimes have a memory limit,
// to stop a program that grows a string without bound, and a metered
// runtime's appends are ordinary (strAppend).
func TestJITStrAppend(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	src := strings.Replace(strAppendTestSrc, "out.push(tooLong())\n", "", 1)
	want := strAppendTestWant[:strings.LastIndexByte(strAppendTestWant, '|')]
	for _, marking := range []bool{false, true} {
		for _, c := range []jitStressConfig{
			{},
			{threshold: true},
			{threshold: true, budget: 1},
			{threshold: true, budget: 5, deopt: 2},
			{threshold: true, budget: 3, deopt: 7},
		} {
			t.Run(fmt.Sprintf("marking %v, %+v", marking, c), func(t *testing.T) {
				if marking {
					jitMarkingForTest(t)
				}
				r := jitRuntimeForTest(t, Config{JIT: true})
				r.jitStress = c
				v, err := r.Run(compileForTest(t, src))
				if err != nil {
					t.Fatal(err)
				}
				if got := v.String().Go(); got != want {
					t.Errorf("%s", firstDifference(got, want))
				}
				if r.JITStats().SSAEntries == 0 {
					t.Errorf("no native entry")
				}
			})
		}
	}
}

// Native code's appends make no node each: a run of them is one string and its
// right side.
func TestJITStrAppendAllocations(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	r := jitRuntimeForTest(t, Config{JIT: true})
	if _, err := r.Run(compileForTest(t, `
		function appends(n) {
			var s = ""
			var sink = { push: function (x) { s += x } }
			for (var i = 0; i < n; i++) sink.push("<x>")
			return s.length
		}
		for (var k = 0; k < 40; k++) appends(5000)`)); err != nil {
		t.Fatal(err)
	}
	if r.JITStats().SSAEntries == 0 {
		t.Fatal("no native entry")
	}
	var a, b runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&a)
	v, err := r.Run(compileForTest(t, `appends(10000)`))
	runtime.ReadMemStats(&b)
	if err != nil || v.Number() != 30000 {
		t.Fatalf("= %v, %v", v, err)
	}
	if n := b.Mallocs - a.Mallocs; n > 1000 && allocsOrdinary() {
		t.Errorf("10,000 appends made %d allocations", n)
	}
}
