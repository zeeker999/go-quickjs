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

func TestJITStrAppendDeclaration(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprint(marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			r := jitRuntimeForTest(t, Config{JIT: true})
			r.SetJITWarmUp(1, 1)
			v, err := r.Run(compileForTest(t, strAppendDeclarationSrc))
			if err != nil {
				t.Fatal(err)
			}
			if got := v.String().Go(); got != "abcdY" {
				t.Errorf("got %q, want abcdY", got)
			}
			if r.JITStats().SSAEntries == 0 {
				t.Fatal("no native entry")
			}
		})
	}
}

// Native code's appends make no node each: a run of them is one string and its
// right side.
func TestJITStrAppendTailGrowth(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	const src = `function growTail() {
		var s = "a".repeat(70)
		const add = x => { s += x }
		for (let i = 0; i < 10000; i++) add("\u00e9x")
		return s
	}
	growTail()`
	want := strings.Repeat("a", 70) + strings.Repeat("\u00e9x", 10000)
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking %v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			r := jitRuntimeForTest(t, Config{JIT: true})
			r.jitStress = jitStressConfig{threshold: true, budget: 3, deopt: 7}
			v, err := r.Run(compileForTest(t, src))
			if err != nil {
				t.Fatal(err)
			}
			if !v.IsString() || v.String().Go() != want {
				t.Fatal("growing the tail changed its text")
			}
			if r.JITStats().SSAEntries == 0 {
				t.Fatal("no native entry")
			}
		})
	}
}

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
	// A number is made a string where it is added, with no exit to Go.
	if _, err := r.Run(compileForTest(t, `
		function numbers(n) {
			var s = ""
			var sink = { push: function (x) { s += x } }
			for (var i = 0; i < n; i++) sink.push(i & 7)
			return s.length
		}
		for (var k = 0; k < 40; k++) numbers(5000)`)); err != nil {
		t.Fatal(err)
	}
	before := r.JITStats()
	if v, err := r.Run(compileForTest(t, `numbers(10000)`)); err != nil || v.Number() != 10000 {
		t.Fatalf("= %v, %v", v, err)
	}
	after := r.JITStats()
	if hosts, entries := after.Hosts-before.Hosts, after.SSAEntries-before.SSAEntries; hosts > 16 || entries == 0 || entries > 16 {
		t.Errorf("10,000 numbers appended: %d exits to Go, %d native entries", hosts, entries)
	}
}

// A pointer word stored natively while the collector marks, each way there is
// of it -- nothing over nothing, which the store routine goes straight back
// from, a pointer over nothing, nothing over a pointer, a pointer over a
// pointer -- in a property, an element and a literal, with every register
// native code keeps given junk across the stores that reach Go. Each answer
// is the interpreter's.
func TestJITStoreWords(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	const src = `
		function step(o, a, i, box, other) {
			o.p = i
			o.q = i & 1 ? box : i
			o.r = i & 2 ? null : i
			o.s = i & 1 ? box : other
			a[0] = i
			a[1] = i & 1 ? box : i
			a[2] = i & 1 ? other : box
			var lit = { x: i, y: box, z: i + 1, w: 's' }
			return o.p + (typeof o.q === 'object' ? o.q.v : o.q) + (o.r === null ? 1 : o.r) + o.s.v +
				a[0] + (typeof a[1] === 'object' ? a[1].v : a[1]) + a[2].v + lit.x + lit.y.v + lit.z + lit.w.length
		}
		function run(n) {
			var o = { p: 0, q: 0, r: 0, s: null }, a = [0, 0, 0], box = { v: 3 }, other = { v: 5 }, s = 0
			for (var i = 0; i < n; i++) s = (s + step(o, a, i, box, other)) | 0
			last = o
			return s
		}
		var got = [], last
		for (var k = 0; k < 40; k++) got.push(run(2001))
		got.join()`
	r := New(Config{})
	defer r.Close()
	v, err := r.Run(compileForTest(t, src))
	if err != nil {
		t.Fatal(err)
	}
	want := v.String().Go()
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking %v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
				jitClobberStoresForTest(t)
			}
			r := jitRuntimeForTest(t, Config{JIT: true})
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
			// The last store over a pointer was a number's: the pointer's word
			// went, which no script sees, and with it the barrier that shades
			// what it held.
			o := r.global.getOwn(r.atoms.intern("last")).value().Object()
			if q := o.getOwn(r.atoms.intern("q")).value(); !q.IsNumber() || q.ref != nil {
				t.Errorf("a number over a pointer left its pointer word: %v, %v", q.num, q.ref)
			}
		})
	}
}
