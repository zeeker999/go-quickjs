//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"fmt"
	"runtime"
	"strings"
	"testing"
	"time"

	jitcompile "github.com/go-quickjs/go-quickjs/internal/jit/compile"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
	"github.com/go-quickjs/go-quickjs/internal/jit/mir"
	"github.com/go-quickjs/go-quickjs/internal/jit/ssa"
)

// linearBlocksSource is a loop of n values, each written in a block of
// its own, an if's, and all of them live across every block: n values
// live across 2n ends of branches.
func linearBlocksSource(name string, n int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "function %s(a, rounds) {\nvar s = 0;\n", name)
	for i := range n {
		fmt.Fprintf(&b, "var v%d = a + %d;\n", i, i)
	}
	b.WriteString("for (var r = 0; r < rounds; r++) {\n")
	for i := range n {
		fmt.Fprintf(&b, "if (a > %d) v%d = v%d * 0.5 + r;\n", i%7, i, i)
	}
	b.WriteString("}\n")
	for i := range n {
		fmt.Fprintf(&b, "s = s + v%d;\n", i)
	}
	b.WriteString("return s;\n}\n")
	return b.String()
}

// linearGuardsSource is n doubles live at once, each multiplied by an
// element of an array whose read a guard checks: n exits that survive
// Optimize, each naming the doubles still to be read.
func linearGuardsSource(name string, n int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "var ARR = []; for (var k = 0; k < %d; k++) ARR.push(k * 0.5);\n", n)
	fmt.Fprintf(&b, "function %s(a, rounds) {\nvar t = 0, arr = ARR;\nfor (var r = 0; r < rounds; r++) {\n", name)
	for i := range n {
		fmt.Fprintf(&b, "var d%d = a * %d + r;\n", i, i+3)
	}
	for i := range n {
		fmt.Fprintf(&b, "t = t + arr[%d] * d%d;\n", i, i)
	}
	b.WriteString("}\nreturn t;\n}\n")
	return b.String()
}

// linearCompile is what compiling function name of src, from its register
// code, takes: the bytes allocated and the time of its lowering, its build
// and Optimize, and its compile for the host, the allocation check off, as
// production's compiles are; with what the program keeps of its liveness
// and the code of its exits' descriptions.
type linearCompile struct {
	lower, build, code uint64
	took               time.Duration
	live, exits        int
}

func linearCompileOf(t *testing.T, src, name string, args func(*Runtime) []Value) linearCompile {
	t.Helper()
	rts := regNativeRuntimes(t, src)
	r := rts[0]
	r.jitMemoryLimit = 1 << 36
	spillMoreCollect(r)
	fn, _ := r.GetProp(Obj(r.Global()), r.Intern(name))
	r.RegNativeCall(fn, args(r), false)
	cl := fn.Object().fn().closure
	e := r.jit.regCache[r.jitRegKeyOf(cl)]
	if e == nil || e.p == nil {
		t.Fatalf("%s: not lowered", name)
	}
	var ms runtime.MemStats
	alloc := func() uint64 {
		runtime.ReadMemStats(&ms)
		return ms.TotalAlloc
	}
	// The least of three lowerings: what the lowering keeps for the next
	// one (its scratch's pool) is there or not as collections run.
	var c linearCompile
	var p *ir.Program
	start := time.Now()
	for k := range 3 {
		a0 := alloc()
		q, err := jitcompile.LowerReg(cl.reg.rc, cl.fn)
		if err != nil {
			t.Fatal(err)
		}
		if a := alloc() - a0; k == 0 || a < c.lower {
			c.lower = a
		}
		p = q
	}
	a1 := alloc()
	f, err := ssa.BuildWith(e.p, r.jitRegFeedbackFor(cl, e.p))
	if err != nil {
		t.Fatal(err)
	}
	jitRegShape(f, e.p, cl, cl.reg)
	ssa.Optimize(f)
	a2 := alloc()
	opts := mir.Settings()
	opts.CheckAllocations = false
	mc, err := mir.CompileWith(new(mir.Workspace), f, jitEncoding, opts)
	if err != nil {
		t.Fatal(err)
	}
	a3 := alloc()
	c.took = time.Since(start)
	c.build, c.code = a2-a1, a3-a2
	c.live = p.Live.Size()
	if len(mc.Exits) > 0 && mc.Exits[0].Frame != nil {
		x := mc.Exits[0].Frame
		c.exits = len(x.Tree)*4 + len(x.Slots)*28
	}
	return c
}

// A function's compile takes memory linear in its size, however its values
// are live: n values live across n ends of branches (blocks), whose first
// states had read every live value and made a phi of each at every one;
// n values live at once with an exit after each of n reads that survives
// Optimize (guards), whose exits' descriptions had each listed every
// value still live; and n values live at once past every register
// (spill). Each of the lowering -- its liveness had been a row of every
// register at every instruction -- the build and Optimize, and the
// compile for the host takes some eight times the memory for eight times
// the values, not sixty-four, and what the program keeps of its liveness
// and the code keeps of its exits' descriptions grows as much: up to
// twice that, where the shared sets' trees take a level more, for each
// sixteen times the slots (ir.Liveness, ssa's slots.go).
func TestJITCompileLinear(t *testing.T) {
	if !jitSSABackend || testing.Short() {
		t.Skip("no SSA backend on this architecture, or short")
	}
	plain := func(*Runtime) []Value { return []Value{Float(3), Float(2)} }
	shapes := []struct {
		name string
		src  func(string, int) string
		args func(r *Runtime) []Value
	}{
		{"blocks", linearBlocksSource, plain},
		{"guards", linearGuardsSource, plain},
		{"spill", func(name string, n int) string { return spillMoreHelpers + spillMoreSource(name, n, 150) }, func(r *Runtime) []Value {
			return []Value{Float(1.25), Float(0.5), Float(7), spillMoreObject(r, 0), Str(NewString("pp")), Float(2)}
		}},
	}
	const small, big = 1000, 8000
	for _, sh := range shapes {
		t.Run(sh.name, func(t *testing.T) {
			var cs [2]linearCompile
			for k, n := range []int{small, big} {
				name := fmt.Sprintf("%s%d", sh.name, n)
				cs[k] = linearCompileOf(t, sh.src(name, n), name, sh.args)
			}
			s, b := cs[0], cs[1]
			t.Logf("%d values: lower %.1f MB, build %.1f MB, code %.1f MB, %v, liveness %d bytes, exits %d bytes",
				small, float64(s.lower)/(1<<20), float64(s.build)/(1<<20), float64(s.code)/(1<<20), s.took.Round(time.Millisecond), s.live, s.exits)
			t.Logf("%d values: lower %.1f MB, build %.1f MB, code %.1f MB, %v, liveness %d bytes, exits %d bytes",
				big, float64(b.lower)/(1<<20), float64(b.build)/(1<<20), float64(b.code)/(1<<20), b.took.Round(time.Millisecond), b.live, b.exits)
			for _, c := range []struct {
				what       string
				small, big uint64
			}{
				{"the lowering", s.lower, b.lower},
				{"the build", s.build, b.build},
				{"the compile", s.code, b.code},
				{"the liveness", uint64(s.live), uint64(b.live)},
				{"the exits' descriptions", uint64(s.exits), uint64(b.exits)},
			} {
				if c.big > 16*c.small {
					t.Errorf("%s: eight times the values took %.1f times the memory", c.what, float64(c.big)/float64(c.small))
				}
			}
		})
	}
}
