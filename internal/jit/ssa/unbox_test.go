package ssa

import (
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/compiler"
	jitcompile "github.com/go-quickjs/go-quickjs/internal/jit/compile"
	"github.com/go-quickjs/go-quickjs/internal/parser"
)

func lowerJS(t *testing.T, src string) *Func {
	t.Helper()
	prog, err := parser.Parse(src, parser.Options{})
	if err != nil {
		t.Fatal(err)
	}
	top, err := compiler.Compile(prog, compiler.Options{})
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range top.Constants {
		if c.Kind == bytecode.ConstFunction {
			p, err := jitcompile.Lower(c.Fn)
			if err != nil {
				t.Fatal(err)
			}
			f, err := Build(p)
			if err != nil {
				t.Fatal(err)
			}
			return f
		}
	}
	t.Fatal("no function")
	return nil
}

// A numeric loop's variables stay unboxed: its phis are Float64, and the only
// tag checks left are at entries, where a slot first comes in.
func TestOptimizeKeepsLoopsUnboxed(t *testing.T) {
	f := lowerJS(t, `function sum(n){let s=0;for(let i=0;i<n;i++)s+=i;return s}`)
	Optimize(f)
	if err := Check(f); err != nil {
		t.Fatal(err)
	}
	tagged, float := 0, 0
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op == OpPhi && v.Type == Tagged {
				tagged++
			}
			if v.Op == OpPhi && v.Type == Float64 {
				float++
			}
			if v.Op == OpUnboxF64 && b.PC >= 0 && b.LoopHeader {
				t.Errorf("a tag check in the loop header b%d: %v", b.ID, v)
			}
		}
	}
	if tagged != 0 || float < 2 {
		t.Fatalf("%d tagged and %d float phis, want 0 and at least 2:\n%s", tagged, float, f)
	}
}

// An entry loads only the locals the interpreter may read from there before
// it writes them, as frame states name only those: the loop's entry, at its
// header, loads n, s and i, not b, which is written after the loop first.
// What runs from each entry is still the slot IR's (TestBuildMatchesSlotIR).
func TestEntriesLoadLiveLocals(t *testing.T) {
	prog, err := parser.Parse(`function f(n){let s=0;for(let i=0;i<n;i++)s=(s+i)|0;let b=s+1;return b}`, parser.Options{})
	if err != nil {
		t.Fatal(err)
	}
	top, err := compiler.Compile(prog, compiler.Options{})
	if err != nil {
		t.Fatal(err)
	}
	var fn *bytecode.Function
	for _, c := range top.Constants {
		if c.Kind == bytecode.ConstFunction {
			fn = c.Fn
		}
	}
	p, err := jitcompile.LowerSSA(fn)
	if err != nil {
		t.Fatal(err)
	}
	f, err := Build(p)
	if err != nil {
		t.Fatal(err)
	}
	if err := Check(f); err != nil {
		t.Fatal(err)
	}
	b := -1
	for i, l := range fn.Locals {
		if l.Name == "b" {
			b = i
		}
	}
	loops := 0
	for _, e := range f.Entries {
		if e.PC != 0 {
			loops++
		}
		for _, v := range e.Block.Values {
			if v.Op == OpLoadSlot && v.Aux < p.LiveLocals && !p.LiveAt(e.PC, v.Aux) {
				t.Errorf("the entry at pc %d loads local %d, dead there:\n%s", e.PC, v.Aux, f)
			}
			if v.Op == OpLoadSlot && e.PC != 0 && v.Aux == b {
				t.Errorf("the loop's entry at pc %d loads b:\n%s", e.PC, f)
			}
		}
	}
	if loops == 0 {
		t.Fatalf("no entry but the function's:\n%s", f)
	}
}

// Build leaves the phis' shadows and the stores' checks to Optimize, which
// would make them again for what it keeps; Finish makes them, once, for
// whatever runs or compiles a Func Optimize has not.
func TestShadowsMadeOnce(t *testing.T) {
	const src = `function f(a,b,c){let t=c?a:b;return t}`
	shadows := func(f *Func) (n int) {
		for _, b := range f.Blocks {
			for _, v := range b.Values {
				if v.Op == OpPhi && v.Type == Source {
					n++
				}
			}
		}
		return n
	}
	f := lowerJS(t, src)
	if n := shadows(f); n != 0 {
		t.Fatalf("Build made %d shadows:\n%s", n, f)
	}
	f.Finish()
	n := shadows(f)
	if err := Check(f); n == 0 || err != nil {
		t.Fatalf("Finish made %d shadows (%v):\n%s", n, err, f)
	}
	f.Finish()
	if again := shadows(f); again != n {
		t.Fatalf("a second Finish made %d shadows, not %d:\n%s", again, n, f)
	}
	g := lowerJS(t, src)
	Optimize(g)
	Optimize(f)
	if err := Check(g); shadows(g) == 0 || shadows(f) != shadows(g) || err != nil {
		t.Fatalf("Optimize made %d shadows alone, %d after Finish (%v):\n%s", shadows(g), shadows(f), err, g)
	}
}

// Optimize unboxes phis at the start of a round, so that the boxes it
// makes cancel in the same round's walk: a function is mostly done in two
// rounds, the second finding nothing, not three. The corpus takes 25
// rounds, 34 the other way.
func TestOptimizeRounds(t *testing.T) {
	rounds := 0
	for _, src := range jsCorpus {
		f := lowerJS(t, src)
		Optimize(f)
		rounds += f.rounds
	}
	if rounds > 2*len(jsCorpus)+2 {
		t.Errorf("%d rounds for %d functions", rounds, len(jsCorpus))
	}
}
