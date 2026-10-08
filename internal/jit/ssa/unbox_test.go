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
