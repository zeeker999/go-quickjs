package ssa

import (
	"testing"

	jitcompile "github.com/go-quickjs/go-quickjs/internal/jit/compile"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

func TestBuildColdTaggedRead(t *testing.T) {
	fn := compiledFunctionForTest(t, `function f(o,n){let x;for(let i=0;i<n;i++)x=o.x;return x}`)
	p, err := jitcompile.LowerReg(fn.RegCode(), fn)
	if err != nil {
		t.Fatal(err)
	}
	read := -1
	for pc, in := range p.Code {
		if in.Op == ir.ReferenceRead {
			read = pc
		}
	}
	if read < 0 {
		t.Fatal("no tagged read")
	}
	f, err := BuildWith(p, siteFeedback{PropertySite{Key: 1, Scan: true}})
	if err != nil {
		t.Fatal(err)
	}
	if err := Check(f); err != nil {
		t.Fatal(err)
	}
	reads := 0
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op == OpCall {
				t.Fatal("cold read added a call branch")
			}
			if v.Op == OpPropCell {
				reads++
				if ir.ExitKind(v.Aux) != ir.HostExit || !v.Scans() || !v.Megamorphic() {
					t.Fatalf("read exit=%v scan=%v lookup=%v", ir.ExitKind(v.Aux), v.Scans(), v.Megamorphic())
				}
			}
		}
	}
	if reads != 1 {
		t.Fatalf("got %d reads, want 1", reads)
	}
	after := p.At(p.Origin[read] + 1)
	for _, e := range f.Entries {
		if e.PC == read+1 || e.PC == after {
			t.Fatal("cold read added a resumption entry")
		}
	}
}
