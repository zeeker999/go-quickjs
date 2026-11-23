package compile

import (
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

func TestLowerRegStoredValues(t *testing.T) {
	fns := regCompiled(t, `
		function property(a, o) { let x = o.x; a[0] = x }
		function numeric(a, o) { a[0] = o.x * 2 }
		function index(a, o) { a[o.x] = 1 }
		function element(a, b) { let x = b[0]; a[0] = x }`)
	for _, tt := range []struct {
		name string
		op   ir.Op
	}{
		{"property", ir.ReferenceRead},
		{"numeric", ir.PropertyRead},
		{"index", ir.PropertyRead},
		{"element", ir.ArrayRead},
	} {
		t.Run(tt.name, func(t *testing.T) {
			fn := fns[tt.name]
			p, err := LowerReg(fn.RegCode(), fn)
			if err != nil {
				t.Fatal(err)
			}
			reads := 0
			for _, in := range p.Code {
				if in.Op != ir.PropertyRead && in.Op != ir.ReferenceRead && in.Op != ir.ArrayRead {
					continue
				}
				reads++
				if in.Op != tt.op || in.Reference {
					t.Errorf("read = %v, reference=%v; want %v, reference=false", in.Op, in.Reference, tt.op)
				}
			}
			if reads != 1 {
				t.Fatalf("got %d reads, want 1", reads)
			}
		})
	}
}
