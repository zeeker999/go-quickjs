package compile

import (
	"strconv"
	"strings"
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
)

// The lowering's asserts for the reserved operations, which nothing emits
// yet: a step-and-branch without its partner at pc+1, and an int32 form
// without its | 0's position, are internal errors before anything is
// lowered; one that is well formed is refused as an operation with no
// lowering, until it has one.
func TestLowerRegReservedAsserts(t *testing.T) {
	fns := regCompiled(t, `function up(n) { for (let i = 0; i < n; i++) {} }
		function arith(x, y) { return x * y }`)
	lowerWith := func(fn *bytecode.Function, pc int, w bytecode.RInstr, aux []uint32) error {
		rc := *fn.RegCode()
		rc.Code = append([]bytecode.RInstr(nil), rc.Code...)
		rc.Code[pc] = w
		rc.Aux = aux
		_, err := LowerReg(&rc, fn)
		return err
	}
	find := func(fn *bytecode.Function, op, next bytecode.ROp) int {
		code := fn.RegCode().Code
		for pc := 0; pc+1 < len(code); pc++ {
			if code[pc].Op() == op && code[pc+1].Op() == next {
				return pc
			}
		}
		t.Fatalf("%s: no %s, %s", fn.Name, op, next)
		return 0
	}
	want := func(err error, msg string) {
		t.Helper()
		if err == nil || !strings.Contains(err.Error(), msg) {
			t.Errorf("lowered with %v, want an error of %q", err, msg)
		}
	}

	up := fns["up"]
	pc := find(up, bytecode.RInc, bytecode.RJLt)
	p := up.RegCode().Code[pc+1]
	want(lowerWith(up, pc, bytecode.MakeR(bytecode.RStepJLt, 1, p.A(), p.B(), 0).WithSC(p.SC()+1), nil),
		"no register lowering for step_jlt")
	want(lowerWith(up, pc, bytecode.MakeR(bytecode.RStepJLt, 1, p.A(), p.B(), 0).WithSC(p.SC()), nil),
		"step_jlt at "+strconv.Itoa(pc)+" without its partner")

	arith := fns["arith"]
	pc = find(arith, bytecode.RMul, bytecode.RRet)
	m := arith.RegCode().Code[pc]
	aux := make([]uint32, len(arith.RegCode().Code))
	add := bytecode.MakeR(bytecode.RAddI32, 0, m.A(), m.B(), m.C())
	want(lowerWith(arith, pc, add, aux), "no register lowering for add_i32")
	want(lowerWith(arith, pc, add, nil), "add_i32 at "+strconv.Itoa(pc)+" without its | 0's position")
	want(lowerWith(arith, pc, bytecode.MakeR(bytecode.RMulI, 0, m.A(), m.B(), 0).WithSC(3), nil),
		"no register lowering for mul_imm")
}
