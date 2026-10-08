package ssa

import (
	"math"

	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

// val is a value at run time, in the representation its type names.
type val struct {
	t ir.Value // Tagged
	f float64  // Float64
	i uint32   // Int32, as bits
	b bool     // Bool
}

// apply computes a pure, non-guard op. It is the definition of each op's
// semantics, shared by the evaluator and constant folding.
func apply(op Op, aux int, a, b val) val {
	switch op {
	case OpBoxF64:
		return val{t: ir.Value{Kind: ir.Number, Bits: math.Float64bits(a.f)}}
	case OpBoxBool:
		return val{t: ir.Bool(a.b)}
	case OpAddF64:
		return val{f: a.f + b.f}
	case OpSubF64:
		return val{f: a.f - b.f}
	case OpMulF64:
		return val{f: a.f * b.f}
	case OpDivF64:
		return val{f: a.f / b.f}
	case OpNegF64:
		return val{f: math.Float64frombits(math.Float64bits(a.f) ^ 1<<63)}
	case OpCmpF64:
		x, y := a.f, b.f
		switch ir.Operator(aux) {
		case ir.Lt:
			return val{b: x < y}
		case ir.Le:
			return val{b: x <= y}
		case ir.Gt:
			return val{b: x > y}
		case ir.Ge:
			return val{b: x >= y}
		case ir.Eq:
			return val{b: x == y}
		case ir.Ne:
			return val{b: x != y}
		}
	case OpNot:
		return val{b: !a.b}
	case OpToInt32:
		return val{i: ir.ToUint32(a.f)}
	case OpAndI32:
		return val{i: a.i & b.i}
	case OpOrI32:
		return val{i: a.i | b.i}
	case OpXorI32:
		return val{i: a.i ^ b.i}
	case OpShlI32:
		return val{i: a.i << (b.i & 31)}
	case OpSarI32:
		return val{i: uint32(int32(a.i) >> (b.i & 31))}
	case OpShrU32:
		return val{i: a.i >> (b.i & 31)}
	case OpNotI32:
		return val{i: ^a.i}
	case OpI32ToF64:
		return val{f: float64(int32(a.i))}
	case OpU32ToF64:
		return val{f: float64(a.i)}
	}
	panic("ssa: apply " + op.String())
}

// truth is JavaScript's truthiness for the kinds the slot IR decides it for.
func truth(v ir.Value) (bool, bool) {
	switch v.Kind {
	case ir.Undefined, ir.Null:
		return false, true
	case ir.Boolean:
		return v.Bits != 0, true
	case ir.Number:
		n := math.Float64frombits(v.Bits)
		return n != 0 && !math.IsNaN(n), true
	}
	return false, false
}

// Evaluate runs f from the entry for a slot IR PC, on slots (locals, then
// operands, as the slot IR evaluator takes them), and returns the exit the
// slot IR evaluator would: the same kind, state, value and live slots. Steps
// is not counted. With pollEvery > 0, every pollEvery-th backward branch
// exits as a budget exit at its loop header, where the function can be
// entered again.
func Evaluate(f *Func, pc int, slots []ir.Value, pollEvery int) (ir.Exit, error) {
	e, ok := f.EntryFor(pc)
	if !ok || len(slots) != f.Locals+f.StackSize {
		return ir.Exit{}, ir.ErrState
	}
	vals := make([]val, f.nextID)
	// An exit rematerializes boxes and constants from their operands, as
	// generated code does: a loop header's state names boxes of its phis,
	// which a poll exits to before the boxes themselves are computed.
	var slot func(v *Value) ir.Value
	slot = func(v *Value) ir.Value {
		switch v.Op {
		case OpConst:
			return v.Const
		case OpBoxF64, OpBoxBool:
			return apply(v.Op, v.Aux, vals[v.Args[0].ID], val{}).t
		}
		return vals[v.ID].t
	}
	exit := func(s *FrameState, kind ir.ExitKind) (ir.Exit, error) {
		for i, v := range s.Slots {
			slots[i] = slot(v)
		}
		return ir.Exit{Kind: kind, State: ir.StateMap{PC: s.PC, Depth: s.Depth}}, nil
	}
	polls := 0
	blk, from := e.Block, -1
	for {
		// Phis take their arguments from the edge just taken, all at once.
		var phis []val
		for _, v := range blk.Values {
			if v.Op != OpPhi {
				break
			}
			phis = append(phis, vals[v.Args[from].ID])
		}
		for i, v := range blk.Values[:len(phis)] {
			vals[v.ID] = phis[i]
		}
		if from >= 0 && blk.Backedge[from] && pollEvery > 0 {
			polls++
			if polls%pollEvery == 0 {
				return exit(blk.Header, ir.BudgetExit)
			}
		}
		for _, v := range blk.Values[len(phis):] {
			var a, b val
			if len(v.Args) > 0 {
				a = vals[v.Args[0].ID]
			}
			if len(v.Args) > 1 {
				b = vals[v.Args[1].ID]
			}
			switch v.Op {
			case OpLoadSlot:
				vals[v.ID] = val{t: slots[v.Aux]}
			case OpConst:
				vals[v.ID] = val{t: v.Const}
			case OpConstF64:
				vals[v.ID] = val{f: math.Float64frombits(v.Const.Bits)}
			case OpUnboxF64:
				if a.t.Kind != ir.Number {
					return exit(v.State, ir.ExitKind(v.Aux))
				}
				vals[v.ID] = val{f: math.Float64frombits(a.t.Bits)}
			case OpTruth:
				t, ok := truth(a.t)
				if !ok {
					return exit(v.State, ir.ExitKind(v.Aux))
				}
				vals[v.ID] = val{b: t}
			case OpCheckInit:
				if a.t.Kind == ir.Uninitialized {
					return exit(v.State, ir.ExitKind(v.Aux))
				}
			case OpCheckScalar:
				if a.t.Kind == ir.Opaque || a.t.Kind == ir.String {
					return exit(v.State, ir.ExitKind(v.Aux))
				}
			default:
				vals[v.ID] = apply(v.Op, v.Aux, a, b)
			}
		}
		switch blk.Kind {
		case BlockReturn:
			return ir.Exit{Kind: ir.Returned, Value: vals[blk.Control.ID].t}, nil
		case BlockExit:
			return exit(blk.State, blk.ExitKind)
		}
		next := blk.Succs[0]
		if blk.Kind == BlockIf && !vals[blk.Control.ID].b {
			next = blk.Succs[1]
		}
		for i, p := range next.Preds {
			if p == blk {
				from = i
				break
			}
		}
		blk = next
	}
}
