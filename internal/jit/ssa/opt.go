package ssa

import (
	"math"

	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

// Optimize runs Phase 2's passes until nothing changes:
//   - trivial phis (every argument the same value, or the phi itself) become
//     that value;
//   - a guard that cannot fail goes: unboxing a boxed number or a numeric
//     constant, truthiness of a boxed boolean, an initialization check of a
//     box or of a constant that is not the uninitialized marker;
//   - ToInt32 of an integer converted to a number is that integer, and a
//     double Not is its operand;
//   - a pure numeric operation on constants is folded, by the same apply the
//     evaluator uses;
//   - within a block, a guard repeating an earlier one on the same value is
//     the earlier one;
//   - values nothing uses are deleted. A guard that can fail is never
//     deleted: it is an effect.
func Optimize(f *Func) {
	for round := 0; round < 32; round++ {
		subst := map[*Value]*Value{}
		find := func(v *Value) *Value {
			for {
				w, ok := subst[v]
				if !ok {
					return v
				}
				v = w
			}
		}
		changed := false
		for _, b := range f.Blocks {
			seen := map[[2]int]*Value{}
			for _, v := range b.Values {
				if _, gone := subst[v]; gone {
					continue
				}
				for i, a := range v.Args {
					v.Args[i] = find(a)
				}
				if w := simplify(f, v); w != nil {
					subst[v] = w
					changed = true
					continue
				}
				if v.Op.isGuard() && v.Op != OpCheckInit {
					key := [2]int{int(v.Op), v.Args[0].ID}
					if first, ok := seen[key]; ok {
						subst[v] = first
						changed = true
						continue
					}
					seen[key] = v
				}
			}
		}
		if len(subst) > 0 {
			rewrite(f, find)
		}
		removed := removeDead(f)
		if !changed && !removed {
			break
		}
	}
	recount(f)
}

// simplify returns a value v can be replaced by, or nil. It may instead
// rewrite v in place (into a constant, or into a guard that cannot fail,
// which removeDead then deletes).
func simplify(f *Func, v *Value) *Value {
	arg := func(i int) *Value { return v.Args[i] }
	switch v.Op {
	case OpPhi:
		var same *Value
		for _, a := range v.Args {
			if a == v || a == same {
				continue
			}
			if same != nil {
				return nil
			}
			same = a
		}
		return same
	case OpUnboxF64:
		switch a := arg(0); {
		case a.Op == OpBoxF64:
			return a.Args[0]
		case a.Op == OpConst && a.Const.Kind == ir.Number:
			v.Op, v.Const, v.Args, v.State = OpConstF64, a.Const, nil, nil
		}
	case OpTruth:
		if a := arg(0); a.Op == OpBoxBool {
			return a.Args[0]
		}
	case OpCheckInit:
		a := arg(0)
		if a.Op == OpBoxF64 || a.Op == OpBoxBool || a.Op == OpConst && a.Const.Kind != ir.Uninitialized {
			v.Op, v.Args, v.State = OpInvalid, nil, nil
		}
	case OpToInt32:
		if a := arg(0); a.Op == OpI32ToF64 || a.Op == OpU32ToF64 {
			return a.Args[0]
		}
	case OpNot:
		if a := arg(0); a.Op == OpNot {
			return a.Args[0]
		}
	case OpAddF64, OpSubF64, OpMulF64, OpDivF64, OpNegF64:
		for _, a := range v.Args {
			if a.Op != OpConstF64 {
				return nil
			}
		}
		var x, y val
		x.f = math.Float64frombits(arg(0).Const.Bits)
		if len(v.Args) > 1 {
			y.f = math.Float64frombits(arg(1).Const.Bits)
		}
		r := apply(v.Op, v.Aux, x, y)
		v.Op, v.Const, v.Args = OpConstF64, ir.Value{Kind: ir.Number, Bits: math.Float64bits(r.f)}, nil
	}
	return nil
}

// rewrite replaces every use of a value with find's answer for it.
func rewrite(f *Func, find func(*Value) *Value) {
	state := func(s *FrameState) {
		if s == nil {
			return
		}
		for i, v := range s.Slots {
			s.Slots[i] = find(v)
		}
	}
	for _, b := range f.Blocks {
		kept := b.Values[:0]
		for _, v := range b.Values {
			if find(v) != v {
				continue
			}
			for i, a := range v.Args {
				v.Args[i] = find(a)
			}
			state(v.State)
			kept = append(kept, v)
		}
		b.Values = kept
		if b.Control != nil {
			b.Control = find(b.Control)
		}
		state(b.State)
		state(b.Header)
	}
}

// removeDead deletes values nothing live uses. Guards that can fail,
// controls, and the frame states of exits, guards and loop headers are what
// keeps values live.
func removeDead(f *Func) bool {
	live := map[*Value]bool{}
	var work []*Value
	mark := func(v *Value) {
		if v != nil && !live[v] {
			live[v] = true
			work = append(work, v)
		}
	}
	markState := func(s *FrameState) {
		if s != nil {
			for _, v := range s.Slots {
				mark(v)
			}
		}
	}
	for _, b := range f.Blocks {
		mark(b.Control)
		markState(b.State)
		markState(b.Header)
		for _, v := range b.Values {
			if v.Op.isGuard() {
				mark(v)
			}
		}
	}
	for len(work) > 0 {
		v := work[len(work)-1]
		work = work[:len(work)-1]
		for _, a := range v.Args {
			mark(a)
		}
		markState(v.State)
	}
	removed := false
	for _, b := range f.Blocks {
		kept := b.Values[:0]
		for _, v := range b.Values {
			if live[v] {
				kept = append(kept, v)
			} else {
				removed = true
			}
		}
		b.Values = kept
	}
	return removed
}

// recount recomputes every value's use count.
func recount(f *Func) {
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			v.Uses = 0
		}
	}
	use := func(v *Value) {
		if v != nil {
			v.Uses++
		}
	}
	state := func(s *FrameState) {
		if s != nil {
			for _, v := range s.Slots {
				use(v)
			}
		}
	}
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			for _, a := range v.Args {
				use(a)
			}
			state(v.State)
		}
		use(b.Control)
		state(b.State)
		state(b.Header)
	}
}
