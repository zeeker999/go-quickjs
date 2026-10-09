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
	// Shadows are remade at the end, for the phis that are left.
	clearShadows(f)
	// Tables are by value ID, not maps: a compile at run time pays for every
	// pass (BenchmarkJITCompile in internal/vm).
	var subst []*Value
	seen := map[[2]int]*Value{}
	for round := 0; round < 32; round++ {
		subst = idTable(subst, f.nextID)
		replaced := false
		find := func(v *Value) *Value {
			for v.ID < len(subst) && subst[v.ID] != nil {
				v = subst[v.ID]
			}
			return v
		}
		changed := false
		for _, b := range f.Blocks {
			clear(seen)
			for _, v := range b.Values {
				if subst[v.ID] != nil {
					continue
				}
				for i, a := range v.Args {
					v.Args[i] = find(a)
				}
				if w := simplify(f, v); w != nil {
					subst[v.ID], replaced = w, true
					changed = true
					continue
				}
				if v.Op.isGuard() && v.Op != OpCheckInit && !v.Op.readsMemory() {
					key := [2]int{int(v.Op), v.Args[0].ID}
					if first, ok := seen[key]; ok {
						subst[v.ID], replaced = first, true
						changed = true
						continue
					}
					seen[key] = v
				}
			}
		}
		if replaced {
			rewrite(f, find)
		}
		if unboxPhis(f) {
			changed = true
		}
		removed := removeDead(f)
		if !changed && !removed {
			break
		}
	}
	shadowMerges(f)
	recount(f)
}

// idTable returns t cleared and n long, reusing its storage.
func idTable[T any](t []T, n int) []T {
	if cap(t) < n {
		return make([]T, n)
	}
	t = t[:n]
	clear(t)
	return t
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

// unboxPhis gives a phi that only ever carries numbers a Float64 phi, so a
// loop's numbers stay unboxed. A candidate's arguments are boxed numbers,
// numeric constants, other candidates, or slots loaded at an entry; a loaded
// slot is unboxed by a guard in its entry block, which exits there if the
// slot is not a number (a speculation, as Phase 3 makes more of). A candidate
// needs a reason: a use that unboxes it, or a candidate with one that it
// flows into. A phi that only carries a number -- x=i, returned -- gains
// nothing unboxed, and its entry guard would exit every time the slot it is
// loaded from holds anything else, such as the reference x held before the
// loop. The old phi becomes a box of the new one, which other passes then
// cancel.
func unboxPhis(f *Func) bool {
	// By value ID: n bounds the IDs of the values there are now; the values
	// this pass makes are numbered from n.
	n := f.nextID
	var phis []*Value // the candidates, in block order
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op == OpPhi && v.Type == Tagged {
				phis = append(phis, v)
			}
		}
	}
	if len(phis) == 0 {
		return false
	}
	// Three tables of flags and three of values, by ID, from scratch the
	// Func keeps.
	flags := idTable(f.scratch.flags, 3*n)
	f.scratch.flags = flags
	cand, unboxedUse, boxed := flags[:n:n], flags[n:2*n:2*n], flags[2*n:]
	vals := idTable(f.scratch.vals, 3*n)
	f.scratch.vals = vals
	for _, v := range phis {
		cand[v.ID] = true
	}
	// Evidence is an unboxing that speculates: a guard, as arithmetic makes.
	// One that exits to Go (a comparison, a property write, a remainder)
	// marks a site that takes any value, which says nothing of the phi's:
	// counting it had the entry speculate a receiver written to a property
	// is a number, and fail every call.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op == OpUnboxF64 && ir.ExitKind(v.Aux) == ir.GuardExit {
				unboxedUse[v.Args[0].ID] = true
			}
		}
	}
	isCand := func(a *Value) bool { return a.ID < n && cand[a.ID] }
	numeric := func(a *Value) bool {
		return a.Op == OpBoxF64 || a.Op == OpConst && a.Const.Kind == ir.Number
	}
	// Filter until stable: a candidate's arguments must be acceptable, and it
	// needs evidence, directly or through candidates. Removing one candidate
	// can disqualify another, so both filters repeat. The result is the same
	// in any order.
	for changed := true; changed; {
		changed = false
		for _, v := range phis {
			if !cand[v.ID] {
				continue
			}
			for _, a := range v.Args {
				if !numeric(a) && !isCand(a) && !(a.Op == OpLoadSlot && a.Block.PC < 0) {
					cand[v.ID] = false
					changed = true
					break
				}
			}
		}
		clear(boxed)
		for _, v := range phis {
			if cand[v.ID] {
				boxed[v.ID] = unboxedUse[v.ID]
			}
		}
		for grew := true; grew; {
			grew = false
			for _, v := range phis {
				if !cand[v.ID] || !boxed[v.ID] {
					continue
				}
				for _, a := range v.Args {
					if isCand(a) && !boxed[a.ID] {
						boxed[a.ID] = true
						grew = true
					}
				}
			}
		}
		for _, v := range phis {
			if cand[v.ID] && !boxed[v.ID] {
				cand[v.ID] = false
				changed = true
			}
		}
	}
	// In block order: value numbers, and so register allocation and code,
	// must not vary from one compilation to the next.
	fp := vals[:n:n]
	var ordered []*Value
	for _, v := range phis {
		if cand[v.ID] {
			p := f.alloc(Value{Op: OpPhi, Type: Float64, Block: v.Block})
			fp[v.ID] = p
			ordered = append(ordered, v)
		}
	}
	if len(ordered) == 0 {
		return false
	}
	unboxed := vals[n : 2*n : 2*n]
	for _, v := range ordered {
		p := fp[v.ID]
		for _, a := range v.Args {
			var x *Value
			switch {
			case a.Op == OpBoxF64:
				x = a.Args[0]
			case a.Op == OpConst:
				// A numeric constant: its number, defined beside it.
				x = unboxed[a.ID]
				if x == nil {
					x = f.alloc(Value{Op: OpConstF64, Type: Float64, Const: a.Const, Block: a.Block})
					insertAfter(a, x)
					unboxed[a.ID] = x
				}
			case isCand(a):
				x = fp[a.ID]
			default:
				x = unboxed[a.ID]
				if x == nil {
					e := a.Block
					x = f.alloc(Value{Op: OpUnboxF64, Type: Float64, Args: f.refsOf(1),
						Aux: int(ir.GuardExit), State: e.Header, Block: e})
					x.Args[0] = a
					e.Values = append(e.Values, x)
					unboxed[a.ID] = x
				}
			}
			p.Args = append(p.Args, x)
		}
	}
	subst := vals[2*n:]
	// Each block's values become its phis, with the new numeric ones beside
	// those they replace, the boxes of those, then the rest, in order:
	// built in buffers the pass reuses.
	var front, boxes, rest []*Value
	for _, b := range f.Blocks {
		front, boxes, rest = front[:0], boxes[:0], rest[:0]
		for _, v := range b.Values {
			if v.Op != OpPhi {
				rest = append(rest, v)
				continue
			}
			front = append(front, v)
			if v.ID < n && fp[v.ID] != nil {
				p := fp[v.ID]
				front = append(front, p)
				box := f.alloc(Value{Op: OpBoxF64, Type: Tagged, Args: f.refsOf(1), Block: b})
				box.Args[0] = p
				boxes = append(boxes, box)
				subst[v.ID] = box
			}
		}
		total := len(front) + len(boxes) + len(rest)
		if cap(b.Values) < total {
			b.Values = make([]*Value, total)
		}
		b.Values = b.Values[:total]
		copy(b.Values, front)
		copy(b.Values[len(front):], boxes)
		copy(b.Values[len(front)+len(boxes):], rest)
	}
	rewrite(f, func(v *Value) *Value {
		if v.ID < n && subst[v.ID] != nil {
			return subst[v.ID]
		}
		return v
	})
	return true
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
			if v.Op == OpLoadCell {
				v.Shadow = v.Args[0]
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
	live := make([]bool, f.nextID)
	var work []*Value
	mark := func(v *Value) {
		if v != nil && !live[v.ID] {
			live[v.ID] = true
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
			if live[v.ID] {
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

// insertAfter places v in a's block right after a.
func insertAfter(a, v *Value) {
	b := a.Block
	for i, w := range b.Values {
		if w == a {
			b.Values = append(b.Values[:i+1], append([]*Value{v}, b.Values[i+1:]...)...)
			return
		}
	}
	panic("ssa: insertAfter: value not in its block")
}
