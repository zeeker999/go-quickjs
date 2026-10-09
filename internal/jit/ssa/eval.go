package ssa

import (
	"errors"
	"fmt"
	"math"
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

// canonicalNaN is the NaN a number cell holds, and numberLimit bounds the
// words that hold numbers (abi.NumberLimit).
const (
	canonicalNaN = 0x7ff8000000000000
	numberLimit  = 0xFFF8000000000000
)

// integer64 reports whether x is an integer an int64 holds exactly.
func integer64(x float64) bool {
	return x >= -1<<63 && x < 1<<63 && math.Trunc(x) == x
}

// index converts an element's key, as the slot IR does: an integer in
// [0, 2**32), negative zero included.
func index(x float64) (uint64, bool) {
	if x < 0 || x > math.MaxUint32 || math.Trunc(x) != x {
		return 0, false
	}
	return uint64(x), true
}

// element is the cell for key in an array view, if it holds a number.
func element(view ir.ArrayView, key float64) (*uint64, bool) {
	i, ok := index(key)
	if !ok || i >= view.DenseLength || view.Data == nil {
		return nil, false
	}
	cell := (*uint64)(unsafe.Add(view.Data, uintptr(i)*16))
	if *cell >= view.NumberLimit {
		return nil, false
	}
	return cell, true
}

// eqTagged is OpEqTagged's comparison, by the words native code sees: two
// numbers compare as numbers; equal words (objects' are one word, strings'
// another) are equal values but for objects, the same only if they are one,
// and strings, left to Go; other words differ, strictly unequal and left to
// Go loosely. It reports false where Go decides.
func eqTagged(x, y ir.Value, strict bool) (bool, bool) {
	if x.Kind == ir.Number && y.Kind == ir.Number {
		return math.Float64frombits(x.Bits) == math.Float64frombits(y.Bits), true
	}
	if x.Kind == y.Kind && (x.Kind == ir.Opaque || x.Kind == ir.String || x.Bits == y.Bits) {
		switch x.Kind {
		case ir.Undefined, ir.Null, ir.Boolean:
			return true, true
		case ir.Opaque:
			return x.Bits == y.Bits, true
		}
		return false, false
	}
	return false, strict
}

// ErrOrigin reports a reference at an exit that is not where its origin
// says: a bug in origin.go.
var ErrOrigin = errors.New("ssa: a reference is not its origin's")

// val is a value at run time, in the representation its type names.
type val struct {
	t ir.Value // Tagged
	f float64  // Float64
	i uint32   // Int32, as bits; Source, a slot's index or -1
	p int      // Ptr: the array view's index
	// cell is a Source that is a heap cell, which an object's table holds.
	cell *ir.Value
	b    bool // Bool
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
	case OpStrictNullish:
		kind := ir.Null
		if aux == 1 {
			kind = ir.Undefined
		}
		return val{b: a.t.Kind == kind}
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
//
// It also checks what generated code relies on to move references: at
// every exit and return, a reference is the value its origin slot held at
// entry (origin.go), and it reports ErrOrigin when one is not.
func Evaluate(f *Func, pc int, slots []ir.Value, pollEvery int) (ir.Exit, error) {
	return EvaluateArrays(f, pc, slots, nil, pollEvery)
}

// EvaluateArrays is Evaluate with arrays, as the slot IR's EvaluateArrays
// takes them: an Opaque value's Bits index arrays, and a view with a
// NumberLimit is an array whose cells it reads and writes in place.
func EvaluateArrays(f *Func, pc int, slots []ir.Value, arrays []ir.ArrayView, pollEvery int) (ir.Exit, error) {
	return EvaluateHeap(f, pc, slots, Heap{Arrays: arrays}, pollEvery)
}

// Heap is what an evaluation reads and writes besides the slots, by an
// Opaque value's Bits: arrays, as the slot IR's views, and objects.
type Heap struct {
	Arrays  []ir.ArrayView
	Objects []Object
	// Global is the object global names are read from, and Lexical the
	// names script-level lexical bindings have.
	Global  *Object
	Lexical map[uint32]bool
	// CharCodeAt is the context's cell for charCodeAt, which holds the
	// intrinsic, or is nil; Strings are strings by their handles: their
	// code units, and whether they are flat.
	CharCodeAt *ir.Value
	Strings    map[uint64]String
	// Word decodes an element's word, for OpElemCell; nil, only numbers'.
	Word func(uint64) ir.Value
}

// String is a string as charCodeAt sees it.
type String struct {
	Units []uint16
	Flat  bool
}

// Object is an object as property operations see it: its shape, whether it
// is an ordinary object whose table may be searched, and its table: each
// entry's key, whether it is plain data and plain writable data, and its
// value.
type Object struct {
	Shape    uintptr
	Ordinary bool
	// HTMLDDA is Annex B's [[IsHTMLDDA]]: == null and == undefined hold.
	HTMLDDA  bool
	Keys     []uint32
	Data     []bool
	Writable []bool
	Uninit   []bool
	Props    []ir.Value
}

// maxScan is abi.MaxScan.
const maxScan = 8

// property finds the property a property operation names in o, as the
// operation's semantics say (OpPropRead), or -1.
func (o *Object) property(v *Value) int {
	if v.Const.Bits != 0 && o.Shape == uintptr(v.Const.Bits) {
		return v.Index
	}
	if !o.Ordinary || len(o.Keys) > maxScan {
		return -1
	}
	for i, k := range o.Keys {
		if k != v.Key {
			continue
		}
		if !o.Data[i] || v.Op == OpPropWrite && !o.Writable[i] {
			return -1
		}
		return i
	}
	return -1
}

// EvaluateHeap is Evaluate with a heap.
func EvaluateHeap(f *Func, pc int, slots []ir.Value, heap Heap, pollEvery int) (ir.Exit, error) {
	arrays := heap.Arrays
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
	origin := Origins(f)
	// The frame is the entry's until an exit writes it.
	traced := func(v *Value) bool {
		x := slot(v)
		if x.Kind != ir.Opaque && x.Kind != ir.String {
			return true
		}
		o := origin.At(v)
		if v.Shadow != nil {
			s := vals[v.Shadow.ID]
			if s.cell != nil {
				return *s.cell == x
			}
			o = int(int32(s.i))
		}
		return o >= 0 && o < len(slots) && slots[o] == x
	}
	exit := func(s *FrameState, kind ir.ExitKind) (ir.Exit, error) {
		for i, v := range s.Slots {
			if !traced(v) {
				return ir.Exit{}, fmt.Errorf("%w: slot %d's %v at pc %d", ErrOrigin, i, v, s.PC)
			}
		}
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
			case OpConstSource:
				vals[v.ID] = val{i: uint32(int32(v.Aux))}
			case OpUnboxF64:
				if a.t.Kind != ir.Number {
					return exit(v.State, ir.ExitKind(v.Aux))
				}
				vals[v.ID] = val{f: math.Float64frombits(a.t.Bits)}
			case OpEqTagged:
				r, ok := eqTagged(a.t, b.t, v.Index == 1)
				if !ok {
					return exit(v.State, ir.ExitKind(v.Aux))
				}
				vals[v.ID] = val{b: r}
			case OpLooseNullish:
				t := a.t
				r := t.Kind == ir.Null || t.Kind == ir.Undefined
				if t.Kind == ir.Opaque && t.Bits < uint64(len(heap.Objects)) {
					r = heap.Objects[t.Bits].HTMLDDA
				}
				vals[v.ID] = val{b: r}
			case OpTruth:
				t, ok := truth(a.t)
				if !ok {
					return exit(v.State, ir.ExitKind(v.Aux))
				}
				vals[v.ID] = val{b: t}
			case OpArrayOf:
				t := a.t
				if t.Kind != ir.Opaque || t.Bits >= uint64(len(arrays)) || arrays[t.Bits].NumberLimit == 0 {
					return exit(v.State, ir.ExitKind(v.Aux))
				}
				vals[v.ID] = val{p: int(t.Bits)}
			case OpElemKey:
				if _, ok := index(a.f); !ok {
					return exit(v.State, ir.ExitKind(v.Aux))
				}
			case OpElemRead, OpElemWrite:
				cell, ok := element(arrays[a.p], b.f)
				if !ok {
					return exit(v.State, ir.ExitKind(v.Aux))
				}
				if v.Op == OpElemRead {
					vals[v.ID] = val{f: math.Float64frombits(*cell)}
					break
				}
				bits := math.Float64bits(vals[v.Args[2].ID].f)
				if math.IsNaN(math.Float64frombits(bits)) {
					bits = canonicalNaN
				}
				*cell = bits
			case OpElemCell:
				view := arrays[a.p]
				i, ok := index(b.f)
				if !ok || i >= view.DenseLength || view.Data == nil {
					return exit(v.State, ir.ExitKind(v.Aux))
				}
				word := *(*uint64)(unsafe.Add(view.Data, uintptr(i)*16))
				t := ir.Value{Kind: ir.Number, Bits: word}
				switch {
				case heap.Word != nil:
					t = heap.Word(word)
				case word >= view.NumberLimit:
					return exit(v.State, ir.ExitKind(v.Aux))
				}
				if t.Kind == ir.Uninitialized {
					// A hole.
					return exit(v.State, ir.ExitKind(v.Aux))
				}
				vals[v.ID] = val{cell: &t}
			case OpLength:
				t := a.t
				switch {
				case t.Kind == ir.Opaque && t.Bits < uint64(len(arrays)) && arrays[t.Bits].NumberLimit != 0:
					vals[v.ID] = val{f: float64(arrays[t.Bits].Length)}
				case t.Kind == ir.String && heap.Strings != nil:
					s, ok := heap.Strings[t.Bits]
					if !ok {
						return exit(v.State, ir.ExitKind(v.Aux))
					}
					vals[v.ID] = val{f: float64(len(s.Units))}
				default:
					return exit(v.State, ir.ExitKind(v.Aux))
				}
			case OpModF64:
				if !integer64(a.f) || !integer64(b.f) || b.f == 0 {
					return exit(v.State, ir.ExitKind(v.Aux))
				}
				vals[v.ID] = val{f: math.Mod(a.f, b.f)}
			case OpObjectOf:
				t := a.t
				if t.Kind != ir.Opaque || t.Bits >= uint64(len(heap.Objects)) {
					return exit(v.State, ir.ExitKind(v.Aux))
				}
				vals[v.ID] = val{p: int(t.Bits)}
			case OpPropRead, OpPropWrite:
				o := &heap.Objects[a.p]
				i := o.property(v)
				if i < 0 || i >= len(o.Props) || o.Props[i].Kind != ir.Number {
					return exit(v.State, ir.ExitKind(v.Aux))
				}
				if v.Op == OpPropRead {
					vals[v.ID] = val{f: math.Float64frombits(o.Props[i].Bits)}
					break
				}
				bits := math.Float64bits(b.f)
				if math.IsNaN(math.Float64frombits(bits)) {
					bits = canonicalNaN
				}
				o.Props[i] = ir.Value{Kind: ir.Number, Bits: bits}
			case OpPropCell:
				o := &heap.Objects[a.p]
				i := o.property(v)
				if i < 0 || i >= len(o.Props) {
					return exit(v.State, ir.ExitKind(v.Aux))
				}
				vals[v.ID] = val{cell: &o.Props[i]}
			case OpGlobalCell:
				g, i := heap.Global, v.Index
				if g == nil || heap.Lexical[v.Key] || i < 0 || i >= len(g.Props) ||
					g.Keys[i] != v.Key || !g.Data[i] || g.Uninit[i] {
					return exit(v.State, ir.ExitKind(v.Aux))
				}
				vals[v.ID] = val{cell: &g.Props[i]}
			case OpStringMethod:
				if a.t.Kind != ir.String || heap.CharCodeAt == nil {
					return exit(v.State, ir.ExitKind(v.Aux))
				}
				vals[v.ID] = val{cell: heap.CharCodeAt}
			case OpStringCode:
				s, ok := heap.Strings[b.t.Bits]
				i, isIndex := index(vals[v.Args[2].ID].f)
				if heap.CharCodeAt == nil || a.t != *heap.CharCodeAt || b.t.Kind != ir.String || !ok || !s.Flat ||
					!isIndex || i >= uint64(len(s.Units)) {
					return exit(v.State, ir.ExitKind(v.Aux))
				}
				vals[v.ID] = val{f: float64(s.Units[i])}
			case OpLoadCell:
				vals[v.ID] = val{t: *a.cell}
			case OpCheckInit:
				if a.t.Kind == ir.Uninitialized {
					return exit(v.State, ir.ExitKind(v.Aux))
				}
			default:
				vals[v.ID] = apply(v.Op, v.Aux, a, b)
			}
		}
		switch blk.Kind {
		case BlockReturn:
			if !traced(blk.Control) {
				return ir.Exit{}, fmt.Errorf("%w: returned %v", ErrOrigin, blk.Control)
			}
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
