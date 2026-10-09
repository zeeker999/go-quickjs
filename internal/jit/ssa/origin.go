package ssa

import "math"

// Native code holds a slot's number word and never its pointer word, so a
// tagged value it moves to another slot, or returns, is either a primitive
// it can write itself or the reference some slot held when native code was
// entered, which only Go can copy. Native code writes the frame only when
// it exits, so at every exit each slot still holds what it held at entry:
// an exit that finds the reference there has Go copy it. A value's origin
// says which slot to look in.
//
// Most values have one origin a compiler can name. A phi can merge two
// slots' values -- a loop that does x=o, entered at its header with x
// already holding a reference -- and two references can share a number
// word, as two objects do, so no test of the word tells them apart. Such a
// phi has a shadow: a Source phi holding, at run time, the slot its value
// came from, or -1 for a primitive. An exit passes the shadow to Go.
//
// A reference loaded from an object (OpLoadCell) came from a heap cell, not
// a slot: its shadow is the cell's address. Native code never stores a
// pointer, so the cell holds the reference until native code exits, when Go
// copies it from there. A phi merging such a value has a shadow too, whose
// argument for it is the cell. This holds only while Go's heap does not
// move and nothing changes the object graph while native code runs, which
// is outside what unsafe.Pointer's rules promise: the user chose it
// (docs/jit-progress.md, 2026-10-08), and it is checked with each new Go.

// Origins of a tagged value, besides a slot's index.
const (
	// OriginScalar: a primitive made natively.
	OriginScalar = -1
	// OriginAmbiguous: a merge of two slots' values, which its shadow tells
	// apart at run time.
	OriginAmbiguous = -2
	originNone      = -3 // a phi not yet reached
	// OriginHeap: loaded from a heap cell, which its shadow is.
	OriginHeap = -4
)

// OriginMap is each tagged value's origin (Origins), by value ID.
type OriginMap struct{ of []int }

// originAbsent marks a value with no origin: one not tagged, or not in the
// function when its origins were found.
const originAbsent = math.MinInt

// Of is v's origin, and whether it has one.
func (m OriginMap) Of(v *Value) (int, bool) {
	if v.ID >= len(m.of) || m.of[v.ID] == originAbsent {
		return 0, false
	}
	return m.of[v.ID], true
}

// At is v's origin, or 0 if it has none.
func (m OriginMap) At(v *Value) int {
	o, _ := m.Of(v)
	return o
}

// Origins returns each tagged value's origin: OriginScalar, or the slot
// whose value at entry it may be -- when it is not a primitive -- or
// OriginAmbiguous.
func Origins(f *Func) OriginMap {
	origin := make([]int, f.nextID)
	for i := range origin {
		origin[i] = originAbsent
	}
	at := func(v *Value) int {
		if v.ID >= len(origin) || origin[v.ID] == originAbsent {
			return 0
		}
		return origin[v.ID]
	}
	var phis []*Value
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Type != Tagged {
				continue
			}
			switch v.Op {
			case OpPhi:
				origin[v.ID] = originNone
				phis = append(phis, v)
			case OpLoadSlot:
				origin[v.ID] = v.Aux
			case OpLoadCell:
				origin[v.ID] = OriginHeap
			default:
				origin[v.ID] = OriginScalar
			}
		}
	}
	for changed := true; changed; {
		changed = false
		for _, p := range phis {
			o := originNone
			for _, a := range p.Args {
				o = joinOrigin(o, at(a))
			}
			if o != origin[p.ID] {
				origin[p.ID] = o
				changed = true
			}
		}
	}
	for _, p := range phis {
		if origin[p.ID] == originNone {
			origin[p.ID] = OriginScalar
		}
	}
	return OriginMap{origin}
}

func joinOrigin(a, b int) int {
	switch {
	case a == OriginHeap && b != originNone || b == OriginHeap && a != originNone:
		// Two cells, or a cell and anything else: known only at run time.
		return OriginAmbiguous
	case a == originNone || a == b:
		return b
	case b == originNone:
		return a
	case a == OriginScalar:
		return b
	case b == OriginScalar:
		return a
	}
	return OriginAmbiguous
}

// shadowMerges gives every ambiguous phi a shadow, and every phi that
// flows into one and may hold a reference, so that each shadow's arguments
// are shadows or constants: a load's slot, or -1 for a primitive. The
// constants are made in the predecessor the argument comes from.
func shadowMerges(f *Func) {
	origin := Origins(f)
	need := map[*Value]bool{}
	var walk func(v *Value)
	walk = func(v *Value) {
		if v.Op != OpPhi || need[v] || origin.At(v) == OriginScalar {
			return
		}
		need[v] = true
		for _, a := range v.Args {
			walk(a)
		}
	}
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op == OpPhi && origin.At(v) == OriginAmbiguous {
				walk(v)
			}
		}
	}
	if len(need) == 0 {
		return
	}
	// In block order, so that values are numbered the same every time.
	var shadows []*Value
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if need[v] && v.Shadow == nil {
				s := f.alloc(Value{Op: OpPhi, Type: Source, Args: f.refsOf(len(v.Args)), Block: b})
				v.Shadow = s
				shadows = append(shadows, v)
			}
		}
	}
	for _, v := range shadows {
		for i, a := range v.Args {
			var s *Value
			switch {
			case a.Shadow != nil:
				s = a.Shadow
			case origin.At(a) >= 0:
				s = f.constSource(v.Block.Preds[i], origin.At(a))
			default:
				s = f.constSource(v.Block.Preds[i], -1)
			}
			v.Shadow.Args[i] = s
			s.Uses++
		}
		insertAfter(v, v.Shadow)
	}
}

// constSource makes a source constant at the end of a block.
func (f *Func) constSource(b *Block, k int) *Value {
	v := f.alloc(Value{Op: OpConstSource, Type: Source, Aux: k, Block: b})
	b.Values = append(b.Values, v)
	return v
}

// clearShadows drops every phi's shadow, for passes to remake them after
// values have been replaced; the shadows themselves are then dead. A loaded
// value keeps its cell, which is its argument as well.
func clearShadows(f *Func) {
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op == OpPhi {
				v.Shadow = nil
			}
		}
	}
}
