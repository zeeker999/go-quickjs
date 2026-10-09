package ssa

import (
	"math"
	"slices"
)

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
// phi has a shadow: a Source phi holding, at run time, where its value came
// from -- the address of the slot's value, which is in the frame, in the
// context for the receiver, or in a captured binding's cell -- or 0 for a
// primitive. An exit passes the shadow to Go, and native code reads the
// pointer word there, the same way wherever it is.
//
// A reference loaded from an object (OpLoadCell) came from a heap cell, not
// a slot: its shadow is the cell's address. The cell holds the reference
// until native code exits, when Go copies it from there: the one store
// native code makes to a heap cell, a property's (OpPropWrite), first
// compares the cell with those of the references live after it
// (storeChecks), and leaves the write to Go on a match. A phi merging such a value has a shadow too, whose
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
	origin := f.ints(f.nextID)
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
			case OpLoadCell, OpKept:
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
// are shadows or constants: a load's slot, or -1 for a primitive, which
// native code makes the slot's address, or 0 (OpConstSource). The
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
		insertAfter(f, v, v.Shadow)
	}
}

// constSource makes a source constant at the end of a block.
func (f *Func) constSource(b *Block, k int) *Value {
	v := f.alloc(Value{Op: OpConstSource, Type: Source, Aux: k, Block: b})
	b.Values = f.appendValue(b.Values, v)
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

// storeChecks gives every property store the cells it must not write: the
// shadows of the tagged values live after it that have one -- references
// loaded from cells, and phis that may hold one. Native code holds such a
// value's number word and reads its pointer word where its shadow says
// when Go needs it; a store to that cell would change the pointer word
// under it. The store compares each with the property's address, and
// leaves the write to Go on a match while the cell holds a pointer word:
// one it does not hold, a primitive's, is not lost -- and a cell a live
// reference was read from cannot have changed since, every store to it
// having left. They are its operands after the object and the value.
func storeChecks(f *Func) {
	stores, live := liveAcross(f)
	a := newAliases()
	for i, s := range stores {
		if s.Op != OpPropWrite {
			continue
		}
		for _, c := range live[i] {
			if sh := c.Shadow; a.may(s.Key, sh) && !slices.Contains(s.Args[2:], sh) {
				s.Args = f.appendValue(s.Args, sh)
				sh.Uses++
			}
		}
	}
}

// aliases says which sources a property store may write: a property's cell
// read through the store's key, a global binding's of that name; through a
// phi, any of its arguments' -- and a keep's, while the collector marks,
// its value's own (OpKeep). Elements, frame slots and the context's cells
// are no property's, and a cell read through another key is another
// property's. Answers are kept, by phi and key.
type aliases struct{ key map[aliasKey]int }

type aliasKey struct {
	v   *Value
	key uint32
}

func newAliases() *aliases { return &aliases{map[aliasKey]int{}} }

// anyKey is a call's key: a callee may write any property.
const anyKey = ^uint32(0)

// may reports whether a store through key may write the cell at s.
func (a *aliases) may(key uint32, s *Value) bool {
	switch s.Op {
	case OpPropCell, OpGlobalCell:
		return key == anyKey || s.Key == key
	case OpElemCell, OpStringMethod, OpConstSource:
		return false
	case OpKeep:
		if x := s.Args[0]; x.Shadow != nil {
			return a.may(key, x.Shadow)
		}
		return false
	case OpPhi:
		// 1 while it is looked at, round a loop; then 0 or 2.
		k := aliasKey{s, key}
		if r, ok := a.key[k]; ok {
			return r == 2
		}
		a.key[k] = 1
		for _, x := range s.Args {
			if a.may(key, x) {
				a.key[k] = 2
				return true
			}
		}
		a.key[k] = 0
		return false
	}
	return true
}

// liveAcross is every property store, and for each the values with a
// shadow -- read from cells, and phis that may hold such -- used after it.
func liveAcross(f *Func) ([]*Value, [][]*Value) {
	var stores, cands []*Value
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			switch {
			case v.Op == OpPropWrite, v.Op == OpCall && v.Calls != nil:
				stores = append(stores, v)
			case v.Type == Tagged && v.Shadow != nil:
				// A kept value's too: while the collector marks, it is still
				// read from its cell (OpKeep).
				cands = append(cands, v)
			}
		}
	}
	if len(stores) == 0 || len(cands) == 0 {
		return stores, make([][]*Value, len(stores))
	}
	nb := 0
	for _, b := range f.Blocks {
		nb = max(nb, b.ID+1)
	}
	pos, cand := f.ints(f.nextID), f.ints(f.nextID)
	for _, b := range f.Blocks {
		for i, v := range b.Values {
			pos[v.ID] = i
		}
	}
	for k, c := range cands {
		cand[c.ID] = k + 1
	}
	// By candidate, then block ID: the last place in the block that uses
	// the candidate -- a value's index, the block's length for its end, -1
	// for its header -- or -2 for none; and whether it is live at the end.
	last, out := f.ints(len(cands)*nb), f.bools(len(cands)*nb)
	for i := range last {
		last[i] = -2
	}
	use := func(a *Value, b *Block, at int) {
		if a.ID < len(cand) && cand[a.ID] != 0 {
			k := (cand[a.ID]-1)*nb + b.ID
			last[k] = max(last[k], at)
		}
	}
	for _, b := range f.Blocks {
		if b.Header != nil {
			for _, s := range b.Header.Slots {
				use(s, b, -1)
			}
		}
		for i, v := range b.Values {
			if v.Op == OpPhi {
				// A phi's argument is used at the end of its predecessor.
				for j, a := range v.Args {
					if a.ID < len(cand) && cand[a.ID] != 0 {
						out[(cand[a.ID]-1)*nb+b.Preds[j].ID] = true
					}
				}
				continue
			}
			for _, a := range v.Args {
				use(a, b, i)
			}
			if v.State != nil {
				for _, s := range v.State.Slots {
					if s != nil {
						use(s, b, i)
					}
				}
			}
		}
		if b.Control != nil {
			use(b.Control, b, len(b.Values))
		}
		if b.State != nil {
			for _, s := range b.State.Slots {
				if s != nil {
					use(s, b, len(b.Values))
				}
			}
		}
	}
	// A candidate is live at the end of a block if it is live at the start
	// of a successor: anywhere but where it is defined, if that uses it or
	// it is live at that one's end.
	var work []*Block
	for k, c := range cands {
		row, uses := out[k*nb:(k+1)*nb], last[k*nb:(k+1)*nb]
		liveIn := func(b *Block) bool { return b != c.Block && (uses[b.ID] != -2 || row[b.ID]) }
		work = work[:0]
		for _, b := range f.Blocks {
			if liveIn(b) {
				work = append(work, b)
			}
		}
		for len(work) > 0 {
			b := work[len(work)-1]
			work = work[:len(work)-1]
			for _, p := range b.Preds {
				if !row[p.ID] {
					row[p.ID] = true
					if liveIn(p) {
						work = append(work, p)
					}
				}
			}
		}
	}
	live := make([][]*Value, len(stores))
	for i, s := range stores {
		at := pos[s.ID]
		for k, c := range cands {
			if c.Block == s.Block && pos[c.ID] > at {
				continue // defined after it
			}
			if !out[k*nb+s.Block.ID] && last[k*nb+s.Block.ID] <= at {
				continue // dead after it
			}
			live[i] = append(live[i], c)
		}
	}
	return stores, live
}

// clearStoreChecks drops the stores' checks, for storeChecks to remake.
func clearStoreChecks(f *Func) {
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op == OpPropWrite {
				for _, a := range v.Args[2:] {
					a.Uses--
				}
				v.Args = v.Args[:2]
			}
		}
	}
}
