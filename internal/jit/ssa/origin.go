package ssa

import (
	"fmt"
	"math"
	"math/bits"
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
	need := f.bools(f.nextID)
	any := false
	var walk func(v *Value)
	walk = func(v *Value) {
		if v.Op != OpPhi || need[v.ID] || origin.At(v) == OriginScalar {
			return
		}
		need[v.ID], any = true, true
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
	if !any {
		return
	}
	// In block order, so that values are numbered the same every time.
	var shadows []*Value
	fresh := f.bools(f.nextID)
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.ID < len(need) && need[v.ID] && v.Shadow == nil {
				s := f.alloc(Value{Op: OpPhi, Type: Source, Args: f.refsOf(len(v.Args)), Block: b})
				v.Shadow, fresh[v.ID] = s, true
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
	}
	// Each shadow right after its phi: a block's, all at once.
	for i := 0; i < len(shadows); {
		b, j := shadows[i].Block, i
		for j < len(shadows) && shadows[j].Block == b {
			j++
		}
		vs := f.refsOf(len(b.Values) + j - i)[:0]
		for _, v := range b.Values {
			vs = append(vs, v)
			if v.ID < len(fresh) && fresh[v.ID] {
				vs = append(vs, v.Shadow)
			}
		}
		b.Values = vs
		i = j
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
type aliases struct{ key, own, elem map[aliasKey]int }

type aliasKey struct {
	v   *Value
	key uint32
}

func newAliases() *aliases {
	return &aliases{map[aliasKey]int{}, map[aliasKey]int{}, map[aliasKey]int{}}
}

// anyKey is a call's key: a callee may write any property.
const anyKey = ^uint32(0)

// may reports whether a store through key may write the cell at s.
func (a *aliases) may(key uint32, s *Value) bool {
	switch s.Op {
	case OpPropCell, OpGlobalCell:
		return key == anyKey || s.Key == key
	case OpElemCell:
		// No property store writes an element; a call's callee may, or
		// pop it.
		return key == anyKey
	case OpStringMethod, OpConstSource:
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

// mayOwnCell reports whether the cell at s may be the result cell of call,
// one that makes nothing but a pool's object (allocOnly): the one cell such
// a call writes, each time it is made, over the last result.
func (a *aliases) mayOwnCell(call, s *Value) bool {
	switch s.Op {
	case OpCallCell:
		return s.Args[0] == call
	case OpKeep:
		if x := s.Args[0]; x.Shadow != nil {
			return a.mayOwnCell(call, x.Shadow)
		}
		return false
	case OpPhi:
		k := aliasKey{s, uint32(call.ID)}
		if r, ok := a.own[k]; ok {
			return r == 2
		}
		a.own[k] = 1
		for _, x := range s.Args {
			if a.mayOwnCell(call, x) {
				a.own[k] = 2
				return true
			}
		}
		a.own[k] = 0
		return false
	}
	return false
}

// mayElemCell reports whether the cell at s may be an element's, or the
// result cell of call, which pops an element (popsElement).
func (a *aliases) mayElemCell(call, s *Value) bool {
	switch s.Op {
	case OpElemCell:
		return true
	case OpCallCell:
		return s.Args[0] == call
	case OpKeep:
		if x := s.Args[0]; x.Shadow != nil {
			return a.mayElemCell(call, x.Shadow)
		}
		return false
	case OpPhi:
		k := aliasKey{s, uint32(call.ID)}
		if r, ok := a.elem[k]; ok {
			return r == 2
		}
		a.elem[k] = 1
		for _, x := range s.Args {
			if a.mayElemCell(call, x) {
				a.elem[k] = 2
				return true
			}
		}
		a.elem[k] = 0
		return false
	}
	return false
}

// liveAcross is every property store, and for each the values with a
// shadow -- read from cells, and phis that may hold such -- used after it.
// Liveness is by bitsets over those values: each block's, live at its
// start and at its end, to a fixed point, then a walk back through each
// block with a store, from its end, past each value: its own value dies,
// what it uses lives.
func liveAcross(f *Func) ([]*Value, [][]*Value) {
	stores, live := liveAcrossSets(f)
	if verifyLiveness {
		refStores, refLive := liveAcrossRef(f)
		if !slices.Equal(stores, refStores) {
			panic("ssa: liveAcross's stores differ from the reference's")
		}
		for i := range live {
			if !slices.Equal(live[i], refLive[i]) {
				panic(fmt.Sprintf("ssa: live across %v: %v, the reference %v", stores[i], live[i], refLive[i]))
			}
		}
	}
	return stores, live
}

// liveAcrossSets is liveAcross.
func liveAcrossSets(f *Func) ([]*Value, [][]*Value) {
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
	live := make([][]*Value, len(stores))
	if len(stores) == 0 || len(cands) == 0 {
		return stores, live
	}
	nb := 0
	for _, b := range f.Blocks {
		nb = max(nb, b.ID+1)
	}
	cand := f.ints(f.nextID)
	for k, c := range cands {
		cand[c.ID] = k + 1
	}
	w := (len(cands) + 63) / 64
	// By block ID: what it uses (before defining it, as SSA has it), what
	// it defines, what its successors' phis take from it, and what is live
	// at its start and at its end.
	sets := make([]uint64, 5*nb*w)
	use, def, phiOut, in, out := sets[:nb*w], sets[nb*w:2*nb*w], sets[2*nb*w:3*nb*w], sets[3*nb*w:4*nb*w], sets[4*nb*w:]
	row := func(s []uint64, b *Block) []uint64 { return s[b.ID*w : (b.ID+1)*w] }
	add := func(s []uint64, v *Value) {
		if v != nil && v.ID < len(cand) && cand[v.ID] != 0 {
			k := cand[v.ID] - 1
			s[k/64] |= 1 << (k % 64)
		}
	}
	states := func(s []uint64, st *FrameState) {
		if st != nil {
			for _, x := range st.Slots {
				add(s, x)
			}
		}
	}
	// uses adds what v uses, a phi's arguments aside.
	uses := func(s []uint64, v *Value) {
		for _, a := range v.Args {
			add(s, a)
		}
		states(s, v.State)
	}
	for _, b := range f.Blocks {
		u, d := row(use, b), row(def, b)
		if b.Header != nil {
			states(u, b.Header)
		}
		for _, v := range b.Values {
			add(d, v)
			if v.Op == OpPhi {
				// A phi's argument is used at the end of its predecessor.
				for j, a := range v.Args {
					add(row(phiOut, b.Preds[j]), a)
				}
				continue
			}
			uses(u, v)
		}
		add(u, b.Control)
		states(u, b.State)
	}
	// The blocks are in reverse post-order: backward, a loop takes a
	// second round or so.
	for changed := true; changed; {
		changed = false
		for i := len(f.Blocks) - 1; i >= 0; i-- {
			b := f.Blocks[i]
			o, n, u, d, p := row(out, b), row(in, b), row(use, b), row(def, b), row(phiOut, b)
			for j := range o {
				x := p[j]
				for _, s := range b.Succs {
					x |= in[s.ID*w+j]
				}
				o[j] = x
				if y := (u[j] | x) &^ d[j]; y != n[j] {
					n[j], changed = y, true
				}
			}
		}
	}
	// Back through each block with a store: what is live after each value.
	cur := make([]uint64, w)
	k := 0
	for _, b := range f.Blocks {
		first := k
		for k < len(stores) && stores[k].Block == b {
			k++
		}
		if first == k {
			continue
		}
		copy(cur, row(out, b))
		add(cur, b.Control)
		states(cur, b.State)
		next := k - 1
		for i := len(b.Values) - 1; i >= 0 && next >= first; i-- {
			v := b.Values[i]
			if v == stores[next] {
				for j, x := range cur {
					for x != 0 {
						bit := bits.TrailingZeros64(x)
						x &= x - 1
						live[next] = append(live[next], cands[j*64+bit])
					}
				}
				next--
			}
			if v.ID < len(cand) && cand[v.ID] != 0 {
				c := cand[v.ID] - 1
				cur[c/64] &^= 1 << (c % 64)
			}
			if v.Op != OpPhi {
				uses(cur, v)
			}
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
