package ssa

import (
	"fmt"
	"slices"

	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
)

// Native code holds a reference's number word, and reads its pointer word
// where it came from when it needs it (origin.go): a reference read from a
// cell needs the cell to keep it. A property store over that cell, while
// the reference is still used, would change it, and used to leave the
// store to Go (storeChecks): reading a field into a variable, then writing
// it -- p = this.queue; this.queue = p.link -- left native code each time.
// V8's code keeps such a value in a register, which its collector finds;
// Go's cannot. So the value is kept instead, before the store, in a cell of
// the context's that the collector scans and no store writes
// (abi.Context.Keep, OpKeep), and the code reads it as a copy whose shadow
// is that cell (OpKept): every use the store reaches, rebuilt as SSA. While
// the collector marks, no pointer may be written: the keep is then the
// value's own source, and the store's checks leave for Go as before.

// keepAcrossStores keeps, before each property store, the values read from
// cells that are used after it (liveAcross), up to abi.MaxKeeps in a
// function, and reports whether it kept any; their phis' shadows are then
// to be made again (shadowMerges).
func keepAcrossStores(f *Func) bool {
	stores, live := liveAcross(f)
	a := newAliases()
	kept := map[*Value][]*Value{}
	var order []*Value
	for i, s := range stores {
		// A store that may write a cell a value used after it was read from
		// (aliases) keeps that value, and every other used after it that has
		// a shadow, and the value it stores, which it reads after the keeps:
		// every pointer word is read before any keep cell is written, since
		// one may be read from a cell another writes -- a value kept at this
		// store the last time round a loop, which this pass, making the
		// keeps, cannot yet tell. So a store keeps all of them or none.
		alias := false
		for _, c := range live[i] {
			alias = alias || c.Op != OpKept && a.may(s.Key, c.Shadow)
		}
		if !alias {
			continue
		}
		var cands []*Value
		for _, c := range append(live[i], s.Args[1]) {
			if c.Type == Tagged && c.Shadow != nil && c.Op != OpKept && !slices.Contains(cands, c) {
				cands = append(cands, c)
			}
		}
		if f.Keeps+len(cands) > abi.MaxKeeps {
			continue
		}
		var refs, keeps, copies []*Value
		b := s.Block
		for _, c := range cands {
			r := f.alloc(Value{Op: OpKeepRef, Type: Ptr, Args: f.refsOf(1), Block: b})
			r.Args[0] = c
			k := f.alloc(Value{Op: OpKeep, Type: Source, Args: f.refsOf(2), Block: b, Index: f.Keeps})
			k.Args[0], k.Args[1] = c, r
			v := f.alloc(Value{Op: OpKept, Type: Tagged, Args: f.refsOf(2), Block: b})
			v.Args[0], v.Args[1], v.Shadow = k, c, k
			f.Keeps++
			refs, keeps, copies = append(refs, r), append(keeps, k), append(copies, v)
			if kept[c] == nil {
				order = append(order, c)
			}
			kept[c] = append(kept[c], v)
		}
		if len(refs) != 0 {
			at := indexOf(b.Values, s)
			ins := append(append(append([]*Value(nil), refs...), keeps...), copies...)
			b.Values = append(b.Values[:at], append(ins, b.Values[at:]...)...)
		}
	}
	for _, c := range order {
		reconstruct(f, c, kept[c])
	}
	return len(order) > 0
}

func indexOf(vs []*Value, v *Value) int {
	for i, w := range vs {
		if w == v {
			return i
		}
	}
	panic("ssa: a value is not in its block")
}

// reconstruct makes each use of c take the definition that reaches it, c's
// own or one of its copies (defs, each defined where it is), with phis
// where they meet, as an SSA builder reads a variable (Braun et al.): a use
// takes the last definition before it in its block, or else what reaches
// the block's start -- from its one predecessor's end, or from a phi of
// every predecessor's, dropped when its arguments are one value.
func reconstruct(f *Func, c *Value, defs []*Value) {
	isDef := map[*Value]bool{c: true}
	for _, d := range defs {
		isDef[d] = true
	}
	// The last definition in a block before index at, or nil.
	before := func(b *Block, at int) *Value {
		for i := min(at, len(b.Values)) - 1; i >= 0; i-- {
			if isDef[b.Values[i]] {
				return b.Values[i]
			}
		}
		return nil
	}
	entry := map[*Block]*Value{}
	alias := map[*Value]*Value{}
	resolve := func(v *Value) *Value {
		for alias[v] != nil {
			v = alias[v]
		}
		return v
	}
	var atStart func(b *Block) *Value
	atEnd := func(b *Block) *Value {
		if d := before(b, len(b.Values)); d != nil {
			return d
		}
		return atStart(b)
	}
	atStart = func(b *Block) *Value {
		if v, ok := entry[b]; ok {
			return resolve(v)
		}
		switch len(b.Preds) {
		case 0:
			panic(fmt.Sprintf("ssa: no definition of %v reaches b%d", c, b.ID))
		case 1:
			v := atEnd(b.Preds[0])
			entry[b] = v
			return v
		}
		phi := f.alloc(Value{Op: OpPhi, Type: Tagged, Args: f.refsOf(len(b.Preds)), Block: b})
		entry[b] = phi
		var same *Value
		trivial := true
		for i, p := range b.Preds {
			a := resolve(atEnd(p))
			phi.Args[i] = a
			if a == phi {
				continue
			}
			if same != nil && same != a {
				trivial = false
			}
			same = a
		}
		if trivial && same != nil {
			alias[phi] = same
			entry[b] = same
			return same
		}
		b.Values = append([]*Value{phi}, b.Values...)
		isDef[phi] = true
		return phi
	}
	reach := func(b *Block, at int) *Value {
		if d := before(b, at); d != nil {
			return d
		}
		return atStart(b)
	}
	// A frame state its users share -- the guards of one instruction do,
	// some before a keep and some after -- is copied before one's slot
	// changes.
	users := map[*FrameState]int{}
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.State != nil {
				users[v.State]++
			}
		}
		if b.State != nil {
			users[b.State]++
		}
	}
	state := func(s *FrameState, b *Block, at int) *FrameState {
		if s == nil {
			return nil
		}
		var out *FrameState
		for i, x := range s.Slots {
			if x != c {
				continue
			}
			r := reach(b, at)
			if r == c {
				continue
			}
			if out == nil {
				out = s
				if users[s] > 1 {
					users[s]--
					out = f.newState(*s)
					out.Slots = f.refsOf(len(s.Slots))
					copy(out.Slots, s.Slots)
					users[out] = 1
				}
			}
			out.Slots[i] = r
		}
		if out == nil {
			return s
		}
		return out
	}
	for _, b := range f.Blocks {
		// Values inserted as phis come first; the loop sees the block as it
		// was, its own phis included.
		values := append([]*Value(nil), b.Values...)
		for _, v := range values {
			at := indexOf(b.Values, v)
			if v.Op == OpPhi {
				for i, a := range v.Args {
					if a == c && !(isDef[v] && v.Type == Tagged && entry[b] == v) {
						v.Args[i] = atEnd(b.Preds[i])
					}
				}
				continue
			}
			if v == c {
				continue
			}
			for i, a := range v.Args {
				if a == c {
					v.Args[i] = reach(b, at)
				}
			}
			v.State = state(v.State, b, at)
		}
		if b.Control == c {
			b.Control = reach(b, len(b.Values))
		}
		b.State = state(b.State, b, len(b.Values))
		if b.Header != nil && b.PC >= 0 {
			// A loop header's state is after its phis.
			phis := 0
			for phis < len(b.Values) && b.Values[phis].Op == OpPhi {
				phis++
			}
			for i, x := range b.Header.Slots {
				if x == c {
					b.Header.Slots[i] = reach(b, phis)
				}
			}
		}
	}
	// Phis made, then found to merge one value, are that value.
	slots := func(s *FrameState) {
		if s != nil {
			for i, x := range s.Slots {
				if x != nil {
					s.Slots[i] = resolve(x)
				}
			}
		}
	}
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			for i, a := range v.Args {
				v.Args[i] = resolve(a)
			}
			slots(v.State)
		}
		if b.Control != nil {
			b.Control = resolve(b.Control)
		}
		slots(b.State)
		slots(b.Header)
	}
}
