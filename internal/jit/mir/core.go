// Package mir compiles the JIT's SSA (internal/jit/ssa) to machine code:
// it orders blocks, computes liveness, allocates registers by linear scan and
// selects instructions, then encodes them (internal/jit/asm). What every
// architecture shares is here (core); each has its code generator
// (amd64.go, arm64.go). Plain code over clever code, verified against the
// SSA evaluator (docs/jit-phase2-design.md).
package mir

import (
	"errors"
	"fmt"
	"math"
	"sort"

	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
	"github.com/go-quickjs/go-quickjs/internal/jit/ssa"
)

// core is a function's analysis and allocation, which code generators for
// every architecture share.
type core struct {
	f     *ssa.Func
	enc   abi.Encoding
	order []*ssa.Block
	// lazy values have no location: constants used only by frame states,
	// and boxes used only by frame states and returns, which are
	// rematerialized where needed.
	lazy map[*ssa.Value]bool
	locs map[*ssa.Value]loc
	// origin is each tagged value's (ssa.Origins): the slot an exit looks
	// in for the reference the value may be.
	origin map[*ssa.Value]int
	// gprs and fprs are the architecture's allocatable registers, by number.
	gprs, fprs []int
}

// prepare checks that f is one the backends compile, and analyses and
// allocates it, with the architecture's allocatable registers.
func prepare(f *ssa.Func, enc abi.Encoding, gprs, fprs []int) (*core, error) {
	c := &core{f: f, enc: enc, gprs: gprs, fprs: fprs}
	if enc.ValueSize != 16 {
		// Slots and elements are found by shifting an index by four.
		return nil, fmt.Errorf("%w: %d-byte values", ErrUnsupported, enc.ValueSize)
	}
	if n := f.Locals + f.StackSize; n > abi.MaxRecords {
		return nil, fmt.Errorf("%w: %d slots", ErrUnsupported, n)
	}
	if f.FrameLocals < 0 || f.FrameLocals > f.Locals {
		return nil, fmt.Errorf("%w: %d frame locals of %d", ErrUnsupported, f.FrameLocals, f.Locals)
	}
	if f.ThisSlot != -1 && (f.ThisSlot < f.FrameLocals || f.ThisSlot >= f.Locals) {
		return nil, fmt.Errorf("%w: the receiver at slot %d", ErrUnsupported, f.ThisSlot)
	}
	if err := capturedUnchanged(f); err != nil {
		return nil, err
	}
	c.origin = ssa.Origins(f)
	for v, o := range c.origin {
		if (o == ssa.OriginAmbiguous || o == ssa.OriginHeap) && v.Shadow == nil {
			return nil, fmt.Errorf("%w: %v merges two slots' values with no shadow", ErrUnsupported, v)
		}
	}
	c.layout()
	c.findLazy()
	if err := c.allocate(); err != nil {
		return nil, err
	}
	return c, nil
}

// ErrUnsupported reports a function the backend does not compile yet.
var ErrUnsupported = errors.New("mir: unsupported")

// Code is a compiled function: its bytes, and the offset of the entry for
// each slot IR PC that has one.
type Code struct {
	Bytes   []byte
	Entries map[int]int
	// Locations lists where each value lives, for tests and debugging.
	Locations string
}

// loc is where a value lives: a register of its class, or a spill slot.
type loc struct {
	reg   int // register number, or -1
	spill int // spill slot, or -1
}

type stubKey struct {
	state *ssa.FrameState
	kind  uint64
}

// capturedUnchanged requires that no instruction write a captured binding,
// or the receiver, past the frame's locals:
// every frame state then holds its value from entry there (in whatever
// representation the optimizer chose), and exits leave those slots alone.
func capturedUnchanged(f *ssa.Func) error {
	for i := f.FrameLocals; i < f.Locals; i++ {
		if f.Written(i) {
			return fmt.Errorf("%w: captured slot %d is written", ErrUnsupported, i)
		}
	}
	return nil
}

// layout orders blocks in reverse post-order from the entries, so that a
// loop's body follows its header.
func (c *core) layout() {
	seen := map[*ssa.Block]bool{}
	var post []*ssa.Block
	var visit func(*ssa.Block)
	visit = func(b *ssa.Block) {
		if seen[b] {
			return
		}
		seen[b] = true
		for i := len(b.Succs) - 1; i >= 0; i-- {
			visit(b.Succs[i])
		}
		post = append(post, b)
	}
	for i := len(c.f.Entries) - 1; i >= 0; i-- {
		visit(c.f.Entries[i].Block)
	}
	for i := len(post) - 1; i >= 0; i-- {
		c.order = append(c.order, post[i])
	}
}

// findLazy marks values that need no location of their own.
func (c *core) findLazy() {
	used := map[*ssa.Value]bool{} // by an argument, a control or a phi
	for _, b := range c.f.Blocks {
		for _, v := range b.Values {
			for _, a := range v.Args {
				used[a] = true
			}
		}
		if b.Control != nil && b.Kind != ssa.BlockReturn {
			used[b.Control] = true
		}
	}
	// A slot loaded at an entry and named only by frame states for that same
	// slot needs no register: an exit leaves that slot as it is.
	elsewhere := map[*ssa.Value]bool{}
	state := func(s *ssa.FrameState) {
		if s == nil {
			return
		}
		for i, v := range s.Slots {
			if !(v.Op == ssa.OpLoadSlot && v.Aux == i) {
				elsewhere[v] = true
			}
		}
	}
	for _, b := range c.f.Blocks {
		for _, v := range b.Values {
			for _, a := range v.Args {
				if a.Op == ssa.OpLoadSlot {
					elsewhere[a] = true
				}
			}
			state(v.State)
		}
		state(b.State)
		state(b.Header)
		if b.Control != nil && b.Control.Op == ssa.OpLoadSlot {
			elsewhere[b.Control] = true
		}
	}
	c.lazy = map[*ssa.Value]bool{}
	for _, b := range c.f.Blocks {
		for _, v := range b.Values {
			switch v.Op {
			case ssa.OpConstSource:
				c.lazy[v] = true
			case ssa.OpConst, ssa.OpBoxF64, ssa.OpBoxBool:
				if !used[v] {
					c.lazy[v] = true
				}
			case ssa.OpLoadSlot:
				if !elsewhere[v] {
					c.lazy[v] = true
				}
			}
		}
	}
}

// allocate computes liveness and assigns locations by linear scan.
func (c *core) allocate() error {
	// Number positions: each block has a start, one position per value, and
	// an end.
	pos := map[*ssa.Value]int{}
	start, end := map[*ssa.Block]int{}, map[*ssa.Block]int{}
	n := 0
	for _, b := range c.order {
		start[b] = n
		n++
		for _, v := range b.Values {
			pos[v] = n
			n++
		}
		end[b] = n
		n++
	}
	// Uses: each value's uses, by the position of the use.
	type use struct {
		v   *ssa.Value
		pos int
	}
	var uses []use
	need := func(v *ssa.Value) bool { return hasResult(v) && !c.lazy[v] }
	useState := func(s *ssa.FrameState, at int) {
		if s == nil {
			return
		}
		for _, v := range s.Slots {
			if c.lazy[v] || remat(v) {
				for _, a := range v.Args {
					uses = append(uses, use{a, at})
				}
				continue
			}
			uses = append(uses, use{v, at})
			if v.Shadow != nil {
				uses = append(uses, use{v.Shadow, at})
			}
		}
	}
	for _, b := range c.order {
		for _, v := range b.Values {
			if v.Op == ssa.OpPhi {
				continue // phi arguments are used at their predecessors' ends
			}
			for _, a := range v.Args {
				uses = append(uses, use{a, pos[v]})
			}
			useState(v.State, pos[v])
		}
		if b.Control != nil {
			if c.lazy[b.Control] {
				for _, a := range b.Control.Args {
					uses = append(uses, use{a, end[b]})
				}
			} else {
				uses = append(uses, use{b.Control, end[b]})
				if b.Control.Shadow != nil {
					uses = append(uses, use{b.Control.Shadow, end[b]})
				}
			}
		}
		useState(b.State, end[b])
		if b.LoopHeader {
			useState(b.Header, start[b])
		}
		for _, s := range b.Succs {
			for i, p := range s.Preds {
				if p != b {
					continue
				}
				for _, phi := range s.Values {
					if phi.Op != ssa.OpPhi {
						break
					}
					uses = append(uses, use{phi.Args[i], end[b]})
				}
				if s.LoopHeader && s.Backedge[i] {
					// A poll on this edge exits to the header's state.
					useState(s.Header, end[b])
				}
			}
		}
	}
	// Block-level liveness, so that values live around a loop stay live for
	// all of it.
	liveIn := map[*ssa.Block]map[*ssa.Value]bool{}
	usesIn := map[*ssa.Block][]*ssa.Value{}
	defsIn := map[*ssa.Block]map[*ssa.Value]bool{}
	blockAt := func(p int) *ssa.Block {
		for _, b := range c.order {
			if p >= start[b] && p <= end[b] {
				return b
			}
		}
		return nil
	}
	for _, b := range c.order {
		defsIn[b] = map[*ssa.Value]bool{}
		for _, v := range b.Values {
			defsIn[b][v] = true
		}
	}
	for _, u := range uses {
		if !need(u.v) {
			continue
		}
		b := blockAt(u.pos)
		if !defsIn[b][u.v] {
			usesIn[b] = append(usesIn[b], u.v)
		}
	}
	for changed := true; changed; {
		changed = false
		for i := len(c.order) - 1; i >= 0; i-- {
			b := c.order[i]
			in := map[*ssa.Value]bool{}
			for _, v := range usesIn[b] {
				in[v] = true
			}
			for _, s := range b.Succs {
				for v := range liveIn[s] {
					if !defsIn[b][v] {
						in[v] = true
					}
				}
			}
			if len(in) != len(liveIn[b]) {
				liveIn[b] = in
				changed = true
			}
		}
	}
	// Intervals: from definition to last use, stretched over every block
	// where the value is live in or out.
	iv := map[*ssa.Value]*interval{}
	get := func(v *ssa.Value) *interval {
		if iv[v] == nil {
			p, ok := pos[v]
			if !ok {
				panic(fmt.Sprintf("use of %v defined outside the function", v))
			}
			if v.Op == ssa.OpPhi {
				p = start[v.Block]
			}
			iv[v] = &interval{v: v, start: p, end: p}
		}
		return iv[v]
	}
	for _, b := range c.order {
		for _, v := range b.Values {
			if need(v) {
				get(v)
			}
		}
	}
	for _, u := range uses {
		if need(u.v) {
			if it := get(u.v); u.pos > it.end {
				it.end = u.pos
			}
		}
	}
	for _, b := range c.order {
		for v := range liveIn[b] {
			it := get(v)
			if start[b] < it.start {
				it.start = start[b]
			}
			if start[b] > it.end {
				it.end = start[b]
			}
		}
		for _, s := range b.Succs {
			for v := range liveIn[s] {
				if !need(v) {
					continue
				}
				if v.Block == s && v.Op == ssa.OpPhi {
					continue
				}
				it := get(v)
				if end[b] > it.end {
					it.end = end[b]
				}
				if start[b] < it.start && !defsIn[b][v] {
					it.start = start[b]
				}
			}
		}
	}
	var all []*interval
	for _, it := range iv {
		all = append(all, it)
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].start != all[j].start {
			return all[i].start < all[j].start
		}
		return all[i].v.ID < all[j].v.ID
	})
	c.locs = map[*ssa.Value]loc{}
	spills := 0
	for _, float := range []bool{false, true} {
		var free []int
		if float {
			free = append(free, c.fprs...)
		} else {
			free = append(free, c.gprs...)
		}
		var active []*interval
		for _, it := range all {
			if isFloat(it.v) != float {
				continue
			}
			kept := active[:0]
			for _, a := range active {
				if a.end < it.start {
					free = append(free, c.locs[a.v].reg)
				} else {
					kept = append(kept, a)
				}
			}
			active = kept
			if len(free) > 0 {
				c.locs[it.v] = loc{reg: free[len(free)-1], spill: -1}
				free = free[:len(free)-1]
				active = append(active, it)
				continue
			}
			// Spill whichever interval ends last.
			victim := it
			vi := -1
			for i, a := range active {
				if a.end > victim.end {
					victim, vi = a, i
				}
			}
			if spills == abi.SpillSlots {
				return fmt.Errorf("%w: more than %d spills", ErrUnsupported, abi.SpillSlots)
			}
			if victim == it {
				c.locs[it.v] = loc{reg: -1, spill: spills}
			} else {
				c.locs[it.v] = loc{reg: c.locs[victim.v].reg, spill: -1}
				c.locs[victim.v] = loc{reg: -1, spill: spills}
				active[vi] = it
			}
			spills++
		}
	}
	return nil
}

// numberWord is a number's bits as a word: NaN canonical.
func (c *core) numberWord(bits uint64) uint64 {
	if math.IsNaN(math.Float64frombits(bits)) {
		return c.enc.CanonicalNaN
	}
	return bits
}

// constWord is a tagged constant's word.
func (c *core) constWord(k ir.Value) uint64 {
	switch k.Kind {
	case ir.Number:
		return c.numberWord(k.Bits)
	case ir.Undefined:
		return c.enc.Undefined
	case ir.Null:
		return c.enc.Null
	case ir.Boolean:
		if k.Bits != 0 {
			return c.enc.True
		}
		return c.enc.False
	case ir.Uninitialized:
		return c.enc.Uninitialized
	}
	panic(fmt.Sprintf("tagged constant of kind %d", k.Kind))
}
func exitKind(aux int) uint64 {
	if ir.ExitKind(aux) == ir.HostExit {
		return abi.ExitHost
	}
	return abi.ExitDeopt
}

// remat reports a value an exit recomputes from its operands rather than
// reads from its location. A poll exits on a back-edge, after the phi moves
// and before the header runs again, so a box the header computed of a phi
// would still hold the last iteration's value; its operand, the phi, holds
// this one's.
func remat(v *ssa.Value) bool {
	return v.Op == ssa.OpBoxF64 || v.Op == ssa.OpBoxBool || v.Op == ssa.OpConst
}

// describe lists each value's location, in block order.
func (c *core) describe() string {
	out := ""
	for _, b := range c.order {
		for _, v := range b.Values {
			l, ok := c.locs[v]
			switch {
			case c.lazy[v]:
				out += fmt.Sprintf("%v:lazy ", v)
			case !ok:
			case l.reg >= 0 && isFloat(v):
				out += fmt.Sprintf("%v:x%d ", v, l.reg)
			case l.reg >= 0:
				out += fmt.Sprintf("%v:r%d ", v, l.reg)
			default:
				out += fmt.Sprintf("%v:spill%d ", v, l.spill)
			}
		}
	}
	return out
}

// isFloat reports a value's register class: true for floating point.
func isFloat(v *ssa.Value) bool { return v.Type == ssa.Float64 }

func hasResult(v *ssa.Value) bool { return v.Type != ssa.None }

// interval is a value's live range over the linear instruction numbering.
type interval struct {
	v          *ssa.Value
	start, end int
}

func (c *core) spillDisp(slot int) int32 { return abi.OffSpill + int32(slot)*8 }

// captured reports whether a slot is past the frame's locals: a captured
// binding, or the receiver.
func (c *core) captured(slot int) bool { return slot >= c.f.FrameLocals && slot < c.f.Locals }

// phiStep is one step of an edge's parallel move: park a value in the
// scratch register of its class (park), or move src into dst's location,
// from scratch when parked.
type phiStep struct {
	park, dst, src *ssa.Value
	parked         bool
}

// phiSchedule orders the parallel move of an edge's phi arguments into the
// phis' locations: a move once no pending move still reads its destination,
// and a cycle broken by parking one source in scratch.
func (c *core) phiSchedule(to *ssa.Block, idx int) []phiStep {
	type move struct{ dst, src *ssa.Value }
	var moves []move
	for _, phi := range to.Values {
		if phi.Op != ssa.OpPhi {
			break
		}
		if _, ok := c.locs[phi]; !ok {
			continue // dead
		}
		src := phi.Args[idx]
		if !c.lazy[src] && c.locs[src] == c.locs[phi] {
			continue
		}
		moves = append(moves, move{phi, src})
	}
	// Locations of different classes never coincide: R10 is not X10.
	same := func(a, b *ssa.Value) bool {
		if c.lazy[a] || c.lazy[b] || isFloat(a) != isFloat(b) {
			return false
		}
		return c.locs[a] == c.locs[b]
	}
	var steps []phiStep
	parked := map[*ssa.Value]bool{}
	for len(moves) > 0 {
		progress := false
		for i := 0; i < len(moves); i++ {
			m := moves[i]
			blocked := false
			for j, o := range moves {
				if j != i && same(o.src, m.dst) && !parked[o.src] {
					blocked = true
					break
				}
			}
			if blocked {
				continue
			}
			steps = append(steps, phiStep{dst: m.dst, src: m.src, parked: parked[m.src]})
			moves = append(moves[:i], moves[i+1:]...)
			i--
			progress = true
		}
		if !progress {
			// Every move is blocked, so they form cycles. Park the value
			// that blocks the first move -- the one in its destination -- in
			// scratch, which frees that destination. Its own move then reads
			// scratch. The cycle unwinds before another needs scratch.
			var blocker *ssa.Value
			for _, o := range moves {
				if same(o.src, moves[0].dst) && !parked[o.src] {
					blocker = o.src
					break
				}
			}
			if blocker == nil {
				panic("phi moves: blocked without a blocker")
			}
			// One scratch register per class: a second value parked while
			// another of its class still waits would overwrite it. A cycle
			// unwinds before the next needs scratch, so this cannot happen;
			// if it did, the function is refused, never miscompiled.
			for _, o := range moves {
				if parked[o.src] && isFloat(o.src) == isFloat(blocker) {
					panic("phi moves: scratch already holds a parked value")
				}
			}
			steps = append(steps, phiStep{park: blocker})
			parked[blocker] = true
		}
	}
	return steps
}
