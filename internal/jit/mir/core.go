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
	"math/bits"
	"runtime"
	"slices"

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
	// rematerialized where needed. lazy, locs and hasLoc are by value ID.
	lazy   []bool
	locs   []loc
	hasLoc []bool
	// origin is each tagged value's (ssa.Origins): the slot an exit looks
	// in for the reference the value may be.
	origin ssa.OriginMap
	// gprs and fprs are the architecture's allocatable registers, by number.
	gprs, fprs []int
	// ws is the workspace the compile takes its tables from, or nil.
	ws *Workspace
	// moves and steps are phiSchedule's, reused from edge to edge.
	moves  []phiMove
	steps  []phiStep
	parked []*ssa.Value
}

// prepare checks that f is one the backends compile, and analyses and
// allocates it, with the architecture's allocatable registers.
func prepare(w *Workspace, f *ssa.Func, enc abi.Encoding, gprs, fprs []int) (*core, error) {
	c := &core{f: f, enc: enc, gprs: gprs, fprs: fprs, ws: w}
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
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if o, ok := c.origin.Of(v); ok && (o == ssa.OriginAmbiguous || o == ssa.OriginHeap) && v.Shadow == nil {
				return nil, fmt.Errorf("%w: %v merges two slots' values with no shadow", ErrUnsupported, v)
			}
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
	// core is the compiler's state, for Locations.
	core *core
}

// Locations lists where each value lives, for tests and debugging: made on
// demand, since compiling at run time never asks.
func (c *Code) Locations() string { return c.core.describe() }

// loc is where a value lives: a register of its class, or a spill slot.
type loc struct {
	reg   int // register number, or -1
	spill int // spill slot, or -1
}

// isLazy reports whether v has no location of its own (core.lazy).
func (c *core) isLazy(v *ssa.Value) bool { return v.ID < len(c.lazy) && c.lazy[v.ID] }

// loc is v's location, and whether it has one.
func (c *core) loc(v *ssa.Value) (loc, bool) {
	if v.ID >= len(c.locs) || !c.hasLoc[v.ID] {
		return loc{}, false
	}
	return c.locs[v.ID], true
}

// locAt is v's location, or the zero location if it has none.
func (c *core) locAt(v *ssa.Value) loc {
	l, _ := c.loc(v)
	return l
}

func (c *core) setLoc(v *ssa.Value, l loc) {
	c.locs[v.ID], c.hasLoc[v.ID] = l, true
}

// valueSet is a set of value IDs.
type valueSet []uint64

func (s valueSet) add(i int)      { s[i>>6] |= 1 << (i & 63) }
func (s valueSet) has(i int) bool { return s[i>>6]&(1<<(i&63)) != 0 }
func (s valueSet) each(f func(int)) {
	for w, word := range s {
		for word != 0 {
			f(w*64 + bits.TrailingZeros64(word))
			word &= word - 1
		}
	}
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
	seen := c.bools(numBlocks(c.f))
	post := c.blockList(len(c.f.Blocks))[:0]
	var visit func(*ssa.Block)
	visit = func(b *ssa.Block) {
		if seen[b.ID] {
			return
		}
		seen[b.ID] = true
		for i := len(b.Succs) - 1; i >= 0; i-- {
			visit(b.Succs[i])
		}
		post = append(post, b)
	}
	for i := len(c.f.Entries) - 1; i >= 0; i-- {
		visit(c.f.Entries[i].Block)
	}
	c.order = c.blockList(len(post))[:0]
	for i := len(post) - 1; i >= 0; i-- {
		c.order = append(c.order, post[i])
	}
}

// findLazy marks values that need no location of their own.
func (c *core) findLazy() {
	n := c.f.NumValues()
	used := c.bools(n) // by an argument, a control or a phi
	for _, b := range c.f.Blocks {
		for _, v := range b.Values {
			for _, a := range v.Args {
				used[a.ID] = true
			}
		}
		if b.Control != nil && b.Kind != ssa.BlockReturn {
			used[b.Control.ID] = true
		}
	}
	// A slot loaded at an entry and named only by frame states for that same
	// slot needs no register: an exit leaves that slot as it is.
	elsewhere := c.bools(n)
	state := func(s *ssa.FrameState) {
		if s == nil {
			return
		}
		for i, v := range s.Slots {
			if !(v.Op == ssa.OpLoadSlot && v.Aux == i) {
				elsewhere[v.ID] = true
			}
		}
	}
	for _, b := range c.f.Blocks {
		for _, v := range b.Values {
			for _, a := range v.Args {
				if a.Op == ssa.OpLoadSlot {
					elsewhere[a.ID] = true
				}
			}
			state(v.State)
		}
		state(b.State)
		state(b.Header)
		if b.Control != nil && b.Control.Op == ssa.OpLoadSlot {
			elsewhere[b.Control.ID] = true
		}
	}
	c.lazy = c.bools(n)
	for _, b := range c.f.Blocks {
		for _, v := range b.Values {
			switch v.Op {
			case ssa.OpConstSource:
				c.lazy[v.ID] = true
			case ssa.OpConst, ssa.OpBoxF64, ssa.OpBoxBool:
				if !used[v.ID] {
					c.lazy[v.ID] = true
				}
			case ssa.OpLoadSlot:
				if !elsewhere[v.ID] {
					c.lazy[v.ID] = true
				}
			}
		}
	}
}

// numBlocks bounds f's block IDs.
func numBlocks(f *ssa.Func) int {
	n := 0
	for _, b := range f.Blocks {
		n = max(n, b.ID+1)
	}
	return n
}

// allocate computes liveness and assigns locations by linear scan. Its
// tables are indexed by value and block ID: a compile at run time pays for
// every pass (BenchmarkJITCompile).
func (c *core) allocate() error {
	nv, nb := c.f.NumValues(), numBlocks(c.f)
	// Number positions: each block has a start, one position per value, and
	// an end. at is the block of each position.
	pos := c.ints(nv)
	for i := range pos {
		pos[i] = -1
	}
	start, end := c.ints(nb), c.ints(nb)
	count := 0
	for _, b := range c.order {
		count += len(b.Values) + 2
	}
	at := c.blockList(count)[:0]
	for _, b := range c.order {
		start[b.ID] = len(at)
		at = append(at, b)
		for _, v := range b.Values {
			pos[v.ID] = len(at)
			at = append(at, b)
		}
		end[b.ID] = len(at)
		at = append(at, b)
	}
	// Uses: each value's uses, by the position of the use.
	uses := c.useList(4 * count)[:0]
	need := func(v *ssa.Value) bool { return hasResult(v) && !c.isLazy(v) }
	useState := func(s *ssa.FrameState, at int) {
		if s == nil {
			return
		}
		for _, v := range s.Slots {
			if c.isLazy(v) || remat(v) {
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
				uses = append(uses, use{a, pos[v.ID]})
			}
			useState(v.State, pos[v.ID])
		}
		if b.Control != nil {
			if c.isLazy(b.Control) {
				for _, a := range b.Control.Args {
					uses = append(uses, use{a, end[b.ID]})
				}
			} else {
				uses = append(uses, use{b.Control, end[b.ID]})
				if b.Control.Shadow != nil {
					uses = append(uses, use{b.Control.Shadow, end[b.ID]})
				}
			}
		}
		useState(b.State, end[b.ID])
		if b.LoopHeader {
			useState(b.Header, start[b.ID])
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
					uses = append(uses, use{phi.Args[i], end[b.ID]})
				}
				if s.LoopHeader && s.Backedge[i] {
					// A poll on this edge exits to the header's state.
					useState(s.Header, end[b.ID])
				}
			}
		}
	}
	// Block-level liveness, so that values live around a loop stay live for
	// all of it. byID finds a value from its ID.
	byID := c.valueList(nv)
	// The three sets of every block share one array.
	words := (nv + 63) / 64
	sets := c.words(3 * nb * words)
	defsIn, usesIn, liveIn := c.setList(nb), c.setList(nb), c.setList(nb)
	for _, b := range c.order {
		at := 3 * b.ID * words
		defsIn[b.ID], usesIn[b.ID], liveIn[b.ID] = sets[at:at+words:at+words], sets[at+words:at+2*words:at+2*words], sets[at+2*words:at+3*words:at+3*words]
		for _, v := range b.Values {
			defsIn[b.ID].add(v.ID)
			byID[v.ID] = v
		}
	}
	for _, u := range uses {
		if !need(u.v) {
			continue
		}
		b := at[u.pos]
		byID[u.v.ID] = u.v
		if !defsIn[b.ID].has(u.v.ID) {
			usesIn[b.ID].add(u.v.ID)
		}
	}
	in := c.words(words)
	for changed := true; changed; {
		changed = false
		for i := len(c.order) - 1; i >= 0; i-- {
			b := c.order[i]
			copy(in, usesIn[b.ID])
			defs := defsIn[b.ID]
			for _, s := range b.Succs {
				if live := liveIn[s.ID]; live != nil {
					for w := range in {
						in[w] |= live[w] &^ defs[w]
					}
				}
			}
			if !slices.Equal(in, liveIn[b.ID]) {
				copy(liveIn[b.ID], in)
				changed = true
			}
		}
	}
	// Intervals: from definition to last use, stretched over every block
	// where the value is live in or out.
	iv := c.intervalList(nv) // by value ID; v is nil until made
	get := func(v *ssa.Value) *interval {
		it := &iv[v.ID]
		if it.v == nil {
			p := pos[v.ID]
			if p < 0 {
				panic(fmt.Sprintf("use of %v defined outside the function", v))
			}
			if v.Op == ssa.OpPhi {
				p = start[v.Block.ID]
			}
			*it = interval{v: v, start: p, end: p}
		}
		return it
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
		bs := start[b.ID]
		liveIn[b.ID].each(func(id int) {
			it := get(byID[id])
			if bs < it.start {
				it.start = bs
			}
			if bs > it.end {
				it.end = bs
			}
		})
		for _, s := range b.Succs {
			if liveIn[s.ID] == nil {
				continue
			}
			liveIn[s.ID].each(func(id int) {
				v := byID[id]
				if !need(v) || v.Block == s && v.Op == ssa.OpPhi {
					return
				}
				it := get(v)
				if end[b.ID] > it.end {
					it.end = end[b.ID]
				}
				if bs < it.start && !defsIn[b.ID].has(id) {
					it.start = bs
				}
			})
		}
	}
	all := c.intervalRefs(nv)[:0]
	for i := range iv {
		if iv[i].v != nil {
			all = append(all, &iv[i])
		}
	}
	slices.SortFunc(all, func(a, b *interval) int {
		if a.start != b.start {
			return a.start - b.start
		}
		return a.v.ID - b.v.ID
	})
	c.locs, c.hasLoc = c.locList(nv), c.bools(nv)
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
					free = append(free, c.locAt(a.v).reg)
				} else {
					kept = append(kept, a)
				}
			}
			active = kept
			if len(free) > 0 {
				c.setLoc(it.v, loc{reg: free[len(free)-1], spill: -1})
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
				c.setLoc(it.v, loc{reg: -1, spill: spills})
			} else {
				c.setLoc(it.v, loc{reg: c.locAt(victim.v).reg, spill: -1})
				c.setLoc(victim.v, loc{reg: -1, spill: spills})
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
			l, ok := c.loc(v)
			switch {
			case c.isLazy(v):
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

// use is a value's use, at the position of the instruction that uses it.
type use struct {
	v   *ssa.Value
	pos int
}

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
// phiMove is a move phiSchedule orders.
type phiMove struct{ dst, src *ssa.Value }

func (c *core) phiSchedule(to *ssa.Block, idx int) []phiStep {
	moves := c.moves[:0]
	for _, phi := range to.Values {
		if phi.Op != ssa.OpPhi {
			break
		}
		if _, ok := c.loc(phi); !ok {
			continue // dead
		}
		src := phi.Args[idx]
		if !c.isLazy(src) && c.locAt(src) == c.locAt(phi) {
			continue
		}
		moves = append(moves, phiMove{phi, src})
	}
	// Locations of different classes never coincide: R10 is not X10.
	same := func(a, b *ssa.Value) bool {
		if c.isLazy(a) || c.isLazy(b) || isFloat(a) != isFloat(b) {
			return false
		}
		return c.locAt(a) == c.locAt(b)
	}
	steps := c.steps[:0]
	c.parked = c.parked[:0]
	isParked := func(v *ssa.Value) bool { return slices.Contains(c.parked, v) }
	for len(moves) > 0 {
		progress := false
		for i := 0; i < len(moves); i++ {
			m := moves[i]
			blocked := false
			for j, o := range moves {
				if j != i && same(o.src, m.dst) && !isParked(o.src) {
					blocked = true
					break
				}
			}
			if blocked {
				continue
			}
			steps = append(steps, phiStep{dst: m.dst, src: m.src, parked: isParked(m.src)})
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
				if same(o.src, moves[0].dst) && !isParked(o.src) {
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
				if isParked(o.src) && isFloat(o.src) == isFloat(blocker) {
					panic("phi moves: scratch already holds a parked value")
				}
			}
			steps = append(steps, phiStep{park: blocker})
			c.parked = append(c.parked, blocker)
		}
	}
	c.moves, c.steps = moves[:0], steps
	return steps
}

// Compile compiles f for the architecture the program runs on, given the
// VM's value encoding: amd64 or arm64.
func Compile(f *ssa.Func, enc abi.Encoding) (*Code, error) {
	if runtime.GOARCH == "arm64" {
		return CompileARM64(f, enc)
	}
	return CompileAMD64(f, enc)
}
