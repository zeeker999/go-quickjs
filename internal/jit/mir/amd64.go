// Package mir compiles the JIT's SSA (internal/jit/ssa) to machine code:
// it orders blocks, computes liveness, allocates registers by linear scan and
// selects instructions, then encodes them (internal/jit/asm). The walking
// skeleton of docs/jit-phase2-design.md: amd64 first, and plain code over
// clever code, verified against the SSA evaluator.
package mir

import (
	"errors"
	"fmt"
	"math"
	"sort"

	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
	"github.com/go-quickjs/go-quickjs/internal/jit/asm/amd64"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
	"github.com/go-quickjs/go-quickjs/internal/jit/ssa"
)

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

// The fixed registers. RSP, RBP and R14 (Go's g) are never touched.
const (
	regCtx    = amd64.RDI
	regLocals = amd64.RSI
	regStack  = amd64.RDX
	scratchA  = amd64.RAX
	scratchC  = amd64.RCX // also shift counts
	scratchB  = amd64.R11
	xScratch0 = amd64.XReg(0)
	xScratch1 = amd64.XReg(1)
)

var (
	gprPool = []amd64.Reg{amd64.RBX, amd64.R8, amd64.R9, amd64.R10, amd64.R12, amd64.R13, amd64.R15}
	xmmPool = func() (rs []amd64.XReg) {
		for r := amd64.XReg(2); r < 16; r++ {
			rs = append(rs, r)
		}
		return
	}()
)

// loc is where a value lives: a register of its class, or a spill slot.
type loc struct {
	reg   int // register number, or -1
	spill int // spill slot, or -1
}

type compiler struct {
	f     *ssa.Func
	enc   abi.Encoding
	a     amd64.Asm
	order []*ssa.Block
	// lazy values have no location: constants used only by frame states,
	// and boxes used only by frame states and returns, which are
	// rematerialized where needed.
	lazy map[*ssa.Value]bool
	locs map[*ssa.Value]loc
	// origin is each tagged value's (ssa.Origins): the slot an exit looks
	// in for the reference the value may be.
	origin map[*ssa.Value]int
	// label of each block's code.
	labels map[*ssa.Block]amd64.Label
	// stubs: one exit per frame state and kind.
	stubs   map[stubKey]amd64.Label
	stubFor []stub
	// cold code, emitted after everything else.
	cold []func()
}

type stubKey struct {
	state *ssa.FrameState
	kind  uint64
}

type stub struct {
	key   stubKey
	label amd64.Label
}

// CompileAMD64 compiles f for amd64, given the VM's value encoding.
func CompileAMD64(f *ssa.Func, enc abi.Encoding) (code *Code, err error) {
	defer func() {
		if v := recover(); v != nil {
			code, err = nil, fmt.Errorf("%w: %v", ErrUnsupported, v)
		}
	}()
	c := &compiler{f: f, enc: enc, labels: map[*ssa.Block]amd64.Label{}, stubs: map[stubKey]amd64.Label{}}
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
	entries := map[int]int{}
	for _, b := range c.order {
		c.labels[b] = c.a.NewLabel()
	}
	for i, b := range c.order {
		c.a.Bind(c.labels[b])
		if b.PC < 0 {
			for _, e := range f.Entries {
				if e.Block == b {
					entries[e.PC] = c.a.Len()
				}
			}
		}
		var next *ssa.Block
		if i+1 < len(c.order) {
			next = c.order[i+1]
		}
		c.block(b, next)
	}
	for i := 0; i < len(c.stubFor); i++ {
		s := c.stubFor[i]
		c.a.Bind(s.label)
		c.exitTo(s.key.state, s.key.kind)
	}
	for i := 0; i < len(c.cold); i++ {
		c.cold[i]()
	}
	bytes, err := c.a.Finish()
	if err != nil {
		return nil, err
	}
	return &Code{Bytes: bytes, Entries: entries, Locations: c.describe()}, nil
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
func (c *compiler) layout() {
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
func (c *compiler) findLazy() {
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

// class reports a value's register class: true for SSE.
func isFloat(v *ssa.Value) bool { return v.Type == ssa.Float64 }

func hasResult(v *ssa.Value) bool { return v.Type != ssa.None }

// interval is a value's live range over the linear instruction numbering.
type interval struct {
	v          *ssa.Value
	start, end int
}

// allocate computes liveness and assigns locations by linear scan.
func (c *compiler) allocate() error {
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
			for _, r := range xmmPool {
				free = append(free, int(r))
			}
		} else {
			for _, r := range gprPool {
				free = append(free, int(r))
			}
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

func (c *compiler) spillDisp(slot int) int32 { return abi.OffSpill + int32(slot)*8 }

// slotAddr is the base register and displacement of a frame slot's word.
// A captured binding's value is found through its cell, into scratch.
func (c *compiler) slotAddr(slot int, ref bool, scratch amd64.Reg) (amd64.Reg, int32) {
	off := c.enc.NumOffset
	if ref {
		off = c.enc.RefOffset
	}
	switch {
	case slot < c.f.FrameLocals:
		return regLocals, int32(slot)*c.enc.ValueSize + off
	case slot == c.f.ThisSlot:
		return regCtx, abi.OffThis + off
	case slot < c.f.Locals:
		c.a.Load(scratch, regCtx, abi.OffUpvalues)
		c.a.Load(scratch, scratch, int32(slot-c.f.FrameLocals)*8)
		c.a.Load(scratch, scratch, c.enc.UpvalueSlot)
		return scratch, off
	}
	return regStack, int32(slot-c.f.Locals)*c.enc.ValueSize + off
}

// captured reports whether a slot is past the frame's locals: a captured
// binding, or the receiver.
func (c *compiler) captured(slot int) bool { return slot >= c.f.FrameLocals && slot < c.f.Locals }

// gpr returns a register holding v's word, loading a spilled or lazy value
// into scratch.
func (c *compiler) gpr(v *ssa.Value, scratch amd64.Reg) amd64.Reg {
	if c.lazy[v] {
		c.materialize(v, scratch)
		return scratch
	}
	l, ok := c.locs[v]
	if !ok {
		panic(fmt.Sprintf("no location for %v (%v)", v, v.Op))
	}
	if l.reg >= 0 {
		return amd64.Reg(l.reg)
	}
	c.a.Load(scratch, regCtx, c.spillDisp(l.spill))
	return scratch
}

// xmm returns an SSE register holding v.
func (c *compiler) xmm(v *ssa.Value, scratch amd64.XReg) amd64.XReg {
	l, ok := c.locs[v]
	if !ok {
		panic(fmt.Sprintf("no location for %v (%v)", v, v.Op))
	}
	if l.reg >= 0 {
		return amd64.XReg(l.reg)
	}
	c.a.LoadSD(scratch, regCtx, c.spillDisp(l.spill))
	return scratch
}

// setG stores src into v's location.
func (c *compiler) setG(v *ssa.Value, src amd64.Reg) {
	l := c.locs[v]
	if l.reg >= 0 {
		if amd64.Reg(l.reg) != src {
			c.a.MovRR(amd64.Reg(l.reg), src)
		}
		return
	}
	c.a.Store(regCtx, c.spillDisp(l.spill), src)
}

// setX stores src into v's location.
func (c *compiler) setX(v *ssa.Value, src amd64.XReg) {
	l := c.locs[v]
	if l.reg >= 0 {
		if amd64.XReg(l.reg) != src {
			c.a.SSEOp(amd64.MovAPD, amd64.XReg(l.reg), src)
		}
		return
	}
	c.a.StoreSD(regCtx, c.spillDisp(l.spill), src)
}

// numberWord is a number's bits as a word: NaN canonical.
func (c *compiler) numberWord(bits uint64) uint64 {
	if math.IsNaN(math.Float64frombits(bits)) {
		return c.enc.CanonicalNaN
	}
	return bits
}

// constWord is a tagged constant's word.
func (c *compiler) constWord(k ir.Value) uint64 {
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

// materialize computes a lazy value's word into dst.
func (c *compiler) materialize(v *ssa.Value, dst amd64.Reg) {
	switch v.Op {
	case ssa.OpConst:
		c.a.MovImm(dst, c.constWord(v.Const))
	case ssa.OpConstSource:
		c.a.MovImm(dst, uint64(int64(v.Aux)))
	case ssa.OpBoxF64:
		c.boxF64(c.xmm(v.Args[0], xScratch1), dst)
	case ssa.OpBoxBool:
		c.boxBool(c.gpr(v.Args[0], scratchB), dst)
	default:
		panic("materialize " + v.Op.String())
	}
}

// boxF64 puts x's bits in dst, with NaN canonical.
func (c *compiler) boxF64(x amd64.XReg, dst amd64.Reg) {
	done := c.a.NewLabel()
	c.a.MovQFromX(dst, x)
	c.a.SSEOp(amd64.UcomiSD, x, x)
	c.a.Jcc(amd64.CondNP, done)
	c.a.MovImm(dst, c.enc.CanonicalNaN)
	c.a.Bind(done)
}

// boxBool puts the word for b (0 or 1) in dst. dst may be b.
func (c *compiler) boxBool(b, dst amd64.Reg) {
	tr, done := c.a.NewLabel(), c.a.NewLabel()
	c.a.Op(amd64.Test, b, b, false)
	c.a.Jcc(amd64.CondNE, tr)
	c.a.MovImm(dst, c.enc.False)
	c.a.Jmp(done)
	c.a.Bind(tr)
	c.a.MovImm(dst, c.enc.True)
	c.a.Bind(done)
}

// stubLabel returns the exit stub for a state and kind.
func (c *compiler) stubLabel(s *ssa.FrameState, kind uint64) amd64.Label {
	k := stubKey{s, kind}
	if l, ok := c.stubs[k]; ok {
		return l
	}
	l := c.a.NewLabel()
	c.stubs[k] = l
	c.stubFor = append(c.stubFor, stub{k, l})
	return l
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

// exitTo writes a frame state into the frame and returns with an exit
// record. A slot holding what was loaded from it at this entry is left as
// it is. Every other slot gets its value's word, boxes and constants
// recomputed from their operands -- unless the value is the reference its
// origin slot held, which Go copies, or the slot holds a reference, whose
// pointer word Go clears: those are records (abi.Record). The frame is as
// it was at entry until here, so the origin slot still holds the
// reference, and a slot this stub has written held none.
func (c *compiler) exitTo(s *ssa.FrameState, kind uint64) {
	c.a.MovImm(scratchC, 0)
	c.a.Store(regCtx, abi.OffRecords, scratchC)
	for i, v := range s.Slots {
		if v.Op == ssa.OpLoadSlot && v.Aux == i || c.captured(i) {
			// Unchanged: a captured binding's own value (capturedUnchanged).
			continue
		}
		var w amd64.Reg
		if remat(v) {
			c.materialize(v, scratchA)
			w = scratchA
		} else {
			w = c.gpr(v, scratchA)
		}
		if v.Shadow != nil {
			// Which slot it came from is known only at run time: Go looks.
			c.appendRecord(uint64(i)|abi.RecordMaybe, c.gprAfter(v.Shadow), 0, false, &w)
			continue
		}
		next := c.a.NewLabel()
		if o, ok := c.origin[v]; ok && o >= 0 {
			scalar := c.a.NewLabel()
			c.isReference(v, w, o, scalar)
			if o != i {
				c.appendRecord(uint64(i), nil, uint64(o), true, nil)
			}
			c.a.Jmp(next)
			c.a.Bind(scalar)
		}
		record := c.a.NewLabel()
		base, disp := c.slotAddr(i, true, scratchC)
		c.a.Load(scratchC, base, disp)
		c.a.Op(amd64.Test, scratchC, scratchC, true)
		c.a.Jcc(amd64.CondNE, record)
		base, disp = c.slotAddr(i, false, scratchC)
		c.a.Store(base, disp, w)
		c.a.Jmp(next)
		c.a.Bind(record)
		c.appendRecord(uint64(i)|abi.RecordScalar, nil, 0, true, &w)
		c.a.Bind(next)
	}
	c.record(kind, uint64(s.PC), uint64(s.Depth))
}

// isReference falls through when v, whose word is in w, is the reference
// slot o held at entry, and jumps to primitive when it is not: when the
// slot held none, or when v is not a load of it and its word is not the
// slot's. Native code makes no word of a reference's kind, so an equal
// word is the slot's value. It uses scratchC.
func (c *compiler) isReference(v *ssa.Value, w amd64.Reg, o int, primitive amd64.Label) {
	base, disp := c.slotAddr(o, true, scratchC)
	c.a.Load(scratchC, base, disp)
	c.a.Op(amd64.Test, scratchC, scratchC, true)
	c.a.Jcc(amd64.CondE, primitive)
	if v.Op != ssa.OpLoadSlot {
		base, disp = c.slotAddr(o, false, scratchC)
		c.a.Load(scratchC, base, disp)
		c.a.Op(amd64.Cmp, w, scratchC, true)
		c.a.Jcc(amd64.CondNE, primitive)
	}
}

// appendRecord adds an abi.Record for slot: its Arg from arg, which
// yields a register once the record's address is in scratchC, or imm when
// useImm; and its Word from word, if not nil, which must not be scratchB
// or scratchC. It uses scratchB and scratchC.
func (c *compiler) appendRecord(slot uint64, arg func() amd64.Reg, imm uint64, useImm bool, word *amd64.Reg) {
	c.a.Load(scratchC, regCtx, abi.OffRecords)
	c.a.ShiftImm(amd64.Shl, scratchC, 5, true)
	c.a.Op(amd64.Add, scratchC, regCtx, true)
	if word != nil {
		c.a.Store(scratchC, abi.OffRecord+16, *word)
	}
	c.a.MovImm(scratchB, slot)
	c.a.Store(scratchC, abi.OffRecord, scratchB)
	if useImm {
		c.a.MovImm(scratchB, imm)
		c.a.Store(scratchC, abi.OffRecord+8, scratchB)
	} else {
		c.a.Store(scratchC, abi.OffRecord+8, arg())
	}
	c.a.Load(scratchC, regCtx, abi.OffRecords)
	c.a.OpImm(amd64.Add, scratchC, 1, true)
	c.a.Store(regCtx, abi.OffRecords, scratchC)
}

// gprAfter returns a function yielding a register holding v, loading it
// into scratchB if it has none; for appendRecord, which has freed scratchB
// by then.
func (c *compiler) gprAfter(v *ssa.Value) func() amd64.Reg {
	return func() amd64.Reg { return c.gpr(v, scratchB) }
}

// record fills the exit record and returns to Go.
func (c *compiler) record(kind, pc, depth uint64) {
	c.a.MovImm(scratchA, kind)
	c.a.Store(regCtx, abi.OffExitKind, scratchA)
	c.a.MovImm(scratchA, pc)
	c.a.Store(regCtx, abi.OffExitPC, scratchA)
	c.a.MovImm(scratchA, depth)
	c.a.Store(regCtx, abi.OffExitDepth, scratchA)
	c.a.Ret()
}

// block emits one block. next is the block laid out after it.
func (c *compiler) block(b *ssa.Block, next *ssa.Block) {
	if b.PC < 0 {
		c.a.Load(regLocals, regCtx, abi.OffLocals)
		c.a.Load(regStack, regCtx, abi.OffStack)
	}
	for _, v := range b.Values {
		if v.Op == ssa.OpPhi || c.lazy[v] {
			continue
		}
		c.value(v, b)
	}
	switch b.Kind {
	case ssa.BlockPlain:
		c.edge(b, b.Succs[0], next)
	case ssa.BlockIf:
		c.branch(b, next)
	case ssa.BlockReturn:
		r := c.gpr(b.Control, scratchA)
		c.a.Store(regCtx, abi.OffRet, r)
		c.a.MovImm(scratchC, 0)
		c.a.Store(regCtx, abi.OffRetFrom, scratchC)
		if s := b.Control.Shadow; s != nil {
			// RetFrom is the source plus one, and 0 for a primitive's -1;
			// Go checks that the slot or cell holds a reference.
			c.a.MovRR(scratchC, c.gpr(s, scratchC))
			c.a.OpImm(amd64.Add, scratchC, 1, true)
			c.a.Store(regCtx, abi.OffRetFrom, scratchC)
		} else if o, ok := c.origin[b.Control]; ok && o >= 0 {
			done := c.a.NewLabel()
			c.isReference(b.Control, r, o, done)
			c.a.MovImm(scratchC, uint64(o)+1)
			c.a.Store(regCtx, abi.OffRetFrom, scratchC)
			c.a.Bind(done)
		}
		c.a.MovImm(scratchA, abi.ExitReturn)
		c.a.Store(regCtx, abi.OffExitKind, scratchA)
		c.a.Ret()
	case ssa.BlockExit:
		c.exitTo(b.State, exitKind(int(b.ExitKind)))
	}
}

// edge emits the moves for an edge's phis, a poll on a back-edge, and a jump
// unless the target is laid out next.
func (c *compiler) edge(from, to, next *ssa.Block) {
	idx := -1
	for i, p := range to.Preds {
		if p == from {
			idx = i
		}
	}
	c.phiMoves(to, idx)
	if to.LoopHeader && to.Backedge[idx] {
		poll := c.a.NewLabel()
		c.a.Load(scratchB, regCtx, abi.OffBackEdges)
		c.a.DecMem(scratchB, 0)
		c.a.Jcc(amd64.CondLE, poll)
		header := to.Header
		c.cold = append(c.cold, func() {
			c.a.Bind(poll)
			// The phis hold the header's values now; the state names them,
			// or values live across the loop.
			c.exitTo(header, abi.ExitPoll)
		})
	}
	if to != next {
		c.a.Jmp(c.labels[to])
	}
}

// phiMoves performs the parallel move of an edge's phi arguments into the
// phis' locations.
func (c *compiler) phiMoves(to *ssa.Block, idx int) {
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
	// Emit a move once no pending move still reads its destination; break a
	// cycle by parking one source in scratch.
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
			c.moveValue(m.dst, m.src, parked[m.src])
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
			if isFloat(blocker) {
				c.a.SSEOp(amd64.MovAPD, xScratch0, c.xmm(blocker, xScratch0))
			} else {
				c.a.MovRR(scratchC, c.gpr(blocker, scratchC))
			}
			parked[blocker] = true
		}
	}
}

// moveValue copies src into dst's location; parked sources are in scratch.
func (c *compiler) moveValue(dst, src *ssa.Value, parked bool) {
	if isFloat(dst) {
		// A spilled source loads through xScratch1: xScratch0 may hold a
		// parked value still to be moved.
		x := xScratch0
		if !parked {
			x = c.xmm(src, xScratch1)
		}
		c.setX(dst, x)
		return
	}
	r := scratchC
	if !parked {
		r = c.gpr(src, scratchA)
	}
	c.setG(dst, r)
}

// branch emits an If block's end, fusing a float comparison into the jump.
func (c *compiler) branch(b *ssa.Block, next *ssa.Block) {
	yes, no := c.a.NewLabel(), c.a.NewLabel()
	ctl := b.Control
	if ctl.Op == ssa.OpCmpF64 && ctl.Uses == 1 && ctl.Block == b {
		x, y := c.xmm(ctl.Args[0], xScratch0), c.xmm(ctl.Args[1], xScratch1)
		switch ir.Operator(ctl.Aux) {
		case ir.Lt:
			c.a.SSEOp(amd64.UcomiSD, y, x)
			c.a.Jcc(amd64.CondA, yes)
		case ir.Le:
			c.a.SSEOp(amd64.UcomiSD, y, x)
			c.a.Jcc(amd64.CondAE, yes)
		case ir.Gt:
			c.a.SSEOp(amd64.UcomiSD, x, y)
			c.a.Jcc(amd64.CondA, yes)
		case ir.Ge:
			c.a.SSEOp(amd64.UcomiSD, x, y)
			c.a.Jcc(amd64.CondAE, yes)
		case ir.Eq:
			c.a.SSEOp(amd64.UcomiSD, x, y)
			c.a.Jcc(amd64.CondNE, no)
			c.a.Jcc(amd64.CondNP, yes)
		case ir.Ne:
			c.a.SSEOp(amd64.UcomiSD, x, y)
			c.a.Jcc(amd64.CondNE, yes)
			c.a.Jcc(amd64.CondP, yes)
		}
		c.a.Jmp(no)
	} else {
		r := c.gpr(ctl, scratchA)
		c.a.Op(amd64.Test, r, r, false)
		c.a.Jcc(amd64.CondNE, yes)
		c.a.Jmp(no)
	}
	c.a.Bind(no)
	c.edge(b, b.Succs[1], nil)
	c.a.Bind(yes)
	c.edge(b, b.Succs[0], next)
}

// arrayOf finds the array a value is, as objectOf finds an object, and
// checks its class.
func (c *compiler) arrayOf(v *ssa.Value, guard func(amd64.Cond)) {
	if !c.objectOf(v, guard) {
		return
	}
	c.a.LoadU8(scratchB, scratchC, c.enc.ObjectClass)
	c.a.OpImm(amd64.Cmp, scratchB, int32(c.enc.ClassArray), false)
	guard(amd64.CondNE)
	c.setG(v, scratchC)
}

// objectOf finds the object a value is, into scratchC: the pointer word of
// the slot it came from, which still holds it (origin.go). A value with an
// object's word is its origin's object, since native code makes no such
// word. It reports false when the value can never be one, having emitted
// the jump to the exit.
func (c *compiler) objectOf(v *ssa.Value, guard func(amd64.Cond)) bool {
	return c.reference(v, v.Args[0], c.enc.Object, guard)
}

// reference finds the reference a is, of the kind whose word is word, into
// scratchC, as objectOf finds an object, exiting to v's state when a is not
// one. It uses scratchA and scratchB.
func (c *compiler) reference(v, a *ssa.Value, word uint64, guard func(amd64.Cond)) bool {
	o := c.origin[a]
	if a.Shadow == nil && o < 0 {
		// A primitive is never a reference.
		c.a.Jmp(c.stubLabel(v.State, exitKind(v.Aux)))
		return false
	}
	w := c.gpr(a, scratchA)
	c.a.MovImm(scratchB, word)
	c.a.Op(amd64.Cmp, w, scratchB, true)
	guard(amd64.CondNE)
	if s := a.Shadow; s != nil {
		// The source is known at run time: a local, a captured binding, the
		// receiver, an operand, or a heap cell (origin.go). scratchC becomes
		// its value's address.
		captured, stack, found := c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel()
		c.a.MovRR(scratchC, c.gpr(s, scratchC))
		c.a.Op(amd64.Test, scratchC, scratchC, true)
		guard(amd64.CondS)
		c.a.OpImm(amd64.Cmp, scratchC, abi.MaxRecords, true)
		c.a.Jcc(amd64.CondAE, found)
		c.a.OpImm(amd64.Cmp, scratchC, int32(c.f.FrameLocals), false)
		c.a.Jcc(amd64.CondGE, captured)
		c.a.ShiftImm(amd64.Shl, scratchC, 4, true)
		c.a.Op(amd64.Add, scratchC, regLocals, true)
		c.a.Jmp(found)
		c.a.Bind(captured)
		c.a.OpImm(amd64.Cmp, scratchC, int32(c.f.Locals), false)
		c.a.Jcc(amd64.CondGE, stack)
		if t := c.f.ThisSlot; t >= 0 {
			notThis := c.a.NewLabel()
			c.a.OpImm(amd64.Cmp, scratchC, int32(t), false)
			c.a.Jcc(amd64.CondNE, notThis)
			c.a.MovRR(scratchC, regCtx)
			c.a.OpImm(amd64.Add, scratchC, abi.OffThis, true)
			c.a.Jmp(found)
			c.a.Bind(notThis)
		}
		c.a.OpImm(amd64.Sub, scratchC, int32(c.f.FrameLocals), false)
		c.a.ShiftImm(amd64.Shl, scratchC, 3, true)
		c.a.Load(scratchB, regCtx, abi.OffUpvalues)
		c.a.Op(amd64.Add, scratchC, scratchB, true)
		c.a.Load(scratchC, scratchC, 0)
		c.a.Load(scratchC, scratchC, c.enc.UpvalueSlot)
		c.a.Jmp(found)
		c.a.Bind(stack)
		c.a.OpImm(amd64.Sub, scratchC, int32(c.f.Locals), false)
		c.a.ShiftImm(amd64.Shl, scratchC, 4, true)
		c.a.Op(amd64.Add, scratchC, regStack, true)
		c.a.Bind(found)
		c.a.Load(scratchC, scratchC, c.enc.RefOffset)
	} else {
		base, disp := c.slotAddr(o, true, scratchC)
		c.a.Load(scratchC, base, disp)
	}
	c.a.Op(amd64.Test, scratchC, scratchC, true)
	guard(amd64.CondE)
	return true
}

// property finds the property a property operation names and leaves the
// address of its value in scratchA. An object of the shape the site's cache
// knows has it at the cached index: the shape settles where it is and what
// it is. Any other ordinary object with a small table is searched for the
// key, unrolled, as the VM's own small objects are; the entry must be plain
// data, and writable for a write. It uses scratchB and scratchC.
func (c *compiler) property(v *ssa.Value, guard func(amd64.Cond)) {
	found, scan := c.a.NewLabel(), c.a.NewLabel()
	if v.Const.Bits != 0 {
		p := c.gpr(v.Args[0], scratchA)
		c.a.Load(scratchB, p, c.enc.ObjectShape)
		c.a.MovImm(scratchA, v.Const.Bits)
		c.a.Op(amd64.Cmp, scratchB, scratchA, true)
		c.a.Jcc(amd64.CondNE, scan)
		p = c.gpr(v.Args[0], scratchA)
		c.a.Load(scratchA, p, c.enc.ObjectProps)
		c.a.OpImm(amd64.Add, scratchA, int32(v.Index)*c.enc.PropertySize+c.enc.PropertyValue, true)
		c.a.Jmp(found)
	}
	c.a.Bind(scan)
	p := c.gpr(v.Args[0], scratchA)
	c.a.LoadU8(scratchB, p, c.enc.ObjectClass)
	c.a.OpImm(amd64.Cmp, scratchB, int32(c.enc.ClassObject), false)
	guard(amd64.CondNE)
	c.a.Load(scratchB, p, c.enc.ObjectProps+8)
	c.a.OpImm(amd64.Cmp, scratchB, abi.MaxScan, true)
	guard(amd64.CondA)
	c.a.Load(scratchA, p, c.enc.ObjectProps)
	mask, want := int32(c.enc.PropNotData), int32(0)
	if v.Op == ssa.OpPropWrite {
		mask, want = int32(c.enc.PropNotWritable), int32(c.enc.PropWritable)
	}
	for k := int32(0); k < abi.MaxScan; k++ {
		next := c.a.NewLabel()
		entry := k * c.enc.PropertySize
		c.a.OpImm(amd64.Cmp, scratchB, k, true)
		guard(amd64.CondBE)
		c.a.LoadU32(scratchC, scratchA, entry+c.enc.PropertyKey)
		c.a.OpImm(amd64.Cmp, scratchC, int32(v.Key), false)
		c.a.Jcc(amd64.CondNE, next)
		c.a.LoadU8(scratchC, scratchA, entry+c.enc.PropertyFlags)
		c.a.OpImm(amd64.And, scratchC, mask, false)
		c.a.OpImm(amd64.Cmp, scratchC, want, false)
		guard(amd64.CondNE)
		c.a.OpImm(amd64.Add, scratchA, entry+c.enc.PropertyValue, true)
		c.a.Jmp(found)
		c.a.Bind(next)
	}
	c.a.Jmp(c.stubLabel(v.State, exitKind(v.Aux)))
	c.a.Bind(found)
}

// length is x.length: a string's, which it keeps rope or not, or an
// array's, the dense count or a sparse array's length when that is larger,
// as Object.arrayLength has it.
func (c *compiler) length(v *ssa.Value, guard func(amd64.Cond)) {
	a := v.Args[0]
	if a.Shadow == nil && c.origin[a] < 0 {
		c.a.Jmp(c.stubLabel(v.State, exitKind(v.Aux)))
		return
	}
	array, have := c.a.NewLabel(), c.a.NewLabel()
	c.a.MovImm(scratchB, c.enc.String)
	c.a.Op(amd64.Cmp, c.gpr(a, scratchA), scratchB, true)
	c.a.Jcc(amd64.CondNE, array)
	c.reference(v, a, c.enc.String, guard)
	c.a.Load(scratchC, scratchC, c.enc.StringLength)
	c.a.Jmp(have)
	c.a.Bind(array)
	c.reference(v, a, c.enc.Object, guard)
	c.a.LoadU8(scratchB, scratchC, c.enc.ObjectClass)
	c.a.OpImm(amd64.Cmp, scratchB, int32(c.enc.ClassArray), false)
	guard(amd64.CondNE)
	p := scratchA
	c.a.MovRR(p, scratchC)
	c.a.Load(scratchC, p, c.enc.ObjectElems+8)
	c.a.LoadU8(scratchB, p, c.enc.ObjectFlags)
	c.a.OpImm(amd64.And, scratchB, int32(c.enc.FlagSparse), false)
	c.a.Jcc(amd64.CondE, have)
	c.a.LoadU32(scratchB, p, c.enc.ObjectArrayLen)
	c.a.Op(amd64.Cmp, scratchB, scratchC, true)
	c.a.Jcc(amd64.CondBE, have)
	c.a.MovRR(scratchC, scratchB)
	c.a.Bind(have)
	c.a.Cvtsi2sd(xScratch0, scratchC, true)
	c.setX(v, xScratch0)
}

// stringCode is charCodeAt called on a string: the callee must be the
// intrinsic, the string flat, and the index an integer below its length;
// the code unit is a byte of an ASCII string's UTF-8, or one of the code
// units a string caches, and anything else exits.
func (c *compiler) stringCode(v *ssa.Value, guard func(amd64.Cond)) {
	if !c.reference(v, v.Args[0], c.enc.Object, guard) {
		return
	}
	c.a.Load(scratchB, regCtx, abi.OffCharCode+c.enc.RefOffset)
	c.a.Op(amd64.Cmp, scratchC, scratchB, true)
	guard(amd64.CondNE)
	if !c.reference(v, v.Args[1], c.enc.String, guard) {
		return
	}
	c.a.Load(scratchB, scratchC, c.enc.StringLeft)
	c.a.Op(amd64.Test, scratchB, scratchB, true)
	guard(amd64.CondNE)
	c.index(v.Args[2], guard)
	c.a.Load(scratchB, scratchC, c.enc.StringLength)
	c.a.Op(amd64.Cmp, scratchA, scratchB, true)
	guard(amd64.CondAE)
	units, done := c.a.NewLabel(), c.a.NewLabel()
	c.a.LoadU8(scratchB, scratchC, c.enc.StringASCII)
	c.a.Op(amd64.Test, scratchB, scratchB, false)
	c.a.Jcc(amd64.CondE, units)
	c.a.Load(scratchB, scratchC, c.enc.StringData)
	c.a.Op(amd64.Add, scratchB, scratchA, true)
	c.a.LoadU8(scratchA, scratchB, 0)
	c.a.Jmp(done)
	c.a.Bind(units)
	c.a.Load(scratchB, scratchC, c.enc.StringU16)
	c.a.Op(amd64.Test, scratchB, scratchB, true)
	guard(amd64.CondE)
	c.a.Op(amd64.Add, scratchA, scratchA, true)
	c.a.Op(amd64.Add, scratchB, scratchA, true)
	c.a.LoadU16(scratchA, scratchB, 0)
	c.a.Bind(done)
	c.a.Cvtsi2sd(xScratch0, scratchA, true)
	c.setX(v, xScratch0)
}

// index converts an element's key, a double, to an index in scratchA,
// failing unless it is an integer in [0, 2**32): truncation and back must
// give the key (NaN and anything past 2**63 do not), and the top half must
// be clear. Negative zero is index 0. It uses scratchB and xScratch1.
func (c *compiler) index(key *ssa.Value, fail func(amd64.Cond)) {
	k := c.xmm(key, xScratch0)
	c.a.Cvttsd2si(scratchA, k)
	c.a.Cvtsi2sd(xScratch1, scratchA, true)
	c.a.SSEOp(amd64.UcomiSD, xScratch1, k)
	fail(amd64.CondNE)
	fail(amd64.CondP)
	c.a.MovRR(scratchB, scratchA)
	c.a.ShiftImm(amd64.Shr, scratchB, 32, true)
	fail(amd64.CondNE)
}

// element turns the index in scratchA into the address of an array's
// element there, in scratchA, and its number word, in scratchB, failing
// unless the element is present and holds a number. It uses scratchC.
func (c *compiler) element(array *ssa.Value, fail func(amd64.Cond)) {
	p := c.gpr(array, scratchC)
	c.a.Load(scratchB, p, c.enc.ObjectElems+8)
	c.a.Op(amd64.Cmp, scratchA, scratchB, true)
	fail(amd64.CondAE)
	c.a.Load(scratchB, p, c.enc.ObjectElems)
	c.a.ShiftImm(amd64.Shl, scratchA, 4, true)
	c.a.Op(amd64.Add, scratchA, scratchB, true)
	c.a.Load(scratchB, scratchA, c.enc.NumOffset)
	c.a.MovImm(scratchC, abi.NumberLimit)
	c.a.Op(amd64.Cmp, scratchB, scratchC, true)
	fail(amd64.CondAE)
}

// value emits one value.
func (c *compiler) value(v *ssa.Value, b *ssa.Block) {
	arg := func(i int) *ssa.Value { return v.Args[i] }
	guard := func(cond amd64.Cond) {
		c.a.Jcc(cond, c.stubLabel(v.State, exitKind(v.Aux)))
	}
	switch v.Op {
	case ssa.OpLoadSlot:
		base, disp := c.slotAddr(v.Aux, false, scratchA)
		c.a.Load(scratchA, base, disp)
		c.setG(v, scratchA)
	case ssa.OpConst:
		c.a.MovImm(scratchA, c.constWord(v.Const))
		c.setG(v, scratchA)
	case ssa.OpConstF64:
		c.a.MovImm(scratchA, v.Const.Bits)
		c.a.MovQToX(xScratch0, scratchA)
		c.setX(v, xScratch0)
	case ssa.OpUnboxF64:
		r := c.gpr(arg(0), scratchB)
		c.a.MovRR(scratchA, r)
		c.a.ShiftImm(amd64.Shr, scratchA, 51, true)
		c.a.OpImm(amd64.Cmp, scratchA, 0x1FFF, false)
		guard(amd64.CondE)
		c.a.MovQToX(xScratch0, r)
		c.setX(v, xScratch0)
	case ssa.OpCheckInit:
		r := c.gpr(arg(0), scratchB)
		c.a.MovImm(scratchA, c.enc.Uninitialized)
		c.a.Op(amd64.Cmp, r, scratchA, true)
		guard(amd64.CondE)
	case ssa.OpTruth:
		c.truth(v, guard)
	case ssa.OpArrayOf:
		c.arrayOf(v, guard)
	case ssa.OpObjectOf:
		if c.objectOf(v, guard) {
			c.setG(v, scratchC)
		}
	case ssa.OpPropRead:
		c.property(v, guard)
		c.a.Load(scratchB, scratchA, c.enc.NumOffset)
		c.a.MovImm(scratchC, abi.NumberLimit)
		c.a.Op(amd64.Cmp, scratchB, scratchC, true)
		guard(amd64.CondAE)
		c.a.MovQToX(xScratch0, scratchB)
		c.setX(v, xScratch0)
	case ssa.OpPropCell:
		c.property(v, guard)
		c.setG(v, scratchA)
	case ssa.OpGlobalCell:
		// No script-level lexical binding of the name, which would shadow
		// it: its bit clear, or past the set's words. Then the binding
		// where it was, the name's, and plain, initialized data.
		unshadowed := c.a.NewLabel()
		word := int32(v.Key >> 6)
		c.a.Load(scratchB, regCtx, abi.OffLexNames)
		c.a.Load(scratchA, scratchB, 8)
		c.a.OpImm(amd64.Cmp, scratchA, word, true)
		c.a.Jcc(amd64.CondBE, unshadowed)
		c.a.Load(scratchB, scratchB, 0)
		c.a.Load(scratchB, scratchB, word*8)
		c.a.MovImm(scratchA, 1<<(v.Key&63))
		c.a.Op(amd64.Test, scratchB, scratchA, true)
		guard(amd64.CondNE)
		c.a.Bind(unshadowed)
		c.a.Load(scratchA, regCtx, abi.OffGlobal)
		c.a.Op(amd64.Test, scratchA, scratchA, true)
		guard(amd64.CondE)
		c.a.Load(scratchB, scratchA, c.enc.ObjectProps+8)
		c.a.OpImm(amd64.Cmp, scratchB, int32(v.Index), true)
		guard(amd64.CondBE)
		c.a.Load(scratchA, scratchA, c.enc.ObjectProps)
		c.a.OpImm(amd64.Add, scratchA, int32(v.Index)*c.enc.PropertySize, true)
		c.a.LoadU32(scratchB, scratchA, c.enc.PropertyKey)
		c.a.OpImm(amd64.Cmp, scratchB, int32(v.Key), false)
		guard(amd64.CondNE)
		c.a.LoadU8(scratchB, scratchA, c.enc.PropertyFlags)
		c.a.OpImm(amd64.And, scratchB, int32(c.enc.PropNotData|c.enc.PropUninit), false)
		guard(amd64.CondNE)
		c.a.OpImm(amd64.Add, scratchA, c.enc.PropertyValue, true)
		c.setG(v, scratchA)
	case ssa.OpStringMethod:
		// A string's charCodeAt is the context's cell, if it holds the
		// intrinsic.
		c.a.MovImm(scratchB, c.enc.String)
		c.a.Op(amd64.Cmp, c.gpr(arg(0), scratchA), scratchB, true)
		guard(amd64.CondNE)
		c.a.Load(scratchB, regCtx, abi.OffCharCode+c.enc.RefOffset)
		c.a.Op(amd64.Test, scratchB, scratchB, true)
		guard(amd64.CondE)
		c.a.MovRR(scratchA, regCtx)
		c.a.OpImm(amd64.Add, scratchA, abi.OffCharCode, true)
		c.setG(v, scratchA)
	case ssa.OpStringCode:
		c.stringCode(v, guard)
	case ssa.OpLoadCell:
		c.a.Load(scratchA, c.gpr(arg(0), scratchA), c.enc.NumOffset)
		c.setG(v, scratchA)
	case ssa.OpPropWrite:
		c.property(v, guard)
		c.a.Load(scratchB, scratchA, c.enc.NumOffset)
		c.a.MovImm(scratchC, abi.NumberLimit)
		c.a.Op(amd64.Cmp, scratchB, scratchC, true)
		guard(amd64.CondAE)
		c.boxF64(c.xmm(arg(1), xScratch0), scratchB)
		c.a.Store(scratchA, c.enc.NumOffset, scratchB)
	case ssa.OpLength:
		c.length(v, guard)
	case ssa.OpElemKey:

		c.index(arg(0), guard)
	case ssa.OpElemRead:
		c.index(arg(1), guard)
		c.element(arg(0), guard)
		c.a.MovQToX(xScratch0, scratchB)
		c.setX(v, xScratch0)
	case ssa.OpElemWrite:
		c.index(arg(1), guard)
		c.element(arg(0), guard)
		c.boxF64(c.xmm(arg(2), xScratch0), scratchB)
		c.a.Store(scratchA, c.enc.NumOffset, scratchB)
	case ssa.OpBoxF64:
		c.boxF64(c.xmm(arg(0), xScratch0), scratchA)
		c.setG(v, scratchA)
	case ssa.OpBoxBool:
		c.boxBool(c.gpr(arg(0), scratchB), scratchA)
		c.setG(v, scratchA)
	case ssa.OpAddF64, ssa.OpSubF64, ssa.OpMulF64, ssa.OpDivF64:
		op := map[ssa.Op]amd64.SSE{ssa.OpAddF64: amd64.AddSD, ssa.OpSubF64: amd64.SubSD,
			ssa.OpMulF64: amd64.MulSD, ssa.OpDivF64: amd64.DivSD}[v.Op]
		x := c.xmm(arg(0), xScratch0)
		if x != xScratch0 {
			c.a.SSEOp(amd64.MovAPD, xScratch0, x)
		}
		c.a.SSEOp(op, xScratch0, c.xmm(arg(1), xScratch1))
		c.setX(v, xScratch0)
	case ssa.OpNegF64:
		x := c.xmm(arg(0), xScratch0)
		if x != xScratch0 {
			c.a.SSEOp(amd64.MovAPD, xScratch0, x)
		}
		c.a.MovImm(scratchA, 1<<63)
		c.a.MovQToX(xScratch1, scratchA)
		c.a.SSEOp(amd64.XorPD, xScratch0, xScratch1)
		c.setX(v, xScratch0)
	case ssa.OpCmpF64:
		if v.Uses == 1 && b.Control == v && b.Kind == ssa.BlockIf {
			return // fused into the branch
		}
		c.compare(v)
	case ssa.OpNot:
		c.a.MovRR32(scratchA, c.gpr(arg(0), scratchA))
		c.a.OpImm(amd64.Xor, scratchA, 1, false)
		c.setG(v, scratchA)
	case ssa.OpToInt32:
		c.toInt32(v)
	case ssa.OpAndI32, ssa.OpOrI32, ssa.OpXorI32:
		op := map[ssa.Op]amd64.ALU{ssa.OpAndI32: amd64.And, ssa.OpOrI32: amd64.Or, ssa.OpXorI32: amd64.Xor}[v.Op]
		c.a.MovRR32(scratchA, c.gpr(arg(0), scratchA))
		c.a.Op(op, scratchA, c.gpr(arg(1), scratchB), false)
		c.setG(v, scratchA)
	case ssa.OpShlI32, ssa.OpSarI32, ssa.OpShrU32:
		op := map[ssa.Op]amd64.Shift{ssa.OpShlI32: amd64.Shl, ssa.OpSarI32: amd64.Sar, ssa.OpShrU32: amd64.Shr}[v.Op]
		c.a.MovRR32(scratchC, c.gpr(arg(1), scratchC))
		c.a.MovRR32(scratchA, c.gpr(arg(0), scratchA))
		c.a.ShiftCL(op, scratchA, false)
		c.setG(v, scratchA)
	case ssa.OpNotI32:
		c.a.MovRR32(scratchA, c.gpr(arg(0), scratchA))
		c.a.Not32(scratchA)
		c.setG(v, scratchA)
	case ssa.OpI32ToF64:
		c.a.Cvtsi2sd(xScratch0, c.gpr(arg(0), scratchA), false)
		c.setX(v, xScratch0)
	case ssa.OpU32ToF64:
		c.a.MovRR32(scratchA, c.gpr(arg(0), scratchA))
		c.a.Cvtsi2sd(xScratch0, scratchA, true)
		c.setX(v, xScratch0)
	default:
		panic("value " + v.Op.String())
	}
}

// compare materializes a float comparison as 0 or 1.
func (c *compiler) compare(v *ssa.Value) {
	x, y := c.xmm(v.Args[0], xScratch0), c.xmm(v.Args[1], xScratch1)
	switch ir.Operator(v.Aux) {
	case ir.Lt:
		c.a.SSEOp(amd64.UcomiSD, y, x)
		c.a.Setcc(amd64.CondA, scratchA)
	case ir.Le:
		c.a.SSEOp(amd64.UcomiSD, y, x)
		c.a.Setcc(amd64.CondAE, scratchA)
	case ir.Gt:
		c.a.SSEOp(amd64.UcomiSD, x, y)
		c.a.Setcc(amd64.CondA, scratchA)
	case ir.Ge:
		c.a.SSEOp(amd64.UcomiSD, x, y)
		c.a.Setcc(amd64.CondAE, scratchA)
	case ir.Eq:
		c.a.SSEOp(amd64.UcomiSD, x, y)
		c.a.Setcc(amd64.CondE, scratchA)
		c.a.Setcc(amd64.CondNP, scratchC)
		c.a.Op(amd64.And, scratchA, scratchC, false)
	case ir.Ne:
		c.a.SSEOp(amd64.UcomiSD, x, y)
		c.a.Setcc(amd64.CondNE, scratchA)
		c.a.Setcc(amd64.CondP, scratchC)
		c.a.Op(amd64.Or, scratchA, scratchC, false)
	}
	c.a.MovZX8(scratchA, scratchA)
	c.setG(v, scratchA)
}

// truth computes JavaScript truthiness for the kinds the slot IR decides it
// for, exiting for the rest.
func (c *compiler) truth(v *ssa.Value, guard func(amd64.Cond)) {
	r := c.gpr(v.Args[0], scratchB)
	number, yes, no, done := c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel(), c.a.NewLabel()
	c.a.MovRR(scratchA, r)
	c.a.ShiftImm(amd64.Shr, scratchA, 51, true)
	c.a.OpImm(amd64.Cmp, scratchA, 0x1FFF, false)
	c.a.Jcc(amd64.CondNE, number)
	for _, w := range []struct {
		word uint64
		to   amd64.Label
	}{{c.enc.True, yes}, {c.enc.False, no}, {c.enc.Undefined, no}, {c.enc.Null, no}} {
		c.a.MovImm(scratchA, w.word)
		c.a.Op(amd64.Cmp, r, scratchA, true)
		c.a.Jcc(amd64.CondE, w.to)
	}
	// Anything else -- the uninitialized marker -- is not decided here.
	c.a.Op(amd64.Cmp, r, r, true)
	guard(amd64.CondE)
	c.a.Bind(number)
	// A number is true unless zero or NaN: ucomisd sets ZF for zero and for
	// NaN, and PF only for NaN.
	c.a.MovQToX(xScratch0, r)
	c.a.SSEOp(amd64.XorPD, xScratch1, xScratch1)
	c.a.SSEOp(amd64.UcomiSD, xScratch0, xScratch1)
	c.a.Setcc(amd64.CondNE, scratchA)
	c.a.Setcc(amd64.CondNP, scratchC)
	c.a.Op(amd64.And, scratchA, scratchC, false)
	c.a.MovZX8(scratchA, scratchA)
	c.a.Jmp(done)
	c.a.Bind(yes)
	c.a.MovImm(scratchA, 1)
	c.a.Jmp(done)
	c.a.Bind(no)
	c.a.MovImm(scratchA, 0)
	c.a.Bind(done)
	c.setG(v, scratchA)
}

// toInt32 is ToUint32's bits: CVTTSD2SI covers |x| < 2^63; beyond, and for
// NaN and the infinities, a cold path works from the exponent and
// significand.
func (c *compiler) toInt32(v *ssa.Value) {
	xv := v.Args[0]
	slow, back := c.a.NewLabel(), c.a.NewLabel()
	x := c.xmm(xv, xScratch0)
	c.a.Cvttsd2si(scratchA, x)
	c.a.MovImm(scratchB, 1<<63)
	c.a.Op(amd64.Cmp, scratchA, scratchB, true)
	c.a.Jcc(amd64.CondE, slow)
	c.a.Bind(back)
	c.a.MovRR32(scratchA, scratchA)
	c.setG(v, scratchA)
	c.cold = append(c.cold, func() {
		zero, positive := c.a.NewLabel(), c.a.NewLabel()
		c.a.Bind(slow)
		c.a.MovQFromX(scratchA, c.xmm(xv, xScratch0))
		// e = exponent; NaN and the infinities give 0.
		c.a.MovRR(scratchC, scratchA)
		c.a.ShiftImm(amd64.Shr, scratchC, 52, true)
		c.a.OpImm(amd64.And, scratchC, 0x7FF, false)
		c.a.OpImm(amd64.Cmp, scratchC, 0x7FF, false)
		c.a.Jcc(amd64.CondE, zero)
		// |x| >= 2^63 here, so e = exponent - 1075 >= 11, and the low 32
		// bits of the integer are the significand's low bits shifted by e,
		// zero once e >= 32; the implicit leading bit lands past bit 62.
		c.a.OpImm(amd64.Sub, scratchC, 1075, false)
		c.a.OpImm(amd64.Cmp, scratchC, 32, false)
		c.a.Jcc(amd64.CondAE, zero)
		c.a.MovRR(scratchB, scratchA) // the sign is bit 63
		c.a.ShiftImm(amd64.Shl, scratchA, 12, true)
		c.a.ShiftImm(amd64.Shr, scratchA, 12, true)
		c.a.ShiftCL(amd64.Shl, scratchA, true)
		c.a.Op(amd64.Test, scratchB, scratchB, true)
		c.a.Jcc(amd64.CondNS, positive)
		c.a.MovRR32(scratchC, scratchA)
		c.a.Op(amd64.Xor, scratchA, scratchA, false)
		c.a.Op(amd64.Sub, scratchA, scratchC, false)
		c.a.Bind(positive)
		c.a.Jmp(back)
		c.a.Bind(zero)
		c.a.Op(amd64.Xor, scratchA, scratchA, false)
		c.a.Jmp(back)
	})
}

// describe lists each value's location, in block order.
func (c *compiler) describe() string {
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
