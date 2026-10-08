package ssa

import (
	"errors"
	"fmt"
	"sort"

	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

// ErrUnsupported reports a program with an operation the SSA builder does
// not yet translate. The function stays in the existing tiers.
var ErrUnsupported = errors.New("ssa: unsupported operation")

// Build translates a validated slot IR program into SSA. Its entries are
// PC 0, each loop header, and each PC a host exit resumes at.
func Build(p *ir.Program) (*Func, error) { return BuildWith(p, nil) }

// Feedback is what the VM knows of a function's sites, which Build would
// otherwise leave to Go.
type Feedback interface {
	// Property is the property a PropertyRead or PropertyWrite at pc names,
	// or false to leave the site to Go.
	Property(pc int) (PropertySite, bool)
	// Global is where the global a BindingRead at pc names was last found,
	// or false to leave the site to Go.
	Global(pc int) (GlobalSite, bool)
}

// GlobalSite is a global read's site: its name, the VM's atom, and the
// index in the global object's table where the binding was found.
type GlobalSite struct {
	Key   uint32
	Index int32
}

// PropertySite is a property site: its key, the VM's atom; and, if the
// site has met objects of one shape, that shape's address, which the VM
// keeps alive and unchanged, with the index of the property -- a writable
// data property, for a write -- in their tables. Shape is 0 otherwise.
type PropertySite struct {
	Key   uint32
	Shape uintptr
	Index int32
}

// BuildWith is Build with what the VM knows of the sites.
func BuildWith(p *ir.Program, fb Feedback) (*Func, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	b := &builder{p: p, fb: fb, f: &Func{Locals: p.Locals, StackSize: p.StackSize, FrameLocals: p.Locals, ThisSlot: -1,
		written: make([]bool, p.Locals+p.StackSize)}, nslots: p.Locals + p.StackSize}
	if err := b.plan(); err != nil {
		return nil, err
	}
	b.translate()
	shadowMerges(b.f)
	return b.f, nil
}

type builder struct {
	p      *ir.Program
	fb     Feedback
	f      *Func
	nslots int

	entryPCs []int
	blockAt  map[int]*Block // by the PC a block starts at
	// endOf is the last PC of each block.
	endOf map[*Block]int

	defs       map[*Block][]*Value
	sealed     map[*Block]bool
	filled     map[*Block]bool
	incomplete map[*Block]map[int]*Value
}

// property is the feedback for a property operation at pc, if any.
func (b *builder) property(pc int) (PropertySite, bool) {
	if op := b.p.Code[pc].Op; b.fb == nil || op != ir.PropertyRead && op != ir.PropertyWrite && op != ir.ReferenceRead {
		return PropertySite{}, false
	}
	return b.fb.Property(pc)
}

// host reports whether the instruction at pc is left to Go whole: it ends
// its block, and the next PC is an entry.
func (b *builder) host(pc int) bool {
	switch b.p.Code[pc].Op {
	case ir.Host, ir.Call:
		return true
	case ir.PropertyRead, ir.PropertyWrite, ir.ReferenceRead:
		_, ok := b.property(pc)
		return !ok
	case ir.BindingRead:
		_, ok := b.global(pc)
		return !ok
	}
	return false
}

// global is the feedback for a global read at pc, if any.
func (b *builder) global(pc int) (GlobalSite, bool) {
	if b.fb == nil || b.p.Code[pc].Op != ir.BindingRead {
		return GlobalSite{}, false
	}
	return b.fb.Global(pc)
}

func reachable(p *ir.Program, pc int) bool {
	return pc >= 0 && pc < len(p.Code) && p.Maps[pc].Depth >= 0
}

// plan finds entries and blocks, and checks every operation is translatable.
func (b *builder) plan() error {
	p := b.p
	entries := map[int]bool{0: true}
	leaders := map[int]bool{0: true}
	for pc, in := range p.Code {
		if !reachable(p, pc) {
			continue
		}
		switch in.Op {
		case ir.Nop, ir.Copy, ir.CopyPair, ir.StoreLoad, ir.Swap, ir.Insert2, ir.Insert3,
			ir.Unary, ir.Update, ir.Return, ir.ArrayRead, ir.ArrayUpdate, ir.ArrayLength, ir.ArrayKey:
		case ir.ArrayWrite, ir.PropertyRead, ir.PropertyWrite, ir.ReferenceRead, ir.BindingRead:
			// What native code does not do exits to Go, which resumes after
			// it.
			entries[pc+1] = true
		case ir.Binary:
			if in.Operator == ir.Eq || in.Operator == ir.Ne {
				// A comparison of non-numbers exits to Go, which resumes after it.
				entries[pc+1] = true
			}
		case ir.Jump:
			leaders[in.Target] = true
			if in.Target <= pc {
				entries[in.Target] = true
			}
		case ir.Branch:
			leaders[in.Target] = true
			if in.Target <= pc {
				entries[in.Target] = true
			}
			if in.Operator == ir.Eq || in.Operator == ir.Ne {
				entries[pc+1], entries[in.Target] = true, true
			}
		case ir.Host, ir.Call:
			entries[pc+1] = true
		default:
			return fmt.Errorf("%w: %d at pc %d", ErrUnsupported, in.Op, pc)
		}
		switch {
		case in.Op == ir.Jump || in.Op == ir.Branch || in.Op == ir.Return || b.host(pc):
			leaders[pc+1] = true
		}
	}
	for pc := range entries {
		if reachable(p, pc) {
			b.entryPCs = append(b.entryPCs, pc)
			leaders[pc] = true
		}
	}
	sort.Ints(b.entryPCs)

	b.blockAt = map[int]*Block{}
	b.endOf = map[*Block]int{}
	var starts []int
	for pc := range leaders {
		if reachable(b.p, pc) {
			starts = append(starts, pc)
		}
	}
	sort.Ints(starts)
	for _, pc := range starts {
		b.blockAt[pc] = b.f.newBlock(pc)
	}
	for i, pc := range starts {
		end := len(p.Code) - 1
		if i+1 < len(starts) {
			end = starts[i+1] - 1
		}
		for q := pc; q <= end; q++ {
			if op := p.Code[q].Op; op == ir.Jump || op == ir.Branch || op == ir.Return || b.host(q) {
				end = q
				break
			}
		}
		blk := b.blockAt[pc]
		b.endOf[blk] = end
		last := p.Code[end]
		switch last.Op {
		case ir.Jump:
			blk.Kind = BlockPlain
			b.edge(blk, b.blockAt[last.Target])
		case ir.Branch:
			blk.Kind = BlockIf
			taken, fall := b.blockAt[last.Target], b.blockAt[end+1]
			if last.When {
				b.edge(blk, taken)
				b.edge(blk, fall)
			} else {
				b.edge(blk, fall)
				b.edge(blk, taken)
			}
		case ir.Return:
			blk.Kind = BlockReturn
		case ir.Host, ir.Call, ir.PropertyRead, ir.PropertyWrite, ir.ReferenceRead, ir.BindingRead:
			if !b.host(end) {
				blk.Kind = BlockPlain
				b.edge(blk, b.blockAt[end+1])
				break
			}
			blk.Kind = BlockExit
		default:
			blk.Kind = BlockPlain
			b.edge(blk, b.blockAt[end+1])
		}
	}
	// Entry blocks load the live slots and go to the block for their PC.
	for _, pc := range b.entryPCs {
		e := b.f.newBlock(-1)
		e.Kind = BlockPlain
		b.edge(e, b.blockAt[pc])
		b.f.Entries = append(b.f.Entries, Entry{PC: pc, Depth: p.Maps[pc].Depth, Block: e})
	}
	// A loop header is reached by an edge from a block at or after it.
	for _, blk := range b.f.Blocks {
		blk.Backedge = make([]bool, len(blk.Preds))
		for i, pred := range blk.Preds {
			if pred.PC >= 0 && blk.PC >= 0 && pred.PC >= blk.PC {
				blk.Backedge[i] = true
				blk.LoopHeader = true
			}
		}
	}
	return nil
}

func (b *builder) edge(from, to *Block) {
	if to == nil {
		panic("ssa: edge to an unreachable PC")
	}
	from.Succs = append(from.Succs, to)
	to.Preds = append(to.Preds, from)
}

// translate fills every block, sealing each once its predecessors are filled
// (Braun et al., "Simple and Efficient Construction of Static Single
// Assignment Form").
func (b *builder) translate() {
	b.defs = map[*Block][]*Value{}
	b.sealed = map[*Block]bool{}
	b.filled = map[*Block]bool{}
	b.incomplete = map[*Block]map[int]*Value{}

	// Reverse post-order from the entries, so that a block's forward
	// predecessors are filled before it.
	var order []*Block
	seen := map[*Block]bool{}
	var visit func(*Block)
	visit = func(blk *Block) {
		if seen[blk] {
			return
		}
		seen[blk] = true
		for i := len(blk.Succs) - 1; i >= 0; i-- {
			visit(blk.Succs[i])
		}
		order = append(order, blk)
	}
	for i := len(b.f.Entries) - 1; i >= 0; i-- {
		visit(b.f.Entries[i].Block)
	}
	for i, j := 0, len(order)-1; i < j; i, j = i+1, j-1 {
		order[i], order[j] = order[j], order[i]
	}
	// Blocks no entry reaches are dropped.
	kept := b.f.Blocks[:0]
	for _, blk := range b.f.Blocks {
		if seen[blk] {
			kept = append(kept, blk)
		}
	}
	b.f.Blocks = kept
	for _, blk := range b.f.Blocks {
		preds := blk.Preds[:0]
		backedge := blk.Backedge[:0]
		for i, pred := range blk.Preds {
			if seen[pred] {
				preds = append(preds, pred)
				backedge = append(backedge, blk.Backedge[i])
			}
		}
		blk.Preds, blk.Backedge = preds, backedge
	}

	for _, blk := range order {
		b.trySeal(blk)
		b.fill(blk)
		b.filled[blk] = true
		for _, s := range blk.Succs {
			b.trySeal(s)
		}
	}
	for _, blk := range order {
		if !b.sealed[blk] {
			b.seal(blk)
		}
	}
	for i, blk := range b.f.Blocks {
		blk.ID = i
	}
}

func (b *builder) trySeal(blk *Block) {
	if b.sealed[blk] {
		return
	}
	for _, p := range blk.Preds {
		if !b.filled[p] {
			return
		}
	}
	b.seal(blk)
}

func (b *builder) seal(blk *Block) {
	// Slot by slot, never in map order: completing a phi can make others,
	// and their numbers must not vary from one build to the next.
	pending := b.incomplete[blk]
	slots := make([]int, 0, len(pending))
	for slot := range pending {
		slots = append(slots, slot)
	}
	sort.Ints(slots)
	for _, slot := range slots {
		b.addPhiOperands(slot, pending[slot])
	}
	delete(b.incomplete, blk)
	b.sealed[blk] = true
}

// assign is an instruction's write of a slot, which Func.Written records;
// write is also how reads record the phis they make.
func (b *builder) assign(slot int, blk *Block, v *Value) {
	b.f.written[slot] = true
	b.write(slot, blk, v)
}

func (b *builder) write(slot int, blk *Block, v *Value) {
	d := b.defs[blk]
	if d == nil {
		d = make([]*Value, b.nslots)
		b.defs[blk] = d
	}
	d[slot] = v
}

func (b *builder) read(slot int, blk *Block) *Value {
	if d := b.defs[blk]; d != nil && d[slot] != nil {
		return d[slot]
	}
	var v *Value
	switch {
	case !b.sealed[blk]:
		v = b.newPhi(blk)
		if b.incomplete[blk] == nil {
			b.incomplete[blk] = map[int]*Value{}
		}
		b.incomplete[blk][slot] = v
	case len(blk.Preds) == 1:
		v = b.read(slot, blk.Preds[0])
	case len(blk.Preds) == 0:
		// A slot read where nothing defined it: an operand slot past an
		// entry's depth. The slot IR never reads one; this keeps the form
		// total.
		v = b.constIn(blk, ir.Value{Kind: ir.Undefined})
	default:
		v = b.newPhi(blk)
		b.write(slot, blk, v)
		b.addPhiOperands(slot, v)
	}
	b.write(slot, blk, v)
	return v
}

func (b *builder) newPhi(blk *Block) *Value {
	v := &Value{ID: b.f.nextID, Op: OpPhi, Type: Tagged, Block: blk}
	b.f.nextID++
	blk.Values = append([]*Value{v}, blk.Values...)
	return v
}

func (b *builder) addPhiOperands(slot int, phi *Value) {
	for _, p := range phi.Block.Preds {
		a := b.read(slot, p)
		a.Uses++
		phi.Args = append(phi.Args, a)
	}
}

// constIn makes a tagged constant at the start of a block.
func (b *builder) constIn(blk *Block, c ir.Value) *Value {
	v := &Value{ID: b.f.nextID, Op: OpConst, Type: Tagged, Const: c, Block: blk}
	b.f.nextID++
	blk.Values = append([]*Value{v}, blk.Values...)
	return v
}

// state captures the frame at a PC: every live slot's current value.
func (b *builder) state(blk *Block, pc int) *FrameState {
	depth := b.p.Maps[pc].Depth
	s := &FrameState{PC: b.p.Maps[pc].PC, Depth: depth, Slots: make([]*Value, b.p.Locals+depth)}
	for i := range s.Slots {
		s.Slots[i] = b.read(i, blk)
		s.Slots[i].Uses++
	}
	return s
}

func (b *builder) fill(blk *Block) {
	f := b.f
	if blk.PC < 0 {
		e, _ := f.entryForBlock(blk)
		blk.Header = &FrameState{PC: b.p.Maps[e.PC].PC, Depth: e.Depth, Slots: make([]*Value, b.p.Locals+e.Depth)}
		for i := range blk.Header.Slots {
			v := f.newValue(blk, OpLoadSlot, Tagged)
			v.Aux = i
			b.write(i, blk, v)
			blk.Header.Slots[i] = v
		}
		return
	}
	if blk.LoopHeader {
		blk.Header = b.state(blk, blk.PC)
	}
	for pc := blk.PC; pc <= b.endOf[blk]; pc++ {
		b.instruction(blk, pc)
	}
}

func (f *Func) entryForBlock(blk *Block) (Entry, bool) {
	for _, e := range f.Entries {
		if e.Block == blk {
			return e, true
		}
	}
	return Entry{}, false
}

func (b *builder) instruction(blk *Block, pc int) {
	f, in := b.f, b.p.Code[pc]
	var st *FrameState
	state := func() *FrameState {
		if st == nil {
			st = b.state(blk, pc)
		}
		return st
	}
	operand := func(o ir.Operand) *Value {
		if o.Slot < 0 {
			v := f.newValue(blk, OpConst, Tagged)
			v.Const = o.Literal
			return v
		}
		return b.read(o.Slot, blk)
	}
	guard := func(op Op, t Type, kind ir.ExitKind, args ...*Value) *Value {
		v := f.newValue(blk, op, t, args...)
		v.Aux = int(kind)
		v.State = state()
		v.State.addUse()
		return v
	}
	number := func(o ir.Operand, kind ir.ExitKind) *Value {
		if o.Slot < 0 && o.Literal.Kind == ir.Number {
			v := f.newValue(blk, OpConstF64, Float64)
			v.Const = o.Literal
			return v
		}
		return guard(OpUnboxF64, Float64, kind, operand(o))
	}
	boxF := func(x *Value) *Value { return f.newValue(blk, OpBoxF64, Tagged, x) }
	boxB := func(x *Value) *Value { return f.newValue(blk, OpBoxBool, Tagged, x) }
	one := func() *Value {
		v := f.newValue(blk, OpConstF64, Float64)
		v.Const = ir.Float(1)
		return v
	}

	if in.Check {
		guard(OpCheckInit, None, ir.GuardExit, b.read(in.CheckSlot, blk))
	}
	switch in.Op {
	case ir.Nop, ir.Jump:
	case ir.Copy:
		b.assign(in.Dest, blk, operand(in.Left))
	case ir.CopyPair:
		l, r := operand(in.Left), operand(in.Right)
		b.assign(in.Dest, blk, l)
		b.assign(in.Extra, blk, r)
	case ir.StoreLoad:
		b.assign(in.Dest, blk, operand(in.Left))
		b.assign(in.Extra, blk, operand(in.Right))
	case ir.Swap:
		d, e := b.read(in.Dest, blk), b.read(in.Extra, blk)
		b.assign(in.Dest, blk, e)
		b.assign(in.Extra, blk, d)
	case ir.Insert3:
		n := in.Dest
		x, y, z := b.read(n, blk), b.read(n+1, blk), b.read(n+2, blk)
		b.assign(n, blk, z)
		b.assign(n+1, blk, x)
		b.assign(n+2, blk, y)
		b.assign(n+3, blk, z)
	case ir.Insert2:
		n := in.Dest
		x, y := b.read(n, blk), b.read(n+1, blk)
		b.assign(n, blk, y)
		b.assign(n+1, blk, x)
		b.assign(n+2, blk, y)
	case ir.Binary:
		kind := ir.GuardExit
		if in.Operator == ir.Eq || in.Operator == ir.Ne {
			kind = ir.HostExit
		}
		x, y := number(in.Left, kind), number(in.Right, kind)
		b.assign(in.Dest, blk, b.binary(blk, in.Operator, x, y, boxF, boxB))
	case ir.Unary:
		if in.Operator == ir.Not {
			t := guard(OpTruth, Bool, ir.GuardExit, operand(in.Left))
			b.assign(in.Dest, blk, boxB(f.newValue(blk, OpNot, Bool, t)))
			break
		}
		x := number(in.Left, ir.GuardExit)
		switch in.Operator {
		case ir.Neg:
			x = f.newValue(blk, OpNegF64, Float64, x)
		case ir.Int32:
			x = f.newValue(blk, OpI32ToF64, Float64, f.newValue(blk, OpToInt32, Int32, x))
		case ir.BitNot:
			x = f.newValue(blk, OpI32ToF64, Float64, f.newValue(blk, OpNotI32, Int32, f.newValue(blk, OpToInt32, Int32, x)))
		}
		b.assign(in.Dest, blk, boxF(x))
	case ir.Update:
		old := operand(in.Left)
		x := guard(OpUnboxF64, Float64, ir.GuardExit, old)
		op := OpAddF64
		if in.Operator == ir.Sub {
			op = OpSubF64
		}
		v := boxF(f.newValue(blk, op, Float64, x, one()))
		b.assign(in.Dest, blk, v)
		if in.Extra >= 0 {
			if in.Postfix {
				v = old
			}
			b.assign(in.Extra, blk, v)
		}
	case ir.Branch:
		var c *Value
		if in.Operator == ir.Truth {
			c = guard(OpTruth, Bool, ir.GuardExit, operand(in.Left))
		} else {
			kind := ir.GuardExit
			if in.Operator == ir.Eq || in.Operator == ir.Ne {
				kind = ir.HostExit
			}
			c = f.newValue(blk, OpCmpF64, Bool, number(in.Left, kind), number(in.Right, kind))
			c.Aux = int(in.Operator)
		}
		blk.Control = c
		c.Uses++
	case ir.Return:
		v := operand(in.Left)
		guard(OpCheckInit, None, ir.GuardExit, v)
		blk.Control = v
		v.Uses++
	case ir.ArrayRead, ir.ArrayUpdate:
		// Every check exits to the state before the instruction, so their
		// order is free; the key's update is written only once the read
		// has succeeded, before the element, which wins if both name one
		// slot.
		array := guard(OpArrayOf, Ptr, ir.GuardExit, operand(in.Left))
		var key, updated *Value
		if in.Op == ir.ArrayUpdate {
			old := guard(OpUnboxF64, Float64, ir.GuardExit, operand(in.Right))
			op := OpAddF64
			if in.Operator == ir.Sub {
				op = OpSubF64
			}
			updated = f.newValue(blk, op, Float64, old, one())
			key = updated
			if in.Postfix {
				key = old
			}
		} else {
			key = number(in.Right, ir.GuardExit)
		}
		elem := guard(OpElemRead, Float64, ir.GuardExit, array, key)
		if updated != nil {
			b.assign(in.Extra, blk, boxF(updated))
		}
		b.assign(in.Dest, blk, boxF(elem))
	case ir.ArrayLength:
		array := guard(OpArrayOf, Ptr, ir.GuardExit, operand(in.Left))
		b.assign(in.Dest, blk, boxF(f.newValue(blk, OpArrayLen, Float64, array)))
	case ir.ArrayKey:
		guard(OpArrayOf, Ptr, ir.GuardExit, operand(in.Left))
		guard(OpElemKey, None, ir.GuardExit, number(in.Right, ir.GuardExit))
	case ir.ArrayWrite:
		// Go stores what native code does not: a value that is not a
		// number, or into an element that does not hold one.
		array := guard(OpArrayOf, Ptr, ir.HostExit, operand(in.Left))
		key, value := number(in.Right, ir.HostExit), number(in.Third, ir.HostExit)
		guard(OpElemWrite, None, ir.HostExit, array, key, value)
	case ir.BindingRead:
		site, ok := b.global(pc)
		if !ok {
			blk.ExitKind = ir.HostExit
			blk.State = state()
			blk.State.addUse()
			break
		}
		cell := guard(OpGlobalCell, Source, ir.HostExit)
		cell.Index, cell.Key = int(site.Index), site.Key
		v := f.newValue(blk, OpLoadCell, Tagged, cell)
		v.Shadow = cell
		b.assign(in.Dest, blk, v)
	case ir.PropertyRead, ir.PropertyWrite, ir.ReferenceRead:
		site, ok := b.property(pc)
		if !ok {
			blk.ExitKind = ir.HostExit
			blk.State = state()
			blk.State.addUse()
			break
		}
		object := guard(OpObjectOf, Ptr, ir.HostExit, operand(in.Left))
		if in.Op == ir.ReferenceRead {
			cell := guard(OpPropCell, Source, ir.HostExit, object)
			cell.Const, cell.Index, cell.Key = ir.Value{Bits: uint64(site.Shape)}, int(site.Index), site.Key
			v := f.newValue(blk, OpLoadCell, Tagged, cell)
			v.Shadow = cell
			b.assign(in.Dest, blk, v)
			break
		}
		if in.Op == ir.PropertyRead {
			v := guard(OpPropRead, Float64, ir.HostExit, object)
			v.Const, v.Index, v.Key = ir.Value{Bits: uint64(site.Shape)}, int(site.Index), site.Key
			b.assign(in.Dest, blk, boxF(v))
			break
		}
		v := guard(OpPropWrite, None, ir.HostExit, object, number(in.Right, ir.HostExit))
		v.Const, v.Index, v.Key = ir.Value{Bits: uint64(site.Shape)}, int(site.Index), site.Key
	case ir.Host, ir.Call:
		blk.ExitKind = ir.HostExit
		blk.State = state()
		blk.State.addUse()
	}
}

func (s *FrameState) addUse() {}

// binary is a slot IR binary operator on two unboxed numbers.
func (b *builder) binary(blk *Block, op ir.Operator, x, y *Value, boxF, boxB func(*Value) *Value) *Value {
	f := b.f
	switch op {
	case ir.Add:
		return boxF(f.newValue(blk, OpAddF64, Float64, x, y))
	case ir.Sub:
		return boxF(f.newValue(blk, OpSubF64, Float64, x, y))
	case ir.Mul:
		return boxF(f.newValue(blk, OpMulF64, Float64, x, y))
	case ir.Div:
		return boxF(f.newValue(blk, OpDivF64, Float64, x, y))
	case ir.Lt, ir.Le, ir.Gt, ir.Ge, ir.Eq, ir.Ne:
		c := f.newValue(blk, OpCmpF64, Bool, x, y)
		c.Aux = int(op)
		return boxB(c)
	}
	l := f.newValue(blk, OpToInt32, Int32, x)
	r := f.newValue(blk, OpToInt32, Int32, y)
	var v *Value
	switch op {
	case ir.BitAnd:
		v = f.newValue(blk, OpAndI32, Int32, l, r)
	case ir.BitOr:
		v = f.newValue(blk, OpOrI32, Int32, l, r)
	case ir.BitXor:
		v = f.newValue(blk, OpXorI32, Int32, l, r)
	case ir.Shl:
		v = f.newValue(blk, OpShlI32, Int32, l, r)
	case ir.Shr:
		v = f.newValue(blk, OpSarI32, Int32, l, r)
	case ir.UShr:
		return boxF(f.newValue(blk, OpU32ToF64, Float64, f.newValue(blk, OpShrU32, Int32, l, r)))
	default:
		panic(fmt.Sprintf("ssa: binary operator %d", op))
	}
	return boxF(f.newValue(blk, OpI32ToF64, Float64, v))
}
