package ssa

import (
	"errors"
	"fmt"
	"slices"

	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
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
	// Generic reports an operation at pc whose speculation failed before,
	// which is built so that what it did not expect goes to Go, which
	// resumes after it, rather than deoptimizing: an element read by its
	// cell, an arithmetic operation or a comparison of what are not numbers
	// by Go. EntryGeneric reports an entry whose speculation failed,
	// by its PC in the function's own code (FrameState.PC): the slots it
	// loads are not taken for numbers there.
	Generic(pc int) bool
	EntryGeneric(pc int) bool
	// Inline is the call at pc to inline, if the VM has seen it call one
	// function it may be: see InlineSite.
	Inline(pc int) (InlineSite, bool)
	// NativeCalls are the functions the call at pc may call natively, if
	// any: those the VM has seen it call, at most a few (see CallSite).
	NativeCalls(pc int) []CallSite
}

// CallSite is a function the VM has seen a call call whose native code a
// caller's may call (mir's native calls): the function object's address,
// which the call checks it calls; its closure's, for Go to make its frame
// from; the address of the cell its code's entry is in, 0 while it has
// none, and of the count of the calls made to it natively, if any, which
// the VM keeps alive; the call's argument count and whether it
// passes a receiver; the callee's parameters, locals and operand slots;
// its receiver's slot, or -1 if it reads none; and whether a receiver that
// is not an object needs coercing, which only Go does.
type CallSite struct {
	Callee, Closure, Entry, Count uintptr
	Argc                          int
	Method                        bool
	Params, LocalCount, MaxStack  int
	ThisSlot                      int
	Coerce                        bool
}

// InlineSite is a call the VM has seen call one function, whose program,
// lowered for inlining, Inlinable accepts: that program and its own
// feedback; the function object's address, which the VM keeps alive and
// the call checks it still calls, and its closure's, for Go to make its
// frame from at an exit inside it (InlineState); the call's argument
// count, and whether it passes a receiver, as a method call does; and the
// callee's parameter count, its receiver's slot, or -1 if it reads none,
// and whether a receiver that is not an object needs coercing, which only
// Go does.
type InlineSite struct {
	Program         *ir.Program
	Feedback        Feedback
	Callee, Closure uintptr
	Argc            int
	Method          bool
	Params          int
	ThisSlot        int
	Coerce          bool
}

// Inlining's bounds: a callee's instructions, all callees' in a function,
// the calls a function inlines, and how deep: a callee's calls are inlined
// in it, as V8 inlines them, to this many levels.
const (
	maxInline      = 48
	maxInlineTotal = 192
	maxInlines     = 8
	maxInlineDepth = 3
)

// Inlinable reports whether a callee may be inlined: a small program that
// reads and writes properties and elements, computes, branches forward
// and returns, which native code can do all of, with no operation it
// leaves to Go, which would have Go finish the call every time. An exit
// inside it makes its frame (InlineState), so that what it did before is
// not done again.
func Inlinable(p *ir.Program) bool {
	calls, ok := InlineCalls(p)
	return ok && len(calls) == 0
}

// InlineCalls reports whether a callee may be inlined if the operations it
// leaves to Go, which it returns by PC, are calls inlined in it too: it is
// Inlinable but for those.
func InlineCalls(p *ir.Program) ([]int, bool) {
	if p == nil || p.Validate() != nil || len(p.Code) > maxInline || len(p.Globals) != 0 {
		return nil, false
	}
	var calls []int
	for pc, in := range p.Code {
		if !reachable(p, pc) {
			continue
		}
		switch in.Op {
		case ir.Call:
			return nil, false
		case ir.Host:
			calls = append(calls, pc)
		case ir.Jump, ir.Branch:
			if in.Target <= pc {
				return nil, false
			}
		}
	}
	return calls, true
}

// GlobalSite is a global read's site: its name, the VM's atom, and the
// index in the global object's table where the binding was found. Fixed
// says the binding can never change -- a data property neither writable
// nor configurable, as undefined, NaN and Infinity are -- and Constant is
// its value then: the read checks the binding is still where it was, and
// uses the constant.
type GlobalSite struct {
	Key      uint32
	Index    int32
	Fixed    bool
	Constant ir.Value
}

// PropertySite is a property site: its key, the VM's atom; and, if the
// site has met objects of one shape, that shape's address, which the VM
// keeps alive and unchanged, with the index of the property -- a writable
// data property, for a write -- in their tables. Shape is 0 otherwise.
//
// A read may have found the property on a prototype, a method most often:
// Holders then name the objects from the receiver's prototype on, one or
// two, each with the shape it had, and Index is the property's in the last
// one's table. The receiver's shape says it has no such property of its
// own; each holder's, that the one before it has none, and the last's
// where it is.
//
// A read may have met objects of other shapes too, as V8's polymorphic
// inline caches keep up to four: Cases are those, each with its own
// holders and index, which the read checks after Shape's.
type PropertySite struct {
	Key     uint32
	Shape   uintptr
	Index   int32
	Holders [2]Holder
	Cases   []PropertyCase
}

// PropertyCase is one more shape a read's site met (PropertySite.Cases):
// the receiver's shape, the prototypes the property was found on, none if
// it is the receiver's own, and its index in the last one's table.
type PropertyCase struct {
	Shape   uintptr
	Index   int32
	Holders [2]Holder
}

// Holder is a prototype a property read was answered by, or passed through,
// at its address, which the VM keeps alive, with the shape it had: 0 for
// none.
type Holder struct{ Object, Shape uintptr }

// BuildWith is Build with what the VM knows of the sites.
func BuildWith(p *ir.Program, fb Feedback) (*Func, error) { return build(nil, p, fb) }

func build(w *Workspace, p *ir.Program, fb Feedback) (*Func, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	p = referenceReads(p, fb)
	f := &Func{Locals: p.Locals, StackSize: p.StackSize, FrameLocals: p.Locals, ThisSlot: -1, ws: w}
	f.written = f.bools(p.Locals + p.StackSize)
	b := &builder{p: p, fb: fb, f: f, nslots: p.Locals + p.StackSize}
	b.root = &frame{p: p, fb: fb}
	b.cur = b.root
	if err := b.plan(); err != nil {
		return nil, err
	}
	b.translate()
	shadowMerges(b.f)
	storeChecks(b.f)
	return b.f, nil
}

type builder struct {
	// p and fb are the program the block being translated is in, and its
	// feedback: the function's own (root), or an inlined callee's.
	p      *ir.Program
	fb     Feedback
	f      *Func
	nslots int

	// root is the function's own frame, cur the one translated now;
	// inlined are the callees, by the PC of their call, inlines how many
	// calls are inlined, at every level; frameOf is each block's frame, by
	// block ID, nil for the function's own.
	root, cur *frame
	inlined   map[int]*frame
	inlines   int
	// calls are the calls made natively (OpCall), by PC: their blocks go
	// on to the next instruction's.
	calls   map[int][]*CallSite
	frameOf []*frame

	entryPCs []int
	blockAt  []*Block // by the PC a block starts at
	// endOf is the last PC of each block, by block ID.
	endOf []int

	// By block ID, which is unique while the builder runs: each block's
	// definitions by slot, whether it is sealed and filled, and the phis
	// it made before it was sealed.
	defs       [][]*Value
	sealed     []bool
	filled     []bool
	incomplete [][]slotPhi
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

// referenceReads is p with the element reads whose speculation failed
// (Feedback's Generic) carried by their cells, as the reads of elements
// used as more than numbers are. p itself is not changed.
func referenceReads(p *ir.Program, fb Feedback) *ir.Program {
	if fb == nil {
		return p
	}
	var q *ir.Program
	for pc, in := range p.Code {
		if in.Op != ir.ArrayRead || in.Reference || !reachable(p, pc) || !fb.Generic(pc) {
			continue
		}
		if q == nil {
			c := *p
			c.Code = append([]ir.Instruction(nil), p.Code...)
			q = &c
		}
		q.Code[pc].Reference = true
	}
	if q == nil {
		return p
	}
	return q
}

// generic reports an arithmetic operation or a comparison at pc whose
// speculation failed (Feedback's Generic): what are not numbers go to Go,
// which resumes after it, as they do for == and %.
func (b *builder) generic(pc int) bool {
	switch in := b.p.Code[pc]; {
	case in.Op == ir.Binary, in.Op == ir.Branch && in.Operator != ir.Truth:
		return b.fb != nil && b.fb.Generic(pc)
	}
	return false
}

// frame is a program the builder translates: the function's own, or a
// callee inlined at one of its calls.
type frame struct {
	p  *ir.Program
	fb Feedback
	// base is a callee's first slot among the builder's, past the
	// function's own and earlier callees'; blockAt its blocks, by its PC.
	base    int
	blockAt []*Block
	site    InlineSite
	// call is the state at the call, where every exit inside the callee
	// goes: Go makes the call, or the interpreter does, from the start. The
	// call's result goes to slot result, and code goes on at cont, the
	// block after the call. start sets the callee's slots: its arguments,
	// its receiver and undefined (init, by its slot), when the call has
	// been checked.
	call      *FrameState
	result    int
	cont      *Block
	start     *Block
	init      []*Value
	callBlock *Block // the caller's block that ends at the call
	// parent is the frame of the call, the function's own or a callee's;
	// depth how many inlined frames this is inside, itself included; and
	// inlined the callees inlined in it, by the PC of their call.
	parent  *frame
	depth   int
	inlined map[int]*frame
}

// inlinedAt is the callee inlined at pc in the frame being translated.
func (b *builder) inlinedAt(pc int) *frame {
	if b.cur == b.root {
		return b.inlined[pc]
	}
	return b.cur.inlined[pc]
}

// enter makes fr the frame being translated.
func (b *builder) enter(fr *frame) {
	b.cur, b.p, b.fb = fr, fr.p, fr.fb
}

// frameAt is the frame blk is in.
func (b *builder) frameAt(blk *Block) *frame {
	if blk.ID < len(b.frameOf) && b.frameOf[blk.ID] != nil {
		return b.frameOf[blk.ID]
	}
	return b.root
}

// inlineAt is the callee to inline at pc in the frame being translated, if
// any: a call the VM has seen call one function (Feedback's Inline), whose
// failures have not made it generic, within inlining's bounds; with the
// calls it makes inlined in it, every one, or it is not.
func (b *builder) inlineAt(pc int, total *int) (*frame, bool) {
	if b.fb == nil || b.p.Code[pc].Op != ir.Host || b.inlines >= maxInlines || b.cur.depth >= maxInlineDepth || b.fb.Generic(pc) {
		return nil, false
	}
	site, ok := b.fb.Inline(pc)
	var calls []int
	if ok {
		calls, ok = InlineCalls(site.Program)
	}
	if !ok || *total+len(site.Program.Code) > maxInlineTotal ||
		site.Argc < 0 || site.Params < 0 || pc+1 >= len(b.p.Code) || !reachable(b.p, pc+1) {
		return nil, false
	}
	depth, after := b.p.Maps[pc].Depth, b.p.Maps[pc+1].Depth
	callee := site.Argc + 1
	if site.Method {
		callee++
	}
	if depth < callee || after != depth-callee+1 || site.ThisSlot >= 0 && !site.Method {
		return nil, false
	}
	q := referenceReads(site.Program, site.Feedback)
	// Its slots follow the function's and earlier callees': every slot's
	// number stays below abi.MaxRecords, as records name slots by number.
	if b.nslots+q.Locals+q.StackSize > abi.MaxRecords {
		return nil, false
	}
	nslots, before, inlines := b.nslots, *total, b.inlines
	*total += len(site.Program.Code)
	fr := &frame{p: q, fb: site.Feedback, site: site, result: b.cur.base + b.p.Locals + after - 1, base: b.nslots,
		parent: b.cur, depth: b.cur.depth + 1}
	b.nslots += q.Locals + q.StackSize
	b.inlines++
	if len(calls) != 0 {
		cur := b.cur
		b.enter(fr)
		for _, at := range calls {
			child, ok := b.inlineAt(at, total)
			if !ok {
				b.enter(cur)
				b.nslots, *total, b.inlines = nslots, before, inlines
				return nil, false
			}
			if fr.inlined == nil {
				fr.inlined = map[int]*frame{}
			}
			fr.inlined[at] = child
		}
		b.enter(cur)
	}
	return fr, true
}

// planInline makes an inlined callee's blocks, between the block that ends
// at its call and cont, the block after the call: the start block, then a
// block at every target of a branch and after every branch, return and
// exit. A return goes to cont; an exit, wherever it is, goes to the call's
// state (frame.call), where Go makes the call and native code goes on
// after it.
func (b *builder) planInline(fr *frame, call, cont *Block) {
	defer b.enter(b.root)
	b.enter(fr)
	p := fr.p
	fr.cont = cont
	leaders := b.f.bools(len(p.Code) + 1)
	leaders[0] = true
	for pc, in := range p.Code {
		if !reachable(p, pc) {
			continue
		}
		switch {
		case in.Op == ir.Jump || in.Op == ir.Branch:
			leaders[in.Target], leaders[pc+1] = true, true
		case in.Op == ir.Return || b.host(pc):
			leaders[pc+1] = true
		}
	}
	newBlock := func(pc, end int) *Block {
		blk := b.f.newBlock(pc)
		for len(b.frameOf) <= blk.ID {
			b.frameOf = append(b.frameOf, nil)
			b.endOf = append(b.endOf, 0)
		}
		b.frameOf[blk.ID], b.endOf[blk.ID] = fr, end
		return blk
	}
	fr.start = newBlock(0, -1)
	fr.start.Kind = BlockPlain
	fr.blockAt = make([]*Block, len(p.Code)+1)
	var starts []int
	for pc := range p.Code {
		if leaders[pc] && reachable(p, pc) {
			starts = append(starts, pc)
		}
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
		fr.blockAt[pc] = newBlock(pc, end)
	}
	b.edge(call, fr.start)
	b.edge(fr.start, fr.blockAt[0])
	for _, pc := range starts {
		blk := fr.blockAt[pc]
		end := b.endOf[blk.ID]
		switch last := p.Code[end]; {
		case last.Op == ir.Jump:
			blk.Kind = BlockPlain
			b.edge(blk, fr.blockAt[last.Target])
		case last.Op == ir.Branch:
			blk.Kind = BlockIf
			taken, fall := fr.blockAt[last.Target], fr.blockAt[end+1]
			if last.When {
				b.edge(blk, taken)
				b.edge(blk, fall)
			} else {
				b.edge(blk, fall)
				b.edge(blk, taken)
			}
		case last.Op == ir.Return:
			blk.Kind = BlockPlain
			b.edge(blk, cont)
		case fr.inlined[end] != nil:
			// Into the callee inlined in it, whose returns go to the
			// block after it.
			blk.Kind = BlockPlain
			fr.inlined[end].callBlock = blk
		case b.host(end):
			blk.Kind = BlockExit
		default:
			blk.Kind = BlockPlain
			b.edge(blk, fr.blockAt[end+1])
		}
	}
	for _, pc := range sortedKeys(fr.inlined) {
		b.planInline(fr.inlined[pc], fr.inlined[pc].callBlock, fr.blockAt[pc+1])
	}
}

// inlineCall checks the call at pc calls the function inlined there, and
// gives the callee's start its slots: its arguments, past which undefined,
// its receiver, and undefined in every other.
func (b *builder) inlineCall(blk *Block, pc int, fr *frame, guard func(Op, Type, ir.ExitKind, ...*Value) *Value, state func() *FrameState) {
	site := fr.site
	sp := b.cur.base + b.p.Locals + b.p.Maps[pc].Depth
	// Another callee is called by Go, from the call's state, as a call that
	// is not inlined is, and native code goes on after it.
	object := guard(OpObjectOf, Ptr, ir.HostExit, b.read(sp-site.Argc-1, blk))
	same := guard(OpSameObject, None, ir.HostExit, object)
	same.Const = ir.Value{Bits: uint64(site.Callee)}
	if site.Coerce && site.ThisSlot >= 0 {
		// A receiver that is not an object is coerced, which Go does.
		guard(OpObjectOf, Ptr, ir.HostExit, b.read(sp-site.Argc-2, blk))
	}
	// Room past the operands for the callee's frame, which an exit inside
	// it writes there, and a context for Go to make it from, and one for
	// each inlined caller it is in.
	room := guard(OpFrameRoom, None, ir.HostExit)
	room.Index = fr.base - b.root.p.Locals + fr.p.Locals + fr.p.StackSize
	room.Const = ir.Value{Bits: uint64(fr.depth)}
	fr.call = state()
	undefined := b.f.newValue(blk, OpConst, Tagged)
	undefined.Const = ir.Value{Kind: ir.Undefined}
	fr.init = b.f.refsOf(fr.p.Locals + fr.p.StackSize)
	for s := range fr.init {
		fr.init[s] = undefined
	}
	for i := 0; i < site.Params && i < site.Argc; i++ {
		fr.init[i] = b.read(sp-site.Argc+i, blk)
	}
	if site.ThisSlot >= 0 {
		fr.init[site.ThisSlot] = b.read(sp-site.Argc-2, blk)
	}
}

// shifted is a callee's instruction with its slots made the builder's,
// base past its own.
func shifted(in ir.Instruction, base int) ir.Instruction {
	for _, o := range []*ir.Operand{&in.Left, &in.Right, &in.Third} {
		if o.Slot >= 0 {
			o.Slot += base
		}
	}
	in.Dest += base
	if in.Extra >= 0 {
		in.Extra += base
	}
	if in.Check {
		in.CheckSlot += base
	}
	return in
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
	// By PC, one past the end included.
	entries, leaders := b.f.bools(len(p.Code)+2), b.f.bools(len(p.Code)+2)
	entries[0], leaders[0] = true, true
	total := 0
	for pc := range p.Code {
		if !reachable(p, pc) {
			continue
		}
		if fr, ok := b.inlineAt(pc, &total); ok {
			if b.inlined == nil {
				b.inlined = map[int]*frame{}
			}
			b.inlined[pc] = fr
		}
	}
	for pc, in := range p.Code {
		if !reachable(p, pc) {
			continue
		}
		switch in.Op {
		case ir.ArrayRead:
			if in.Reference {
				// Go reads what native code does not, and resumes after it.
				entries[pc+1] = true
			}
		case ir.Nop, ir.Copy, ir.CopyPair, ir.StoreLoad, ir.Swap, ir.Insert2, ir.Insert3,
			ir.Unary, ir.Update, ir.Return, ir.ArrayUpdate, ir.ArrayLength, ir.ArrayKey:
		case ir.ArrayWrite, ir.PropertyRead, ir.PropertyWrite, ir.ReferenceRead, ir.BindingRead,
			ir.StringMethod, ir.StringCode:
			// What native code does not do exits to Go, which resumes after
			// it. A fixed global's check only deoptimizes: the binding can
			// never move, so it never fails, and what follows sees its
			// constant rather than a merge with a slot loaded at an entry.
			if site, ok := b.global(pc); !ok || !site.Fixed {
				entries[pc+1] = true
			}
		case ir.Binary:
			if in.Operator == ir.Eq || in.Operator == ir.Ne || in.Operator == ir.Mod || b.generic(pc) {
				// A comparison or remainder of non-numbers exits to Go, which
				// resumes after it.
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
			if in.Operator == ir.Eq || in.Operator == ir.Ne || b.generic(pc) {
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
	for pc, entry := range entries {
		if entry && reachable(p, pc) {
			b.entryPCs = append(b.entryPCs, pc)
			leaders[pc] = true
		}
	}

	b.blockAt = make([]*Block, len(p.Code)+2)
	var starts []int
	for pc, leader := range leaders {
		if leader && reachable(b.p, pc) {
			starts = append(starts, pc)
		}
	}
	for _, pc := range starts {
		b.blockAt[pc] = b.f.newBlock(pc)
	}
	b.endOf = make([]int, len(b.f.Blocks))
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
		b.endOf[blk.ID] = end
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
			if fr := b.inlined[end]; fr != nil {
				// Into the callee, whose returns go to the block after it.
				blk.Kind = BlockPlain
				fr.callBlock = blk
				break
			}
			if !b.host(end) {
				blk.Kind = BlockPlain
				b.edge(blk, b.blockAt[end+1])
				break
			}
			if calls := b.nativeCalls(end); len(calls) != 0 && b.f.Keeps < abi.MaxKeeps {
				if b.calls == nil {
					b.calls = map[int][]*CallSite{}
				}
				b.calls[end] = calls
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
	b.frameOf = make([]*frame, len(b.f.Blocks))
	for _, pc := range sortedKeys(b.inlined) {
		b.planInline(b.inlined[pc], b.inlined[pc].callBlock, b.blockAt[pc+1])
	}
	// Entry blocks load the live slots and go to the block for their PC.
	for _, pc := range b.entryPCs {
		e := b.f.newBlock(-1)
		e.Kind = BlockPlain
		e.Generic = b.fb != nil && b.fb.EntryGeneric(int(p.Maps[pc].PC))
		b.edge(e, b.blockAt[pc])
		b.f.Entries = append(b.f.Entries, Entry{PC: pc, Depth: p.Maps[pc].Depth, Block: e})
	}
	// A loop header is reached by an edge from a block at or after it, in
	// one frame: a callee's blocks have their own PCs, and no loops.
	for _, blk := range b.f.Blocks {
		blk.Backedge = b.f.bools(len(blk.Preds))
		for i, pred := range blk.Preds {
			if pred.PC >= 0 && blk.PC >= 0 && pred.PC >= blk.PC && b.frameAt(pred) == b.root && b.frameAt(blk) == b.root {
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
	from.Succs = b.f.appendBlock(from.Succs, to)
	to.Preds = b.f.appendBlock(to.Preds, from)
}

// translate fills every block, sealing each once its predecessors are filled
// (Braun et al., "Simple and Efficient Construction of Static Single
// Assignment Form").
func (b *builder) translate() {
	n := len(b.f.Blocks)
	b.defs = make([][]*Value, n)
	b.sealed, b.filled = b.f.bools(n), b.f.bools(n)
	b.incomplete = make([][]slotPhi, n)

	// Reverse post-order from the entries, so that a block's forward
	// predecessors are filled before it.
	order := make([]*Block, 0, n)
	seen := b.f.bools(n)
	var visit func(*Block)
	visit = func(blk *Block) {
		if seen[blk.ID] {
			return
		}
		seen[blk.ID] = true
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
	// Blocks no entry reaches are dropped; the others are kept in this
	// order, which every later pass walks (Func.Blocks).
	b.f.Blocks = append(b.f.Blocks[:0], order...)
	for _, blk := range b.f.Blocks {
		preds := blk.Preds[:0]
		backedge := blk.Backedge[:0]
		for i, pred := range blk.Preds {
			if seen[pred.ID] {
				preds = append(preds, pred)
				backedge = append(backedge, blk.Backedge[i])
			}
		}
		blk.Preds, blk.Backedge = preds, backedge
	}

	for _, blk := range order {
		b.trySeal(blk)
		b.enter(b.frameAt(blk))
		b.fill(blk)
		b.enter(b.root)
		b.filled[blk.ID] = true
		for _, s := range blk.Succs {
			b.trySeal(s)
		}
	}
	for _, blk := range order {
		if !b.sealed[blk.ID] {
			b.seal(blk)
		}
	}
	for i, blk := range b.f.Blocks {
		blk.ID = i
	}
}

func (b *builder) trySeal(blk *Block) {
	if b.sealed[blk.ID] {
		return
	}
	for _, p := range blk.Preds {
		if !b.filled[p.ID] {
			return
		}
	}
	b.seal(blk)
}

// slotPhi is a phi a block made for a slot before it was sealed.
type slotPhi struct {
	slot int
	phi  *Value
}

func (b *builder) seal(blk *Block) {
	// Slot by slot: completing a phi can make others, and their numbers
	// must not vary from one build to the next.
	pending := b.incomplete[blk.ID]
	slices.SortFunc(pending, func(x, y slotPhi) int { return x.slot - y.slot })
	for _, p := range pending {
		b.addPhiOperands(p.slot, p.phi)
	}
	b.incomplete[blk.ID] = nil
	b.sealed[blk.ID] = true
}

// assign is an instruction's write of a slot, which Func.Written records;
// write is also how reads record the phis they make.
func (b *builder) assign(slot int, blk *Block, v *Value) {
	if slot < len(b.f.written) {
		b.f.written[slot] = true
	}
	b.write(slot, blk, v)
}

func (b *builder) write(slot int, blk *Block, v *Value) {
	d := b.defs[blk.ID]
	if d == nil {
		d = b.f.refsOf(b.nslots)
		b.defs[blk.ID] = d
	}
	d[slot] = v
}

func (b *builder) read(slot int, blk *Block) *Value {
	if d := b.defs[blk.ID]; d != nil && d[slot] != nil {
		return d[slot]
	}
	var v *Value
	switch {
	case !b.sealed[blk.ID]:
		v = b.newPhi(blk)
		b.incomplete[blk.ID] = append(b.incomplete[blk.ID], slotPhi{slot, v})
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
	v := b.f.alloc(Value{Op: OpPhi, Type: Tagged, Block: blk})
	prepend(b.f, blk, v)
	return v
}

// prepend puts v first in blk's values.
func prepend(f *Func, blk *Block, v *Value) {
	blk.Values = f.appendValue(blk.Values, nil)
	copy(blk.Values[1:], blk.Values)
	blk.Values[0] = v
}

func (b *builder) addPhiOperands(slot int, phi *Value) {
	preds := phi.Block.Preds
	if phi.Args == nil {
		phi.Args = b.f.refsOf(len(preds))[:0]
	}
	for _, p := range preds {
		a := b.read(slot, p)
		a.Uses++
		phi.Args = append(phi.Args, a)
	}
}

// constIn makes a tagged constant at the start of a block.
func (b *builder) constIn(blk *Block, c ir.Value) *Value {
	v := b.f.alloc(Value{Op: OpConst, Type: Tagged, Const: c, Block: blk})
	prepend(b.f, blk, v)
	return v
}

// equality is an Eq or Ne as a bool, of any two values: with null or
// undefined (nullish), or as OpEqTagged compares them. It returns nil for
// any other operation.
func (b *builder) equality(blk *Block, in ir.Instruction, operand func(ir.Operand) *Value, guard func(Op, Type, ir.ExitKind, ...*Value) *Value) *Value {
	if in.Operator != ir.Eq && in.Operator != ir.Ne {
		return nil
	}
	if c := b.nullish(blk, in, operand, guard); c != nil {
		return c
	}
	c := guard(OpEqTagged, Bool, ir.HostExit, operand(in.Left), operand(in.Right))
	if in.Strict {
		c.Index = 1
	}
	if in.Operator == ir.Ne {
		c = b.f.newValue(blk, OpNot, Bool, c)
	}
	return c
}

// nullish is an Eq or Ne with null or undefined, as a bool, whatever the
// other operand is: compared natively, rather than taken for a number.
// It returns nil for any other comparison.
func (b *builder) nullish(blk *Block, in ir.Instruction, operand func(ir.Operand) *Value, guard func(Op, Type, ir.ExitKind, ...*Value) *Value) *Value {
	if in.Operator != ir.Eq && in.Operator != ir.Ne {
		return nil
	}
	isNullish := func(v *Value) bool {
		return v.Op == OpConst && (v.Const.Kind == ir.Null || v.Const.Kind == ir.Undefined)
	}
	x, y := operand(in.Left), operand(in.Right)
	if isNullish(x) {
		x, y = y, x
	}
	if !isNullish(y) {
		return nil
	}
	var c *Value
	if in.Strict {
		c = b.f.newValue(blk, OpStrictNullish, Bool, x)
		if y.Const.Kind == ir.Undefined {
			c.Aux = 1
		}
	} else {
		c = guard(OpLooseNullish, Bool, ir.HostExit, x)
	}
	if in.Operator == ir.Ne {
		c = b.f.newValue(blk, OpNot, Bool, c)
	}
	return c
}

// state captures the frame at a PC: every live slot's current value. Inside
// an inlined callee it is the caller's at the call and the callee's
// (InlineState).
func (b *builder) state(blk *Block, pc int) *FrameState {
	if fr := b.cur; fr != b.root {
		call, depth := fr.call, b.p.Maps[pc].Depth
		s := b.f.newState(FrameState{PC: call.PC, Depth: call.Depth, Site: call.Site, Slots: b.f.refsOf(fr.base + b.p.Locals + depth)})
		for i, v := range call.Slots {
			if v != nil {
				s.Slots[i] = v
				v.Uses++
			}
		}
		for i := fr.base; i < len(s.Slots); i++ {
			s.Slots[i] = b.read(i, blk)
			s.Slots[i].Uses++
		}
		s.Inline = &InlineState{Parent: call.Inline, Closure: fr.site.Closure, Base: fr.base, Locals: b.p.Locals, ThisSlot: fr.site.ThisSlot,
			PC: b.p.Maps[pc].PC, Depth: depth, Site: pc}
		return s
	}
	depth := b.p.Maps[pc].Depth
	s := b.f.newState(FrameState{PC: b.p.Maps[pc].PC, Depth: depth, Slots: b.f.refsOf(b.p.Locals + depth), Site: pc})
	for i := range s.Slots {
		s.Slots[i] = b.read(i, blk)
		s.Slots[i].Uses++
	}
	return s
}

func (b *builder) fill(blk *Block) {
	f := b.f
	if fr := b.cur; fr != b.root && blk == fr.start {
		for s, v := range fr.init {
			b.write(fr.base+s, blk, v)
		}
		return
	}
	if blk.PC < 0 {
		e, _ := f.entryForBlock(blk)
		blk.Header = f.newState(FrameState{PC: b.p.Maps[e.PC].PC, Depth: e.Depth, Slots: f.refsOf(b.p.Locals + e.Depth), Site: -1})
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
		blk.Header.Site = -1
	}
	for pc := blk.PC; pc <= b.endOf[blk.ID]; pc++ {
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
	if b.cur != b.root {
		in = shifted(in, b.cur.base)
	}
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
		if b.cur != b.root {
			// Inside an inlined callee Go does the operation, in the
			// callee's frame, and native code goes on where it can.
			kind = ir.HostExit
		}
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
		if c := b.equality(blk, in, operand, guard); c != nil {
			b.assign(in.Dest, blk, boxB(c))
			break
		}
		kind := ir.GuardExit
		if in.Operator == ir.Eq || in.Operator == ir.Ne || in.Operator == ir.Mod || b.generic(pc) {
			kind = ir.HostExit
		}
		x, y := number(in.Left, kind), number(in.Right, kind)
		if in.Operator == ir.Mod {
			b.assign(in.Dest, blk, boxF(guard(OpModF64, Float64, ir.HostExit, x, y)))
			break
		}
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
		} else if n := b.equality(blk, in, operand, guard); n != nil {
			c = n
		} else {
			kind := ir.GuardExit
			if in.Operator == ir.Eq || in.Operator == ir.Ne || b.generic(pc) {
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
		if b.cur != b.root {
			// The call's result, and on after it.
			b.assign(b.cur.result, blk, v)
			break
		}
		blk.Control = v
		v.Uses++
	case ir.ArrayRead, ir.ArrayUpdate:
		if in.Reference {
			// Whatever the element holds, by its cell; Go reads the rest.
			array := guard(OpArrayOf, Ptr, ir.HostExit, operand(in.Left))
			cell := guard(OpElemCell, Source, ir.HostExit, array, number(in.Right, ir.HostExit))
			v := f.newValue(blk, OpLoadCell, Tagged, cell)
			v.Shadow = cell
			b.assign(in.Dest, blk, v)
			break
		}
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
		b.assign(in.Dest, blk, boxF(guard(OpLength, Float64, ir.GuardExit, operand(in.Left))))
	case ir.ArrayKey:
		guard(OpArrayOf, Ptr, ir.GuardExit, operand(in.Left))
		guard(OpElemKey, None, ir.GuardExit, number(in.Right, ir.GuardExit))
	case ir.ArrayWrite:
		// Go stores what native code does not: a value that is not a
		// number, or into an element that does not hold one.
		array := guard(OpArrayOf, Ptr, ir.HostExit, operand(in.Left))
		key, value := number(in.Right, ir.HostExit), number(in.Third, ir.HostExit)
		guard(OpElemWrite, None, ir.HostExit, array, key, value)
	case ir.StringMethod:
		cell := guard(OpStringMethod, Source, ir.HostExit, operand(in.Left))
		v := f.newValue(blk, OpLoadCell, Tagged, cell)
		v.Shadow = cell
		b.assign(in.Dest, blk, v)
	case ir.StringCode:
		code := guard(OpStringCode, Float64, ir.HostExit, operand(in.Left), operand(in.Right), number(in.Third, ir.HostExit))
		b.assign(in.Dest, blk, boxF(code))
	case ir.BindingRead:
		site, ok := b.global(pc)
		if !ok {
			blk.ExitKind = ir.HostExit
			blk.State = state()
			blk.State.addUse()
			break
		}
		kind := ir.HostExit
		if site.Fixed {
			kind = ir.GuardExit
		}
		cell := guard(OpGlobalCell, Source, kind)
		cell.Index, cell.Key = int(site.Index), site.Key
		if site.Fixed {
			k := f.newValue(blk, OpConst, Tagged)
			k.Const = site.Constant
			b.assign(in.Dest, blk, k)
			break
		}
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
		// A read the receiver's prototypes answered checks them too; a
		// write, which makes a property of the receiver's own, never meets
		// one.
		var holders *[2]Holder
		if site.Holders[0].Object != 0 && site.Shape != 0 && in.Op != ir.PropertyWrite {
			holders = new([2]Holder)
			*holders = site.Holders
		} else if site.Holders[0].Object != 0 {
			site.Shape = 0
		}
		var cases []PropertyCase
		if site.Shape != 0 && in.Op != ir.PropertyWrite {
			cases = site.Cases
		}
		if in.Op == ir.ReferenceRead {
			cell := guard(OpPropCell, Source, ir.HostExit, object)
			cell.Const, cell.Index, cell.Key = ir.Value{Bits: uint64(site.Shape)}, int(site.Index), site.Key
			cell.Holders, cell.Cases = holders, cases
			v := f.newValue(blk, OpLoadCell, Tagged, cell)
			v.Shadow = cell
			b.assign(in.Dest, blk, v)
			break
		}
		if in.Op == ir.PropertyRead {
			v := guard(OpPropRead, Float64, ir.HostExit, object)
			v.Const, v.Index, v.Key = ir.Value{Bits: uint64(site.Shape)}, int(site.Index), site.Key
			v.Holders, v.Cases = holders, cases
			b.assign(in.Dest, blk, boxF(v))
			break
		}
		v := guard(OpPropWrite, None, ir.HostExit, object, operand(in.Right))
		v.Const, v.Index, v.Key = ir.Value{Bits: uint64(site.Shape)}, int(site.Index), site.Key
	case ir.Host, ir.Call:
		if fr := b.inlinedAt(pc); fr != nil {
			b.inlineCall(blk, pc, fr, guard, state)
			break
		}
		if calls := b.calls[pc]; calls != nil && b.cur == b.root {
			// The call's operands, from the receiver or the function on; its
			// result is the slot below them, its pointer word kept.
			n := calls[0].Argc + 1
			if calls[0].Method {
				n++
			}
			sp := b.p.Locals + b.p.Maps[pc].Depth
			args := make([]*Value, n)
			for i := range args {
				args[i] = b.read(sp-n+i, blk)
			}
			call := guard(OpCall, Tagged, ir.HostExit, args...)
			if b.f.Keeps < abi.MaxKeeps {
				call.Calls, call.Index = calls, b.f.Keeps
				b.f.Keeps++
			}
			cell := f.newValue(blk, OpCallCell, Source, call)
			r := f.newValue(blk, OpKept, Tagged, cell, call)
			r.Shadow = cell
			b.write(b.p.Locals+b.p.Maps[pc+1].Depth-1, blk, r)
			break
		}
		blk.ExitKind = ir.HostExit
		blk.State = state()
		blk.State.addUse()
	}
}

// nativeCalls are the functions the call at pc may call natively, if any
// (CallSite): those the VM has seen it call, if speculation has not given
// up on it and there is an entry after it to go on at.
func (b *builder) nativeCalls(pc int) []*CallSite {
	if b.fb == nil || pc+1 >= len(b.p.Code) || !reachable(b.p, pc+1) || b.fb.Generic(pc) {
		return nil
	}
	var calls []*CallSite
	for _, site := range b.fb.NativeCalls(pc) {
		depth, after := b.p.Maps[pc].Depth, b.p.Maps[pc+1].Depth
		operands := site.Argc + 1
		if site.Method {
			operands++
		}
		if site.Argc < 0 || depth < operands || after != depth-operands+1 || site.ThisSlot >= 0 && !site.Method {
			continue
		}
		if site.ThisSlot < 0 {
			// A receiver it never reads needs no coercing.
			site.Coerce = false
		}
		c := new(CallSite)
		*c = site
		calls = append(calls, c)
	}
	return calls
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

// sortedKeys is a map's keys in order, so that blocks are made the same
// every time.
func sortedKeys(m map[int]*frame) []int {
	keys := make([]int, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return keys
}
