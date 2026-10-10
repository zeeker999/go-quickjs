// Package compile lowers VM bytecode into the JIT's slot IR. It is separate
// from executable-memory support and does not modify shared function templates.
package compile

import (
	"fmt"
	"slices"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

// Compiler work and scratch size are bounded even for permanently refused
// functions. Runtime code/metadata ownership is handled by the optional VM tier.
const (
	// MaxInstructions bounds control-flow analysis and IR allocation.
	MaxInstructions = ir.MaxInstructions
	// MaxSlots bounds the locals and operand scratch storage together.
	MaxSlots = ir.MaxSlots
)

// Refusal identifies the first unsupported construct or invalid state map.
// PC is -1 for a function-wide refusal.
type Refusal struct {
	PC     int
	Reason string
}

func (e *Refusal) Error() string {
	if e.PC < 0 {
		return "jit: " + e.Reason
	}
	return fmt.Sprintf("jit: pc %d: %s", e.PC, e.Reason)
}

func refuse(pc int, why string) error { return &Refusal{PC: pc, Reason: why} }

// Lower checks eligibility, validates control-flow stack depths, and produces
// one atomic IR instruction and pre-instruction map per bytecode instruction.
// No executable memory is allocated. Unsupported code, even when unreachable,
// is refused. Captured own locals are excluded; read-only upvalues are snapshots
// refreshed by the VM after host operations.
// Property keys are source-name indices; the VM resolves them to its runtime's
// scalar property identifiers before emission or evaluation with borrowed views.
func Lower(fn *bytecode.Function) (*ir.Program, error) {
	return lowerRecovered(fn, lowering{})
}

// LowerSSA lowers for the new pipeline (internal/jit/ssa), whose costs are
// not the slot IR emitters': it reads properties and the receiver in place
// where the VM's caches know the shape, so it keeps every property
// operation and the receiver, which Lower keeps only in loops worth a view's
// preparation; and it reads globals where the VM found them, with no
// binding-view slots.
func LowerSSA(fn *bytecode.Function) (*ir.Program, error) {
	return lowerRecovered(fn, lowering{ssa: true})
}

// LowerSSAInline is LowerSSA for a callee the new pipeline inlines into its
// caller (ssa.InlineSite): it lowers what a function called alone would not
// be worth compiling for, host operations without a loop.
func LowerSSAInline(fn *bytecode.Function) (*ir.Program, error) {
	return lowerRecovered(fn, lowering{ssa: true, callee: true})
}

// LowerCalls additionally retains guarded numeric fields across call boundaries.
// Its caller must refresh borrowed views after every potentially effectful call.
func LowerCalls(fn *bytecode.Function) (*ir.Program, error) {
	return lowerRecovered(fn, lowering{calls: true})
}

// LowerCallee permits small guarded functions whose entry costs are amortized
// by an encoded caller. It keeps reference fields on the coordinator's host
// path and does not change the standalone profitability policy.
func LowerCallee(fn *bytecode.Function) (*ir.Program, error) {
	return lowerRecovered(fn, lowering{calls: true, callee: true})
}

// lowering is how a function is lowered: for which entry points, and for
// which pipeline.
type lowering struct{ calls, callee, ssa bool }

// lowerRecovered turns a panic in analysis into a refusal: an optional tier
// must never take down its host, and the function then runs in the existing
// tiers.
func lowerRecovered(fn *bytecode.Function, m lowering) (p *ir.Program, err error) {
	defer func() {
		if v := recover(); v != nil {
			p, err = nil, refuse(-1, fmt.Sprintf("internal error: %v", v))
		}
	}()
	return lowerImpl(fn, m)
}

// lowerImpl is lowerFunction, or a test's replacement for it.
var lowerImpl = lowerFunction

// SSAConstruct has the new pipeline compile functions that construct, `new
// C(...)`, which its code makes natively where it can and leaves to Go
// otherwise. It is off until built-in constructors and
// Function.prototype.call are made natively too: DeltaBlue's constructors,
// which use both, leave native code at each construction, and run 43%
// slower than the tree tier runs them (docs/jit-progress.md, P7w).
var SSAConstruct = false

func lowerFunction(fn *bytecode.Function, m lowering) (*ir.Program, error) {
	if fn == nil {
		return nil, refuse(-1, "nil function")
	}
	switch {
	case fn.TopLevel || fn.IsModule:
		return nil, refuse(-1, "top-level or module code")
	case fn.Async || fn.Generator:
		return nil, refuse(-1, "async or generator function")
	case fn.Kind != bytecode.KindNormal && fn.Kind != bytecode.KindArrow && fn.Kind != bytecode.KindMethod:
		return nil, refuse(-1, "special function kind")
	case fn.HasDirectEval || len(fn.EvalScopes) != 0:
		return nil, refuse(-1, "direct eval")
	case fn.UsesArguments || fn.MappedArguments:
		return nil, refuse(-1, "arguments object")
	case !fn.HasSimpleParams || fn.HasRest || fn.ParamsAreLexical:
		return nil, refuse(-1, "non-simple parameters")
	case len(fn.Code) == 0 || len(fn.Code) > MaxInstructions:
		return nil, refuse(-1, "instruction budget")
	case fn.LocalCount < 0 || fn.MaxStack < 0 || fn.LocalCount > MaxSlots || fn.MaxStack > MaxSlots || len(fn.Upvalues) > MaxSlots || fn.LocalCount+fn.MaxStack+len(fn.Upvalues) > MaxSlots:
		return nil, refuse(-1, "slot budget")
	case fn.ParamCount < 0 || fn.ParamCount > fn.LocalCount || len(fn.Locals) != fn.LocalCount:
		return nil, refuse(-1, "invalid local layout")
	}
	for _, local := range fn.Locals {
		if local.Captured {
			return nil, refuse(-1, "captured local")
		}
	}
	effects := make([]effect, len(fn.Code))
	host, loop, indexed, property, bitwise, this := false, false, false, false, false, false
	for pc, in := range fn.Code {
		e, err := describe(fn, pc, in)
		if err != nil {
			return nil, err
		}
		effects[pc] = e
		if (in.Op == bytecode.OpNew || in.Op == bytecode.OpSetGlobal || in.Op == bytecode.OpSetGlobalStrict) && (!m.ssa || !SSAConstruct) {
			// The new pipeline constructs natively (mir's native calls),
			// and leaves an assignment to a global to Go, as the code that
			// constructs -- a benchmark's driver, planner = new Planner()
			// -- does; the old one does neither.
			return nil, refuse(pc, "unsupported opcode "+in.Op.String())
		}
		host = host || in.Op == bytecode.OpNew || in.Op == bytecode.OpSetGlobal || in.Op == bytecode.OpSetGlobalStrict || in.Op == bytecode.OpNewArray || in.Op == bytecode.OpCall || in.Op == bytecode.OpCallMethod || in.Op == bytecode.OpGetGlobal || in.Op == bytecode.OpGetPropThis || in.Op == bytecode.OpPushThis || in.Op == bytecode.OpGetProp || in.Op == bytecode.OpSetProp
		property = property || in.Op == bytecode.OpPushThis || in.Op == bytecode.OpGetProp || in.Op == bytecode.OpSetProp
		this = this || in.Op == bytecode.OpPushThis
		raw := uint32(in.Op)
		switch in.Op {
		case bytecode.OpBinLocal, bytecode.OpBinImm:
			raw = in.B
		case bytecode.OpLocalBinImm:
			raw = in.A >> 24
		}
		op, _ := operator(raw)
		host = host || raw == uint32(bytecode.OpMod)
		bitwise = bitwise || op >= ir.BitAnd && op <= ir.UShr || in.Op == bytecode.OpBitNot
		loop = loop || e.branch && int(in.A) <= pc
		indexed = indexed || in.Op == bytecode.OpGetIndex || in.Op == bytecode.OpSetIndex || in.Op == bytecode.OpGetLocalIndex || in.Op == bytecode.OpGetLocalIndexUpdate
	}
	// Tiny host-only wrappers pay the bridge overhead without enough native work.
	if host && !loop && !indexed && !m.callee {
		return nil, refuse(-1, "host operations without native loop or array work")
	}
	if property && !indexed && !bitwise && !m.callee && !m.ssa {
		return nil, refuse(-1, "property operations without native array or bitwise work")
	}
	maps := make([]ir.StateMap, len(fn.Code))
	for pc := range maps {
		maps[pc] = ir.StateMap{PC: uint32(pc), Depth: -1}
	}
	maps[0].Depth = 0
	queue := make([]int, 1, len(fn.Code))
	queue[0] = 0
	join := func(pc, depth int) error {
		if pc >= len(maps) {
			return refuse(pc-1, "fallthrough past function")
		}
		if maps[pc].Depth >= 0 {
			if maps[pc].Depth != depth {
				return refuse(pc, "inconsistent stack depth at join")
			}
			return nil
		}
		maps[pc].Depth = depth
		queue = append(queue, pc)
		return nil
	}
	for head := 0; head < len(queue); head++ {
		pc := queue[head]
		e, depth := effects[pc], maps[pc].Depth
		if depth < e.need {
			return nil, refuse(pc, "operand stack underflow")
		}
		nextDepth := depth + e.delta
		if nextDepth > fn.MaxStack {
			return nil, refuse(pc, "operand stack exceeds MaxStack")
		}
		if e.branch {
			targetDepth := nextDepth
			if e.keep {
				targetDepth = depth
			}
			if err := join(int(fn.Code[pc].A), targetDepth); err != nil {
				return nil, err
			}
		}
		if !e.terminal {
			if err := join(pc+1, nextDepth); err != nil {
				return nil, err
			}
		}
	}
	p := &ir.Program{Locals: fn.LocalCount + len(fn.Upvalues), StackSize: fn.MaxStack, This: this, Maps: maps, Code: make([]ir.Instruction, len(fn.Code))}
	if this {
		if p.Locals+p.StackSize == MaxSlots {
			p.This = false
		} else {
			p.Locals++
		}
	}
	for pc, in := range fn.Code {
		if maps[pc].Depth >= 0 {
			p.Code[pc] = lower(fn, in, p.Locals+maps[pc].Depth, p.This)
		}
	}
	selectNumericProperties(p, m.ssa)
	if m.ssa {
		liveLocals(fn, p, effects)
	}
	if m.ssa {
		for pc, in := range p.Code {
			if in.Op == ir.Host && p.Maps[pc].Depth >= 0 {
				// The new pipeline computes a remainder; the old one leaves
				// it to Go.
				if mod, ok := lowerMod(fn.Code[pc], p.Locals+p.Maps[pc].Depth); ok {
					p.Code[pc] = mod
				}
			}
			if in.Op == ir.BindingRead {
				// The operand is the binding view's slot in the old pipeline;
				// the new one reads the global object, through its context.
				p.Code[pc].Left = ir.Literal(ir.Value{Kind: ir.Undefined})
			}
			if b := fn.Code[pc]; in.Op == ir.Host && b.Op == bytecode.OpGetPropThis && p.Maps[pc].Depth >= 0 {
				// A method's read, which keeps its receiver below it: the
				// new pipeline reads it where the site's cache found it,
				// on a prototype too, and carries it by its cell to the call.
				sp := p.Locals + p.Maps[pc].Depth
				p.Code[pc] = ir.Instruction{Op: ir.ReferenceRead, Left: ir.Slot(sp - 1), Dest: sp, Key: b.A}
			}
		}
		return p, nil
	}
	selectPropertyLoops(fn, p, m.calls, m.callee)
	selectGlobalSlots(fn, p)
	return p, nil
}

// Numeric global reads borrow live cells, rather than snapshotting their values.
// Reserve one handle per name; excess bindings retain the original host path.
func selectGlobalSlots(fn *bytecode.Function, p *ir.Program) {
	base := p.Locals
	for pc, in := range p.Code {
		if in.Op != ir.BindingRead {
			continue
		}
		i := 0
		for i < len(p.Globals) && p.Globals[i] != in.Key {
			i++
		}
		if i == len(p.Globals) {
			if base+p.StackSize+len(p.Globals) == MaxSlots {
				p.Code[pc] = ir.Instruction{Op: ir.Host}
				continue
			}
			p.Globals = append(p.Globals, in.Key)
		}
		in.Left = ir.Slot(base + i)
		p.Code[pc] = in
	}
	if len(p.Globals) == 0 {
		return
	}
	p.Locals += len(p.Globals)
	for pc, old := range p.Code {
		if p.Maps[pc].Depth < 0 || old.Op == ir.Host {
			continue
		}
		in := lower(fn, fn.Code[pc], p.Locals+p.Maps[pc].Depth, p.This)
		if old.Op == ir.BindingRead {
			in.Left = old.Left
		}
		if old.Op == ir.ReferenceRead {
			in.Op = ir.ReferenceRead
		}
		p.Code[pc] = in
	}
}

// Borrowing object tables and another receiver root costs something on every
// entry. Keep mixed loops on their existing bridge until their remaining host
// operations are covered; a native field loop must amortize that preparation.
func selectPropertyLoops(fn *bytecode.Function, p *ir.Program, calls, callee bool) bool {
	// Preparing a reference graph cannot amortize a read performed only once
	// before the loop. Keep those entries on the existing cheap Go bridge.
	loopReference := false
	for pc, in := range p.Code {
		if (in.Op == ir.Jump || in.Op == ir.Branch) && in.Target <= pc {
			for _, body := range p.Code[in.Target : pc+1] {
				loopReference = loopReference || body.Op == ir.ReferenceRead
			}
		}
	}
	if !loopReference {
		for pc, in := range p.Code {
			if in.Op == ir.ReferenceRead {
				p.Code[pc] = ir.Instruction{Op: ir.Host}
			}
		}
	}
	hosts := make([]int, len(p.Code)+1)
	fields, loop, mixed, fieldCount := false, false, false, 0
	for pc, in := range p.Code {
		hosts[pc+1] = hosts[pc]
		if in.Op == ir.Host {
			hosts[pc+1]++
		}
		fields = fields || in.Op == ir.PropertyRead || in.Op == ir.PropertyWrite || in.Op == ir.ReferenceRead
		if in.Op == ir.PropertyRead || in.Op == ir.PropertyWrite || in.Op == ir.ReferenceRead {
			fieldCount++
		}
	}
	for pc, in := range p.Code {
		if (in.Op == ir.Jump || in.Op == ir.Branch) && in.Target <= pc {
			loop = true
			mixed = mixed || hosts[pc+1] != hosts[in.Target]
		}
	}
	if callee || calls && mixed {
		for pc, in := range p.Code {
			if in.Op == ir.ReferenceRead {
				p.Code[pc] = ir.Instruction{Op: ir.Host}
			}
		}
	}
	fields = fields && (loop || callee) && (!mixed || calls)
	if p.This && !fields && hosts[len(p.Code)]+fieldCount != 0 {
		p.This = false
		p.Locals--
		for pc, in := range fn.Code {
			if p.Maps[pc].Depth >= 0 {
				p.Code[pc] = lower(fn, in, p.Locals+p.Maps[pc].Depth, false)
			}
		}
		selectNumericProperties(p, false)
	}
	if !fields {
		for pc, in := range p.Code {
			if in.Op == ir.PropertyRead || in.Op == ir.PropertyWrite || in.Op == ir.ReferenceRead {
				p.Code[pc] = ir.Instruction{Op: ir.Host}
			}
		}
	}
	return loop && !mixed
}

// This bounded backward scan chooses field reads whose results feed numeric
// operations or reference receivers. It is a profitability hint; every native
// read still checks its live type and permissions before committing. With
// cells, the new pipeline carries any value a read finds by where it found
// it, so a read no native operation consumes -- a call's callee or argument,
// which Go takes from the frame -- stays native there, as a global's cell or
// a reference, rather than add an exit.
func selectNumericProperties(p *ir.Program, cells bool) {
	const number, reference uint8 = 1, 2
	var needed [MaxSlots]uint8
	read := func(o ir.Operand, kind uint8) {
		if o.Slot >= 0 {
			needed[o.Slot] |= kind
		}
	}
	take := func(slot int) uint8 { k := needed[slot]; needed[slot] = 0; return k }
	for pc := len(p.Code) - 1; pc >= 0; pc-- {
		if p.Maps[pc].Depth < 0 {
			continue
		}
		in := p.Code[pc]
		switch in.Op {
		case ir.PropertyRead, ir.BindingRead:
			kind := take(in.Dest)
			if kind&number == 0 && !(cells && in.Op == ir.BindingRead) {
				if in.Op == ir.PropertyRead && (kind&reference != 0 || cells) {
					in.Op = ir.ReferenceRead
					p.Code[pc] = in
				} else {
					p.Code[pc] = ir.Instruction{Op: ir.Host}
				}
			}
			if in.Op != ir.BindingRead {
				read(in.Left, reference)
			}
		case ir.Copy:
			read(in.Left, take(in.Dest))
		case ir.CopyPair:
			left, right := take(in.Dest), take(in.Extra)
			read(in.Left, left)
			read(in.Right, right)
		case ir.StoreLoad:
			read(in.Right, take(in.Extra))
			read(in.Left, take(in.Dest))
		case ir.Swap:
			needed[in.Dest], needed[in.Extra] = needed[in.Extra], needed[in.Dest]
		case ir.Binary:
			take(in.Dest)
			if !(cells && (in.Operator == ir.Eq || in.Operator == ir.Ne)) {
				// The new pipeline compares any value: equality takes a read
				// as it finds it, as a reference if it is one.
				read(in.Left, number)
				read(in.Right, number)
			}
		case ir.Unary, ir.Update:
			take(in.Dest)
			if in.Op == ir.Update && in.Extra >= 0 {
				take(in.Extra)
			}
			if in.Operator != ir.Not {
				read(in.Left, number)
			}
		case ir.ArrayRead, ir.ArrayUpdate, ir.ArrayLength:
			if kind := take(in.Dest); cells && in.Op == ir.ArrayRead && kind&number == 0 {
				in.Reference = true
				p.Code[pc] = in
			}
			if in.Op == ir.ArrayUpdate {
				take(in.Extra)
			}
			read(in.Left, reference)
			if in.Op != ir.ArrayLength {
				read(in.Right, number)
			}
		case ir.ArrayWrite, ir.ArrayKey:
			read(in.Left, reference)
			read(in.Right, number)
			if in.Op == ir.ArrayWrite {
				read(in.Third, number)
			}
		case ir.PropertyWrite:
			read(in.Left, reference)
			if !cells {
				// The new pipeline stores any value.
				read(in.Right, number)
			}
		case ir.Branch:
			if in.Operator != ir.Truth && !(cells && (in.Operator == ir.Eq || in.Operator == ir.Ne)) {
				read(in.Left, number)
				read(in.Right, number)
			}
		case ir.Host:
			clear(needed[p.Locals:])
		case ir.Insert2, ir.Insert3:
			n, last := in.Dest, 2
			if in.Op == ir.Insert2 {
				last = 1
			}
			kind := needed[n] | needed[n+last+1]
			for i := 0; i < last; i++ {
				needed[n+i] = needed[n+i+1]
			}
			needed[n+last], needed[n+last+1] = kind, 0
		}
	}
}

// liveLocals fills p's Live: for each reachable instruction, the locals the
// code from there may read before writing them, as a backward dataflow
// over the instructions, each a branch to its A, a fall to the next, or
// both. Only the instructions describe takes reach here; of those, the
// ones below name locals, in A and B, and no other reads or writes one by
// index -- a host operation runs the instruction itself, so a local it
// names is read as the instruction reads it. describe takes no exception
// handler: one that it comes to take must be a successor here of every
// instruction it guards, or a catch would find its locals unwritten.
func liveLocals(fn *bytecode.Function, p *ir.Program, effects []effect) {
	n := fn.LocalCount
	if n == 0 {
		return
	}
	words := (n + 63) / 64
	live := make([]uint64, len(fn.Code)*words)
	at := func(pc int) []uint64 { return live[pc*words : (pc+1)*words] }
	read := func(s []uint64, i uint32) { s[i/64] |= 1 << (i % 64) }
	kill := func(s []uint64, i uint32) { s[i/64] &^= 1 << (i % 64) }
	in := make([]uint64, words)
	for changed := true; changed; {
		changed = false
		for pc := len(fn.Code) - 1; pc >= 0; pc-- {
			if p.Maps[pc].Depth < 0 {
				continue
			}
			clear(in)
			e, ins := effects[pc], fn.Code[pc]
			if !e.terminal && pc+1 < len(fn.Code) {
				for w, x := range at(pc + 1) {
					in[w] |= x
				}
			}
			if e.branch {
				for w, x := range at(int(ins.A)) {
					in[w] |= x
				}
			}
			switch ins.Op {
			case bytecode.OpSetLocal, bytecode.OpInitLocal, bytecode.OpPutLocal, bytecode.OpClearLocal:
				kill(in, ins.A)
			case bytecode.OpSetLocalGet:
				// The write, then the read, which sees it if they are one.
				kill(in, ins.A)
				if ins.B != ins.A {
					read(in, ins.B)
				}
			case bytecode.OpGetLocal, bytecode.OpGetLocalCheck, bytecode.OpSetLocalCheck, bytecode.OpIncLocal,
				bytecode.OpDecLocal, bytecode.OpUpdateLocal, bytecode.OpBinLocal:
				// set_local_check reads the binding first: its dead zone.
				read(in, ins.A)
			case bytecode.OpGetLocal2, bytecode.OpGetLocalIndex:
				read(in, ins.A)
				read(in, ins.B)
			case bytecode.OpGetLocalIndexUpdate:
				read(in, ins.A)
				read(in, ins.B>>2)
			case bytecode.OpLocalBinImm:
				read(in, ins.A&(1<<24-1))
			}
			if s := at(pc); !slices.Equal(s, in) {
				copy(s, in)
				changed = true
			}
		}
	}
	p.Live, p.LiveLocals, p.LiveWords = live, n, words
}

type effect struct {
	need, delta int
	branch      bool
	keep        bool
	terminal    bool
}

func describe(fn *bytecode.Function, pc int, in bytecode.Instr) (effect, error) {
	bad := func(why string) (effect, error) { return effect{}, refuse(pc, why) }
	local := func(index uint32) bool { return uint64(index) < uint64(fn.LocalCount) }
	switch in.Op {
	case bytecode.OpGetLocal, bytecode.OpSetLocal, bytecode.OpPutLocal,
		bytecode.OpInitLocal, bytecode.OpGetLocalCheck, bytecode.OpSetLocalCheck,
		bytecode.OpClearLocal, bytecode.OpIncLocal, bytecode.OpDecLocal,
		bytecode.OpUpdateLocal, bytecode.OpBinLocal:
		if !local(in.A) {
			return bad("local index out of bounds")
		}
	case bytecode.OpGetLocal2, bytecode.OpSetLocalGet, bytecode.OpGetLocalIndex:
		if !local(in.A) || !local(in.B) {
			return bad("local index out of bounds")
		}
	case bytecode.OpLocalBinImm:
		if !local(in.A & (1<<24 - 1)) {
			return bad("local index out of bounds")
		}
	}
	if in.Op == bytecode.OpGetLocalIndexUpdate && (!local(in.A) || !local(in.B>>2)) {
		return bad("local index out of bounds")
	}
	if (in.Op == bytecode.OpGetUpvalue || in.Op == bytecode.OpGetUpvalueCheck) && uint64(in.A) >= uint64(len(fn.Upvalues)) {
		return bad("upvalue index out of bounds")
	}
	switch in.Op {
	case bytecode.OpNewArray:
		if in.A > MaxSlots {
			return bad("array literal budget")
		}
		return effect{need: int(in.A), delta: 1 - int(in.A)}, nil
	case bytecode.OpGetUpvalue, bytecode.OpGetUpvalueCheck, bytecode.OpGetLocalIndex, bytecode.OpGetLocalIndexUpdate:
		return effect{delta: 1}, nil
	case bytecode.OpGetIndex:
		return effect{need: 2, delta: -1}, nil
	case bytecode.OpSetIndex:
		return effect{need: 3, delta: -3}, nil
	case bytecode.OpGetLength:
		return effect{need: 1}, nil
	case bytecode.OpToPropertyKeyOfBase:
		return effect{need: 2}, nil
	case bytecode.OpInsert2:
		return effect{need: 2, delta: 1}, nil
	case bytecode.OpInsert3:
		return effect{need: 3, delta: 1}, nil
	case bytecode.OpCall, bytecode.OpCallMethod, bytecode.OpNew:
		if in.A > MaxSlots {
			return bad("argument budget")
		}
		n := int(in.A) + 1
		if in.Op == bytecode.OpCallMethod {
			n++
		}
		return effect{need: n, delta: 1 - n}, nil
	case bytecode.OpGetGlobal, bytecode.OpGetPropThis, bytecode.OpGetProp, bytecode.OpSetProp, bytecode.OpSetGlobal, bytecode.OpSetGlobalStrict:
		if uint64(in.A) >= uint64(len(fn.Names)) || in.B == 0 || in.B > fn.PropSites {
			return bad("invalid property site")
		}
		if in.Op == bytecode.OpSetProp {
			return effect{need: 2, delta: -2}, nil
		}
		if in.Op == bytecode.OpSetGlobal || in.Op == bytecode.OpSetGlobalStrict {
			return effect{need: 1, delta: -1}, nil
		}
		if in.Op == bytecode.OpGetPropThis {
			return effect{need: 1, delta: 1}, nil
		}
		if in.Op == bytecode.OpGetProp {
			return effect{need: 1}, nil
		}
		return effect{delta: 1}, nil
	case bytecode.OpPushThis:
		return effect{delta: 1}, nil
	case bytecode.OpNop, bytecode.OpEndParams, bytecode.OpClearLocal:
		return effect{}, nil
	case bytecode.OpPushConst:
		if uint64(in.A) >= uint64(len(fn.Constants)) || fn.Constants[in.A].Kind != bytecode.ConstNumber {
			return bad("non-number or invalid constant")
		}
		return effect{delta: 1}, nil
	case bytecode.OpPushInt, bytecode.OpPushUndef, bytecode.OpPushNull,
		bytecode.OpPushTrue, bytecode.OpPushFalse, bytecode.OpPushUninitialized,
		bytecode.OpGetLocal, bytecode.OpGetLocalCheck:
		return effect{delta: 1}, nil
	case bytecode.OpGetLocal2:
		return effect{delta: 2}, nil
	case bytecode.OpSetLocal, bytecode.OpInitLocal:
		return effect{need: 1, delta: -1}, nil
	case bytecode.OpSetLocalCheck:
		if !fn.Locals[in.A].Mutable {
			return bad("assignment to immutable local")
		}
		return effect{need: 1, delta: -1}, nil
	case bytecode.OpPutLocal, bytecode.OpSetLocalGet, bytecode.OpSwap:
		need := 1
		if in.Op == bytecode.OpSwap {
			need = 2
		}
		return effect{need: need}, nil
	case bytecode.OpDup:
		return effect{need: 1, delta: 1}, nil
	case bytecode.OpDup2:
		return effect{need: 2, delta: 2}, nil
	case bytecode.OpDrop:
		return effect{need: 1, delta: -1}, nil
	case bytecode.OpBitAnd, bytecode.OpBitOr, bytecode.OpBitXor, bytecode.OpShl, bytecode.OpShr, bytecode.OpUShr,
		bytecode.OpAdd, bytecode.OpSub, bytecode.OpMul, bytecode.OpDiv, bytecode.OpMod,
		bytecode.OpLt, bytecode.OpLe, bytecode.OpGt, bytecode.OpGe,
		bytecode.OpEq, bytecode.OpNe, bytecode.OpStrictEq, bytecode.OpStrictNe:
		return effect{need: 2, delta: -1}, nil
	case bytecode.OpNeg, bytecode.OpPos, bytecode.OpInc, bytecode.OpDec,
		bytecode.OpToNumber, bytecode.OpToNumeric, bytecode.OpNot, bytecode.OpBitNot:
		return effect{need: 1}, nil
	case bytecode.OpIncLocal, bytecode.OpDecLocal:
		return effect{}, nil
	case bytecode.OpUpdateLocal:
		if in.B & ^uint32(bytecode.UpdateDec|bytecode.UpdatePostfix) != 0 {
			return bad("invalid update flags")
		}
		return effect{delta: 1}, nil
	case bytecode.OpBinImm, bytecode.OpBinLocal, bytecode.OpLocalBinImm:
		raw := in.B
		if in.Op == bytecode.OpLocalBinImm {
			raw = in.A >> 24
		}
		op, ok := operator(raw)
		if raw != uint32(bytecode.OpMod) && (!ok || op > ir.Mul && (op < ir.BitAnd || op > ir.UShr)) {
			return bad("unsupported fused arithmetic operator")
		}
		if in.Op == bytecode.OpLocalBinImm {
			return effect{delta: 1}, nil
		}
		return effect{need: 1}, nil
	case bytecode.OpJump, bytecode.OpJumpIfFalse, bytecode.OpJumpIfTrue,
		bytecode.OpJumpIfFalseKeep, bytecode.OpJumpIfTrueKeep, bytecode.OpJumpIfCmpFalse:
		if uint64(in.A) >= uint64(len(fn.Code)) {
			return bad("branch target out of bounds")
		}
		e := effect{branch: true, need: 1, delta: -1}
		if in.Op == bytecode.OpJump {
			e.need, e.delta, e.terminal = 0, 0, true
		} else if in.Op == bytecode.OpJumpIfFalseKeep || in.Op == bytecode.OpJumpIfTrueKeep {
			e.keep = true
		} else if in.Op == bytecode.OpJumpIfCmpFalse {
			op, ok := operator(in.B)
			if !ok || op < ir.Lt || op > ir.Ne {
				return bad("unsupported fused comparison operator")
			}
			e.need, e.delta = 2, -2
		}
		return e, nil
	case bytecode.OpReturn:
		return effect{need: 1, delta: -1, terminal: true}, nil
	case bytecode.OpReturnUndef:
		return effect{terminal: true}, nil
	}
	return bad("unsupported opcode " + in.Op.String())
}

// strictOp reports whether a raw comparison opcode is === or !==.
func strictOp(raw uint32) bool {
	return raw == uint32(bytecode.OpStrictEq) || raw == uint32(bytecode.OpStrictNe)
}

func operator(raw uint32) (ir.Operator, bool) {
	switch raw {
	case uint32(bytecode.OpBitAnd):
		return ir.BitAnd, true
	case uint32(bytecode.OpBitOr):
		return ir.BitOr, true
	case uint32(bytecode.OpBitXor):
		return ir.BitXor, true
	case uint32(bytecode.OpShl):
		return ir.Shl, true
	case uint32(bytecode.OpShr):
		return ir.Shr, true
	case uint32(bytecode.OpUShr):
		return ir.UShr, true
	case uint32(bytecode.OpAdd):
		return ir.Add, true
	case uint32(bytecode.OpSub):
		return ir.Sub, true
	case uint32(bytecode.OpMul):
		return ir.Mul, true
	case uint32(bytecode.OpDiv):
		return ir.Div, true
	case uint32(bytecode.OpLt):
		return ir.Lt, true
	case uint32(bytecode.OpLe):
		return ir.Le, true
	case uint32(bytecode.OpGt):
		return ir.Gt, true
	case uint32(bytecode.OpGe):
		return ir.Ge, true
	case uint32(bytecode.OpEq), uint32(bytecode.OpStrictEq):
		return ir.Eq, true
	case uint32(bytecode.OpNe), uint32(bytecode.OpStrictNe):
		return ir.Ne, true
	}
	return 0, false
}

// lowerMod is a remainder's instruction, in each form the bytecode has it:
// the operator, with a local, with an immediate, and a local's with an
// immediate.
func lowerMod(in bytecode.Instr, sp int) (ir.Instruction, bool) {
	mod := uint32(bytecode.OpMod)
	top := ir.Slot(sp - 1)
	switch {
	case in.Op == bytecode.OpMod:
		return ir.Instruction{Op: ir.Binary, Operator: ir.Mod, Dest: sp - 2, Left: ir.Slot(sp - 2), Right: top}, true
	case in.Op == bytecode.OpBinLocal && in.B == mod:
		return ir.Instruction{Op: ir.Binary, Operator: ir.Mod, Dest: sp - 1, Left: top, Right: ir.Slot(int(in.A))}, true
	case in.Op == bytecode.OpBinImm && in.B == mod:
		return ir.Instruction{Op: ir.Binary, Operator: ir.Mod, Dest: sp - 1, Left: top, Right: ir.Literal(ir.Float(float64(int32(in.A))))}, true
	case in.Op == bytecode.OpLocalBinImm && in.A>>24 == mod:
		return ir.Instruction{Op: ir.Binary, Operator: ir.Mod, Dest: sp, Left: ir.Slot(int(in.A & (1<<24 - 1))), Right: ir.Literal(ir.Float(float64(int32(in.B))))}, true
	}
	return ir.Instruction{}, false
}

func lower(fn *bytecode.Function, in bytecode.Instr, sp int, this bool) ir.Instruction {
	copyTo := func(dest int, source ir.Operand) ir.Instruction {
		return ir.Instruction{Op: ir.Copy, Dest: dest, Left: source}
	}
	number := func(n int32) ir.Operand { return ir.Literal(ir.Float(float64(n))) }
	top := ir.Slot(sp - 1)
	switch in.Op {
	case bytecode.OpGetUpvalue, bytecode.OpGetUpvalueCheck:
		slot := fn.LocalCount + int(in.A)
		out := copyTo(sp, ir.Slot(slot))
		out.Check, out.CheckSlot = in.Op == bytecode.OpGetUpvalueCheck, slot
		return out
	case bytecode.OpGetIndex:
		return ir.Instruction{Op: ir.ArrayRead, Dest: sp - 2, Left: ir.Slot(sp - 2), Right: top}
	case bytecode.OpGetLocalIndex:
		return ir.Instruction{Op: ir.ArrayRead, Dest: sp, Left: ir.Slot(int(in.A)), Right: ir.Slot(int(in.B))}
	case bytecode.OpGetLocalIndexUpdate:
		return ir.Instruction{Op: ir.ArrayUpdate, Dest: sp, Extra: int(in.B >> 2), Left: ir.Slot(int(in.A)), Right: ir.Slot(int(in.B >> 2)), Postfix: in.B&bytecode.UpdatePostfix != 0, Operator: ir.Operator(in.B & bytecode.UpdateDec)}
	case bytecode.OpSetIndex:
		return ir.Instruction{Op: ir.ArrayWrite, Left: ir.Slot(sp - 3), Right: ir.Slot(sp - 2), Third: top}
	case bytecode.OpGetLength:
		return ir.Instruction{Op: ir.ArrayLength, Left: top, Dest: sp - 1}
	case bytecode.OpToPropertyKeyOfBase:
		return ir.Instruction{Op: ir.ArrayKey, Left: ir.Slot(sp - 2), Right: top}
	case bytecode.OpInsert2:
		return ir.Instruction{Op: ir.Insert2, Dest: sp - 2}
	case bytecode.OpInsert3:
		return ir.Instruction{Op: ir.Insert3, Dest: sp - 3}
	case bytecode.OpCallMethod:
		if in.A == 1 && stringKernel(fn) {
			for _, name := range fn.Names {
				if name == "charCodeAt" {
					return ir.Instruction{Op: ir.StringCode, Left: ir.Slot(sp - 2), Right: ir.Slot(sp - 3), Third: top, Dest: sp - 3}
				}
			}
		}
		return ir.Instruction{Op: ir.Host}
	case bytecode.OpGetPropThis:
		if fn.Names[in.A] == "charCodeAt" && stringKernel(fn) {
			return ir.Instruction{Op: ir.StringMethod, Left: top, Dest: sp}
		}
		return ir.Instruction{Op: ir.Host}
	case bytecode.OpNewArray, bytecode.OpCall, bytecode.OpMod, bytecode.OpNew, bytecode.OpSetGlobal, bytecode.OpSetGlobalStrict:
		return ir.Instruction{Op: ir.Host}
	case bytecode.OpGetGlobal:
		return ir.Instruction{Op: ir.BindingRead, Left: ir.Literal(ir.Value{Kind: ir.Opaque}), Dest: sp, Key: in.A}
	case bytecode.OpPushThis:
		if !this {
			return ir.Instruction{Op: ir.Host}
		}
		slot := fn.LocalCount + len(fn.Upvalues)
		return ir.Instruction{Op: ir.Copy, Left: ir.Slot(slot), Dest: sp, Check: true, CheckSlot: slot}
	case bytecode.OpGetProp:
		return ir.Instruction{Op: ir.PropertyRead, Left: top, Dest: sp - 1, Key: in.A}
	case bytecode.OpSetProp:
		return ir.Instruction{Op: ir.PropertyWrite, Left: ir.Slot(sp - 2), Right: top, Key: in.A}
	case bytecode.OpPushConst:
		return copyTo(sp, ir.Literal(ir.Float(fn.Constants[in.A].Num)))
	case bytecode.OpPushInt:
		return copyTo(sp, number(int32(in.A)))
	case bytecode.OpPushUndef, bytecode.OpPushNull, bytecode.OpPushUninitialized, bytecode.OpPushTrue, bytecode.OpPushFalse:
		v := ir.Value{Kind: ir.Undefined}
		switch in.Op {
		case bytecode.OpPushNull:
			v.Kind = ir.Null
		case bytecode.OpPushUninitialized:
			v.Kind = ir.Uninitialized
		case bytecode.OpPushTrue:
			v = ir.Bool(true)
		case bytecode.OpPushFalse:
			v = ir.Bool(false)
		}
		return copyTo(sp, ir.Literal(v))
	case bytecode.OpGetLocal, bytecode.OpGetLocalCheck:
		out := copyTo(sp, ir.Slot(int(in.A)))
		out.Check, out.CheckSlot = in.Op == bytecode.OpGetLocalCheck, int(in.A)
		return out
	case bytecode.OpGetLocal2:
		return ir.Instruction{Op: ir.CopyPair, Dest: sp, Extra: sp + 1, Left: ir.Slot(int(in.A)), Right: ir.Slot(int(in.B))}
	case bytecode.OpSetLocal, bytecode.OpInitLocal, bytecode.OpPutLocal, bytecode.OpSetLocalCheck:
		out := copyTo(int(in.A), top)
		out.Check, out.CheckSlot = in.Op == bytecode.OpSetLocalCheck, int(in.A)
		return out
	case bytecode.OpSetLocalGet:
		return ir.Instruction{Op: ir.StoreLoad, Dest: int(in.A), Extra: sp - 1, Left: top, Right: ir.Slot(int(in.B))}
	case bytecode.OpClearLocal:
		return copyTo(int(in.A), ir.Literal(ir.Value{Kind: ir.Undefined}))
	case bytecode.OpDup:
		return copyTo(sp, top)
	case bytecode.OpDup2:
		return ir.Instruction{Op: ir.CopyPair, Dest: sp, Extra: sp + 1, Left: ir.Slot(sp - 2), Right: top}
	case bytecode.OpSwap:
		return ir.Instruction{Op: ir.Swap, Dest: sp - 2, Extra: sp - 1}
	case bytecode.OpBitAnd, bytecode.OpBitOr, bytecode.OpBitXor, bytecode.OpShl, bytecode.OpShr, bytecode.OpUShr,
		bytecode.OpAdd, bytecode.OpSub, bytecode.OpMul, bytecode.OpDiv,
		bytecode.OpLt, bytecode.OpLe, bytecode.OpGt, bytecode.OpGe,
		bytecode.OpEq, bytecode.OpNe, bytecode.OpStrictEq, bytecode.OpStrictNe:
		op, _ := operator(uint32(in.Op))
		return ir.Instruction{Op: ir.Binary, Operator: op, Dest: sp - 2, Left: ir.Slot(sp - 2), Right: top, Strict: strictOp(uint32(in.Op))}
	case bytecode.OpBinLocal:
		if in.B == uint32(bytecode.OpMod) {
			return ir.Instruction{Op: ir.Host}
		}
		op, _ := operator(in.B)
		return ir.Instruction{Op: ir.Binary, Operator: op, Dest: sp - 1, Left: top, Right: ir.Slot(int(in.A)), Strict: strictOp(in.B)}
	case bytecode.OpBinImm:
		if in.B == uint32(bytecode.OpMod) {
			return ir.Instruction{Op: ir.Host}
		}
		if in.B == uint32(bytecode.OpBitOr) && in.A == 0 {
			return ir.Instruction{Op: ir.Unary, Operator: ir.Int32, Left: top, Dest: sp - 1}
		}
		op, _ := operator(in.B)
		return ir.Instruction{Op: ir.Binary, Operator: op, Dest: sp - 1, Left: top, Right: number(int32(in.A)), Strict: strictOp(in.B)}
	case bytecode.OpLocalBinImm:
		if in.A>>24 == uint32(bytecode.OpMod) {
			return ir.Instruction{Op: ir.Host}
		}
		if in.A>>24 == uint32(bytecode.OpBitOr) && in.B == 0 {
			return ir.Instruction{Op: ir.Unary, Operator: ir.Int32, Left: ir.Slot(int(in.A & (1<<24 - 1))), Dest: sp}
		}
		op, _ := operator(in.A >> 24)
		return ir.Instruction{Op: ir.Binary, Operator: op, Dest: sp, Left: ir.Slot(int(in.A & (1<<24 - 1))), Right: number(int32(in.B)), Strict: strictOp(in.A >> 24)}
	case bytecode.OpNeg, bytecode.OpPos, bytecode.OpToNumber, bytecode.OpToNumeric, bytecode.OpNot, bytecode.OpBitNot:
		op := ir.Pos
		if in.Op == bytecode.OpNeg {
			op = ir.Neg
		} else if in.Op == bytecode.OpBitNot {
			op = ir.BitNot
		} else if in.Op == bytecode.OpNot {
			op = ir.Not
		}
		return ir.Instruction{Op: ir.Unary, Operator: op, Dest: sp - 1, Left: top}
	case bytecode.OpInc, bytecode.OpDec, bytecode.OpIncLocal, bytecode.OpDecLocal, bytecode.OpUpdateLocal:
		dest, extra := sp-1, -1
		if in.Op == bytecode.OpIncLocal || in.Op == bytecode.OpDecLocal || in.Op == bytecode.OpUpdateLocal {
			dest = int(in.A)
		}
		if in.Op == bytecode.OpUpdateLocal {
			extra = sp
		}
		op := ir.Add
		if in.Op == bytecode.OpDec || in.Op == bytecode.OpDecLocal || in.Op == bytecode.OpUpdateLocal && in.B&bytecode.UpdateDec != 0 {
			op = ir.Sub
		}
		return ir.Instruction{Op: ir.Update, Operator: op, Dest: dest, Extra: extra, Left: ir.Slot(dest), Postfix: in.Op == bytecode.OpUpdateLocal && in.B&bytecode.UpdatePostfix != 0}
	case bytecode.OpJump:
		return ir.Instruction{Op: ir.Jump, Target: int(in.A)}
	case bytecode.OpJumpIfTrue, bytecode.OpJumpIfFalse, bytecode.OpJumpIfTrueKeep, bytecode.OpJumpIfFalseKeep:
		return ir.Instruction{Op: ir.Branch, Operator: ir.Truth, Target: int(in.A), Left: top, When: in.Op == bytecode.OpJumpIfTrue || in.Op == bytecode.OpJumpIfTrueKeep}
	case bytecode.OpJumpIfCmpFalse:
		op, _ := operator(in.B)
		return ir.Instruction{Op: ir.Branch, Operator: op, Target: int(in.A), Left: ir.Slot(sp - 2), Right: top, Strict: strictOp(in.B)}
	case bytecode.OpReturn:
		return ir.Instruction{Op: ir.Return, Left: top}
	case bytecode.OpReturnUndef:
		return ir.Instruction{Op: ir.Return, Left: ir.Literal(ir.Value{Kind: ir.Undefined})}
	case bytecode.OpNop, bytecode.OpEndParams, bytecode.OpDrop:
		// A pop writes no slot: the next map's depth is the whole effect.
		return ir.Instruction{Op: ir.Nop}
	}
	// describe accepted an opcode this switch does not translate. Becoming a
	// Nop with describe's stack effect would be a silent miscompile; the panic
	// is a refusal (lowerRecovered).
	panic(fmt.Sprintf("jit: no lowering for %v", in.Op))
}

// An intrinsic snapshot and an extra root must amortize their entry cost.
// One-off character helpers retain the existing call path.
func stringKernel(fn *bytecode.Function) bool {
	if len(fn.Code) >= 64 {
		return true
	}
	for pc, in := range fn.Code {
		switch in.Op {
		case bytecode.OpJump, bytecode.OpJumpIfFalse, bytecode.OpJumpIfTrue,
			bytecode.OpJumpIfFalseKeep, bytecode.OpJumpIfTrueKeep, bytecode.OpJumpIfCmpFalse:
			if in.A <= uint32(pc) {
				return true
			}
		}
	}
	return false
}
