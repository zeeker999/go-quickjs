// Package compile lowers VM bytecode into the JIT's slot IR. It is separate
// from executable-memory support and does not modify shared function templates.
package compile

import (
	"fmt"

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
func Lower(fn *bytecode.Function) (*ir.Program, error) {
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
	host, loop, indexed, property, bitwise := false, false, false, false, false
	for pc, in := range fn.Code {
		e, err := describe(fn, pc, in)
		if err != nil {
			return nil, err
		}
		effects[pc] = e
		host = host || in.Op == bytecode.OpCall || in.Op == bytecode.OpCallMethod || in.Op == bytecode.OpGetGlobal || in.Op == bytecode.OpGetPropThis || in.Op == bytecode.OpPushThis || in.Op == bytecode.OpGetProp || in.Op == bytecode.OpSetProp
		property = property || in.Op == bytecode.OpPushThis || in.Op == bytecode.OpGetProp || in.Op == bytecode.OpSetProp
		raw := uint32(in.Op)
		switch in.Op {
		case bytecode.OpBinLocal, bytecode.OpBinImm:
			raw = in.B
		case bytecode.OpLocalBinImm:
			raw = in.A >> 24
		}
		op, _ := operator(raw)
		bitwise = bitwise || op >= ir.BitAnd && op <= ir.UShr || in.Op == bytecode.OpBitNot
		loop = loop || e.branch && int(in.A) <= pc
		indexed = indexed || in.Op == bytecode.OpGetIndex || in.Op == bytecode.OpSetIndex || in.Op == bytecode.OpGetLocalIndex || in.Op == bytecode.OpGetLocalIndexUpdate
	}
	// Tiny host-only wrappers pay the bridge overhead without enough native work.
	if host && !loop && !indexed {
		return nil, refuse(-1, "host operations without native loop or array work")
	}
	// Property-heavy object loops run faster in the tree tier. Require work
	// that benefits from the new coverage before paying property boundaries.
	if property && !indexed && !bitwise {
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
	p := &ir.Program{Locals: fn.LocalCount + len(fn.Upvalues), StackSize: fn.MaxStack, Maps: maps, Code: make([]ir.Instruction, len(fn.Code))}
	for pc, in := range fn.Code {
		if maps[pc].Depth >= 0 {
			p.Code[pc] = lower(fn, in, p.Locals+maps[pc].Depth)
		}
	}
	return p, nil
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
	case bytecode.OpCall, bytecode.OpCallMethod:
		if in.A > MaxSlots {
			return bad("argument budget")
		}
		n := int(in.A) + 1
		if in.Op == bytecode.OpCallMethod {
			n++
		}
		return effect{need: n, delta: 1 - n}, nil
	case bytecode.OpGetGlobal, bytecode.OpGetPropThis, bytecode.OpGetProp, bytecode.OpSetProp:
		if uint64(in.A) >= uint64(len(fn.Names)) || in.B == 0 || in.B > fn.PropSites {
			return bad("invalid property site")
		}
		if in.Op == bytecode.OpSetProp {
			return effect{need: 2, delta: -2}, nil
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
		bytecode.OpAdd, bytecode.OpSub, bytecode.OpMul, bytecode.OpDiv,
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
		if !ok || op > ir.Mul && (op < ir.BitAnd || op > ir.UShr) {
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

func lower(fn *bytecode.Function, in bytecode.Instr, sp int) ir.Instruction {
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
	case bytecode.OpCall, bytecode.OpCallMethod, bytecode.OpGetGlobal, bytecode.OpGetPropThis, bytecode.OpPushThis, bytecode.OpGetProp, bytecode.OpSetProp:
		return ir.Instruction{Op: ir.Host}
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
		return ir.Instruction{Op: ir.Binary, Operator: op, Dest: sp - 2, Left: ir.Slot(sp - 2), Right: top}
	case bytecode.OpBinLocal:
		op, _ := operator(in.B)
		return ir.Instruction{Op: ir.Binary, Operator: op, Dest: sp - 1, Left: top, Right: ir.Slot(int(in.A))}
	case bytecode.OpBinImm:
		if in.B == uint32(bytecode.OpBitOr) && in.A == 0 {
			return ir.Instruction{Op: ir.Unary, Operator: ir.Int32, Left: top, Dest: sp - 1}
		}
		op, _ := operator(in.B)
		return ir.Instruction{Op: ir.Binary, Operator: op, Dest: sp - 1, Left: top, Right: number(int32(in.A))}
	case bytecode.OpLocalBinImm:
		if in.A>>24 == uint32(bytecode.OpBitOr) && in.B == 0 {
			return ir.Instruction{Op: ir.Unary, Operator: ir.Int32, Left: ir.Slot(int(in.A & (1<<24 - 1))), Dest: sp}
		}
		op, _ := operator(in.A >> 24)
		return ir.Instruction{Op: ir.Binary, Operator: op, Dest: sp, Left: ir.Slot(int(in.A & (1<<24 - 1))), Right: number(int32(in.B))}
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
		return ir.Instruction{Op: ir.Branch, Operator: op, Target: int(in.A), Left: ir.Slot(sp - 2), Right: top}
	case bytecode.OpReturn:
		return ir.Instruction{Op: ir.Return, Left: top}
	case bytecode.OpReturnUndef:
		return ir.Instruction{Op: ir.Return, Left: ir.Literal(ir.Value{Kind: ir.Undefined})}
	}
	return ir.Instruction{Op: ir.Nop}
}
