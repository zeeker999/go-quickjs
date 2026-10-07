//go:build quickjs_jit && !android && !ios && darwin

package jit

import (
	"encoding/binary"
	"fmt"

	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

// R0 owns programState and R2 scalar scratch. R8 holds the remaining budget,
// R9 the pre-instruction PC. R1, R3-R7 and F0-F1 are scratch.
// SP, FP, LR, R18 and Go's R28 remain untouched; no native calls occur.
type arm64Program struct {
	programAssembler
	guard, budget, returned int
}

func programInstructions(p *ir.Program) ([]byte, []int, error) {
	a := &arm64Program{}
	for range p.Code {
		a.label()
	}
	a.guard, a.budget, a.returned = a.label(), a.label(), a.label()
	for pc, in := range p.Code {
		if p.Maps[pc].Depth < 0 {
			continue
		}
		a.mark(pc)
		a.immediate(9, uint64(pc))
		a.compareImmediate(8, 0)
		a.conditional(0, a.budget)
		if in.Check {
			a.memory(true, false, 4, 2, in.CheckSlot*16+8)
			a.compareImmediate(4, uint32(ir.Uninitialized))
			a.conditional(0, a.guard)
		}
		switch in.Op {
		case ir.Nop:
		case ir.Copy:
			a.load(in.Left, 3, 4)
			a.store(in.Dest, 3, 4)
		case ir.CopyPair:
			a.load(in.Left, 3, 4)
			a.load(in.Right, 5, 6)
			a.store(in.Dest, 3, 4)
			a.store(in.Extra, 5, 6)
		case ir.StoreLoad:
			a.load(in.Left, 3, 4)
			a.store(in.Dest, 3, 4)
			a.load(in.Right, 3, 4)
			a.store(in.Extra, 3, 4)
		case ir.Swap:
			a.load(ir.Slot(in.Dest), 3, 4)
			a.load(ir.Slot(in.Extra), 5, 6)
			a.store(in.Dest, 5, 6)
			a.store(in.Extra, 3, 4)
		case ir.Binary:
			a.binary(in.Operator, in.Left, in.Right)
			if in.Operator <= ir.Div {
				a.storeNumber(in.Dest)
			} else {
				a.storeBool(in.Dest)
			}
		case ir.Unary:
			if in.Operator == ir.Not {
				a.truth(in.Left)
				a.immediate(5, 1)
				a.word(0xca050063) // eor x3, x3, x5
				a.storeBool(in.Dest)
			} else {
				a.load(in.Left, 3, 4)
				a.compareImmediate(4, uint32(ir.Number))
				a.conditional(1, a.guard)
				if in.Operator == ir.Neg {
					a.immediate(5, 1<<63)
					a.word(0xca050063)
				}
				a.store(in.Dest, 3, 4)
			}
		case ir.Update:
			if in.Postfix && in.Extra >= 0 {
				a.load(in.Left, 5, 6)
			}
			a.binary(in.Operator, in.Left, ir.Literal(ir.Float(1)))
			a.storeNumber(in.Dest)
			if in.Extra >= 0 {
				if in.Postfix {
					a.store(in.Extra, 5, 6)
				} else {
					a.storeNumber(in.Extra)
				}
			}
		case ir.Jump:
			a.commit()
			a.jump(in.Target)
			continue
		case ir.Branch:
			if in.Operator == ir.Truth {
				a.truth(in.Left)
			} else {
				a.binary(in.Operator, in.Left, in.Right)
			}
			a.commit()
			a.compareImmediate(3, 0)
			condition := uint32(0)
			if in.When {
				condition = 1
			}
			a.conditional(condition, in.Target)
			continue
		case ir.Return:
			a.load(in.Left, 3, 4)
			a.compareImmediate(4, uint32(ir.Null))
			a.conditional(8, a.guard)
			a.memory(false, false, 3, 0, 24)
			a.memory(false, false, 4, 0, 32)
			a.commit()
			a.jump(a.returned)
			continue
		}
		a.commit()
	}
	for _, exit := range []struct {
		label int
		kind  ir.ExitKind
	}{{a.guard, ir.GuardExit}, {a.budget, ir.BudgetExit}, {a.returned, ir.Returned}} {
		a.mark(exit.label)
		a.memory(false, false, 8, 0, 0)
		a.memory(false, false, 9, 0, 16)
		a.immediate(3, uint64(exit.kind))
		a.memory(false, false, 3, 0, 8)
		a.word(0xd65f03c0)
	}
	// External entries initialize the budget register; internal branches go
	// straight to instruction bodies and preserve its current value.
	entries := make([]int, len(p.Code))
	for pc := range p.Code {
		entries[pc] = -1
		if p.Maps[pc].Depth < 0 {
			continue
		}
		entries[pc] = len(a.code)
		a.memory(true, false, 8, 0, 0)
		a.jump(pc)
	}
	if err := a.valid(); err != nil {
		return nil, nil, err
	}
	for _, fixup := range a.fixups {
		delta := a.labels[fixup.label] - fixup.offset
		if fixup.conditional {
			if delta < -1<<20 || delta >= 1<<20 {
				return nil, nil, fmt.Errorf("native conditional branch out of range")
			}
			word := binary.LittleEndian.Uint32(a.code[fixup.offset:])
			word |= (uint32(int32(delta/4)) & 0x7ffff) << 5
			binary.LittleEndian.PutUint32(a.code[fixup.offset:], word)
		} else {
			binary.LittleEndian.PutUint32(a.code[fixup.offset:], 0x14000000|uint32(int32(delta/4))&0x03ffffff)
		}
	}
	return a.code, entries, nil
}

func (a *arm64Program) conditional(condition uint32, label int) {
	a.fixups = append(a.fixups, relocation{offset: len(a.code), label: label, conditional: true})
	a.word(0x54000000 | condition)
}

func (a *arm64Program) jump(label int) {
	a.fixups = append(a.fixups, relocation{offset: len(a.code), label: label})
	a.word(0x14000000)
}

func (a *arm64Program) immediate(reg uint32, bits uint64) {
	a.word(0xd2800000 | uint32(bits&0xffff)<<5 | reg) // movz
	for shift := uint32(1); shift < 4; shift++ {
		part := uint32(bits >> (shift * 16) & 0xffff)
		if part != 0 {
			a.word(0xf2800000 | shift<<21 | part<<5 | reg) // movk
		}
	}
}

func (a *arm64Program) memory(load, fp bool, reg, base uint32, offset int) {
	op := uint32(0xf9000000)
	if fp {
		op = 0xfd000000
	}
	if load {
		op |= 1 << 22
	}
	a.word(op | uint32(offset/8)<<10 | base<<5 | reg)
}

func (a *arm64Program) load(o ir.Operand, bits, kind uint32) {
	if o.Slot < 0 {
		a.immediate(bits, o.Literal.Bits)
		a.immediate(kind, uint64(o.Literal.Kind))
		return
	}
	a.memory(true, false, bits, 2, o.Slot*16)
	a.memory(true, false, kind, 2, o.Slot*16+8)
}

func (a *arm64Program) store(slot int, bits, kind uint32) {
	a.memory(false, false, bits, 2, slot*16)
	a.memory(false, false, kind, 2, slot*16+8)
}

func (a *arm64Program) compareImmediate(reg, value uint32) {
	a.word(0xf100001f | value<<10 | reg<<5)
}

func (a *arm64Program) commit() {
	a.word(0xd1000508) // sub x8, x8, #1
}

func (a *arm64Program) number(o ir.Operand, fp uint32) {
	if o.Slot < 0 {
		if o.Literal.Kind != ir.Number {
			a.jump(a.guard)
		}
		a.immediate(7, o.Literal.Bits)
		a.word(0x9e670000 | 7<<5 | fp) // fmov dN, x7
		return
	}
	a.memory(true, false, 4, 2, o.Slot*16+8)
	a.compareImmediate(4, uint32(ir.Number))
	a.conditional(1, a.guard)
	a.memory(true, true, fp, 2, o.Slot*16)
}

func (a *arm64Program) binary(op ir.Operator, left, right ir.Operand) {
	a.number(left, 0)
	a.number(right, 1)
	if op <= ir.Div {
		word := [...]uint32{0x1e612800, 0x1e613800, 0x1e610800, 0x1e611800}[op]
		a.word(word)
		return
	}
	yes, nan, done := a.label(), a.label(), a.label()
	a.immediate(3, 0)
	a.word(0x1e612000)    // fcmp d0, d1
	a.conditional(6, nan) // b.vs: unordered
	condition := [...]uint32{3, 9, 8, 2, 0, 1}[op-ir.Lt]
	a.conditional(condition, yes)
	a.jump(done)
	a.mark(yes)
	a.immediate(3, 1)
	a.jump(done)
	a.mark(nan)
	nanResult := uint64(0)
	if op == ir.Ne {
		nanResult = 1
	}
	a.immediate(3, nanResult)
	a.mark(done)
}

func (a *arm64Program) storeNumber(slot int) {
	a.memory(false, true, 0, 2, slot*16)
	a.immediate(4, uint64(ir.Number))
	a.memory(false, false, 4, 2, slot*16+8)
}

func (a *arm64Program) storeBool(slot int) {
	a.immediate(4, uint64(ir.Boolean))
	a.store(slot, 3, 4)
}

func (a *arm64Program) truth(o ir.Operand) {
	number, zero, done := a.label(), a.label(), a.label()
	a.load(o, 3, 4)
	a.compareImmediate(4, uint32(ir.Number))
	a.conditional(0, number)
	a.compareImmediate(4, uint32(ir.Boolean))
	a.conditional(0, done)
	a.compareImmediate(4, uint32(ir.Undefined))
	a.conditional(0, zero)
	a.compareImmediate(4, uint32(ir.Null))
	a.conditional(0, zero)
	a.jump(a.guard)
	a.mark(number)
	a.immediate(5, 1<<63-1)
	a.word(0x8a050063) // and x3, x3, x5: absolute IEEE bits
	a.compareImmediate(3, 0)
	a.conditional(0, zero)
	a.immediate(5, 0x7ff0000000000000)
	a.word(0xeb05007f) // cmp x3, x5
	a.conditional(8, zero)
	a.immediate(3, 1)
	a.jump(done)
	a.mark(zero)
	a.immediate(3, 0)
	a.mark(done)
}
