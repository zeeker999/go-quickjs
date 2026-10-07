//go:build quickjs_jit && !android && !ios && darwin

package jit

import (
	"encoding/binary"
	"fmt"

	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

// R0 owns programState and R2 scalar scratch. R8 holds the remaining budget,
// R9 the pre-instruction PC. R1, R3-R7 and F0-F1 are scratch; F2-F7 and
// F16-F23 cache scalar bits, R10-R15 cache the first six kinds. All are
// spilled on every exit.
// SP, FP, LR, R18 and Go's R28 remain untouched; no native calls occur.
type arm64Program struct {
	programAssembler
	guard, budget, returned int
}

func programInstructions(p *ir.Program) ([]byte, []int, error) {
	a := &arm64Program{}
	a.allocateRegisters(p, []int{2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23})
	for range p.Code {
		a.label()
	}
	a.guard, a.budget, a.returned = a.label(), a.label(), a.label()
	initialize := a.label()
	for pc, in := range p.Code {
		if p.Maps[pc].Depth < 0 {
			continue
		}
		a.mark(pc)
		a.immediate(9, uint64(pc))
		a.compareImmediate(8, 0)
		a.conditional(0, a.budget)
		if in.Check {
			a.loadKind(in.CheckSlot, 4)
			a.compareImmediate(4, uint32(ir.Uninitialized))
			a.conditional(0, a.guard)
		}
		switch in.Op {
		case ir.Nop:
		case ir.Copy:
			a.loadScalar(in.Left, 0, 4)
			a.storeScalar(in.Dest, 0, 4)
		case ir.CopyPair:
			a.loadScalar(in.Left, 0, 4)
			a.loadScalar(in.Right, 1, 6)
			a.storeScalar(in.Dest, 0, 4)
			a.storeScalar(in.Extra, 1, 6)
		case ir.StoreLoad:
			a.loadScalar(in.Left, 0, 4)
			a.storeScalar(in.Dest, 0, 4)
			a.loadScalar(in.Right, 0, 4)
			a.storeScalar(in.Extra, 0, 4)
		case ir.Swap:
			a.loadScalar(ir.Slot(in.Dest), 0, 4)
			a.loadScalar(ir.Slot(in.Extra), 1, 6)
			a.storeScalar(in.Dest, 1, 6)
			a.storeScalar(in.Extra, 0, 4)
		case ir.Binary:
			fp := a.binary(in.Operator, in.Left, in.Right, in.Dest)
			if in.Operator <= ir.Div {
				a.storeNumber(in.Dest, fp)
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
			fp := a.binary(in.Operator, in.Left, ir.Literal(ir.Float(1)), in.Dest)
			a.storeNumber(in.Dest, fp)
			if in.Extra >= 0 {
				if in.Postfix {
					a.store(in.Extra, 5, 6)
				} else {
					a.storeNumber(in.Extra, fp)
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
				left := a.number(in.Left, 0)
				right := a.number(in.Right, 1)
				a.word(0x1e602000 | right<<16 | left<<5) // fcmp dN, dM
				a.commit()                               // sub does not change condition flags
				condition := arm64Comparison(in.Operator)
				if !in.When {
					condition ^= 1
				}
				a.conditional(condition, in.Target)
				continue
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
		for slot, reg := range a.registers {
			if reg >= 0 {
				a.memory(false, true, uint32(reg), 2, slot*16)
				if reg < 8 {
					a.memory(false, false, uint32(reg+8), 2, slot*16+8)
				}
			}
		}
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
		a.fixups = append(a.fixups, relocation{offset: len(a.code), label: pc, address: true})
		a.word(0x10000010) // adr x16, body
		a.jump(initialize)
	}
	a.mark(initialize)
	a.memory(true, false, 8, 0, 0)
	for slot, reg := range a.registers {
		if reg >= 0 {
			a.memory(true, true, uint32(reg), 2, slot*16)
			if reg < 8 {
				a.memory(true, false, uint32(reg+8), 2, slot*16+8)
			}
		}
	}
	a.word(0xd61f0200) // br x16
	if err := a.valid(); err != nil {
		return nil, nil, err
	}
	for _, fixup := range a.fixups {
		delta := a.labels[fixup.label] - fixup.offset
		if fixup.address {
			if delta < -1<<20 || delta >= 1<<20 {
				return nil, nil, fmt.Errorf("native entry address out of range")
			}
			word := uint32(0x10000010) | (uint32(delta)&3)<<29 | ((uint32(delta)>>2)&0x7ffff)<<5
			binary.LittleEndian.PutUint32(a.code[fixup.offset:], word)
		} else if fixup.conditional {
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
	if reg := a.registers[o.Slot]; reg >= 0 {
		a.word(0x9e660000 | uint32(reg)<<5 | bits) // fmov xN, dM
	} else {
		a.memory(true, false, bits, 2, o.Slot*16)
	}
	a.loadKind(o.Slot, kind)
}

func (a *arm64Program) store(slot int, bits, kind uint32) {
	if reg := a.registers[slot]; reg >= 0 {
		a.word(0x9e670000 | bits<<5 | uint32(reg)) // fmov dN, xM
	} else {
		a.memory(false, false, bits, 2, slot*16)
	}
	a.storeKind(slot, kind)
}

func (a *arm64Program) loadKind(slot int, reg uint32) {
	if cache := a.registers[slot]; cache >= 0 && cache < 8 {
		a.word(0xaa0003e0 | uint32(cache+8)<<16 | reg) // mov xN, xM
	} else {
		a.memory(true, false, reg, 2, slot*16+8)
	}
}

func (a *arm64Program) storeKind(slot int, reg uint32) {
	if cache := a.registers[slot]; cache >= 0 && cache < 8 {
		a.word(0xaa0003e0 | reg<<16 | uint32(cache+8))
	} else {
		a.memory(false, false, reg, 2, slot*16+8)
	}
}

func (a *arm64Program) loadScalar(o ir.Operand, fp, kind uint32) {
	if o.Slot < 0 {
		a.immediate(7, o.Literal.Bits)
		a.word(0x9e670000 | 7<<5 | fp)
		a.immediate(kind, uint64(o.Literal.Kind))
	} else {
		if reg := a.registers[o.Slot]; reg >= 0 {
			a.word(0x1e604000 | uint32(reg)<<5 | fp)
		} else {
			a.memory(true, true, fp, 2, o.Slot*16)
		}
		a.loadKind(o.Slot, kind)
	}
}

func (a *arm64Program) storeScalar(slot int, fp, kind uint32) {
	if reg := a.registers[slot]; reg >= 0 {
		a.word(0x1e604000 | fp<<5 | uint32(reg))
	} else {
		a.memory(false, true, fp, 2, slot*16)
	}
	a.storeKind(slot, kind)
}

func (a *arm64Program) compareImmediate(reg, value uint32) {
	a.word(0xf100001f | value<<10 | reg<<5)
}

func (a *arm64Program) commit() {
	a.word(0xd1000508) // sub x8, x8, #1
}

func (a *arm64Program) number(o ir.Operand, fp uint32) uint32 {
	if o.Slot < 0 {
		if o.Literal.Kind != ir.Number {
			a.jump(a.guard)
		}
		a.immediate(7, o.Literal.Bits)
		a.word(0x9e670000 | 7<<5 | fp) // fmov dN, x7
		return fp
	}
	if reg := a.registers[o.Slot]; reg >= 0 && reg < 8 {
		a.compareImmediate(uint32(reg+8), uint32(ir.Number))
	} else {
		a.loadKind(o.Slot, 4)
		a.compareImmediate(4, uint32(ir.Number))
	}
	a.conditional(1, a.guard)
	if reg := a.registers[o.Slot]; reg >= 0 {
		return uint32(reg)
	}
	a.memory(true, true, fp, 2, o.Slot*16)
	return fp
}

func (a *arm64Program) binary(op ir.Operator, left, right ir.Operand, dest int) uint32 {
	l := a.number(left, 0)
	r := a.number(right, 1)
	if op <= ir.Div {
		fp := uint32(0)
		if reg := a.registers[dest]; reg >= 0 {
			fp = uint32(reg)
		}
		word := [...]uint32{0x1e602800, 0x1e603800, 0x1e600800, 0x1e601800}[op]
		a.word(word | r<<16 | l<<5 | fp)
		return fp
	}
	a.word(0x1e602000 | r<<16 | l<<5)                // fcmp dN, dM
	a.word(0x9a9f07e3 | (arm64Comparison(op)^1)<<12) // cset x3, condition
	return 0
}

func arm64Comparison(op ir.Operator) uint32 {
	// FCMP's unordered flags (N=0,Z=0,C=1,V=1) make these conditions
	// false for every comparison except !=, without a separate NaN branch.
	return [...]uint32{4, 9, 12, 10, 0, 1}[op-ir.Lt] // MI, LS, GT, GE, EQ, NE
}

func (a *arm64Program) storeNumber(slot int, fp uint32) {
	if reg := a.registers[slot]; reg >= 0 {
		if uint32(reg) != fp {
			a.word(0x1e604000 | fp<<5 | uint32(reg))
		}
	} else {
		a.memory(false, true, fp, 2, slot*16)
	}
	a.immediate(4, uint64(ir.Number))
	a.storeKind(slot, 4)
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
