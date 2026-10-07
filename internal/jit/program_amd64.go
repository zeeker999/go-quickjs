//go:build quickjs_jit && !android && !ios && (linux || windows)

package jit

import (
	"encoding/binary"

	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

// DI owns programState and SI scalar scratch. R10 holds the remaining budget,
// R11 the pre-instruction PC. AX, CX, DX, R8, R9 and X0-X1 are scratch.
// SP, BP and Go's R14 remain untouched; no native calls occur.
type amd64Program struct {
	programAssembler
	guard, budget, returned int
}

func programInstructions(p *ir.Program) ([]byte, []int, error) {
	a := &amd64Program{}
	for range p.Code {
		a.label()
	}
	a.guard, a.budget, a.returned = a.label(), a.label(), a.label()
	for pc, in := range p.Code {
		if p.Maps[pc].Depth < 0 {
			continue
		}
		a.mark(pc)
		a.bytes(0x41, 0xbb) // movl $pc, r11d
		a.word(uint32(pc))
		a.bytes(0x4d, 0x85, 0xd2) // testq r10, r10
		a.conditional(4, a.budget)
		if in.Check {
			a.kindCompare(in.CheckSlot, ir.Uninitialized)
			a.conditional(4, a.guard)
		}
		switch in.Op {
		case ir.Nop:
		case ir.Copy:
			a.load(in.Left, 0, 2)
			a.store(in.Dest, 0, 2)
		case ir.CopyPair:
			a.load(in.Left, 0, 2)
			a.load(in.Right, 8, 9)
			a.store(in.Dest, 0, 2)
			a.store(in.Extra, 8, 9)
		case ir.StoreLoad:
			a.load(in.Left, 0, 2)
			a.store(in.Dest, 0, 2)
			a.load(in.Right, 0, 2)
			a.store(in.Extra, 0, 2)
		case ir.Swap:
			a.load(ir.Slot(in.Dest), 0, 2)
			a.load(ir.Slot(in.Extra), 8, 9)
			a.store(in.Dest, 8, 9)
			a.store(in.Extra, 0, 2)
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
				a.bytes(0x48, 0x83, 0xf0, 1) // xorq $1, ax
				a.storeBool(in.Dest)
			} else {
				a.load(in.Left, 0, 2)
				a.bytes(0x48, 0x83, 0xfa, byte(ir.Number))
				a.conditional(5, a.guard)
				if in.Operator == ir.Neg {
					a.immediate(8, 1<<63)
					a.bytes(0x4c, 0x31, 0xc0) // xorq r8, ax
				}
				a.store(in.Dest, 0, 2)
			}
		case ir.Update:
			if in.Postfix && in.Extra >= 0 {
				a.load(in.Left, 8, 9)
			}
			a.binary(in.Operator, in.Left, ir.Literal(ir.Float(1)))
			a.storeNumber(in.Dest)
			if in.Extra >= 0 {
				if in.Postfix {
					a.store(in.Extra, 8, 9)
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
			a.bytes(0x48, 0x85, 0xc0) // testq ax, ax
			condition := byte(4)
			if in.When {
				condition = 5
			}
			a.conditional(condition, in.Target)
			continue
		case ir.Return:
			a.load(in.Left, 0, 2)
			a.bytes(0x48, 0x83, 0xfa, byte(ir.Null))
			a.conditional(7, a.guard) // opaque/uninitialized are not primitive returns
			a.memory(0x89, 0, 7, 24)
			a.memory(0x89, 2, 7, 32)
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
		a.memory(0x89, 10, 7, 0)
		a.memory(0x89, 11, 7, 16)
		a.stateImmediate(8, uint32(exit.kind))
		a.bytes(0xc3)
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
		a.memory(0x8b, 10, 7, 0)
		a.jump(pc)
	}
	if err := a.valid(); err != nil {
		return nil, nil, err
	}
	for _, fixup := range a.fixups {
		delta := a.labels[fixup.label] - (fixup.offset + 4)
		binary.LittleEndian.PutUint32(a.code[fixup.offset:], uint32(int32(delta)))
	}
	return a.code, entries, nil
}

func (a *amd64Program) conditional(condition byte, label int) {
	a.bytes(0x0f, 0x80+condition)
	a.fixups = append(a.fixups, relocation{offset: len(a.code), label: label})
	a.word(0)
}

func (a *amd64Program) jump(label int) {
	a.bytes(0xe9)
	a.fixups = append(a.fixups, relocation{offset: len(a.code), label: label})
	a.word(0)
}

func (a *amd64Program) immediate(reg byte, bits uint64) {
	a.bytes(0x48|reg>>3, 0xb8|reg&7)
	a.quad(bits)
}

func (a *amd64Program) memory(op, reg, base byte, offset uint32) {
	a.bytes(0x48|(reg>>3)<<2, op, 0x80|(reg&7)<<3|base)
	a.word(offset)
}

func (a *amd64Program) load(o ir.Operand, bits, kind byte) {
	if o.Slot < 0 {
		a.immediate(bits, o.Literal.Bits)
		a.immediate(kind, uint64(o.Literal.Kind))
		return
	}
	a.memory(0x8b, bits, 6, uint32(o.Slot*16))
	a.memory(0x8b, kind, 6, uint32(o.Slot*16+8))
}

func (a *amd64Program) store(slot int, bits, kind byte) {
	a.memory(0x89, bits, 6, uint32(slot*16))
	a.memory(0x89, kind, 6, uint32(slot*16+8))
}

func (a *amd64Program) stateImmediate(offset, value uint32) {
	a.bytes(0x48, 0xc7, 0x87)
	a.word(offset)
	a.word(value)
}

func (a *amd64Program) kindCompare(slot int, kind ir.Kind) {
	a.bytes(0x48, 0x83, 0xbe)
	a.word(uint32(slot*16 + 8))
	a.bytes(byte(kind))
}

func (a *amd64Program) commit() { a.bytes(0x49, 0xff, 0xca) } // decq r10

func (a *amd64Program) number(o ir.Operand, xmm byte) {
	if o.Slot < 0 {
		if o.Literal.Kind != ir.Number {
			a.jump(a.guard)
		}
		a.immediate(0, o.Literal.Bits)
		a.bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0|xmm<<3) // movq ax, xmm
		return
	}
	a.kindCompare(o.Slot, ir.Number)
	a.conditional(5, a.guard)
	a.bytes(0xf2, 0x0f, 0x10, 0x86|xmm<<3)
	a.word(uint32(o.Slot * 16))
}

func (a *amd64Program) binary(op ir.Operator, left, right ir.Operand) {
	a.number(left, 0)
	a.number(right, 1)
	if op <= ir.Div {
		instruction := [...]byte{0x58, 0x5c, 0x59, 0x5e}[op]
		a.bytes(0xf2, 0x0f, instruction, 0xc1)
		return
	}
	// UCOMISD sets parity on unordered operands. Handle that explicitly rather
	// than reversing an ordered condition, which would mishandle NaNs.
	nan, done := a.label(), a.label()
	a.immediate(0, 0)
	a.bytes(0x66, 0x0f, 0x2e, 0xc1)
	a.conditional(10, nan)
	condition := [...]byte{2, 6, 7, 3, 4, 5}[op-ir.Lt]
	a.bytes(0x0f, 0x90+condition, 0xc0)
	a.jump(done)
	a.mark(nan)
	nanResult := uint64(0)
	if op == ir.Ne {
		nanResult = 1
	}
	a.immediate(0, nanResult)
	a.mark(done)
}

func (a *amd64Program) storeNumber(slot int) {
	a.bytes(0xf2, 0x0f, 0x11, 0x86)
	a.word(uint32(slot * 16))
	a.immediate(2, uint64(ir.Number))
	a.memory(0x89, 2, 6, uint32(slot*16+8))
}

func (a *amd64Program) storeBool(slot int) {
	a.immediate(2, uint64(ir.Boolean))
	a.store(slot, 0, 2)
}

func (a *amd64Program) truth(o ir.Operand) {
	number, zero, done := a.label(), a.label(), a.label()
	a.load(o, 0, 2)
	a.bytes(0x48, 0x83, 0xfa, byte(ir.Number))
	a.conditional(4, number)
	a.bytes(0x48, 0x83, 0xfa, byte(ir.Boolean))
	a.conditional(4, done)
	a.bytes(0x48, 0x83, 0xfa, byte(ir.Undefined))
	a.conditional(4, zero)
	a.bytes(0x48, 0x83, 0xfa, byte(ir.Null))
	a.conditional(4, zero)
	a.jump(a.guard)
	a.mark(number)
	a.immediate(8, 1<<63-1)
	a.bytes(0x4c, 0x21, 0xc0) // andq r8, ax: absolute IEEE bits
	a.bytes(0x48, 0x85, 0xc0)
	a.conditional(4, zero)
	a.immediate(8, 0x7ff0000000000000)
	a.bytes(0x4c, 0x39, 0xc0)
	a.conditional(7, zero) // absolute bits above +Inf are NaNs
	a.immediate(0, 1)
	a.jump(done)
	a.mark(zero)
	a.immediate(0, 0)
	a.mark(done)
}
