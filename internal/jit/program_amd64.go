//go:build quickjs_jit && !android && !ios && (linux || windows)

package jit

import (
	"encoding/binary"

	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

// DI owns programState and SI scalar scratch. R10 holds the remaining budget,
// R11 the exit PC (and a region's cached view), CX borrowed views.
// AX, DX, R8, R9 and X0-X1 are scratch;
// X2-X14 cache scalar bits; X15 holds tentative array updates; BX, R12, R13 and R15 cache the first four kinds.
// All are spilled on every exit. This uses Go's ABI,
// including on Windows, rather than the platform C calling convention.
// SP, BP and Go's R14 remain untouched; no native calls occur.
type amd64Program struct {
	programAssembler
	guard, budget, returned, host int
}

func programInstructions(p *ir.Program) ([]byte, []int, error) {
	a := &amd64Program{}
	a.allocateRegisters(p, []int{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14})
	for range p.Code {
		a.label()
	}
	common := [4]int{a.label(), a.label(), a.label(), a.label()}
	a.regions(p)
	initialize := a.label()
	for mode := 0; mode < 2; mode++ {
		a.fast = mode == 1
		for pc, in := range p.Code {
			if p.Maps[pc].Depth < 0 {
				continue
			}
			a.pc = pc
			if a.starts[pc] || pc > 0 && a.tails[pc-1] == 1 {
				a.arrayCacheID = -1
			}
			a.guard, a.budget, a.returned, a.host = a.exit(pc, ir.GuardExit), a.exit(pc, ir.BudgetExit), a.exit(pc, ir.Returned), a.exit(pc, ir.HostExit)
			if a.fast {
				a.mark(a.fastBodies[pc])
			} else {
				a.mark(pc)
				a.bytes(0x4d, 0x85, 0xd2) // testq r10,r10
				a.conditional(4, a.budget)
			}
			if in.Check {
				a.kindCompare(in.CheckSlot, ir.Uninitialized)
				a.conditional(4, a.guard)
			}
			switch in.Op {
			case ir.Nop:
			case ir.Host:
				a.jump(a.host)
				continue
			case ir.Insert3:
				a.load(ir.Slot(in.Dest+2), 8, 9)
				a.store(in.Dest+3, 8, 9)
				a.load(ir.Slot(in.Dest+1), 0, 2)
				a.store(in.Dest+2, 0, 2)
				a.load(ir.Slot(in.Dest), 0, 2)
				a.store(in.Dest+1, 0, 2)
				a.store(in.Dest, 8, 9)
			case ir.ArrayRead, ir.ArrayWrite, ir.ArrayKey, ir.ArrayLength, ir.ArrayUpdate:
				a.array(in)
			case ir.Copy:
				a.loadScalar(in.Left, 0, 2)
				a.storeScalar(in.Dest, 0, 2)
			case ir.CopyPair:
				a.loadScalar(in.Left, 0, 2)
				a.loadScalar(in.Right, 1, 9)
				a.storeScalar(in.Dest, 0, 2)
				a.storeScalar(in.Extra, 1, 9)
			case ir.StoreLoad:
				a.loadScalar(in.Left, 0, 2)
				a.storeScalar(in.Dest, 0, 2)
				a.loadScalar(in.Right, 0, 2)
				a.storeScalar(in.Extra, 0, 2)
			case ir.Swap:
				a.loadScalar(ir.Slot(in.Dest), 0, 2)
				a.loadScalar(ir.Slot(in.Extra), 1, 9)
				a.storeScalar(in.Dest, 1, 9)
				a.storeScalar(in.Extra, 0, 2)
			case ir.Binary:
				fp := a.binary(in.Operator, in.Left, in.Right, in.Dest)
				if in.Operator <= ir.Div {
					a.storeNumber(in.Dest, fp)
				} else {
					a.storeBool(in.Dest)
				}
			case ir.Unary:
				if in.Operator == ir.Int32 {
					fp := a.number(in.Left, 0)
					a.immediate(0, 0xc1e0000000000000)
					a.bytes(0x66, 0x48, 0x0f, 0x6e, 0xc8)
					a.fpBinary(0x66, 0x2e, fp, 1)
					a.conditional(2, a.guard)
					a.immediate(0, 0x41dfffffffc00000)
					a.bytes(0x66, 0x48, 0x0f, 0x6e, 0xc8)
					a.fpBinary(0x66, 0x2e, fp, 1)
					a.conditional(7, a.guard)
					a.bytes(0xf2, 0x48|fp>>3, 0x0f, 0x2c, 0xc0|fp&7)
					a.bytes(0xf2, 0x48, 0x0f, 0x2a, 0xc0)
					a.storeNumber(in.Dest, 0)
				} else if in.Operator == ir.Not {
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
				fp := a.binary(in.Operator, in.Left, ir.Literal(ir.Float(1)), in.Dest)
				a.storeNumber(in.Dest, fp)
				if in.Extra >= 0 {
					if in.Postfix {
						a.store(in.Extra, 8, 9)
					} else {
						a.storeNumber(in.Extra, fp)
					}
				}
			case ir.Jump:
				a.commit()
				a.jump(a.target(in.Target))
				continue
			case ir.Branch:
				if in.Operator == ir.Truth {
					a.truth(in.Left)
				} else {
					a.binary(in.Operator, in.Left, in.Right, -1)
				}
				a.commit()
				a.bytes(0x48, 0x85, 0xc0) // testq ax, ax
				condition := byte(4)
				if in.When {
					condition = 5
				}
				a.conditional(condition, a.target(in.Target))
				a.jump(a.fastEntries[pc+1])
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
			if a.tails[pc] == 1 {
				a.jump(a.fastEntries[pc+1])
			}
		}
	}
	a.fast = false
	for _, exit := range a.exits {
		a.mark(exit.label)
		a.bytes(0x41, 0xbb)
		a.word(uint32(exit.pc))
		if exit.refund != 0 {
			a.bytes(0x49, 0x81, 0xc2)
			a.word(uint32(exit.refund))
		}
		a.jump(common[exit.kind])
	}
	for pc := range p.Code {
		if p.Maps[pc].Depth < 0 {
			continue
		}
		a.mark(a.fastEntries[pc])
		a.bytes(0x49, 0x81, 0xfa)
		a.word(uint32(a.tails[pc]))
		a.conditional(2, pc)
		a.bytes(0x49, 0x81, 0xea)
		a.word(uint32(a.tails[pc]))
		a.jump(a.fastBodies[pc])
	}
	for _, exit := range []struct {
		label int
		kind  ir.ExitKind
	}{{common[ir.GuardExit], ir.GuardExit}, {common[ir.BudgetExit], ir.BudgetExit}, {common[ir.Returned], ir.Returned}, {common[ir.HostExit], ir.HostExit}} {
		a.mark(exit.label)
		for slot, reg := range a.registers {
			if reg >= 0 {
				a.fpMemory(0x11, byte(reg), uint32(slot*16))
				if kind := a.kindRegister(slot); kind >= 0 {
					a.memory(0x89, byte(kind), 6, uint32(slot*16+8))
				}
			}
		}
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
		a.bytes(0x48, 0x8d, 0x05) // leaq body(%rip), ax
		a.fixups = append(a.fixups, relocation{offset: len(a.code), label: a.target(pc)})
		a.word(0)
		a.jump(initialize)
	}
	a.mark(initialize)
	a.memory(0x8b, 10, 7, 0)
	for slot, reg := range a.registers {
		if reg >= 0 {
			a.fpMemory(0x10, byte(reg), uint32(slot*16))
			if kind := a.kindRegister(slot); kind >= 0 {
				a.memory(0x8b, byte(kind), 6, uint32(slot*16+8))
			}
		}
	}
	a.bytes(0xff, 0xe0) // jmp ax
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
	a.bytes(0x48|(reg>>3)<<2|base>>3, op, 0x80|(reg&7)<<3|base&7)
	a.word(offset)
}

func (a *amd64Program) load(o ir.Operand, bits, kind byte) {
	if o.Slot < 0 {
		a.immediate(bits, o.Literal.Bits)
		a.immediate(kind, uint64(o.Literal.Kind))
		return
	}
	if reg := a.registers[o.Slot]; reg >= 0 {
		a.bytes(0x66, 0x48|byte(reg>>3)<<2|bits>>3, 0x0f, 0x7e, 0xc0|byte(reg&7)<<3|bits&7)
	} else {
		a.memory(0x8b, bits, 6, uint32(o.Slot*16))
	}
	a.loadKind(o.Slot, kind)
}

func (a *amd64Program) store(slot int, bits, kind byte) {
	if reg := a.registers[slot]; reg >= 0 {
		a.bytes(0x66, 0x48|byte(reg>>3)<<2|bits>>3, 0x0f, 0x6e, 0xc0|byte(reg&7)<<3|bits&7)
	} else {
		a.memory(0x89, bits, 6, uint32(slot*16))
	}
	a.storeKind(slot, kind)
}

func (a *amd64Program) kindRegister(slot int) int {
	if reg := a.registers[slot]; reg >= 2 && reg < 6 {
		return [...]int{3, 12, 13, 15}[reg-2]
	}
	return -1
}

func (a *amd64Program) move(dest, source byte) {
	a.bytes(0x48|(source>>3)<<2|dest>>3, 0x89, 0xc0|(source&7)<<3|dest&7)
}

func (a *amd64Program) loadKind(slot int, kind byte) {
	if reg := a.kindRegister(slot); reg >= 0 {
		a.move(kind, byte(reg))
	} else {
		a.memory(0x8b, kind, 6, uint32(slot*16+8))
	}
}

func (a *amd64Program) storeKind(slot int, kind byte) {
	if a.fast && a.unchanged[a.pc][slot] {
		return
	}
	if reg := a.kindRegister(slot); reg >= 0 {
		a.move(byte(reg), kind)
	} else {
		a.memory(0x89, kind, 6, uint32(slot*16+8))
	}
}

func (a *amd64Program) loadScalar(o ir.Operand, xmm, kind byte) {
	if o.Slot < 0 {
		a.immediate(0, o.Literal.Bits)
		a.bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0|xmm<<3)
		a.immediate(kind, uint64(o.Literal.Kind))
	} else {
		if reg := a.registers[o.Slot]; reg >= 0 {
			a.fpMove(xmm, byte(reg))
		} else {
			a.fpMemory(0x10, xmm, uint32(o.Slot*16))
		}
		a.loadKind(o.Slot, kind)
	}
}

func (a *amd64Program) storeScalar(slot int, xmm, kind byte) {
	if reg := a.registers[slot]; reg >= 0 {
		a.fpMove(byte(reg), xmm)
	} else {
		a.fpMemory(0x11, xmm, uint32(slot*16))
	}
	a.storeKind(slot, kind)
}

func (a *amd64Program) stateImmediate(offset, value uint32) {
	a.bytes(0x48, 0xc7, 0x87)
	a.word(offset)
	a.word(value)
}

func (a *amd64Program) kindCompare(slot int, kind ir.Kind) {
	if reg := a.kindRegister(slot); reg >= 0 {
		a.bytes(0x48|byte(reg>>3), 0x83, 0xf8|byte(reg&7), byte(kind))
		return
	}
	a.bytes(0x48, 0x83, 0xbe)
	a.word(uint32(slot*16 + 8))
	a.bytes(byte(kind))
}

func (a *amd64Program) commit() {
	if !a.fast {
		a.bytes(0x49, 0xff, 0xca)
	}
} // decq r10

func (a *amd64Program) number(o ir.Operand, xmm byte) byte {
	if o.Slot < 0 {
		if o.Literal.Kind != ir.Number {
			a.jump(a.guard)
		}
		a.immediate(0, o.Literal.Bits)
		a.bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0|xmm<<3) // movq ax, xmm
		return xmm
	}
	if !a.known(o, ir.Number) {
		a.kindCompare(o.Slot, ir.Number)
		a.conditional(5, a.guard)
	}
	if reg := a.registers[o.Slot]; reg >= 0 {
		return byte(reg)
	}
	a.fpMemory(0x10, xmm, uint32(o.Slot*16))
	return xmm
}

func (a *amd64Program) fpMemory(op, xmm byte, offset uint32) {
	a.bytes(0xf2)
	if xmm >= 8 {
		a.bytes(0x44)
	}
	a.bytes(0x0f, op, 0x86|(xmm&7)<<3)
	a.word(offset)
}

func (a *amd64Program) fpMove(dest, source byte) {
	a.bytes(0xf2)
	if dest >= 8 || source >= 8 {
		a.bytes(0x40 | (dest>>3)<<2 | source>>3)
	}
	a.bytes(0x0f, 0x10, 0xc0|(dest&7)<<3|source&7)
}

func (a *amd64Program) binary(op ir.Operator, left, right ir.Operand, dest int) byte {
	l := a.number(left, 0)
	r := a.number(right, 1)
	if op <= ir.Div {
		fp := byte(0)
		if reg := a.registers[dest]; reg >= 0 {
			fp = byte(reg)
		}
		// SSE2 overwrites its left operand. Preserve an aliased right input
		// until the operation has completed, then store the result normally.
		if fp == r && fp != l {
			fp = 0
		}
		if fp != l {
			a.fpMove(fp, l)
		}
		instruction := [...]byte{0x58, 0x5c, 0x59, 0x5e}[op]
		a.fpBinary(0xf2, instruction, fp, r)
		return fp
	}
	// UCOMISD sets parity on unordered operands. Mask those out for <, <=,
	// and ==; include them for !=. > and >= are already false when unordered.
	a.immediate(0, 0)
	a.fpBinary(0x66, 0x2e, l, r)
	condition := [...]byte{2, 6, 7, 3, 4, 5}[op-ir.Lt]
	a.bytes(0x0f, 0x90+condition, 0xc0)
	switch op {
	case ir.Lt, ir.Le, ir.Eq:
		a.bytes(0x0f, 0x9b, 0xc2, 0x20, 0xd0) // setnp dl; andb dl, al
	case ir.Ne:
		a.bytes(0x0f, 0x9a, 0xc2, 0x08, 0xd0) // setp dl; orb dl, al
	}
	return 0
}

func (a *amd64Program) fpBinary(prefix, op, left, right byte) {
	a.bytes(prefix)
	if left >= 8 || right >= 8 {
		a.bytes(0x40 | (left>>3)<<2 | right>>3)
	}
	a.bytes(0x0f, op, 0xc0|(left&7)<<3|right&7)
}

func (a *amd64Program) storeNumber(slot int, fp byte) {
	if reg := a.registers[slot]; reg >= 0 {
		if byte(reg) != fp {
			a.fpMove(byte(reg), fp)
		}
	} else {
		a.fpMemory(0x11, fp, uint32(slot*16))
	}
	if !a.fast || !a.unchanged[a.pc][slot] {
		a.immediate(2, uint64(ir.Number))
		a.storeKind(slot, 2)
	}
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

func (a *amd64Program) array(in ir.Instruction) {
	if a.arrayCached(in.Left) {
		a.move(8, 11)
		a.memory(0x8b, 9, 8, 24)
	} else {
		a.bytes(0x48, 0x85, 0xc9) // testq cx, cx
		a.conditional(4, a.guard)
		a.load(in.Left, 8, 2)
		if !a.known(in.Left, ir.Opaque) {
			a.bytes(0x48, 0x83, 0xfa, byte(ir.Opaque))
			a.conditional(5, a.guard)
		}
		a.bytes(0x49, 0x81, 0xf8)
		a.word(ir.MaxSlots)
		a.conditional(3, a.guard)                      // JAE
		a.bytes(0x49, 0xc1, 0xe0, 5, 0x49, 0x01, 0xc8) // shlq $5,r8; addq cx,r8
		a.memory(0x8b, 9, 8, 24)
		a.bytes(0x4d, 0x85, 0xc9)
		a.conditional(4, a.guard)
		if a.fast {
			a.move(11, 8)
		}
	}
	if in.Op == ir.ArrayLength {
		a.memory(0x8b, 0, 8, 16)
		a.bytes(0xf2, 0x48, 0x0f, 0x2a, 0xc0) // cvtsi2sd ax,x0
		a.storeNumber(in.Dest, 0)
		return
	}
	fp := a.number(in.Right, 0)
	if in.Op == ir.ArrayUpdate {
		a.fpMove(15, fp)
		a.immediate(0, 0x3ff0000000000000)
		a.bytes(0x66, 0x48, 0x0f, 0x6e, 0xc8)
		op := byte(0x58)
		if in.Operator == ir.Sub {
			op = 0x5c
		}
		a.fpBinary(0xf2, op, 15, 1)
		if !in.Postfix {
			fp = 15
		}
	}
	a.bytes(0xf2, 0x48|fp>>3, 0x0f, 0x2c, 0xc0|fp&7) // cvttsd2si xmm,ax
	a.immediate(2, 0xffffffff)
	a.bytes(0x48, 0x39, 0xd0)
	a.conditional(7, a.guard) // rejects negative, overflowing, NaN
	a.bytes(0xf2, 0x48, 0x0f, 0x2a, 0xc8)
	a.fpBinary(0x66, 0x2e, fp, 1)
	a.conditional(5, a.guard)
	a.conditional(10, a.guard)
	if in.Op == ir.ArrayKey {
		return
	}
	a.memory(0x8b, 2, 8, 8)
	a.bytes(0x48, 0x39, 0xd0)
	a.conditional(3, a.guard)
	a.memory(0x8b, 8, 8, 0)
	a.bytes(0x4d, 0x85, 0xc0)
	a.conditional(4, a.guard)
	a.bytes(0x48, 0xc1, 0xe0, 4, 0x49, 0x01, 0xc0)
	a.memory(0x8b, 0, 8, 0)
	a.bytes(0x4c, 0x39, 0xc8)
	a.conditional(3, a.guard)
	if in.Op == ir.ArrayWrite {
		fp = a.number(in.Third, 0)
		a.fpBinary(0x66, 0x2e, fp, fp)
		ordered, end := a.label(), a.label()
		a.conditional(11, ordered)
		a.immediate(0, 0x7ff8000000000000)
		a.memory(0x89, 0, 8, 0)
		a.jump(end)
		a.mark(ordered)
		a.bytes(0xf2)
		a.bytes(0x41|(fp>>3)<<2, 0x0f, 0x11, (fp&7)<<3) // movsd xmm,(r8)
		a.mark(end)
	} else {
		if in.Op == ir.ArrayUpdate {
			a.storeNumber(in.Extra, 15)
		}
		a.bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0)
		a.storeNumber(in.Dest, 0)
	}
}
