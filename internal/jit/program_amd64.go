//go:build quickjs_jit && !android && !ios && (linux || windows)

package jit

import (
	"encoding/binary"
	"math"

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

func programInstructions(p *ir.Program, dispatch bool) ([]byte, []int, error) {
	a := &amd64Program{}
	a.allocateRegisters(p, []int{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14})
	for range p.Code {
		a.label()
	}
	common := [5]int{a.label(), a.label(), a.label(), a.label(), a.label()}
	a.regions(p)
	initialize := a.label()
	for mode := 0; mode < 2; mode++ {
		a.fast = mode == 1
		for pc, in := range p.Code {
			if p.Maps[pc].Depth < 0 {
				continue
			}
			a.pc = pc
			a.beginInteger(in)
			if a.starts[pc] || pc > 0 && a.tails[pc-1] == 1 {
				a.arrayCacheID = -1
				a.propertyCacheID = -1
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
			case ir.Call:
				if dispatch {
					a.memory(0x8b, 0, 7, 8)
					a.bytes(0x48, 0x85, 0xc0)
					a.conditional(4, a.host)
					a.stateImmediate(24, in.Key)
					a.jump(a.exit(pc, ir.CallExit))
				} else {
					a.jump(a.host)
				}
				continue
			case ir.Insert2, ir.Insert3:
				last := 2
				if in.Op == ir.Insert2 {
					last = 1
				}
				a.load(ir.Slot(in.Dest+last), 8, 9)
				a.store(in.Dest+last+1, 8, 9)
				for i := last - 1; i >= 0; i-- {
					a.load(ir.Slot(in.Dest+i), 0, 2)
					a.store(in.Dest+i+1, 0, 2)
				}
				a.store(in.Dest, 8, 9)
			case ir.ArrayRead, ir.ArrayWrite, ir.ArrayKey, ir.ArrayLength, ir.ArrayUpdate:
				a.array(in)
			case ir.PropertyRead, ir.PropertyWrite, ir.BindingRead:
				a.property(in)
			case ir.ReferenceRead:
				a.reference(in)
			case ir.StringMethod, ir.StringCode:
				a.string(in)
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
				if in.Operator < ir.Lt || in.Operator > ir.Ne {
					a.storeNumber(in.Dest, fp)
				} else {
					a.storeBool(in.Dest)
				}
			case ir.Unary:
				if in.Operator == ir.Int32 || in.Operator == ir.BitNot {
					a.integer(in.Left)
					if in.Operator == ir.BitNot {
						a.bytes(0xf7, 0xd0) // notl ax
					}
					a.bytes(0xf2, 0x0f, 0x2a, 0xc0) // cvtsi2sd eax,x0
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
				a.bytes(0x48, 0x83, 0xfa, byte(ir.Uninitialized))
				a.conditional(4, a.guard)
				a.memory(0x89, 0, 7, 24)
				a.memory(0x89, 2, 7, 32)
				a.commit()
				a.jump(a.returned)
				continue
			}
			a.finishInteger(in)
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
	exitKinds := []ir.ExitKind{ir.GuardExit, ir.BudgetExit, ir.Returned, ir.HostExit}
	if dispatch {
		exitKinds = append(exitKinds, ir.CallExit)
	}
	for _, kind := range exitKinds {
		exit := struct {
			label int
			kind  ir.ExitKind
		}{common[kind], kind}
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
		if dispatch && (exit.kind == ir.Returned || exit.kind == ir.CallExit) {
			a.memory(0x8b, 0, 7, 8)
		}
		a.stateImmediate(8, uint32(exit.kind))
		if dispatch && exit.kind == ir.Returned {
			a.bytes(0x48, 0x85, 0xc0, 0x74, 0x02, 0xff, 0xe0) // test ax; jz ret; jmp ax
		} else if dispatch && exit.kind == ir.CallExit {
			a.bytes(0xff, 0xe0)
		}
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
	for _, conversion := range a.conversions {
		a.integerSlow(conversion)
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
	if op >= ir.BitAnd && op <= ir.UShr {
		return a.bitwise(op, left, right)
	}
	guard := a.guard
	if op == ir.Eq || op == ir.Ne {
		a.guard = a.host
	}
	l := a.number(left, 0)
	r := a.number(right, 1)
	a.guard = guard
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

// integer leaves ToUint32's bits in EAX. CVTTSD2SI's overflow sentinel selects
// a significand path for large doubles and nonfinite values. R8 and CX survive:
// they hold the first operand and borrowed views. SSE2 shifts avoid using CL.
func (a *amd64Program) integer(o ir.Operand) {
	if a.integerCached(o) {
		return
	}
	if o.Slot < 0 && o.Literal.Kind == ir.Number {
		a.immediate(0, uint64(ir.ToUint32(math.Float64frombits(o.Literal.Bits))))
		return
	}
	fp := a.number(o, 0)
	if a.boundedInteger(o) {
		a.bytes(0xf2, 0x48|fp>>3, 0x0f, 0x2c, 0xc0|fp&7) // cvttsd2si xmm,rax: no overflow possible
		return
	}
	slow, done := a.label(), a.label()
	a.conversions = append(a.conversions, integerConversion{entry: slow, done: done, fp: uint32(fp)})
	a.bytes(0xf2, 0x48|fp>>3, 0x0f, 0x2c, 0xc0|fp&7) // cvttsd2si xmm,rax
	// Subtracting one overflows only for CVTTSD2SI's MinInt64 sentinel.
	a.bytes(0x48, 0x83, 0xf8, 1) // cmp rax,1
	a.conditional(0, slow)       // JO
	a.mark(done)
}

// Keep the rare significand conversion out of the ordinary instruction body.
func (a *amd64Program) integerSlow(conversion integerConversion) {
	fp := byte(conversion.fp)
	zero, right, shifted := a.label(), a.label(), a.label()
	a.mark(conversion.entry)
	a.bytes(0x66, 0x49|(fp>>3)<<2, 0x0f, 0x7e, 0xc1|(fp&7)<<3) // movq xmm,r9: sign + exponent
	a.move(0, 9)
	a.move(2, 9)
	a.bytes(0x48, 0xc1, 0xea, 52, 0x81, 0xe2)
	a.word(2047)
	a.bytes(0x81, 0xea)
	a.word(1023)
	a.bytes(0x83, 0xfa, 84)
	a.conditional(3, zero) // unsigned: also rejects negative exponent
	a.bytes(0x48, 0xc1, 0xe0, 12, 0x48, 0xc1, 0xe8, 12)
	a.bytes(0x48, 0x0f, 0xba, 0xe8, 52) // bts rax,52: implicit leading bit
	a.bytes(0x83, 0xea, 52)
	a.conditional(8, right)               // JS: unbiased exponent < 52
	a.bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0) // movq rax,x0
	a.bytes(0x66, 0x0f, 0x6e, 0xca)       // movd edx,x1
	a.fpBinary(0x66, 0xf3, 0, 1)          // psllq x0,x1
	a.jump(shifted)
	a.mark(right)
	a.bytes(0xf7, 0xda) // neg edx
	a.bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0)
	a.bytes(0x66, 0x0f, 0x6e, 0xca)
	a.fpBinary(0x66, 0xd3, 0, 1) // psrlq x0,x1
	a.mark(shifted)
	a.bytes(0x66, 0x0f, 0x7e, 0xc0)   // movd x0,eax
	a.bytes(0x4d, 0x85, 0xc9)         // testq r9,r9
	a.conditional(9, conversion.done) // JNS
	a.bytes(0xf7, 0xd8)               // neg eax
	a.jump(conversion.done)
	a.mark(zero)
	a.immediate(0, 0)
	a.jump(conversion.done)
}

func (a *amd64Program) bitwise(op ir.Operator, left, right ir.Operand) byte {
	a.integer(left)
	if right.Slot < 0 && right.Literal.Kind == ir.Number {
		value := ir.ToUint32(math.Float64frombits(right.Literal.Bits))
		if op <= ir.BitXor {
			a.bytes([...]byte{0x25, 0x0d, 0x35}[op-ir.BitAnd]) // op eax,imm32
			a.word(value)
		} else if shift := byte(value & 31); shift != 0 {
			a.bytes(0xc1, [...]byte{0xe0, 0xf8, 0xe8}[op-ir.Shl], shift)
		}
	} else {
		a.move(8, 0)
		a.integer(right)
		a.move(2, 0)
		a.move(0, 8)
		if op <= ir.BitXor {
			a.bytes([...]byte{0x21, 0x09, 0x31}[op-ir.BitAnd], 0xd0) // op edx,eax
		} else {
			a.bytes(0x83, 0xe2, 31)
			a.bytes(0x66, 0x0f, 0x6e, 0xc0, 0x66, 0x0f, 0x6e, 0xca)
			a.fpBinary(0x66, [...]byte{0xf2, 0xe2, 0xd2}[op-ir.Shl], 0, 1)
			a.bytes(0x66, 0x0f, 0x7e, 0xc0)
		}
	}
	if op == ir.UShr {
		a.bytes(0x89, 0xc0) // zero-extend the unsigned result before 64-bit conversion
		a.bytes(0xf2, 0x48, 0x0f, 0x2a, 0xc0)
	} else {
		a.bytes(0xf2, 0x0f, 0x2a, 0xc0)
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

func (a *amd64Program) reference(in ir.Instruction) {
	a.bytes(0x48, 0x85, 0xc9)
	a.conditional(4, a.host)
	a.load(in.Left, 0, 2)
	if !a.known(in.Left, ir.Opaque) {
		a.bytes(0x48, 0x83, 0xfa, byte(ir.Opaque))
		a.conditional(5, a.host)
	}
	a.bytes(0x48, 0x3d)
	a.word(ir.MaxSlots)
	a.conditional(3, a.host)
	a.bytes(0x48, 0x8d, 0x04, 0x80, 0x48, 0xc1, 0xe0, 3, 0x48, 0x01, 0xc8)
	a.move(8, 0)
	a.memory(0x8b, 2, 8, 24)
	a.bytes(0x48, 0x85, 0xd2)
	a.conditional(5, a.host)
	a.memory(0x8b, 2, 8, 32)
	a.bytes(0x48, 0x85, 0xd2)
	a.conditional(5, a.host)
	a.memory(0x8b, 2, 8, 16)
	a.bytes(0x48, 0x85, 0xd2)
	a.conditional(4, a.host)
	a.memory(0x8b, 9, 8, 8)
	a.bytes(0x49, 0x83, 0xf9, ir.MaxProperties)
	a.conditional(7, a.host)
	a.bytes(0x4d, 0x85, 0xc9)
	a.conditional(4, a.host)
	a.memory(0x8b, 8, 8, 0)
	a.bytes(0x4d, 0x85, 0xc0)
	a.conditional(4, a.host)
	search, found := a.label(), a.label()
	a.mark(search)
	a.bytes(0x41, 0x81, 0x38)
	a.word(in.Key)
	a.conditional(4, found)
	a.bytes(0x49, 0x83, 0xc0, 40, 0x49, 0xff, 0xc9)
	a.conditional(5, search)
	a.jump(a.host)
	a.mark(found)
	a.memory(0x8b, 2, 8, 24)
	a.bytes(0x48, 0x85, 0xd2)
	a.conditional(4, a.host)
	a.bytes(0x81, 0x3a)
	a.word(in.Key)
	a.conditional(5, a.host)
	a.bytes(0xf6, 0x42, 4, 0xf8)
	a.conditional(5, a.host)
	for _, offset := range []uint32{8, 16} {
		a.memory(0x8b, 0, 8, offset)
		a.memory(0x3b, 0, 2, offset)
		a.conditional(5, a.host)
	}
	a.memory(0x8b, 0, 8, 32)
	a.bytes(0x48, 0x3d)
	a.word(ir.MaxSlots)
	a.conditional(3, a.host)
	a.immediate(2, uint64(ir.Opaque))
	a.store(in.Dest, 0, 2)
}

func (a *amd64Program) property(in ir.Instruction) {
	guard := a.guard
	a.guard = a.host
	defer func() { a.guard = guard }()
	if in.Op != ir.BindingRead && a.propertyCached(in.Left, in.Key) {
		a.move(8, 11)
		if in.Op == ir.PropertyWrite {
			a.bytes(0x41, 0xf6, 0x40, 4, 1)
			a.conditional(4, a.host)
		}
		a.memory(0x8b, 0, 8, 8)
	} else {
		referenceTable, attributes := a.label(), a.label()
		a.bytes(0x48, 0x85, 0xc9)
		a.conditional(4, a.host)
		a.load(in.Left, 0, 2)
		if !a.known(in.Left, ir.Opaque) {
			a.bytes(0x48, 0x83, 0xfa, byte(ir.Opaque))
			a.conditional(5, a.host)
		}
		a.bytes(0x48, 0x3d)
		a.word(ir.MaxSlots)
		a.conditional(3, a.host)
		a.bytes(0x48, 0x8d, 0x04, 0x80, 0x48, 0xc1, 0xe0, 3, 0x48, 0x01, 0xc8) // view = cx + ax*40
		a.memory(0x8b, 2, 0, 24)
		a.bytes(0x48, 0x85, 0xd2)
		a.conditional(5, a.host) // array permission excludes property access
		a.memory(0x8b, 2, 0, 32)
		a.bytes(0x48, 0x85, 0xd2)
		if in.Op == ir.BindingRead {
			a.conditional(4, a.host)
		} else {
			a.conditional(4, referenceTable)
		}
		a.bytes(0x66, 0x4c, 0x0f, 0x6e, 0xfa) // movq dx,x15: numeric tag boundary
		a.memory(0x8b, 9, 0, 8)
		a.bytes(0x49, 0x83, 0xf9, ir.MaxProperties)
		a.conditional(7, a.host)
		a.bytes(0x4d, 0x85, 0xc9)
		a.conditional(4, a.host)
		a.memory(0x8b, 8, 0, 0)
		a.bytes(0x4d, 0x85, 0xc0)
		a.conditional(4, a.host)
		search, found := a.label(), a.label()
		a.mark(search)
		a.bytes(0x41, 0x81, 0x38) // cmpl key,(r8)
		a.word(in.Key)
		a.conditional(4, found)
		a.bytes(0x49, 0x83, 0xc0, 24, 0x49, 0xff, 0xc9)
		a.conditional(5, search)
		a.jump(a.host)
		a.mark(found)
		if in.Op != ir.BindingRead {
			a.jump(attributes)
			a.mark(referenceTable)
			a.memory(0x8b, 2, 0, 16)
			a.bytes(0x48, 0x85, 0xd2)
			a.conditional(4, a.host)
			a.bytes(0x66, 0x4c, 0x0f, 0x6e, 0xfa)
			a.memory(0x8b, 9, 0, 8)
			a.bytes(0x49, 0x83, 0xf9, ir.MaxProperties)
			a.conditional(7, a.host)
			a.bytes(0x4d, 0x85, 0xc9)
			a.conditional(4, a.host)
			a.memory(0x8b, 8, 0, 0)
			a.bytes(0x4d, 0x85, 0xc0)
			a.conditional(4, a.host)
			refSearch, refFound := a.label(), a.label()
			a.mark(refSearch)
			a.bytes(0x41, 0x81, 0x38)
			a.word(in.Key)
			a.conditional(4, refFound)
			a.bytes(0x49, 0x83, 0xc0, 40, 0x49, 0xff, 0xc9)
			a.conditional(5, refSearch)
			a.jump(a.host)
			a.mark(refFound)
			a.memory(0x8b, 8, 8, 24)
			a.bytes(0x4d, 0x85, 0xc0)
			a.conditional(4, a.host)
			a.bytes(0x41, 0x81, 0x38)
			a.word(in.Key)
			a.conditional(5, a.host)
			a.mark(attributes)
		}

		a.bytes(0x41, 0xf6, 0x40, 4, 0xf8) // testb invalid attributes,4(r8)
		a.conditional(5, a.host)
		if in.Op == ir.PropertyWrite {
			a.bytes(0x41, 0xf6, 0x40, 4, 1)
			a.conditional(4, a.host)
		}
		a.memory(0x8b, 0, 8, 8)
		a.bytes(0x66, 0x4c, 0x0f, 0x7e, 0xfa) // movq x15,dx
		a.bytes(0x48, 0x39, 0xd0)
		a.conditional(3, a.host)
		if in.Op != ir.BindingRead {
			a.move(11, 8)
		}
	}
	if in.Op != ir.PropertyWrite {
		a.bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0)
		a.storeNumber(in.Dest, 0)
		return
	}
	fp := a.number(in.Right, 0)
	a.fpBinary(0x66, 0x2e, fp, fp)
	ordered, end := a.label(), a.label()
	a.conditional(11, ordered)
	a.immediate(0, 0x7ff8000000000000)
	a.memory(0x89, 0, 8, 8)
	a.jump(end)
	a.mark(ordered)
	a.bytes(0xf2, 0x41|(fp>>3)<<2, 0x0f, 0x11, 0x40|(fp&7)<<3, 8)
	a.mark(end)
}

func (a *amd64Program) string(in ir.Instruction) {
	guard := a.guard
	a.guard = a.host
	defer func() { a.guard = guard }()
	a.arrayCacheID, a.propertyCacheID = -1, -1
	view := func(o ir.Operand, kind ir.Kind) {
		a.bytes(0x48, 0x85, 0xc9) // testq cx,cx
		a.conditional(4, a.host)
		a.load(o, 8, 2)
		a.bytes(0x48, 0x83, 0xfa, byte(kind)) // cmpq dx,kind
		a.conditional(5, a.host)
		a.bytes(0x49, 0x81, 0xf8)
		a.word(ir.MaxSlots)
		a.conditional(3, a.host)
		a.bytes(0x4f, 0x8d, 0x04, 0x80) // lea (r8,r8,4),r8
		a.bytes(0x49, 0xc1, 0xe0, 3, 0x49, 0x01, 0xc8)
	}
	if in.Op == ir.StringMethod {
		view(in.Left, ir.String)
		a.memory(0x8b, 0, 8, 32)
		a.bytes(0x48, 0x85, 0xc0)
		a.conditional(4, a.host)
		a.bytes(0x48, 0x3d)
		a.word(ir.MaxSlots)
		a.conditional(7, a.host)
		a.bytes(0x48, 0xff, 0xc8) // decq ax
		a.immediate(2, uint64(ir.Opaque))
		a.store(in.Dest, 0, 2)
		return
	}
	view(in.Left, ir.Opaque)
	a.memory(0x8b, 9, 8, 8)
	a.immediate(0, ir.CharCodeAtBuiltin)
	a.bytes(0x49, 0x39, 0xc1) // cmpq r9,ax
	a.conditional(5, a.host)
	view(in.Right, ir.String)
	a.move(11, 8)
	fp := a.number(in.Third, 0)
	a.bytes(0xf2, 0x48|fp>>3, 0x0f, 0x2c, 0xc0|fp&7)
	a.immediate(2, 0xffffffff)
	a.bytes(0x48, 0x39, 0xd0)
	a.conditional(7, a.host)
	a.bytes(0xf2, 0x48, 0x0f, 0x2a, 0xc8)
	a.fpBinary(0x66, 0x2e, fp, 1)
	a.conditional(5, a.host)
	a.conditional(10, a.host)
	a.memory(0x8b, 2, 11, 8)
	a.bytes(0x48, 0x39, 0xd0)
	a.conditional(3, a.host)
	a.memory(0x8b, 9, 11, 16)
	a.memory(0x8b, 8, 11, 0)
	a.bytes(0x4d, 0x85, 0xc0)
	a.conditional(4, a.host)
	ascii, done := a.label(), a.label()
	a.bytes(0x49, 0x83, 0xf9, 1)
	a.conditional(4, ascii)
	a.bytes(0x49, 0x83, 0xf9, 2)
	a.conditional(5, a.host)
	a.bytes(0x4d, 0x8d, 0x04, 0x40) // lea (r8,ax,2),r8
	a.bytes(0x41, 0x0f, 0xb7, 0x00) // movzwl (r8),ax
	a.jump(done)
	a.mark(ascii)
	a.bytes(0x49, 0x01, 0xc0)       // addq ax,r8
	a.bytes(0x41, 0x0f, 0xb6, 0x00) // movzbl (r8),ax
	a.mark(done)
	a.bytes(0xf2, 0x0f, 0x2a, 0xc0) // cvtsi2sd ax,x0
	a.storeNumber(in.Dest, 0)
}

func (a *amd64Program) array(in ir.Instruction) {
	if in.Op == ir.ArrayWrite {
		guard := a.guard
		a.guard = a.host
		defer func() { a.guard = guard }()
	}
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
		a.bytes(0x4f, 0x8d, 0x04, 0x80)                // lea (r8,r8,4),r8
		a.bytes(0x49, 0xc1, 0xe0, 3, 0x49, 0x01, 0xc8) // 40-byte view
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
	if in.Op == ir.ArrayWrite {
		a.memory(0x8b, 2, 8, 32)
		a.bytes(0x66, 0x4c, 0x0f, 0x6e, 0xfa) // movq dx,x15
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
	if in.Op == ir.ArrayWrite {
		numeric := a.label()
		a.conditional(2, numeric)             // JB
		a.bytes(0x66, 0x4c, 0x0f, 0x7e, 0xfa) // movq x15,dx
		a.bytes(0x48, 0x39, 0xd0)
		a.conditional(5, a.guard)
		a.mark(numeric)
	} else {
		a.conditional(3, a.guard)
	}
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
