//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package jit

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/bits"

	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

// R0 owns programState and R2 scalar scratch. R8 holds the remaining budget,
// R9 the exit PC, R1 borrowed array views. R16 caches a view within a region.
// R3-R7 and F0-F1 are scratch; F24 holds tentative index updates. F2-F7 and
// F16-F23 cache scalar bits, R10-R15 cache the first six kinds. All are
// spilled on every exit. R19-R25 retain integer conversions within a region.
// SP, FP, LR, R18 and Go's R28 remain untouched; no native calls occur.
type arm64Program struct {
	programAssembler
	guard, budget, returned, host int
	integerShadows                [7]int
	integerNext                   int
}

func arm64Instructions(p *ir.Program) ([]byte, []int, error) {
	a := &arm64Program{}
	a.allocateRegisters(p, []int{2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23})
	for range p.Code {
		a.label()
	}
	common := [4]int{a.label(), a.label(), a.label(), a.label()}
	a.regions(p)
	a.inferIntegerResults(p)
	initialize := a.label()
	for mode := 0; mode < 2; mode++ {
		a.fast = mode == 1
		for pc, in := range p.Code {
			if p.Maps[pc].Depth < 0 {
				continue
			}
			a.pc = pc
			a.beginInteger(in)
			if !a.fast || a.starts[pc] || pc > 0 && a.tails[pc-1] == 1 {
				for i := range a.integerShadows {
					a.integerShadows[i] = -1
				}
				a.integerNext = 0
			}
			if a.starts[pc] || pc > 0 && a.tails[pc-1] == 1 {
				a.arrayCacheID = -1
				a.propertyCacheID = -1
			}
			a.guard, a.budget, a.returned, a.host = a.exit(pc, ir.GuardExit), a.exit(pc, ir.BudgetExit), a.exit(pc, ir.Returned), a.exit(pc, ir.HostExit)
			if a.fast {
				a.mark(a.fastBodies[pc])
			} else {
				a.mark(pc)
				a.compareImmediate(8, 0)
				a.conditional(0, a.budget)
			}
			if in.Check {
				a.loadKind(in.CheckSlot, 4)
				a.compareImmediate(4, uint32(ir.Uninitialized))
				a.conditional(0, a.guard)
			}
			switch in.Op {
			case ir.Nop:
			case ir.Host:
				a.jump(a.host)
				continue
			case ir.Call:
				a.jump(a.host)
				continue
			case ir.Insert2, ir.Insert3:
				last := 2
				if in.Op == ir.Insert2 {
					last = 1
				}
				a.load(ir.Slot(in.Dest+last), 5, 6)
				a.store(in.Dest+last+1, 5, 6)
				for i := last - 1; i >= 0; i-- {
					a.load(ir.Slot(in.Dest+i), 3, 4)
					a.store(in.Dest+i+1, 3, 4)
				}
				a.store(in.Dest, 5, 6)
			case ir.ArrayRead, ir.ArrayWrite, ir.ArrayKey, ir.ArrayLength, ir.ArrayUpdate:
				a.array(in)
			case ir.PropertyRead, ir.PropertyWrite, ir.BindingRead:
				a.property(in)
			case ir.ReferenceRead:
				a.reference(in)
			case ir.StringMethod, ir.StringCode:
				a.string(in)
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
				if in.Operator < ir.Lt || in.Operator > ir.Ne {
					a.storeNumber(in.Dest, fp)
				} else {
					a.storeBool(in.Dest)
				}
			case ir.Unary:
				if in.Operator == ir.Int32 || in.Operator == ir.BitNot {
					a.integer(in.Left)
					if in.Operator == ir.BitNot {
						a.word(0x2a2303e3) // mvn w3,w3
					}
					a.word(0x1e620060) // scvtf d0,w3
					a.storeNumber(in.Dest, 0)
				} else if in.Operator == ir.Not {
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
				a.jump(a.target(in.Target))
				continue
			case ir.Branch:
				if in.Operator == ir.Truth {
					a.truth(in.Left)
				} else {
					guard := a.guard
					if in.Operator == ir.Eq || in.Operator == ir.Ne {
						a.guard = a.host
					}
					left := a.number(in.Left, 0)
					right := a.number(in.Right, 1)
					a.guard = guard
					a.word(0x1e602000 | right<<16 | left<<5) // fcmp dN, dM
					a.commit()                               // sub does not change condition flags
					condition := arm64Comparison(in.Operator)
					if !in.When {
						condition ^= 1
					}
					a.conditional(condition, a.target(in.Target))
					a.jump(a.fastEntries[pc+1])
					continue
				}
				a.commit()
				a.compareImmediate(3, 0)
				condition := uint32(0)
				if in.When {
					condition = 1
				}
				a.conditional(condition, a.target(in.Target))
				a.jump(a.fastEntries[pc+1])
				continue
			case ir.Return:
				a.load(in.Left, 3, 4)
				a.compareImmediate(4, uint32(ir.Uninitialized))
				a.conditional(0, a.guard)
				a.memory(false, false, 3, 0, 24)
				a.memory(false, false, 4, 0, 32)
				a.commit()
				a.jump(a.returned)
				continue
			}
			a.finishInteger(in)
			if a.fast && a.integerOrigin >= 0 && a.integerResults[pc] {
				a.retainIntegerOrigin(a.integerOrigin)
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
		a.immediate(9, uint64(exit.pc))
		if exit.refund != 0 {
			a.word(0x91000108 | uint32(exit.refund)<<10)
		} // add x8,x8,#refund
		a.jump(common[exit.kind])
	}
	for pc := range p.Code {
		if p.Maps[pc].Depth < 0 {
			continue
		}
		a.mark(a.fastEntries[pc])
		a.compareImmediate(8, uint32(a.tails[pc]))
		a.conditional(3, pc)                         // LO: exact small-budget path
		a.word(0xd1000108 | uint32(a.tails[pc])<<10) // sub x8,x8,#tail
		a.jump(a.fastBodies[pc])
	}
	exitKinds := []ir.ExitKind{ir.GuardExit, ir.BudgetExit, ir.Returned, ir.HostExit}
	for _, kind := range exitKinds {
		exit := struct {
			label int
			kind  ir.ExitKind
		}{common[kind], kind}
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
		a.fixups = append(a.fixups, relocation{offset: len(a.code), label: a.target(pc), address: true})
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
	for _, conversion := range a.conversions {
		a.integerSlow(conversion)
	}
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
	if a.fast && a.unchanged[a.pc][slot] {
		return
	}
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
	if a.fast {
		return
	}
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
	if !a.known(o, ir.Number) {
		if reg := a.registers[o.Slot]; reg >= 0 && reg < 8 {
			a.compareImmediate(uint32(reg+8), uint32(ir.Number))
		} else {
			a.loadKind(o.Slot, 4)
			a.compareImmediate(4, uint32(ir.Number))
		}
		a.conditional(1, a.guard)
	}
	if reg := a.registers[o.Slot]; reg >= 0 {
		return uint32(reg)
	}
	a.memory(true, true, fp, 2, o.Slot*16)
	return fp
}

func (a *arm64Program) binary(op ir.Operator, left, right ir.Operand, dest int) uint32 {
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

// integer leaves ToUint32's bits in W3. The common path truncates to int64;
// large values use their IEEE significand modulo 2^32, including NaNs/infinities.
// R6 and the checked view in R16 survive so a second operand can be converted.
func (a *arm64Program) integer(o ir.Operand) {
	if a.integerCached(o) {
		return
	}
	if a.fast && o.Slot >= 0 {
		origin := int(a.origins[a.pc][o.Slot])
		for i, cached := range a.integerShadows {
			if origin == cached {
				a.word(0x2a0003e3 | uint32(19+i)<<16) // mov w3,wN: retained ToUint32 bits
				return
			}
		}
	}
	if o.Slot < 0 && o.Literal.Kind == ir.Number {
		a.immediate(3, uint64(ir.ToUint32(math.Float64frombits(o.Literal.Bits))))
		return
	}
	fp := a.number(o, 0)
	if a.boundedInteger(o) {
		a.word(0x9e780003 | fp<<5) // fcvtzs x3,dN: proved finite and within int64
		a.retainInteger(o)
		return
	}
	slow, done := a.label(), a.label()
	a.conversions = append(a.conversions, integerConversion{entry: slow, done: done, fp: fp})
	a.word(0x9e780003 | fp<<5) // fcvtzs x3,dN
	// Saturated values have different top two bits. Accept the signed
	// 62-bit range directly; NaNs convert to zero, as ToUint32 requires.
	a.word(0xca030464) // eor x4,x3,x3,lsl #1
	a.compareImmediate(4, 0)
	a.conditional(4, slow) // MI
	a.mark(done)
	a.retainInteger(o)
}

func (a *arm64Program) retainInteger(o ir.Operand) {
	if !a.fast || o.Slot < 0 {
		return
	}
	a.retainIntegerOrigin(int(a.origins[a.pc][o.Slot]))
}

func (a *arm64Program) retainIntegerOrigin(origin int) {
	i := a.integerNext
	a.integerShadows[i] = origin
	a.integerNext = (i + 1) % len(a.integerShadows)
	a.word(0x2a0303e0 | uint32(19+i)) // mov wN,w3: shadow scalar bits, never a Go pointer
}

// Cold conversion blocks stay after the entry/exit code so ordinary integer
// operands fall through without jumping over a significand conversion body.
func (a *arm64Program) integerSlow(conversion integerConversion) {
	zero, right, sign := a.label(), a.label(), a.label()
	a.mark(conversion.entry)
	a.word(0x9e660007 | conversion.fp<<5)       // fmov x7,dN
	a.word(0xd3400004 | 52<<16 | 62<<10 | 7<<5) // ubfx x4,x7,#52,#11
	a.word(0xd1000084 | 1023<<10)               // sub x4,x4,#bias
	a.compareImmediate(4, 84)
	a.conditional(2, zero)             // too large, nonfinite, or exponent below zero
	a.word(0xd3400003 | 51<<10 | 7<<5) // ubfx x3,x7,#0,#52
	a.immediate(5, 1<<52)
	a.word(0xaa050063) // orr x3,x3,x5: restore implicit leading bit
	a.word(0xd1000084 | 52<<10)
	a.compareImmediate(4, 0)
	a.conditional(11, right)
	a.word(0x9ac42063) // lslv x3,x3,x4
	a.jump(sign)
	a.mark(right)
	a.word(0xcb0403e4) // neg x4,x4
	a.word(0x9ac42463) // lsrv x3,x3,x4
	a.mark(sign)
	a.compareImmediate(7, 0)
	a.conditional(10, conversion.done) // nonnegative IEEE bits
	a.word(0x4b0303e3)                 // neg w3,w3
	a.jump(conversion.done)
	a.mark(zero)
	a.immediate(3, 0)
	a.jump(conversion.done)
}

func (a *arm64Program) bitwise(op ir.Operator, left, right ir.Operand) uint32 {
	a.integer(left)
	if right.Slot < 0 && right.Literal.Kind == ir.Number {
		value := ir.ToUint32(math.Float64frombits(right.Literal.Bits))
		if op <= ir.BitXor {
			if immediate, ok := arm64LogicalImmediate(value); ok {
				a.word([...]uint32{0x12000063, 0x32000063, 0x52000063}[op-ir.BitAnd] | immediate)
			} else {
				a.immediate(4, uint64(value))
				a.word([...]uint32{0x0a040063, 0x2a040063, 0x4a040063}[op-ir.BitAnd])
			}
		} else if shift := value & 31; shift != 0 {
			word := uint32(0x53000063 | shift<<16 | 31<<10) // lsr w3,w3,#shift
			if op == ir.Shl {
				word = 0x53000063 | ((32-shift)&31)<<16 | (31-shift)<<10
			} else if op == ir.Shr {
				word = 0x13000063 | shift<<16 | 31<<10 // asr w3,w3,#shift
			}
			a.word(word)
		}
	} else {
		a.word(0x2a0303e6) // mov w6,w3
		a.integer(right)
		word := [...]uint32{0x0a0300c3, 0x2a0300c3, 0x4a0300c3,
			0x1ac320c3, 0x1ac328c3, 0x1ac324c3}[op-ir.BitAnd]
		a.word(word) // AND/OR/XOR or a variable 32-bit shift, inherently masked to 31
	}
	convert := uint32(0x1e620060) // scvtf d0,w3
	if op == ir.UShr {
		convert = 0x1e630060 // ucvtf d0,w3
	}
	a.word(convert)
	return 0
}

// A logical immediate repeats a rotated run of ones in a power-of-two element.
// Zero and all ones have no encoding and use an ordinary register operand.
func arm64LogicalImmediate(value uint32) (uint32, bool) {
	for size := uint32(2); size <= 32; size *= 2 {
		mask := uint32((uint64(1) << size) - 1)
		unit := value & mask
		ones := uint32(bits.OnesCount32(unit))
		if ones == 0 || ones == size {
			continue
		}
		repeated := unit
		for shift := size; shift < 32; shift += size {
			repeated |= unit << shift
		}
		if repeated != value {
			continue
		}
		run := (uint32(1) << ones) - 1
		for rotate := uint32(0); rotate < size; rotate++ {
			if ((run>>rotate | run<<(size-rotate)) & mask) == unit {
				return rotate<<16 | (((^(size-1)<<1)|(ones-1))&63)<<10, true
			}
		}
	}
	return 0, false
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
	if !a.fast || !a.unchanged[a.pc][slot] {
		a.immediate(4, uint64(ir.Number))
		a.storeKind(slot, 4)
	}
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

func (a *arm64Program) reference(in ir.Instruction) {
	a.compareImmediate(1, 0)
	a.conditional(0, a.host)
	a.load(in.Left, 3, 4)
	if !a.known(in.Left, ir.Opaque) {
		a.compareImmediate(4, uint32(ir.Opaque))
		a.conditional(1, a.host)
	}
	a.compareImmediate(3, ir.MaxSlots)
	a.conditional(2, a.host)
	a.word(0x8b030863)
	a.word(0x8b030c23) // add x3,x1,handle*40
	a.memory(true, false, 7, 3, 24)
	a.compareImmediate(7, 0)
	a.conditional(1, a.host)
	a.memory(true, false, 7, 3, 32)
	a.compareImmediate(7, 0)
	a.conditional(1, a.host)
	a.memory(true, false, 7, 3, 16)
	a.compareImmediate(7, 0)
	a.conditional(0, a.host)
	a.memory(true, false, 4, 3, 8)
	a.compareImmediate(4, ir.MaxProperties)
	a.conditional(8, a.host)
	a.compareImmediate(4, 0)
	a.conditional(0, a.host)
	a.memory(true, false, 3, 3, 0)
	a.compareImmediate(3, 0)
	a.conditional(0, a.host)
	a.immediate(7, uint64(in.Key))
	search, found := a.label(), a.label()
	a.mark(search)
	a.word(0xb9400065) // ldr w5,[x3]: permission key
	a.word(0x6b0700bf)
	a.conditional(0, found)
	a.word(0x9100a063) // add x3,x3,#40
	a.word(0xd1000484)
	a.compareImmediate(4, 0)
	a.conditional(1, search)
	a.jump(a.host)
	a.mark(found)
	a.memory(true, false, 5, 3, 24)
	a.compareImmediate(5, 0)
	a.conditional(0, a.host)
	a.word(0xb94000a6) // ldr w6,[x5]: live key
	a.word(0x6b0700df) // cmp w6,w7
	a.conditional(1, a.host)
	a.word(0x394010a6) // ldrb w6,[x5,#4]: live flags
	a.immediate(7, 0xf8)
	a.word(0x6a0700df)
	a.conditional(1, a.host)
	for _, offset := range []int{8, 16} {
		a.memory(true, false, 6, 3, offset)
		a.memory(true, false, 7, 5, offset)
		a.word(0xeb0700df) // cmp x6,x7: snapshot bits/reference
		a.conditional(1, a.host)
	}
	a.memory(true, false, 3, 3, 32)
	a.compareImmediate(3, ir.MaxSlots)
	a.conditional(2, a.host)
	a.immediate(4, uint64(ir.Opaque))
	a.store(in.Dest, 3, 4)
}

func (a *arm64Program) property(in ir.Instruction) {
	guard := a.guard
	a.guard = a.host
	defer func() { a.guard = guard }()
	if in.Op != ir.BindingRead && a.propertyCached(in.Left, in.Key) {
		a.word(0xaa1003e3) // mov x3,x16: checked cell in this region
		if in.Op == ir.PropertyWrite {
			a.word(0x39401065)
			a.immediate(7, 1)
			a.word(0x6a0700bf)
			a.conditional(0, a.host)
		}
		a.memory(true, false, 7, 3, 8)
	} else {
		referenceTable, attributes := a.label(), a.label()
		a.compareImmediate(1, 0)
		a.conditional(0, a.host)
		a.load(in.Left, 3, 4)
		if !a.known(in.Left, ir.Opaque) {
			a.compareImmediate(4, uint32(ir.Opaque))
			a.conditional(1, a.host)
		}
		a.compareImmediate(3, ir.MaxSlots)
		a.conditional(2, a.host)
		a.word(0x8b030863) // add x3,x3,x3,lsl #2
		a.word(0x8b030c23) // add x3,x1,x3,lsl #3: 40-byte view
		a.memory(true, false, 6, 3, 24)
		a.compareImmediate(6, 0)
		a.conditional(1, a.host) // array permission excludes property access
		a.memory(true, false, 6, 3, 32)
		a.compareImmediate(6, 0)
		if in.Op == ir.BindingRead {
			a.conditional(0, a.host)
		} else {
			a.conditional(0, referenceTable)
		}
		a.memory(true, false, 4, 3, 8)
		a.compareImmediate(4, ir.MaxProperties)
		a.conditional(8, a.host)
		a.compareImmediate(4, 0)
		a.conditional(0, a.host)
		a.memory(true, false, 3, 3, 0)
		a.compareImmediate(3, 0)
		a.conditional(0, a.host)
		a.immediate(7, uint64(in.Key))
		search, found := a.label(), a.label()
		a.mark(search)
		a.word(0xb9400065) // ldr w5,[x3]: property key
		a.word(0x6b0700bf) // cmp w5,w7
		a.conditional(0, found)
		a.word(0x91006063) // add x3,x3,#24
		a.word(0xd1000484) // sub x4,x4,#1
		a.compareImmediate(4, 0)
		a.conditional(1, search)
		a.jump(a.host)
		a.mark(found)
		if in.Op != ir.BindingRead {
			a.jump(attributes)
			a.mark(referenceTable)
			a.memory(true, false, 6, 3, 16)
			a.compareImmediate(6, 0)
			a.conditional(0, a.host)
			a.memory(true, false, 4, 3, 8)
			a.compareImmediate(4, ir.MaxProperties)
			a.conditional(8, a.host)
			a.compareImmediate(4, 0)
			a.conditional(0, a.host)
			a.memory(true, false, 3, 3, 0)
			a.compareImmediate(3, 0)
			a.conditional(0, a.host)
			a.immediate(7, uint64(in.Key))
			refSearch, refFound := a.label(), a.label()
			a.mark(refSearch)
			a.word(0xb9400065)
			a.word(0x6b0700bf)
			a.conditional(0, refFound)
			a.word(0x9100a063)
			a.word(0xd1000484)
			a.compareImmediate(4, 0)
			a.conditional(1, refSearch)
			a.jump(a.host)
			a.mark(refFound)
			a.memory(true, false, 3, 3, 24)
			a.compareImmediate(3, 0)
			a.conditional(0, a.host)
			a.word(0xb9400065)
			a.word(0x6b0700bf)
			a.conditional(1, a.host)
			a.mark(attributes)
		}

		a.word(0x39401065) // ldrb w5,[x3,#4]: property flags
		a.immediate(7, 0xf8)
		a.word(0x6a0700bf) // tst w5,w7: only ordinary data attributes
		a.conditional(1, a.host)
		if in.Op == ir.PropertyWrite {
			a.immediate(7, 1)
			a.word(0x6a0700bf)
			a.conditional(0, a.host)
		}
		a.memory(true, false, 7, 3, 8)
		a.word(0xeb0600ff) // cmp x7,x6: existing numeric value
		a.conditional(2, a.host)
		if in.Op != ir.BindingRead {
			a.word(0xaa0303f0) // mov x16,x3
		}
	}
	if in.Op != ir.PropertyWrite {
		a.word(0x9e6700e0) // fmov d0,x7
		a.storeNumber(in.Dest, 0)
		return
	}
	fp := a.number(in.Right, 0)
	a.word(0x1e602000 | fp<<16 | fp<<5)
	ordered, end := a.label(), a.label()
	a.conditional(7, ordered)
	a.immediate(7, 0x7ff8000000000000)
	a.memory(false, false, 7, 3, 8)
	a.jump(end)
	a.mark(ordered)
	a.memory(false, true, fp, 3, 8)
	a.mark(end)
}

// array resolves a borrowed view and guards every condition before any write.
// X3 is the view/cell address, X5 the index, X6 the numeric tag boundary.
func (a *arm64Program) string(in ir.Instruction) {
	guard := a.guard
	a.guard = a.host
	defer func() { a.guard = guard }()
	a.arrayCacheID, a.propertyCacheID = -1, -1
	view := func(o ir.Operand, kind ir.Kind) {
		a.compareImmediate(1, 0)
		a.conditional(0, a.host)
		a.load(o, 3, 4)
		a.compareImmediate(4, uint32(kind))
		a.conditional(1, a.host)
		a.compareImmediate(3, ir.MaxSlots)
		a.conditional(2, a.host)
		a.word(0x8b030863) // add x3,x3,x3,lsl #2
		a.word(0x8b030c23) // add x3,x1,x3,lsl #3
	}
	if in.Op == ir.StringMethod {
		view(in.Left, ir.String)
		a.memory(true, false, 3, 3, 32)
		a.compareImmediate(3, 0)
		a.conditional(0, a.host)
		a.compareImmediate(3, ir.MaxSlots)
		a.conditional(8, a.host) // HI
		a.word(0xd1000463)       // sub x3,x3,#1
		a.immediate(4, uint64(ir.Opaque))
		a.store(in.Dest, 3, 4)
		return
	}
	view(in.Left, ir.Opaque)
	a.memory(true, false, 7, 3, 8)
	a.immediate(5, ir.CharCodeAtBuiltin)
	a.word(0xeb0500ff) // cmp x7,x5
	a.conditional(1, a.host)
	view(in.Right, ir.String)
	a.word(0xaa0303f0) // mov x16,x3
	fp := a.number(in.Third, 0)
	a.word(0x1e790005 | fp<<5) // fcvtzu w5,dN
	a.word(0x1e6300a1)         // ucvtf d1,w5
	a.word(0x1e612000 | fp<<5) // fcmp dN,d1
	a.conditional(1, a.host)
	a.memory(true, false, 7, 16, 8)
	a.word(0xeb0700bf) // cmp x5,x7
	a.conditional(2, a.host)
	a.memory(true, false, 7, 16, 16)
	a.memory(true, false, 3, 16, 0)
	a.compareImmediate(3, 0)
	a.conditional(0, a.host)
	ascii, done := a.label(), a.label()
	a.compareImmediate(7, 1)
	a.conditional(0, ascii)
	a.compareImmediate(7, 2)
	a.conditional(1, a.host)
	a.word(0x8b050463) // add x3,x3,x5,lsl #1
	a.word(0x79400063) // ldrh w3,[x3]
	a.jump(done)
	a.mark(ascii)
	a.word(0x8b050063) // add x3,x3,x5
	a.word(0x39400063) // ldrb w3,[x3]
	a.mark(done)
	a.word(0x1e630060) // ucvtf d0,w3
	a.storeNumber(in.Dest, 0)
}

func (a *arm64Program) array(in ir.Instruction) {
	if in.Op == ir.ArrayWrite {
		guard := a.guard
		a.guard = a.host
		defer func() { a.guard = guard }()
	}
	if a.arrayCached(in.Left) {
		a.word(0xaa1003e3) // mov x3,x16: same view within this region
		a.memory(true, false, 6, 3, 24)
	} else {
		a.compareImmediate(1, 0)
		a.conditional(0, a.guard)
		a.load(in.Left, 3, 4)
		if !a.known(in.Left, ir.Opaque) {
			a.compareImmediate(4, uint32(ir.Opaque))
			a.conditional(1, a.guard)
		}
		a.compareImmediate(3, ir.MaxSlots)
		a.conditional(2, a.guard) // HS
		a.word(0x8b030863)        // add x3,x3,x3,lsl #2
		a.word(0x8b030c23)        // add x3,x1,x3,lsl #3: 40-byte view
		a.memory(true, false, 6, 3, 24)
		a.compareImmediate(6, 0)
		a.conditional(0, a.guard)
		if a.fast {
			a.word(0xaa0303f0) // mov x16,x3
		}
	}
	if in.Op == ir.ArrayLength {
		a.memory(true, false, 5, 3, 16)
		a.word(0x9e6300a0) // ucvtf d0, x5
		a.storeNumber(in.Dest, 0)
		return
	}
	if in.Op == ir.ArrayWrite {
		a.memory(true, false, 17, 3, 32)
	}
	fp := a.number(in.Right, 0)
	if in.Op == ir.ArrayUpdate {
		a.immediate(7, 0x3ff0000000000000)
		a.word(0x9e6700e1) // fmov d1, x7
		op := uint32(0x1e602800)
		if in.Operator == ir.Sub {
			op = 0x1e603800
		}
		a.word(op | 1<<16 | fp<<5 | 24) // tentative updated index in d24
		if !in.Postfix {
			fp = 24
		}
	}
	a.word(0x1e790005 | fp<<5) // fcvtzu w5, dN
	a.word(0x1e6300a1)         // ucvtf d1, w5
	a.word(0x1e612000 | fp<<5) // fcmp dN, d1
	a.conditional(1, a.guard)
	if in.Op == ir.ArrayKey {
		return
	}
	a.memory(true, false, 7, 3, 8)
	a.word(0xeb0700bf) // cmp x5, x7
	a.conditional(2, a.guard)
	a.memory(true, false, 3, 3, 0)
	a.compareImmediate(3, 0)
	a.conditional(0, a.guard)
	a.word(0x8b051063) // add x3, x3, x5, lsl #4
	a.memory(true, false, 7, 3, 0)
	a.word(0xeb0600ff) // cmp x7, x6
	if in.Op == ir.ArrayWrite {
		numeric := a.label()
		a.conditional(3, numeric) // LO
		a.word(0xeb1100ff)        // cmp x7,x17: approved pointer-free hole
		a.conditional(1, a.guard)
		a.mark(numeric)
	} else {
		a.conditional(2, a.guard)
	}
	if in.Op == ir.ArrayWrite {
		fp = a.number(in.Third, 0)
		a.word(0x1e602000 | fp<<16 | fp<<5) // fcmp dN, dN
		done := a.label()
		a.conditional(7, done) // VC: ordered
		a.immediate(7, 0x7ff8000000000000)
		a.word(0x9e6700e0)
		fp = 0
		// Ordered inputs branch around the canonical NaN store.
		a.memory(false, true, fp, 3, 0)
		end := a.label()
		a.jump(end)
		a.mark(done)
		fp = a.number(in.Third, 0)
		a.memory(false, true, fp, 3, 0)
		a.mark(end)
	} else {
		if in.Op == ir.ArrayUpdate {
			a.storeNumber(in.Extra, 24)
		}
		a.word(0x9e6700e0) // fmov d0, x7
		a.storeNumber(in.Dest, 0)
	}
}
