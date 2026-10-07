//go:build quickjs_jit && !android && !ios && darwin

package jit

import (
	"encoding/binary"
	"fmt"

	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

// R0 owns programState and R2 scalar scratch. R8 holds the remaining budget,
// R9 the exit PC, R1 borrowed array views. R16 caches a view within a region.
// R3-R7 and F0-F1 are scratch; F24 holds tentative index updates. F2-F7 and
// F16-F23 cache scalar bits, R10-R15 cache the first six kinds. All are
// spilled on every exit.
// SP, FP, LR, R18 and Go's R28 remain untouched; no native calls occur.
type arm64Program struct {
	programAssembler
	guard, budget, returned, host int
}

func programInstructions(p *ir.Program) ([]byte, []int, error) {
	a := &arm64Program{}
	a.allocateRegisters(p, []int{2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23})
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
			case ir.Insert3:
				a.load(ir.Slot(in.Dest+2), 5, 6)
				a.store(in.Dest+3, 5, 6)
				a.load(ir.Slot(in.Dest+1), 3, 4)
				a.store(in.Dest+2, 3, 4)
				a.load(ir.Slot(in.Dest), 3, 4)
				a.store(in.Dest+1, 3, 4)
				a.store(in.Dest, 5, 6)
			case ir.ArrayRead, ir.ArrayWrite, ir.ArrayKey, ir.ArrayLength, ir.ArrayUpdate:
				a.array(in)
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
				if in.Operator == ir.Int32 {
					fp := a.number(in.Left, 0)
					a.immediate(7, 0xc1e0000000000000)
					a.word(0x9e6700e1)
					a.word(0x1e612000 | fp<<5)
					a.conditional(4, a.guard)
					a.conditional(6, a.guard)
					a.immediate(7, 0x41dfffffffc00000)
					a.word(0x9e6700e1)
					a.word(0x1e612000 | fp<<5)
					a.conditional(12, a.guard)
					a.word(0x1e780007 | fp<<5) // fcvtzs w7,dN
					a.word(0x1e6200e0)         // scvtf d0,w7
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
					left := a.number(in.Left, 0)
					right := a.number(in.Right, 1)
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
				a.compareImmediate(4, uint32(ir.Null))
				a.conditional(8, a.guard)
				a.memory(false, false, 3, 0, 24)
				a.memory(false, false, 4, 0, 32)
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
	for _, exit := range []struct {
		label int
		kind  ir.ExitKind
	}{{common[ir.GuardExit], ir.GuardExit}, {common[ir.BudgetExit], ir.BudgetExit}, {common[ir.Returned], ir.Returned}, {common[ir.HostExit], ir.HostExit}} {
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

// array resolves a borrowed view and guards every condition before any write.
// X3 is the view/cell address, X5 the index, X6 the numeric tag boundary.
func (a *arm64Program) array(in ir.Instruction) {
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
		a.word(0x8b031423)        // add x3, x1, x3, lsl #5
		a.memory(true, false, 6, 3, 24)
		a.compareImmediate(6, 0)
		a.conditional(0, a.guard)
		if a.fast {
			a.word(0xaa0303f0)
		} // mov x16,x3
	}
	if in.Op == ir.ArrayLength {
		a.memory(true, false, 5, 3, 16)
		a.word(0x9e6300a0) // ucvtf d0, x5
		a.storeNumber(in.Dest, 0)
		return
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
	a.conditional(2, a.guard)
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
