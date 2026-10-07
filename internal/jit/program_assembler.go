//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package jit

import (
	"encoding/binary"
	"fmt"
	"sort"

	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

type relocation struct {
	offset, label int
	conditional   bool
	address       bool
}

type programAssembler struct {
	code      []byte
	labels    []int
	fixups    []relocation
	registers [ir.MaxSlots]int
}

// Keep frequently accessed scalar bits in FP registers, independently of kind.
// Entry loads and exit spills make every PC resumable without changing maps.
func (a *programAssembler) allocateRegisters(p *ir.Program, available []int) {
	var uses [ir.MaxSlots]int
	read := func(o ir.Operand) {
		if o.Slot >= 0 {
			uses[o.Slot]++
		}
	}
	for pc, in := range p.Code {
		if p.Maps[pc].Depth < 0 {
			continue
		}
		switch in.Op {
		case ir.Copy, ir.Unary, ir.Update:
			read(in.Left)
			uses[in.Dest]++
			if in.Op == ir.Update && in.Extra >= 0 {
				uses[in.Extra]++
			}
		case ir.CopyPair, ir.StoreLoad:
			read(in.Left)
			read(in.Right)
			uses[in.Dest]++
			uses[in.Extra]++
		case ir.Swap:
			uses[in.Dest] += 2
			uses[in.Extra] += 2
		case ir.Binary:
			read(in.Left)
			read(in.Right)
			uses[in.Dest]++
		case ir.Branch:
			read(in.Left)
			if in.Operator != ir.Truth {
				read(in.Right)
			}
		case ir.Return:
			read(in.Left)
		}
	}
	slots := make([]int, 0, p.Locals+p.StackSize)
	for slot := range a.registers {
		a.registers[slot] = -1
		if uses[slot] > 1 {
			slots = append(slots, slot)
		}
	}
	sort.Slice(slots, func(i, j int) bool {
		if uses[slots[i]] == uses[slots[j]] {
			return slots[i] < slots[j]
		}
		return uses[slots[i]] > uses[slots[j]]
	})
	for i, slot := range slots[:min(len(slots), len(available))] {
		a.registers[slot] = available[i]
	}
}

func (a *programAssembler) label() int {
	a.labels = append(a.labels, -1)
	return len(a.labels) - 1
}

func (a *programAssembler) mark(label int)      { a.labels[label] = len(a.code) }
func (a *programAssembler) bytes(bytes ...byte) { a.code = append(a.code, bytes...) }
func (a *programAssembler) word(word uint32)    { a.code = binary.LittleEndian.AppendUint32(a.code, word) }
func (a *programAssembler) quad(word uint64)    { a.code = binary.LittleEndian.AppendUint64(a.code, word) }

func (a *programAssembler) valid() error {
	if len(a.code) == 0 || len(a.code) > MaxCodeBytes {
		return fmt.Errorf("native code budget: %d bytes", len(a.code))
	}
	for _, fixup := range a.fixups {
		if a.labels[fixup.label] < 0 {
			return fmt.Errorf("unbound native branch")
		}
	}
	return nil
}
