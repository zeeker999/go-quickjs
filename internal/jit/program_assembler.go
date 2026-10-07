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
	code                           []byte
	labels                         []int
	fixups                         []relocation
	registers                      [ir.MaxSlots]int
	fast                           bool
	fastBodies, fastEntries, tails []int
	starts                         []bool
	facts                          [][ir.MaxSlots]int8
	unchanged                      [][ir.MaxSlots]bool
	origins                        [][ir.MaxSlots]uint16
	arrayCacheID                   int
	pc                             int
	exits                          []programExit
	conversions                    []integerConversion
}

type integerConversion struct {
	entry, done int
	fp          uint32
}

type programExit struct {
	label, pc, refund int
	kind              ir.ExitKind
}

// Precharge straight-line regions once. A small remaining budget uses the
// exact per-instruction path, and a guard refunds the uncommitted suffix.
func (a *programAssembler) regions(p *ir.Program) {
	n := len(p.Code)
	a.fastBodies, a.fastEntries, a.tails = make([]int, n), make([]int, n), make([]int, n)
	starts := make([]bool, n)
	a.starts = starts
	starts[0] = true
	for pc, in := range p.Code {
		if p.Maps[pc].Depth < 0 {
			continue
		}
		if in.Op == ir.Jump || in.Op == ir.Branch {
			starts[in.Target] = true
		}
		if (in.Op == ir.Jump || in.Op == ir.Branch || in.Op == ir.Return || in.Op == ir.Host) && pc+1 < n {
			starts[pc+1] = true
		}
	}
	for pc := n - 1; pc >= 0; pc-- {
		a.fastBodies[pc], a.fastEntries[pc] = a.label(), a.label()
		if p.Maps[pc].Depth < 0 {
			continue
		}
		a.tails[pc] = 1
		in := p.Code[pc]
		if pc+1 < n && a.tails[pc+1] < 4095 && !starts[pc+1] && in.Op != ir.Jump && in.Op != ir.Branch && in.Op != ir.Return && in.Op != ir.Host {
			a.tails[pc] += a.tails[pc+1]
		}
	}
	a.inferKinds(p)
}

func (a *programAssembler) target(pc int) int {
	if a.starts[pc] {
		return a.fastEntries[pc]
	}
	return pc
}

func (a *programAssembler) exit(pc int, kind ir.ExitKind) int {
	label := a.label()
	refund := 0
	if a.fast {
		refund = a.tails[pc]
		if kind == ir.Returned {
			refund--
		}
	}
	a.exits = append(a.exits, programExit{label, pc, refund, kind})
	return label
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
		case ir.Binary, ir.ArrayRead, ir.ArrayUpdate:
			read(in.Left)
			read(in.Right)
			uses[in.Dest]++
		case ir.ArrayWrite, ir.ArrayKey:
			read(in.Left)
			read(in.Right)
			if in.Op == ir.ArrayWrite {
				read(in.Third)
			}
		case ir.ArrayLength:
			read(in.Left)
			uses[in.Dest]++
		case ir.Insert2, ir.Insert3:
			width := 3
			if in.Op == ir.Insert2 {
				width = 2
			}
			for slot := in.Dest; slot <= in.Dest+width; slot++ {
				uses[slot] += 2
			}
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

// Facts hold only within a straight-line region. External entry into its middle
// uses the generic path, so no assumption can leak across an arbitrary resume.
func (a *programAssembler) inferKinds(p *ir.Program) {
	a.facts = make([][ir.MaxSlots]int8, len(p.Code))
	a.unchanged = make([][ir.MaxSlots]bool, len(p.Code))
	a.origins = make([][ir.MaxSlots]uint16, len(p.Code))
	var kinds [ir.MaxSlots]int8
	var origins [ir.MaxSlots]uint16
	next := uint16(ir.MaxSlots)
	reset := func() {
		for i := range kinds {
			kinds[i] = -1
			origins[i] = uint16(i)
		}
		next = uint16(ir.MaxSlots)
	}
	reset()
	read := func(o ir.Operand) int8 {
		if o.Slot < 0 {
			return int8(o.Literal.Kind)
		}
		return kinds[o.Slot]
	}
	refine := func(o ir.Operand, k ir.Kind) {
		if o.Slot < 0 {
			return
		}
		origin := origins[o.Slot]
		for i := range kinds {
			if origins[i] == origin {
				kinds[i] = int8(k)
			}
		}
	}
	write := func(dest int, k int8) { kinds[dest] = k; origins[dest] = next; next++ }
	for pc, in := range p.Code {
		if p.Maps[pc].Depth < 0 {
			continue
		}
		if a.starts[pc] || pc > 0 && a.tails[pc-1] == 1 {
			reset()
		}
		a.facts[pc] = kinds
		a.origins[pc] = origins
		before := kinds
		simple := false
		switch in.Op {
		case ir.Copy:
			k := read(in.Left)
			origin := next
			next++
			if in.Left.Slot >= 0 {
				origin = origins[in.Left.Slot]
			}
			kinds[in.Dest], origins[in.Dest] = k, origin
			simple = true
		case ir.CopyPair:
			l, r := read(in.Left), read(in.Right)
			lo, ro := next, next+1
			next += 2
			if in.Left.Slot >= 0 {
				lo = origins[in.Left.Slot]
			}
			if in.Right.Slot >= 0 {
				ro = origins[in.Right.Slot]
			}
			kinds[in.Dest], kinds[in.Extra] = l, r
			origins[in.Dest], origins[in.Extra] = lo, ro
		case ir.StoreLoad:
			write(in.Dest, read(in.Left))
			write(in.Extra, read(in.Right))
		case ir.Swap:
			kinds[in.Dest], kinds[in.Extra] = kinds[in.Extra], kinds[in.Dest]
			origins[in.Dest], origins[in.Extra] = origins[in.Extra], origins[in.Dest]
		case ir.Binary:
			refine(in.Left, ir.Number)
			refine(in.Right, ir.Number)
			before = kinds
			k := int8(ir.Number)
			if in.Operator >= ir.Lt && in.Operator <= ir.Ne {
				k = int8(ir.Boolean)
			}
			write(in.Dest, k)
			simple = true
		case ir.Unary:
			if in.Operator != ir.Not {
				refine(in.Left, ir.Number)
			}
			before = kinds
			k := int8(ir.Number)
			if in.Operator == ir.Not {
				k = int8(ir.Boolean)
			}
			write(in.Dest, k)
			simple = true
		case ir.Update:
			refine(in.Left, ir.Number)
			before = kinds
			write(in.Dest, int8(ir.Number))
			if in.Extra >= 0 {
				write(in.Extra, int8(ir.Number))
			}
			simple = true
		case ir.ArrayRead, ir.ArrayWrite, ir.ArrayUpdate, ir.ArrayKey, ir.ArrayLength:
			refine(in.Left, ir.Opaque)
			if in.Op != ir.ArrayLength {
				refine(in.Right, ir.Number)
			}
			if in.Op == ir.ArrayWrite {
				refine(in.Third, ir.Number)
			}
			before = kinds
			if in.Op == ir.ArrayRead || in.Op == ir.ArrayUpdate || in.Op == ir.ArrayLength {
				write(in.Dest, int8(ir.Number))
			}
			if in.Op == ir.ArrayUpdate {
				write(in.Extra, int8(ir.Number))
			}
			simple = true
		case ir.Insert2, ir.Insert3:
			n, last := in.Dest, 2
			if in.Op == ir.Insert2 {
				last = 1
			}
			t := kinds[n+last]
			for i := last; i > 0; i-- {
				write(n+i, kinds[n+i-1])
			}
			write(n, t)
			write(n+last+1, t)
		}
		if simple {
			for slot := range kinds {
				a.unchanged[pc][slot] = before[slot] >= 0 && before[slot] == kinds[slot]
			}
		}
	}
}

func (a *programAssembler) known(o ir.Operand, kind ir.Kind) bool {
	return a.fast && (o.Slot < 0 && o.Literal.Kind == kind || o.Slot >= 0 && a.facts[a.pc][o.Slot] == int8(kind))
}

func (a *programAssembler) arrayCached(o ir.Operand) bool {
	if !a.fast || o.Slot < 0 {
		a.arrayCacheID = -1
		return false
	}
	id := int(a.origins[a.pc][o.Slot])
	hit := a.arrayCacheID == id
	a.arrayCacheID = id
	return hit
}
