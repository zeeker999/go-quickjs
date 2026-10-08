//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package jit

import (
	"math"
	"reflect"
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

func rangeProgram() *ir.Program {
	p := &ir.Program{Locals: 4, Code: []ir.Instruction{
		{Op: ir.Binary, Operator: ir.BitAnd, Left: ir.Slot(0), Right: ir.Literal(ir.Float(8191)), Dest: 1},
		{Op: ir.Copy, Left: ir.Slot(1), Dest: 2},
		{Op: ir.Binary, Operator: ir.Mul, Left: ir.Slot(1), Right: ir.Slot(2), Dest: 2},
		{Op: ir.Binary, Operator: ir.Add, Left: ir.Slot(2), Right: ir.Literal(ir.Float(4194304)), Dest: 2},
		{Op: ir.Binary, Operator: ir.Shr, Left: ir.Slot(2), Right: ir.Literal(ir.Float(13)), Dest: 3},
		{Op: ir.Return, Left: ir.Slot(3)},
	}}
	for pc := range p.Code {
		p.Maps = append(p.Maps, ir.StateMap{PC: uint32(pc)})
	}
	return p
}

func TestNativeIntegerRanges(t *testing.T) {
	p := rangeProgram()
	var a programAssembler
	a.regions(p)
	if a.ranges[1][1] != 13 || a.ranges[2][2] != 13 || a.ranges[3][2] != 26 || a.ranges[4][2] != 27 {
		t.Fatal("bounded limb arithmetic not inferred")
	}
	c := newTestCode(t, p)
	values := []ir.Value{
		{}, {Kind: ir.Opaque, Bits: 1}, ir.Float(math.Copysign(0, -1)), ir.Float(math.NaN()),
		ir.Float(math.Inf(1)), ir.Float(math.Inf(-1)), ir.Float(math.SmallestNonzeroFloat64),
		ir.Float(-8192.75), ir.Float(8191), ir.Float(1 << 31), ir.Float(1 << 32),
		ir.Float(math.Nextafter(1<<62, 0)), ir.Float(1 << 63), ir.Float(-1 << 63), ir.Float(1e100),
	}
	for pc := range p.Code {
		for budget := uint64(0); budget <= uint64(len(p.Code)+1); budget++ {
			for _, value := range values {
				// External entry after a supposedly bounded producer must accept
				// arbitrary values and retain the full conversion semantics.
				wantSlots := []ir.Value{value, value, value, value}
				gotSlots := append([]ir.Value(nil), wantSlots...)
				want, err := p.Evaluate(wantSlots, pc, budget)
				if err != nil {
					t.Fatal(err)
				}
				got, err := c.Run(gotSlots, pc, budget)
				if err != nil || got != want || !reflect.DeepEqual(gotSlots, wantSlots) {
					t.Fatalf("pc %d budget %d value %+v: %+v/%+v slots %v/%v error %v", pc, budget, value, got, want, gotSlots, wantSlots, err)
				}
			}
		}
	}
}

func TestIntegerRangeInvalidation(t *testing.T) {
	p := &ir.Program{Locals: 3, Code: []ir.Instruction{
		{Op: ir.Binary, Operator: ir.UShr, Left: ir.Slot(0), Right: ir.Literal(ir.Float(16)), Dest: 1},
		{Op: ir.StoreLoad, Left: ir.Slot(1), Right: ir.Slot(2), Dest: 2, Extra: 1},
		{Op: ir.Unary, Operator: ir.Neg, Left: ir.Slot(1), Dest: 2},
		{Op: ir.Binary, Operator: ir.Div, Left: ir.Slot(1), Right: ir.Slot(0), Dest: 2},
		{Op: ir.Jump, Target: 5},
		{Op: ir.Binary, Operator: ir.Shr, Left: ir.Slot(1), Right: ir.Literal(ir.Float(1)), Dest: 2},
		{Op: ir.Return, Left: ir.Slot(2)},
	}}
	for pc := range p.Code {
		p.Maps = append(p.Maps, ir.StateMap{PC: uint32(pc)})
	}
	var a programAssembler
	a.regions(p)
	if a.ranges[1][1] != 16 || a.ranges[2][1] != 16 || a.ranges[2][2] != 16 || a.ranges[3][2] != 16 ||
		a.ranges[4][2] != 0 || a.ranges[5][1] != 0 {
		t.Fatal("sequential aliasing, division or jump reset lost range semantics")
	}
	if boundedExponent(62) != 62 || boundedExponent(63) != 0 {
		t.Fatal("overflow-range boundary changed")
	}
}

func TestNativeIntegerAliases(t *testing.T) {
	values := []ir.Value{ir.Float(-1), ir.Float(3.75), ir.Float(1 << 31), ir.Float(1e100), ir.Float(math.NaN()), ir.Float(math.Inf(-1)), {}, {Kind: ir.Opaque, Bits: 3}}
	for first := ir.BitAnd; first <= ir.UShr; first++ {
		for second := ir.BitAnd; second <= ir.UShr; second++ {
			p := &ir.Program{Locals: 4, Code: []ir.Instruction{
				{Op: ir.Binary, Operator: first, Left: ir.Slot(0), Right: ir.Slot(3), Dest: 1},
				{Op: ir.Copy, Left: ir.Slot(1), Dest: 2},
				{Op: ir.Binary, Operator: second, Left: ir.Slot(2), Right: ir.Slot(2), Dest: 1},
				{Op: ir.CopyPair, Left: ir.Slot(1), Right: ir.Slot(3), Dest: 2, Extra: 3},
				{Op: ir.Unary, Operator: ir.BitNot, Left: ir.Slot(2), Dest: 1},
				{Op: ir.StoreLoad, Left: ir.Slot(1), Right: ir.Slot(2), Dest: 2, Extra: 1},
				{Op: ir.Unary, Operator: ir.Int32, Left: ir.Slot(1), Dest: 2},
				{Op: ir.Return, Left: ir.Slot(2)},
			}}
			for pc := range p.Code {
				p.Maps = append(p.Maps, ir.StateMap{PC: uint32(pc)})
			}
			c := newTestCode(t, p)
			for pc := range p.Code {
				for budget := uint64(0); budget <= uint64(len(p.Code)+1); budget++ {
					for _, v := range values {
						wantSlots := []ir.Value{v, v, v, ir.Float(13)}
						gotSlots := append([]ir.Value(nil), wantSlots...)
						want, err := p.Evaluate(wantSlots, pc, budget)
						if err != nil {
							t.Fatal(err)
						}
						got, err := c.Run(gotSlots, pc, budget)
						if err != nil || got != want || !reflect.DeepEqual(gotSlots, wantSlots) {
							t.Fatalf("operators %d/%d pc %d budget %d value %+v: %+v/%+v slots %v/%v error %v", first, second, pc, budget, v, got, want, gotSlots, wantSlots, err)
						}
					}
				}
			}
		}
	}
}

func TestNativeRetainedIntegers(t *testing.T) {
	p := &ir.Program{Locals: 10}
	for i := 0; i < 8; i++ {
		p.Code = append(p.Code,
			ir.Instruction{Op: ir.Unary, Operator: ir.Int32, Left: ir.Slot(i), Dest: 8},
			ir.Instruction{Op: ir.Binary, Operator: ir.Add, Left: ir.Slot(8), Right: ir.Literal(ir.Float(1)), Dest: 9},
		)
	}
	for i := 0; i < 8; i++ {
		p.Code = append(p.Code,
			ir.Instruction{Op: ir.Binary, Operator: ir.BitXor, Left: ir.Slot(i), Right: ir.Slot(9), Dest: 9},
			ir.Instruction{Op: ir.Copy, Left: ir.Slot(9), Dest: i},
		)
	}
	p.Code = append(p.Code, ir.Instruction{Op: ir.Return, Left: ir.Slot(9)})
	for pc := range p.Code {
		p.Maps = append(p.Maps, ir.StateMap{PC: uint32(pc)})
	}
	c := newTestCode(t, p)
	for pc := range p.Code {
		for budget := uint64(0); budget <= uint64(len(p.Code)+1); budget++ {
			wantSlots := make([]ir.Value, p.Locals)
			for i := range wantSlots {
				wantSlots[i] = ir.Float(math.Ldexp(float64(13+i), i*13) * math.Pow(-1, float64(i)))
			}
			gotSlots := append([]ir.Value(nil), wantSlots...)
			want, err := p.Evaluate(wantSlots, pc, budget)
			if err != nil {
				t.Fatal(err)
			}
			got, err := c.Run(gotSlots, pc, budget)
			if err != nil || got != want || !reflect.DeepEqual(gotSlots, wantSlots) {
				t.Fatalf("retained conversions pc %d budget %d: %+v/%+v slots %v/%v error %v", pc, budget, got, want, gotSlots, wantSlots, err)
			}
		}
	}
}

// Insert3 hands out four origins per instruction, so a long run of them
// alternating with bitwise operations once overran the integer-use table
// (arm64 only uses it, but the analysis is shared).
func TestIntegerResultsOriginBound(t *testing.T) {
	p := &ir.Program{StackSize: 4}
	add := func(in ir.Instruction, depth int) {
		p.Code = append(p.Code, in)
		p.Maps = append(p.Maps, ir.StateMap{PC: uint32(len(p.Maps)), Depth: depth})
	}
	for i := 0; i < 3; i++ {
		add(ir.Instruction{Op: ir.Copy, Dest: i, Left: ir.Literal(ir.Float(float64(i + 5)))}, i)
	}
	for len(p.Code)+3 <= ir.MaxInstructions {
		add(ir.Instruction{Op: ir.Insert3, Dest: 0}, 3)
		add(ir.Instruction{Op: ir.Binary, Operator: ir.BitOr, Dest: 2, Left: ir.Slot(2), Right: ir.Slot(3)}, 4)
	}
	add(ir.Instruction{Op: ir.Return, Left: ir.Slot(2)}, 3)
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	a := &programAssembler{}
	for range p.Code {
		a.label()
	}
	a.regions(p)
	if a.originCount <= ir.MaxInstructions*2+ir.MaxSlots {
		t.Fatalf("origin count %d does not exercise the old bound", a.originCount)
	}
	a.inferIntegerResults(p)
}

// Kind inference names every operation: one it did not know would keep facts
// across a write and remove a guard, so it is refused instead.
func TestInferKindsRefusesUnknownOperations(t *testing.T) {
	p := &ir.Program{Locals: 1, Code: []ir.Instruction{{Op: ir.Op(250)}, {Op: ir.Return, Left: ir.Slot(0)}}, Maps: []ir.StateMap{{PC: 0}, {PC: 1}}}
	defer func() {
		if recover() == nil {
			t.Fatal("kind inference accepted an unknown operation")
		}
	}()
	a := &programAssembler{}
	for range p.Code {
		a.label()
	}
	a.regions(p)
}
