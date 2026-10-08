//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package jit

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"reflect"
	"runtime"
	"sync"
	"testing"
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	jitcompile "github.com/go-quickjs/go-quickjs/internal/jit/compile"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

func newTestCode(t testing.TB, p *ir.Program) *Code {
	t.Helper()
	c, err := Compile(p)
	if err != nil {
		t.Fatalf("native compilation must succeed: %v", err)
	}
	t.Cleanup(func() {
		if err := c.Close(); err != nil {
			t.Error(err)
		}
	})
	return c
}

func constantProgram() *ir.Program {
	return &ir.Program{Code: []ir.Instruction{{Op: ir.Return, Left: ir.Literal(ir.Float(42))}}, Maps: []ir.StateMap{{PC: 0}}}
}

func TestNativeProgramABI(t *testing.T) {
	var s programState
	var v ir.Value
	if unsafe.Sizeof(s) != 40 || unsafe.Offsetof(s.remaining) != 0 || unsafe.Offsetof(s.reason) != 8 ||
		unsafe.Offsetof(s.pc) != 16 || unsafe.Offsetof(s.value) != 24 || unsafe.Sizeof(v) != 16 || unsafe.Offsetof(v.Kind) != 8 {
		t.Fatal("native program ABI changed")
	}
	var view ir.ArrayView
	if unsafe.Sizeof(view) != 40 || unsafe.Offsetof(view.DenseLength) != 8 || unsafe.Offsetof(view.Length) != 16 || unsafe.Offsetof(view.NumberLimit) != 24 || unsafe.Offsetof(view.WritableHole) != 32 {
		t.Fatal("native array ABI changed")
	}
	var cell ir.PropertyCell
	if unsafe.Sizeof(cell) != 24 || unsafe.Offsetof(cell.Flags) != 4 || unsafe.Offsetof(cell.Bits) != 8 || unsafe.Offsetof(cell.Reference) != 16 {
		t.Fatal("native property ABI changed")
	}
	var ref ir.ReferenceCell
	if unsafe.Sizeof(ref) != 40 || unsafe.Offsetof(ref.Bits) != 8 || unsafe.Offsetof(ref.Reference) != 16 || unsafe.Offsetof(ref.Cell) != 24 || unsafe.Offsetof(ref.Handle) != 32 {
		t.Fatal("native reference ABI changed")
	}
}

func TestNativeStringCode(t *testing.T) {
	p := &ir.Program{Locals: 3, StackSize: 1, Code: []ir.Instruction{
		{Op: ir.StringMethod, Left: ir.Slot(0), Dest: 3},
		{Op: ir.StringCode, Left: ir.Slot(3), Right: ir.Slot(0), Third: ir.Slot(2), Dest: 3},
		{Op: ir.Return, Left: ir.Slot(3)},
	}, Maps: []ir.StateMap{{PC: 0}, {PC: 1, Depth: 1}, {PC: 2, Depth: 1}}}
	c := newTestCode(t, p)
	bytes := []byte("abcdef")
	units := []uint16{0x61, 0x1234, 0xd83d, 0xde00, 0xd800, 0xdc00}
	for _, width := range []uint64{0, 1, 2, 3} {
		for _, grant := range []uint64{0, 18, ir.MaxSlots + 1} {
			for _, index := range []float64{-1, math.Copysign(0, -1), 0, 1, 5, 6, 1.5, math.NaN(), math.Inf(1)} {
				views := make([]ir.ArrayView, ir.MaxSlots)
				views[5] = ir.ArrayView{Data: unsafe.Pointer(&bytes[0]), DenseLength: 6, Length: width, WritableHole: grant}
				if width == 2 {
					views[5].Data = unsafe.Pointer(&units[0])
				}
				views[17].DenseLength = ir.CharCodeAtBuiltin
				for pc := 0; pc < 3; pc++ {
					for budget := uint64(0); budget < 5; budget++ {
						x := []ir.Value{{Kind: ir.String, Bits: 5}, ir.Float(7), ir.Float(index), {Kind: ir.Opaque, Bits: 17}}
						y := append([]ir.Value(nil), x...)
						want, err := p.EvaluateArrays(x, views, pc, budget)
						if err != nil {
							t.Fatal(err)
						}
						got, err := c.RunEncodedArrays(y, views, pc, budget)
						if err != nil || want != got || !reflect.DeepEqual(x, y) {
							t.Fatalf("width %d grant %d index %v pc %d budget %d: %+v/%+v, slots %+v/%+v, %v", width, grant, index, pc, budget, want, got, x, y, err)
						}
					}
				}
			}
		}
	}
}

func TestNativeProgramReferences(t *testing.T) {
	for _, dest := range []int{0, 1, 2} {
		p := &ir.Program{Locals: 3, Code: []ir.Instruction{
			{Op: ir.ReferenceRead, Left: ir.Slot(0), Dest: dest, Key: 0xdeadbeef},
			{Op: ir.Return, Left: ir.Slot(dest)},
		}, Maps: []ir.StateMap{{PC: 0}, {PC: 1}}}
		c := newTestCode(t, p)
		for _, count := range []uint64{0, 1, 3, 8, 9} {
			for _, flags := range []uint8{0, 1, 7, 8, 16, 32, 64, 128} {
				for _, change := range []string{"none", "bits", "reference", "key", "nil cell", "handle"} {
					cell := ir.PropertyCell{Key: 0xdeadbeef, Flags: flags, Bits: 0xfff8000000000010, Reference: unsafe.Pointer(new(int))}
					refs := make([]ir.ReferenceCell, 9)
					for i := range refs {
						refs[i].Key = uint32(i)
					}
					refs[2] = ir.ReferenceCell{Key: cell.Key, Bits: cell.Bits, Reference: cell.Reference, Cell: &cell, Handle: 31}
					switch change {
					case "bits":
						cell.Bits++
					case "reference":
						cell.Reference = unsafe.Pointer(new(int))
					case "key":
						cell.Key++
					case "nil cell":
						refs[2].Cell = nil
					case "handle":
						refs[2].Handle = ir.MaxSlots
					}
					views := make([]ir.ArrayView, ir.MaxSlots)
					views[17] = ir.ArrayView{Data: unsafe.Pointer(&refs[0]), DenseLength: count, Length: 0xfff8000000000000}
					for _, source := range []ir.Value{{Kind: ir.Opaque, Bits: 17}, {Kind: ir.Opaque, Bits: ir.MaxSlots}, ir.Float(17)} {
						for pc := 0; pc < 2; pc++ {
							for budget := uint64(0); budget < 3; budget++ {
								x := []ir.Value{source, ir.Float(7), ir.Float(9)}
								y := append([]ir.Value(nil), x...)
								want, err := p.EvaluateArrays(x, views, pc, budget)
								if err != nil {
									t.Fatal(err)
								}
								got, err := c.RunEncodedArrays(y, views, pc, budget)
								if err != nil || want != got || !reflect.DeepEqual(x, y) {
									t.Fatalf("dest %d count %d flags %x %s source %+v pc %d budget %d: %+v/%+v, %v", dest, count, flags, change, source, pc, budget, want, got, err)
								}
							}
						}
						if _, err := c.RunEncodedArrays([]ir.Value{source, {}, {}}, views[:1], 0, 1); !errors.Is(err, ir.ErrState) {
							t.Fatal(err)
						}
					}
				}
			}
		}
		got, err := c.RunArrays([]ir.Value{{Kind: ir.Opaque, Bits: 17}, {}, {}}, nil, 0, 2)
		if err != nil || got.Kind != ir.HostExit || got.Steps != 0 {
			t.Fatalf("missing permissions: %+v, %v", got, err)
		}
	}
}

func TestNativeProgramProperties(t *testing.T) {
	for _, borrowed := range []bool{false, true} {
		for _, op := range []ir.Op{ir.PropertyRead, ir.PropertyWrite, ir.BindingRead} {
			for _, dest := range []int{0, 1, 2} {
				p := &ir.Program{Locals: 3, Code: []ir.Instruction{
					{Op: op, Left: ir.Slot(0), Right: ir.Slot(1), Dest: dest, Key: 0xdeadbeef},
					{Op: ir.Return, Left: ir.Slot(dest)},
				}, Maps: []ir.StateMap{{PC: 0}, {PC: 1}}}
				c := newTestCode(t, p)
				for _, count := range []uint64{0, 1, 3, 8, 9} {
					for _, flags := range []uint8{0, 1, 7, 8, 16, 32, 64, 128} {
						for _, bits := range []uint64{math.Float64bits(3), 0xfff8000000000001} {
							for _, value := range []ir.Value{ir.Float(7), ir.Float(math.Copysign(0, -1)), {Kind: ir.Number, Bits: 0xfff8000000000100}, {Kind: ir.Opaque}} {
								for _, budget := range []uint64{0, 1, 2} {
									a, b := make([]ir.PropertyCell, 9), make([]ir.PropertyCell, 9)
									for i := range a {
										a[i] = ir.PropertyCell{Key: uint32(i), Flags: 7, Bits: math.Float64bits(11)}
									}
									a[2] = ir.PropertyCell{Key: 0xdeadbeef, Flags: flags, Bits: bits}
									copy(b, a)
									va, vb := make([]ir.ArrayView, ir.MaxSlots), make([]ir.ArrayView, ir.MaxSlots)
									va[17] = ir.ArrayView{Data: unsafe.Pointer(&a[0]), DenseLength: count, WritableHole: 0xfff8000000000000}
									vb[17] = va[17]
									vb[17].Data = unsafe.Pointer(&b[0])
									if borrowed {
										ra, rb := make([]ir.ReferenceCell, 9), make([]ir.ReferenceCell, 9)
										for i := range a {
											ra[i] = ir.ReferenceCell{Key: a[i].Key, Cell: &a[i]}
											rb[i] = ir.ReferenceCell{Key: b[i].Key, Cell: &b[i]}
										}
										va[17] = ir.ArrayView{Data: unsafe.Pointer(&ra[0]), DenseLength: count, Length: 0xfff8000000000000}
										vb[17] = ir.ArrayView{Data: unsafe.Pointer(&rb[0]), DenseLength: count, Length: 0xfff8000000000000}
									}

									x := []ir.Value{{Kind: ir.Opaque, Bits: 17}, value, ir.Float(5)}
									y := append([]ir.Value(nil), x...)
									want, err := p.EvaluateArrays(x, va, 0, budget)
									if err != nil {
										t.Fatal(err)
									}
									got, err := c.RunArrays(y, vb, 0, budget)
									if err != nil || got != want || !reflect.DeepEqual(x, y) || !reflect.DeepEqual(a, b) {
										t.Fatalf("op %v dest %d count %d flags %x bits %x value %+v budget %d: %+v/%+v %v", op, dest, count, flags, bits, value, budget, got, want, err)
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

func TestNativeProgramPropertyPermission(t *testing.T) {
	p := &ir.Program{Locals: 1, StackSize: 1, Code: []ir.Instruction{
		{Op: ir.PropertyRead, Left: ir.Slot(0), Dest: 1, Key: 3},
		{Op: ir.Return, Left: ir.Slot(1)},
	}, Maps: []ir.StateMap{{PC: 0}, {PC: 1, Depth: 1}}}
	c := newTestCode(t, p)
	if got, err := c.RunArrays([]ir.Value{{Kind: ir.Opaque}, ir.Float(0)}, nil, 0, 2); err != nil || got.Kind != ir.HostExit || got.Steps != 0 {
		t.Fatalf("absent field views: %+v, %v", got, err)
	}
	cell := ir.PropertyCell{Key: 3, Flags: 7, Bits: math.Float64bits(11)}
	for _, permission := range []uint64{0, 1, 0xfff8000000000000} {
		views := make([]ir.ArrayView, ir.MaxSlots)
		views[0] = ir.ArrayView{Data: unsafe.Pointer(&cell), DenseLength: 1, NumberLimit: permission, WritableHole: 0xfff8000000000000}
		slots := []ir.Value{{Kind: ir.Opaque}, ir.Float(0)}
		want, err := p.EvaluateArrays(append([]ir.Value(nil), slots...), views, 0, 2)
		if err != nil {
			t.Fatal(err)
		}
		got, err := c.RunArrays(slots, views, 0, 2)
		if err != nil || got != want || permission != 0 && got.Kind != ir.HostExit || permission == 0 && got.Value != ir.Float(11) {
			t.Fatalf("property read with array permission %x: %+v/%+v, %v", permission, got, want, err)
		}
	}
}

func TestNativeProgramPropertyCache(t *testing.T) {
	p := &ir.Program{Locals: 6, Code: []ir.Instruction{
		{Op: ir.Copy, Left: ir.Slot(0), Dest: 5},
		{Op: ir.PropertyRead, Left: ir.Slot(0), Dest: 4, Key: 3},
		{Op: ir.PropertyWrite, Left: ir.Slot(5), Right: ir.Slot(2), Key: 3},
		{Op: ir.PropertyRead, Left: ir.Slot(5), Dest: 4, Key: 3},
		{Op: ir.ArrayRead, Left: ir.Slot(3), Right: ir.Slot(1), Dest: 4},
		{Op: ir.BindingRead, Left: ir.Slot(0), Dest: 4, Key: 3},
		{Op: ir.ArrayRead, Left: ir.Slot(3), Right: ir.Slot(1), Dest: 4},
		{Op: ir.PropertyRead, Left: ir.Slot(0), Dest: 4, Key: 9},
		{Op: ir.PropertyRead, Left: ir.Slot(5), Dest: 4, Key: 3},
		{Op: ir.PropertyRead, Left: ir.Slot(0), Dest: 4, Key: 3},
		{Op: ir.Return, Left: ir.Slot(4)},
	}}
	for pc := range p.Code {
		p.Maps = append(p.Maps, ir.StateMap{PC: uint32(pc)})
	}
	c := newTestCode(t, p)
	for _, flags := range []uint8{0, 1, 7} {
		for pc := range p.Code {
			for budget := uint64(0); budget <= uint64(len(p.Code)+1); budget++ {
				a := []ir.PropertyCell{{Key: 3, Flags: flags, Bits: math.Float64bits(2)}, {Key: 9, Flags: 7, Bits: math.Float64bits(11)}}
				b := append([]ir.PropertyCell(nil), a...)
				array := [2]uint64{math.Float64bits(13), 0}
				va, vb := make([]ir.ArrayView, ir.MaxSlots), make([]ir.ArrayView, ir.MaxSlots)
				va[0] = ir.ArrayView{Data: unsafe.Pointer(&a[0]), DenseLength: 2, WritableHole: 0xfff8000000000000}
				va[1] = ir.ArrayView{Data: unsafe.Pointer(&array), DenseLength: 1, Length: 1, NumberLimit: 0xfff8000000000000}
				copy(vb, va)
				vb[0].Data = unsafe.Pointer(&b[0])
				x := []ir.Value{{Kind: ir.Opaque}, ir.Float(0), ir.Float(7), {Kind: ir.Opaque, Bits: 1}, ir.Float(0), {Kind: ir.Opaque}}
				y := append([]ir.Value(nil), x...)
				want, err := p.EvaluateArrays(x, va, pc, budget)
				if err != nil {
					t.Fatal(err)
				}
				got, err := c.RunArrays(y, vb, pc, budget)
				if err != nil || got != want || !reflect.DeepEqual(x, y) || !reflect.DeepEqual(a, b) {
					t.Fatalf("cache flags %x pc %d budget %d: %+v/%+v %v", flags, pc, budget, got, want, err)
				}
			}
		}
	}
}

func TestNativeProgramArrays(t *testing.T) {
	type cell struct {
		bits uint64
		ref  unsafe.Pointer
	}
	for _, op := range []ir.Op{ir.ArrayRead, ir.ArrayWrite, ir.ArrayUpdate, ir.ArrayLength, ir.ArrayKey} {
		for _, key := range []float64{0, math.Copysign(0, -1), 1, 2, -1, 0.5, 4294967295, 4294967296, math.NaN(), math.Inf(1)} {
			for _, postfix := range []bool{false, true} {
				for _, bits := range []uint64{math.Float64bits(3), 0xfff8000000000001, math.Float64bits(math.NaN())} {
					in := ir.Instruction{Op: op, Left: ir.Slot(0), Right: ir.Slot(1), Third: ir.Slot(2), Dest: 3, Extra: 1, Operator: ir.Add, Postfix: postfix}
					p := &ir.Program{Locals: 4, Code: []ir.Instruction{in, {Op: ir.Return, Left: ir.Slot(3)}}, Maps: []ir.StateMap{{PC: 0}, {PC: 1}}}
					c := newTestCode(t, p)
					for _, budget := range []uint64{0, 1, 2} {
						a, b := []cell{{bits: bits}, {bits: math.Float64bits(7)}}, []cell{{bits: bits}, {bits: math.Float64bits(7)}}
						va, vb := make([]ir.ArrayView, ir.MaxSlots), make([]ir.ArrayView, ir.MaxSlots)
						va[0] = ir.ArrayView{Data: unsafe.Pointer(&a[0]), DenseLength: 2, Length: 4, NumberLimit: 0xfff8000000000000}
						vb[0] = va[0]
						vb[0].Data = unsafe.Pointer(&b[0])
						x := []ir.Value{{Kind: ir.Opaque}, ir.Float(key), ir.Float(math.NaN()), ir.Float(19)}
						y := append([]ir.Value(nil), x...)
						want, err := p.EvaluateArrays(x, va, 0, budget)
						if err != nil {
							t.Fatal(err)
						}
						got, err := c.RunArrays(y, vb, 0, budget)
						if err != nil || !reflect.DeepEqual(got, want) || !reflect.DeepEqual(x, y) || !reflect.DeepEqual(a, b) {
							t.Fatalf("op %v key %v postfix %v bits %x budget %d: exit %+v/%+v slots %v/%v cells %v/%v err %v", op, key, postfix, bits, budget, got, want, y, x, b, a, err)
						}
					}
				}
			}
		}
	}
}

func TestNativeProgramArrayViewCache(t *testing.T) {
	p := &ir.Program{Locals: 4, Code: []ir.Instruction{
		{Op: ir.ArrayRead, Left: ir.Slot(0), Right: ir.Slot(1), Dest: 2},
		{Op: ir.ArrayRead, Left: ir.Literal(ir.Value{Kind: ir.Opaque, Bits: 1}), Right: ir.Slot(1), Dest: 3},
		{Op: ir.ArrayRead, Left: ir.Slot(0), Right: ir.Slot(1), Dest: 2},
		{Op: ir.Binary, Operator: ir.Add, Left: ir.Slot(2), Right: ir.Slot(3), Dest: 2},
		{Op: ir.Return, Left: ir.Slot(2)},
	}, Maps: []ir.StateMap{{PC: 0}, {PC: 1}, {PC: 2}, {PC: 3}, {PC: 4}}}
	c := newTestCode(t, p)
	cells := [2][2]uint64{{math.Float64bits(2), 0}, {math.Float64bits(5), 0}}
	views := make([]ir.ArrayView, ir.MaxSlots)
	for i := 0; i < 2; i++ {
		views[i] = ir.ArrayView{Data: unsafe.Pointer(&cells[i][0]), DenseLength: 1, Length: 1, NumberLimit: 0xfff8000000000000}
	}
	for _, handle := range []uint64{0, 1, 255, 256, ^uint64(0)} {
		for _, budget := range []uint64{0, 1, 2, 3, 4, 5} {
			x := []ir.Value{{Kind: ir.Opaque, Bits: handle}, ir.Float(0), ir.Float(0), ir.Float(0)}
			y := append([]ir.Value(nil), x...)
			want, _ := p.EvaluateArrays(x, views, 0, budget)
			got, err := c.RunArrays(y, views, 0, budget)
			if err != nil || got != want || !reflect.DeepEqual(x, y) {
				t.Fatalf("handle %d budget %d: got %+v %v want %+v", handle, budget, got, err, want)
			}
		}
	}
	x := []ir.Value{{Kind: ir.Opaque}, ir.Float(0), ir.Float(0), ir.Float(0)}
	if got, err := c.RunArrays(x, nil, 0, 5); err != nil || got.Kind != ir.GuardExit || got.Steps != 0 {
		t.Fatalf("nil views: %+v %v", got, err)
	}
	if _, err := c.RunArrays(x, views[:1], 0, 5); !errors.Is(err, ir.ErrState) {
		t.Fatalf("invalid views: %v", err)
	}
}

func TestNativeProgramHoleWrites(t *testing.T) {
	const hole = 0xfff8000000000008
	type cell struct {
		bits uint64
		ref  unsafe.Pointer
	}
	p := &ir.Program{Locals: 3, Code: []ir.Instruction{
		{Op: ir.ArrayWrite, Left: ir.Slot(0), Right: ir.Slot(1), Third: ir.Slot(2)},
		{Op: ir.Return, Left: ir.Slot(2)},
	}, Maps: []ir.StateMap{{PC: 0}, {PC: 1}}}
	c := newTestCode(t, p)
	for _, bits := range []uint64{hole, 0xfff8000000000001, math.Float64bits(7)} {
		for _, permission := range []uint64{0, hole} {
			for _, value := range []float64{3, math.NaN(), math.Copysign(0, -1)} {
				for _, budget := range []uint64{0, 1, 2} {
					a, b := cell{bits: bits}, cell{bits: bits}
					va, vb := make([]ir.ArrayView, ir.MaxSlots), make([]ir.ArrayView, ir.MaxSlots)
					va[1] = ir.ArrayView{Data: unsafe.Pointer(&a), DenseLength: 1, Length: 1, NumberLimit: 0xfff8000000000000, WritableHole: permission}
					vb[1] = va[1]
					vb[1].Data = unsafe.Pointer(&b)
					x := []ir.Value{{Kind: ir.Opaque, Bits: 1}, ir.Float(0), ir.Float(value)}
					y := append([]ir.Value(nil), x...)
					want, err := p.EvaluateArrays(x, va, 0, budget)
					if err != nil {
						t.Fatal(err)
					}
					got, err := c.RunArrays(y, vb, 0, budget)
					if err != nil || got != want || a != b || !reflect.DeepEqual(x, y) || b.ref != nil {
						t.Fatalf("hole %x permission %x budget %d: %+v/%+v, %v", bits, permission, budget, got, want, err)
					}
				}
			}
		}
	}
}

func TestNativeProgramInt32(t *testing.T) {
	p := &ir.Program{Locals: 1, Code: []ir.Instruction{{Op: ir.Unary, Operator: ir.Int32, Left: ir.Slot(0), Dest: 0}, {Op: ir.Return, Left: ir.Slot(0)}}, Maps: []ir.StateMap{{PC: 0}, {PC: 1}}}
	c := newTestCode(t, p)
	for _, n := range []float64{0, math.Copysign(0, -1), 2.9, -2.9, math.MinInt32, math.MaxInt32, math.MinInt32 - 1, math.MaxInt32 + 1, math.NaN(), math.Inf(-1), math.Inf(1)} {
		for _, budget := range []uint64{0, 1, 2} {
			x, y := []ir.Value{ir.Float(n)}, []ir.Value{ir.Float(n)}
			want, _ := p.Evaluate(x, 0, budget)
			got, err := c.Run(y, 0, budget)
			if err != nil || got != want || x[0] != y[0] {
				t.Fatalf("input %v budget %d: %+v %v want %+v", n, budget, got, err, want)
			}
		}
	}
}

func TestNativeProgramBitwise(t *testing.T) {
	values := []float64{0, math.Copysign(0, -1), 2.9, -2.9, math.MinInt32, math.MaxInt32,
		2147483648, 4294967295, 4294967296, 4294967297, -4294967297,
		math.SmallestNonzeroFloat64, math.MaxFloat64, math.NaN(), math.Inf(-1), math.Inf(1)}
	for exponent := 31; exponent <= 86; exponent++ {
		x := math.Ldexp(1, exponent)
		values = append(values, math.Nextafter(x, 0), x, math.Nextafter(x, math.Inf(1)), -math.Nextafter(x, math.Inf(1)))
	}
	random := rand.New(rand.NewPCG(187, 901))
	for range 1000 {
		values = append(values, math.Float64frombits(random.Uint64()))
	}
	for _, op := range []ir.Operator{ir.Int32, ir.BitNot, ir.BitAnd, ir.BitOr, ir.BitXor, ir.Shl, ir.Shr, ir.UShr} {
		for _, dest := range []int{0, 1, 2} {
			for _, literal := range []bool{false, true} {
				in := ir.Instruction{Op: ir.Binary, Operator: op, Left: ir.Slot(0), Right: ir.Slot(1), Dest: dest}
				if op == ir.Int32 || op == ir.BitNot {
					in.Op = ir.Unary
				}
				if literal {
					in.Right = ir.Literal(ir.Float(-4294967297))
				}
				p := &ir.Program{Locals: 3, Code: []ir.Instruction{in, {Op: ir.Return, Left: ir.Slot(dest)}}, Maps: []ir.StateMap{{PC: 0}, {PC: 1}}}
				c := newTestCode(t, p)
				for i, x := range values {
					for _, y := range []float64{values[(i*17+3)%len(values)], float64(i%73 - 36)} {
						for _, budget := range []uint64{0, 1, 2} {
							a := []ir.Value{ir.Float(x), ir.Float(y), ir.Float(31)}
							b := append([]ir.Value(nil), a...)
							want, _ := p.Evaluate(a, 0, budget)
							got, err := c.Run(b, 0, budget)
							if err != nil || got != want || !reflect.DeepEqual(a, b) {
								t.Fatalf("op %v dest %d literal %v x %x y %x budget %d: exit %+v/%+v slots %v/%v error %v", op, dest, literal, math.Float64bits(x), math.Float64bits(y), budget, got, want, b, a, err)
							}
						}
					}
				}
				for _, slot := range []int{0, 1} {
					if in.Op == ir.Unary && slot == 1 || literal && slot == 1 {
						continue
					}
					a := []ir.Value{ir.Float(7), ir.Float(33), ir.Float(31)}
					a[slot] = ir.Value{Kind: ir.Opaque, Bits: 3}
					b := append([]ir.Value(nil), a...)
					want, _ := p.Evaluate(a, 0, 2)
					got, err := c.Run(b, 0, 2)
					if err != nil || got != want || !reflect.DeepEqual(a, b) {
						t.Fatalf("op %v guard slot %d: %+v / %+v %v", op, slot, got, want, err)
					}
				}
			}
		}
	}
}

func TestNativeProgramIntegerResultReuse(t *testing.T) {
	for _, boundary := range []ir.Op{ir.Nop, ir.Host, ir.Jump} {
		p := &ir.Program{Locals: 24}
		for i := 0; i < 10; i++ {
			p.Code = append(p.Code, ir.Instruction{Op: ir.Binary, Operator: ir.UShr, Left: ir.Slot(i), Right: ir.Literal(ir.Float(float64(i))), Dest: i + 12})
		}
		p.Code = append(p.Code, ir.Instruction{Op: boundary, Target: len(p.Code) + 1})
		for i := 0; i < 10; i++ {
			p.Code = append(p.Code,
				ir.Instruction{Op: ir.Copy, Left: ir.Slot(i + 12), Dest: 23},
				ir.Instruction{Op: ir.Binary, Operator: ir.Add, Left: ir.Slot(i), Right: ir.Literal(ir.Float(0.25)), Dest: i},
				ir.Instruction{Op: ir.Binary, Operator: ir.BitXor, Left: ir.Slot(i), Right: ir.Slot(23), Dest: i},
				ir.Instruction{Op: ir.Binary, Operator: ir.BitOr, Left: ir.Slot(i + 12), Right: ir.Slot(i), Dest: i + 12},
			)
		}
		p.Code = append(p.Code, ir.Instruction{Op: ir.Return, Left: ir.Slot(12)})
		p.Maps = make([]ir.StateMap, len(p.Code))
		for pc := range p.Maps {
			p.Maps[pc].PC = uint32(pc)
		}
		c := newTestCode(t, p)
		for _, value := range []float64{math.NaN(), math.Inf(-1), math.MaxFloat64, math.Copysign(0, -1), -4294967297, 2147483648, 3.75} {
			for entry := range p.Code {
				for budget := uint64(0); budget <= uint64(len(p.Code)+1); budget++ {
					for _, bad := range []bool{false, true} {
						a := make([]ir.Value, p.Locals)
						for i := range a {
							a[i] = ir.Float(value + float64(i))
						}
						if bad {
							a[5] = ir.Value{Kind: ir.Opaque, Bits: 0}
						}
						b := append([]ir.Value(nil), a...)
						want, err := p.Evaluate(a, entry, budget)
						if err != nil {
							t.Fatal(err)
						}
						got, err := c.Run(b, entry, budget)
						if err != nil || got != want || !reflect.DeepEqual(a, b) {
							t.Fatalf("boundary %v value %v entry %d budget %d bad %v: native %+v/%v oracle %+v", boundary, value, entry, budget, bad, got, err, want)
						}
					}
				}
			}
		}
	}
}

func TestNativeProgramBitwiseImmediates(t *testing.T) {
	constants := []float64{0, -1, 0x12345678, 4294967297, -4294967297,
		1e20, math.MaxFloat64, math.NaN(), math.Inf(1), math.Inf(-1)}
	// Enumerate every repeated circular run of bits by its bit positions,
	// independently of the backend's logical-immediate encoding algorithm.
	for width := 2; width <= 32; width *= 2 {
		for start := 0; start < width; start++ {
			for length := 1; length < width; length++ {
				var mask uint32
				for bit := 0; bit < 32; bit++ {
					if (bit-start+32)%width < length {
						mask |= 1 << bit
					}
				}
				constants = append(constants, float64(mask))
			}
		}
	}
	for _, op := range []ir.Operator{ir.BitAnd, ir.BitOr, ir.BitXor, ir.Shl, ir.Shr, ir.UShr} {
		right := constants
		if op >= ir.Shl {
			right = append([]float64(nil), constants[:10]...)
			for count := -40; count <= 72; count++ {
				right = append(right, float64(count)+0.25)
			}
		}
		for _, value := range right {
			p := &ir.Program{Locals: 1, Code: []ir.Instruction{
				{Op: ir.Binary, Operator: op, Left: ir.Slot(0), Right: ir.Literal(ir.Float(value)), Dest: 0},
				{Op: ir.Return, Left: ir.Slot(0)},
			}, Maps: []ir.StateMap{{PC: 0}, {PC: 1}}}
			c := newTestCode(t, p)
			for _, input := range []ir.Value{ir.Float(0xabcdef01), ir.Float(-7.9), ir.Float(1e20), ir.Float(math.NaN()), {Kind: ir.Opaque, Bits: 3}} {
				for pc := 0; pc < 2; pc++ {
					for budget := uint64(0); budget <= 2; budget++ {
						x, y := []ir.Value{input}, []ir.Value{input}
						want, _ := p.Evaluate(x, pc, budget)
						got, err := c.Run(y, pc, budget)
						if err != nil || got != want || x[0] != y[0] {
							t.Fatalf("op %v literal %v input %+v pc %d budget %d: native %+v/%v oracle %+v; slots %v/%v", op, value, input, pc, budget, got, err, want, y, x)
						}
					}
				}
			}
			if err := c.Close(); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestNativeProgramBitwiseArrayCache(t *testing.T) {
	p := &ir.Program{Locals: 4, Code: []ir.Instruction{
		{Op: ir.ArrayRead, Left: ir.Slot(0), Right: ir.Slot(1), Dest: 2},
		{Op: ir.Binary, Operator: ir.BitXor, Left: ir.Slot(2), Right: ir.Literal(ir.Float(5)), Dest: 3},
		{Op: ir.ArrayRead, Left: ir.Slot(0), Right: ir.Slot(1), Dest: 2},
		{Op: ir.Binary, Operator: ir.BitXor, Left: ir.Slot(2), Right: ir.Slot(3), Dest: 3},
		{Op: ir.Return, Left: ir.Slot(3)},
	}, Maps: []ir.StateMap{{PC: 0}, {PC: 1}, {PC: 2}, {PC: 3}, {PC: 4}}}
	c := newTestCode(t, p)
	cell := [2]uint64{math.Float64bits(math.Nextafter(math.Ldexp(1, 63), math.Inf(1))), 0}
	views := make([]ir.ArrayView, ir.MaxSlots)
	views[0] = ir.ArrayView{Data: unsafe.Pointer(&cell[0]), DenseLength: 1, Length: 1, NumberLimit: 0xfff8000000000000}
	for pc := range p.Code {
		for budget := uint64(0); budget <= 5; budget++ {
			a := []ir.Value{{Kind: ir.Opaque}, ir.Float(0), ir.Float(7), ir.Float(3)}
			b := append([]ir.Value(nil), a...)
			want, _ := p.EvaluateArrays(a, views, pc, budget)
			got, err := c.RunArrays(b, views, pc, budget)
			if err != nil || got != want || !reflect.DeepEqual(a, b) {
				t.Fatalf("pc %d budget %d: exit %+v/%+v slots %v/%v error %v", pc, budget, got, want, b, a, err)
			}
		}
	}
}

func TestNativeProgramInserts(t *testing.T) {
	values := []ir.Value{ir.Float(37), ir.Float(math.NaN()), ir.Bool(true), {Kind: ir.Null}, {Kind: ir.Uninitialized}, {Kind: ir.Opaque, Bits: 17}}
	for _, op := range []ir.Op{ir.Insert2, ir.Insert3} {
		locals := 2
		if op == ir.Insert3 {
			locals = 3
		}
		p := &ir.Program{Locals: locals, StackSize: 1, Code: []ir.Instruction{{Op: op}, {Op: ir.Return, Left: ir.Slot(locals)}}, Maps: []ir.StateMap{{PC: 0}, {PC: 1, Depth: 1}}}
		c := newTestCode(t, p)
		for _, l := range values {
			for _, r := range values {
				for pc := range p.Code {
					for _, budget := range []uint64{0, 1, 2} {
						a := []ir.Value{l, r, ir.Float(3), ir.Float(9)}[:locals+1]
						b := append([]ir.Value(nil), a...)
						want, _ := p.Evaluate(a, pc, budget)
						got, err := c.Run(b, pc, budget)
						if err != nil || got != want || !reflect.DeepEqual(a, b) {
							t.Fatalf("insert %v pc %d budget %d: %+v/%+v slots %v/%v error %v", op, pc, budget, got, want, b, a, err)
						}
					}
				}
			}
		}
	}
}

func TestNativeProgramMaximumRegion(t *testing.T) {
	p := &ir.Program{Code: make([]ir.Instruction, ir.MaxInstructions), Maps: make([]ir.StateMap, ir.MaxInstructions)}
	for pc := range p.Maps {
		p.Maps[pc].PC = uint32(pc)
	}
	p.Code[len(p.Code)-1] = ir.Instruction{Op: ir.Return, Left: ir.Literal(ir.Float(7))}
	c := newTestCode(t, p)
	for _, budget := range []uint64{0, 1, 4094, 4095, 4096} {
		want, _ := p.Evaluate(nil, 0, budget)
		got, err := c.Run(nil, 0, budget)
		if err != nil || got != want {
			t.Fatalf("budget %d: %+v %v want %+v", budget, got, err, want)
		}
	}
}

func TestNativeProgramMemoryBudget(t *testing.T) {
	p := constantProgram()
	c := newTestCode(t, p)
	owned := c.Size() + c.MetadataSize()
	for _, limit := range []int{-1, 0, owned - 1} {
		if code, err := CompileBudget(p, limit); code != nil || !errors.Is(err, ErrCodeBudget) {
			t.Fatalf("budget %d: %v, %v", limit, code, err)
		}
	}
	exact, err := CompileBudget(p, owned)
	if err != nil {
		t.Fatal(err)
	}
	if err := exact.Close(); err != nil {
		t.Fatal(err)
	}
	if code, err := Compile(nil); code != nil || !errors.Is(err, ErrProgram) {
		t.Fatalf("invalid program: %v, %v", code, err)
	}
}

func TestNativeProgramLifecycle(t *testing.T) {
	p := constantProgram()
	c := newTestCode(t, p)
	if c.Size() != os.Getpagesize() || c.MetadataSize() <= 0 {
		t.Fatalf("owned memory = %d code, %d metadata", c.Size(), c.MetadataSize())
	}
	// Code owns exit maps and entry offsets independently of the mutable IR.
	p.Maps[0] = ir.StateMap{PC: 99, Depth: 99}
	p.Code[0].Left = ir.Literal(ir.Float(99))
	got, err := c.Run(nil, 0, 1)
	if err != nil || got.Kind != ir.Returned || got.Value != ir.Float(42) || got.State.PC != 0 || got.Steps != 1 {
		t.Fatalf("Run = %+v, %v", got, err)
	}
	for _, pc := range []int{-1, 1, math.MaxInt} {
		if _, ok := c.EntryDepth(pc); ok {
			t.Fatalf("invalid entry map %d", pc)
		}
		if _, err := c.Run(nil, pc, 1); !errors.Is(err, ir.ErrState) {
			t.Fatalf("invalid pc %d: %v", pc, err)
		}
		if _, err := c.RunEncodedArrays(nil, nil, pc, 1); !errors.Is(err, ir.ErrState) {
			t.Fatalf("encoded invalid pc %d: %v", pc, err)
		}
	}
	if _, err := c.Run([]ir.Value{{}}, 0, 1); !errors.Is(err, ir.ErrState) {
		t.Fatal("accepted invalid scratch size")
	}
	if _, err := c.Run(nil, 0, MaxIterations+1); !errors.Is(err, ErrIterations) {
		t.Fatal("accepted excessive execution budget")
	}
	if _, err := c.RunEncodedArrays([]ir.Value{{}}, nil, 0, 1); !errors.Is(err, ir.ErrState) {
		t.Fatal("encoded entry accepted invalid scratch size")
	}
	if _, err := c.RunEncodedArrays(nil, make([]ir.ArrayView, 1), 0, 1); !errors.Is(err, ir.ErrState) {
		t.Fatal("encoded entry accepted invalid view-table size")
	}
	if _, err := c.RunEncodedArrays(nil, nil, 0, MaxIterations+1); !errors.Is(err, ErrIterations) {
		t.Fatal("encoded entry accepted excessive execution budget")
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	if c.Size() != 0 || c.MetadataSize() != 0 {
		t.Fatal("closed owner retained memory")
	}
	if _, ok := c.EntryDepth(0); ok {
		t.Fatal("closed code exposed an entry")
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Run(nil, 0, 0); !errors.Is(err, ErrClosed) {
		t.Fatal("entered released native code")
	}
	if _, err := c.RunEncodedArrays(nil, nil, 0, 0); !errors.Is(err, ErrClosed) {
		t.Fatal("encoded entry entered released native code")
	}
}

func TestNativeProgramMultiplePages(t *testing.T) {
	p := &ir.Program{Code: make([]ir.Instruction, 1000), Maps: make([]ir.StateMap, 1000)}
	for pc := range p.Maps {
		p.Maps[pc].PC = uint32(pc)
	}
	p.Code[999] = ir.Instruction{Op: ir.Return, Left: ir.Literal(ir.Float(7))}
	c := newTestCode(t, p)
	if c.Size() <= os.Getpagesize() || c.Size()%os.Getpagesize() != 0 {
		t.Fatalf("multi-page code size = %d", c.Size())
	}
	for _, pc := range []int{0, 498, 999} {
		got, err := c.Run(nil, pc, MaxIterations)
		if err != nil || got.Kind != ir.Returned || got.Value != ir.Float(7) || got.Steps != uint64(1000-pc) {
			t.Fatalf("pc %d: %+v, %v", pc, got, err)
		}
	}
}

func TestNativeProgramHostEntries(t *testing.T) {
	p := &ir.Program{Locals: 1, StackSize: 1,
		Code: []ir.Instruction{
			{Op: ir.Copy, Left: ir.Literal(ir.Float(8)), Dest: 1},
			{Op: ir.Host},
			{Op: ir.Host, Check: true, CheckSlot: 0},
			{Op: ir.Return, Left: ir.Slot(1)},
			{Op: ir.Host},
		}, Maps: []ir.StateMap{{PC: 0}, {PC: 1, Depth: 1}, {PC: 2, Depth: 1}, {PC: 3, Depth: 1}, {PC: 4, Depth: -1}}}
	c := newTestCode(t, p)
	if c.entries[1] != hostProgramEntry || c.entries[2] < 0 || c.entries[4] != -1 {
		t.Fatal("host entry must preserve checked and unreachable entries")
	}
	for _, input := range []ir.Value{ir.Float(3), {Kind: ir.Uninitialized}, {Kind: ir.Opaque, Bits: 1}} {
		for pc := 0; pc < 4; pc++ {
			if depth, ok := c.EntryDepth(pc); !ok || depth != p.Maps[pc].Depth {
				t.Fatalf("host entry %d depth = %d, %v", pc, depth, ok)
			}
			for _, budget := range []uint64{0, 1, 2, MaxIterations} {
				for _, encoded := range []bool{false, true} {
					x, y := []ir.Value{input, ir.Float(7)}, []ir.Value{input, ir.Float(7)}
					want, _ := p.Evaluate(x, pc, budget)
					var got ir.Exit
					var err error
					if encoded {
						got, err = c.RunEncodedArrays(y, nil, pc, budget)
					} else {
						got, err = c.Run(y, pc, budget)
					}
					if err != nil || got != want || x[0] != y[0] || x[1] != y[1] {
						t.Fatalf("host pc %d budget %d encoded %v: native %+v/%v oracle %+v; slots %v/%v", pc, budget, encoded, got, err, want, y, x)
					}
				}
			}
		}
	}
	if _, err := c.Run([]ir.Value{{Kind: ir.Boolean, Bits: 2}, {}}, 1, 1); !errors.Is(err, ir.ErrState) {
		t.Fatal("host entry skipped checked scalar validation")
	}
}

func TestNativeProgramEveryEntry(t *testing.T) {
	// Exercise transfers across a large scratch layout. Every reachable entry
	// needs its own initialized budget register, including entries in the
	// middle of a straight line.
	p := &ir.Program{Locals: ir.MaxSlots,
		Code: []ir.Instruction{
			{Op: ir.CopyPair, Left: ir.Slot(30), Right: ir.Slot(32), Dest: 31, Extra: 200},
			{Op: ir.StoreLoad, Left: ir.Slot(200), Right: ir.Slot(250), Dest: 250, Extra: 30},
			{Op: ir.Swap, Dest: 31, Extra: 200},
			{Op: ir.Binary, Operator: ir.Add, Left: ir.Slot(31), Right: ir.Slot(200), Dest: 0},
			{Op: ir.Return, Left: ir.Slot(0)},
		}, Maps: make([]ir.StateMap, 5)}
	for pc := range p.Maps {
		p.Maps[pc].PC = uint32(pc)
	}
	c := newTestCode(t, p)
	for pc := range p.Maps {
		if depth, ok := c.EntryDepth(pc); !ok || depth != p.Maps[pc].Depth {
			t.Fatalf("entry %d depth = %d, %v", pc, depth, ok)
		}
	}
	for _, value := range []ir.Value{ir.Float(2), ir.Bool(true), {Kind: ir.Opaque, Bits: 123}, {Kind: ir.Uninitialized}} {
		for pc := range p.Code {
			for _, budget := range []uint64{0, 1, 2, 3, MaxIterations} {
				oracle := make([]ir.Value, ir.MaxSlots)
				for i := range oracle {
					oracle[i] = ir.Float(float64(i))
				}
				oracle[30] = value
				slots := append([]ir.Value(nil), oracle...)
				want, err := p.Evaluate(oracle, pc, budget)
				if err != nil {
					t.Fatal(err)
				}
				got, err := c.Run(slots, pc, budget)
				if err != nil || got != want {
					t.Fatalf("pc %d budget %d: native %+v, %v; oracle %+v", pc, budget, got, err, want)
				}
				for i := range slots {
					if slots[i] != oracle[i] {
						t.Fatalf("pc %d budget %d: slot %d native %+v, oracle %+v", pc, budget, i, slots[i], oracle[i])
					}
				}
			}
		}
	}
}

func TestNativeProgramRegisterTransfers(t *testing.T) {
	// Exceed the register pool and cross its boundary with overlapping copies.
	// Compare every entry and budget so initialization, spills, and guards all
	// preserve exact scalar bits, including opaque handles and NaN payloads.
	p := &ir.Program{Locals: 24}
	for slot := 0; slot < p.Locals; slot++ {
		p.Code = append(p.Code, ir.Instruction{Op: ir.Swap, Dest: slot, Extra: (slot + 1) % p.Locals})
	}
	p.Code = append(p.Code,
		ir.Instruction{Op: ir.CopyPair, Left: ir.Slot(0), Right: ir.Slot(14), Dest: 14, Extra: 0},
		ir.Instruction{Op: ir.StoreLoad, Left: ir.Slot(15), Right: ir.Slot(0), Dest: 0, Extra: 15},
		ir.Instruction{Op: ir.Update, Operator: ir.Add, Left: ir.Slot(0), Dest: 0, Extra: 15, Postfix: true},
		ir.Instruction{Op: ir.Return, Left: ir.Slot(15)},
	)
	p.Maps = make([]ir.StateMap, len(p.Code))
	for pc := range p.Maps {
		p.Maps[pc].PC = uint32(pc)
	}
	c := newTestCode(t, p)
	values := []ir.Value{
		ir.Float(0), ir.Float(math.Copysign(0, -1)), ir.Float(math.Inf(1)),
		{Kind: ir.Number, Bits: 0x7ff0000000000123},
		{Kind: ir.Number, Bits: 0xfff8000000004567},
		ir.Bool(true), {Kind: ir.Opaque, Bits: 123}, {Kind: ir.Uninitialized},
	}
	for pc := range p.Code {
		for budget := uint64(0); budget <= uint64(len(p.Code)+1); budget++ {
			oracle := make([]ir.Value, p.Locals)
			for slot := range oracle {
				oracle[slot] = values[slot%len(values)]
			}
			slots := append([]ir.Value(nil), oracle...)
			want, err := p.Evaluate(oracle, pc, budget)
			if err != nil {
				t.Fatal(err)
			}
			got, err := c.Run(slots, pc, budget)
			if err != nil || got != want {
				t.Fatalf("pc %d budget %d: native %+v, %v; oracle %+v", pc, budget, got, err, want)
			}
			for slot := range slots {
				if slots[slot] != oracle[slot] {
					t.Fatalf("pc %d budget %d slot %d: native %+v; oracle %+v", pc, budget, slot, slots[slot], oracle[slot])
				}
			}
		}
	}
}

func TestNativeProgramComparisons(t *testing.T) {
	values := []float64{math.Inf(-1), -1, math.Copysign(0, -1), 0, 1, math.Inf(1), math.NaN()}
	for op := ir.Lt; op <= ir.Ne; op++ {
		for _, branch := range []bool{false, true} {
			for _, when := range []bool{false, true} {
				p := &ir.Program{Locals: 3,
					Code: []ir.Instruction{
						{Op: ir.Binary, Operator: op, Left: ir.Slot(0), Right: ir.Slot(1), Dest: 2},
						{Op: ir.Return, Left: ir.Slot(2)},
						{Op: ir.Return, Left: ir.Literal(ir.Bool(true))},
					}, Maps: []ir.StateMap{{PC: 0}, {PC: 1}, {PC: 2}},
				}
				if branch {
					p.Code[0] = ir.Instruction{Op: ir.Branch, Operator: op, Left: ir.Slot(0), Right: ir.Slot(1), Target: 2, When: when}
					p.Code[1].Left = ir.Literal(ir.Bool(false))
				} else {
					p.Maps[2].Depth = -1
				}
				c := newTestCode(t, p)
				for _, x := range values {
					for _, y := range values {
						oracle := []ir.Value{ir.Float(x), ir.Float(y), ir.Bool(false)}
						slots := append([]ir.Value(nil), oracle...)
						want, err := p.Evaluate(oracle, 0, 2)
						if err != nil {
							t.Fatal(err)
						}
						got, err := c.Run(slots, 0, 2)
						if err != nil || got != want || slots[2] != oracle[2] {
							t.Fatalf("op %d branch %v when %v inputs %g,%g: native %+v, %v; oracle %+v", op, branch, when, x, y, got, err, want)
						}
					}
				}
			}
		}
	}
}

func TestNativeProgramArithmeticAliases(t *testing.T) {
	for op := ir.Add; op <= ir.Div; op++ {
		for _, left := range []int{0, 12, 20} {
			right := left + 1
			for _, dest := range []int{left, right, 23} {
				p := &ir.Program{Locals: 24}
				for slot := 0; slot < p.Locals; slot++ {
					p.Code = append(p.Code, ir.Instruction{Op: ir.Copy, Left: ir.Slot(slot), Dest: slot})
				}
				// Put the middle inputs in extended FP registers, and the last
				// pair outside the pool, by making earlier slots more frequent.
				prime := 0
				if left == 12 {
					prime = 6
				} else if left == 20 {
					prime = 14
				}
				for slot := 0; slot < prime; slot++ {
					for range 4 {
						p.Code = append(p.Code, ir.Instruction{Op: ir.Copy, Left: ir.Slot(slot), Dest: slot})
					}
				}
				p.Code = append(p.Code,
					ir.Instruction{Op: ir.Binary, Operator: op, Left: ir.Slot(left), Right: ir.Slot(right), Dest: dest},
					ir.Instruction{Op: ir.Return, Left: ir.Slot(dest)},
				)
				p.Maps = make([]ir.StateMap, len(p.Code))
				for pc := range p.Maps {
					p.Maps[pc].PC = uint32(pc)
				}
				c := newTestCode(t, p)
				oracle := make([]ir.Value, p.Locals)
				for slot := range oracle {
					oracle[slot] = ir.Float(float64(slot + 2))
				}
				slots := append([]ir.Value(nil), oracle...)
				want, err := p.Evaluate(oracle, 0, MaxIterations)
				if err != nil {
					t.Fatal(err)
				}
				got, err := c.Run(slots, 0, MaxIterations)
				if err != nil || got != want {
					t.Fatalf("op %d left %d right %d dest %d: native %+v, %v; oracle %+v", op, left, right, dest, got, err, want)
				}
				for slot := range slots {
					if slots[slot] != oracle[slot] {
						t.Fatalf("op %d left %d right %d dest %d slot %d: native %+v; oracle %+v", op, left, right, dest, slot, slots[slot], oracle[slot])
					}
				}
			}
		}
	}
}

//go:noinline
func runProgramGrowingStack(c *Code, slots []ir.Value, depth int) (uint64, error) {
	var padding [128]uint64
	for i := range padding {
		padding[i] = uint64(i + depth)
	}
	var steps uint64
	if depth != 0 {
		var err error
		steps, err = runProgramGrowingStack(c, slots, depth-1)
		if err != nil {
			return 0, err
		}
	} else {
		exit, err := c.Run(slots, 0, MaxIterations)
		if err != nil || exit.Kind != ir.BudgetExit || exit.State.PC != uint32(MaxIterations%uint64(len(c.maps))) || exit.Steps != MaxIterations {
			return 0, fmt.Errorf("bounded native loop: %+v, %v", exit, err)
		}
		steps = exit.Steps
	}
	for _, value := range padding {
		steps += value
	}
	return steps, nil
}

func TestNativeProgramBoundary(t *testing.T) {
	// Exercise the full register pool, including cached kinds, while Go grows
	// and scans stacks. Copies preserve handles and NaN payloads exactly.
	p := &ir.Program{Locals: 24}
	for slot := 0; slot < p.Locals; slot++ {
		p.Code = append(p.Code, ir.Instruction{Op: ir.Copy, Left: ir.Slot(slot), Dest: slot})
	}
	p.Code = append(p.Code, ir.Instruction{Op: ir.Jump, Target: 0})
	p.Maps = make([]ir.StateMap, len(p.Code))
	for pc := range p.Maps {
		p.Maps[pc].PC = uint32(pc)
	}
	c := newTestCode(t, p)
	slots := make([]ir.Value, p.Locals)
	for slot := range slots {
		slots[slot] = ir.Float(float64(slot))
	}
	slots[1] = ir.Value{Kind: ir.Opaque, Bits: 123}
	slots[2] = ir.Value{Kind: ir.Number, Bits: 0x7ff0000000004567}
	wantSlots := append([]ir.Value(nil), slots...)
	for _, procs := range []int{1, 2} {
		t.Run(fmt.Sprint(procs), func(t *testing.T) {
			previous := runtime.GOMAXPROCS(procs)
			defer runtime.GOMAXPROCS(previous)
			stop, done := make(chan struct{}), make(chan struct{})
			go func() {
				defer close(done)
				for {
					select {
					case <-stop:
						return
					default:
						runtime.GC()
						var stacks [16 << 10]byte
						runtime.Stack(stacks[:], true)
					}
				}
			}()
			defer func() { close(stop); <-done }()
			want := MaxIterations
			for depth := 0; depth <= 64; depth++ {
				for i := 0; i < 128; i++ {
					want += uint64(i + depth)
				}
			}
			for i := 0; i < 128; i++ {
				got, err := runProgramGrowingStack(c, slots, 64)
				if err != nil || got != want {
					t.Fatalf("native stack growth: %d, %v; want %d", got, err, want)
				}
				for slot := range slots {
					if slots[slot] != wantSlots[slot] {
						t.Fatalf("native stack growth changed slot %d: %+v; want %+v", slot, slots[slot], wantSlots[slot])
					}
				}
				if i%8 == 0 {
					runtime.Gosched()
				}
			}
		})
	}
}

func TestNativeProgramIndependentOwners(t *testing.T) {
	p := constantProgram()
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 32 {
				c, err := Compile(p)
				if err != nil {
					t.Error(err)
					return
				}
				got, err := c.Run(nil, 0, 1)
				closeErr := c.Close()
				if err != nil || closeErr != nil || got.Kind != ir.Returned || got.Value != ir.Float(42) {
					t.Errorf("native owner: %+v, %v, %v", got, err, closeErr)
					return
				}
			}
		}()
	}
	wg.Wait()
}

func FuzzNativeProgram(f *testing.F) {
	for _, code := range [][]bytecode.Instr{
		{{Op: bytecode.OpReturnUndef}},
		{{Op: bytecode.OpGetLocal2, B: 1}, {Op: bytecode.OpAdd}, {Op: bytecode.OpReturn}},
		{{Op: bytecode.OpGetLocal2, B: 1}, {Op: bytecode.OpBitAnd}, {Op: bytecode.OpReturn}},
		{{Op: bytecode.OpGetLocal2, B: 1}, {Op: bytecode.OpBitOr}, {Op: bytecode.OpReturn}},
		{{Op: bytecode.OpGetLocal2, B: 1}, {Op: bytecode.OpBitXor}, {Op: bytecode.OpReturn}},
		{{Op: bytecode.OpGetLocal2, B: 1}, {Op: bytecode.OpShl}, {Op: bytecode.OpReturn}},
		{{Op: bytecode.OpGetLocal2, B: 1}, {Op: bytecode.OpShr}, {Op: bytecode.OpReturn}},
		{{Op: bytecode.OpGetLocal2, B: 1}, {Op: bytecode.OpUShr}, {Op: bytecode.OpReturn}},
		{{Op: bytecode.OpGetLocal}, {Op: bytecode.OpBitNot}, {Op: bytecode.OpReturn}},
		{{Op: bytecode.OpJump}},
	} {
		data := make([]byte, 16+9*len(code))
		binary.LittleEndian.PutUint64(data, math.Float64bits(2))
		binary.LittleEndian.PutUint64(data[8:], math.Float64bits(3))
		for pc, in := range code {
			data[16+pc*9] = byte(in.Op)
			binary.LittleEndian.PutUint32(data[16+pc*9+1:], in.A)
			binary.LittleEndian.PutUint32(data[16+pc*9+5:], in.B)
		}
		f.Add(data)
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 16 || len(data) > 512 {
			return
		}
		fn := &bytecode.Function{LocalCount: 2, Locals: make([]bytecode.LocalDesc, 2), MaxStack: 4, HasSimpleParams: true,
			Constants: []bytecode.Constant{{Kind: bytecode.ConstNumber, Num: math.NaN()}}}
		for i := 16; i+9 <= len(data); i += 9 {
			fn.Code = append(fn.Code, bytecode.Instr{Op: bytecode.Op(data[i]), A: binary.LittleEndian.Uint32(data[i+1:]), B: binary.LittleEndian.Uint32(data[i+5:])})
		}
		p, err := jitcompile.Lower(fn)
		if err != nil {
			return
		}
		c := newTestCode(t, p)
		oracle := make([]ir.Value, p.Locals+p.StackSize)
		oracle[0] = ir.Value{Kind: ir.Number, Bits: binary.LittleEndian.Uint64(data)}
		oracle[1] = ir.Value{Kind: ir.Number, Bits: binary.LittleEndian.Uint64(data[8:])}
		slots := append([]ir.Value(nil), oracle...)
		want, err := p.Evaluate(oracle, 0, 64)
		if err != nil {
			t.Fatal(err)
		}
		var got ir.Exit
		if len(data)&1 == 0 {
			got, err = c.RunEncodedArrays(slots, nil, 0, 64)
		} else {
			got, err = c.Run(slots, 0, 64)
		}
		same := func(a, b ir.Value) bool {
			return a == b || a.Kind == ir.Number && b.Kind == ir.Number && sameFloat(math.Float64frombits(a.Bits), math.Float64frombits(b.Bits))
		}
		if err != nil || got.Kind != want.Kind || got.State != want.State || got.Steps != want.Steps || !same(got.Value, want.Value) {
			t.Fatalf("native %+v, %v; oracle %+v", got, err, want)
		}
		for i := range slots {
			if !same(slots[i], oracle[i]) {
				t.Fatalf("native slot %d %+v, oracle %+v", i, slots[i], oracle[i])
			}
		}
	})
}
