//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package jit

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
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
	if unsafe.Sizeof(view) != 32 || unsafe.Offsetof(view.DenseLength) != 8 || unsafe.Offsetof(view.Length) != 16 || unsafe.Offsetof(view.NumberLimit) != 24 {
		t.Fatal("native array ABI changed")
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
	}
	if _, err := c.Run([]ir.Value{{}}, 0, 1); !errors.Is(err, ir.ErrState) {
		t.Fatal("accepted invalid scratch size")
	}
	if _, err := c.Run(nil, 0, MaxIterations+1); !errors.Is(err, ErrIterations) {
		t.Fatal("accepted excessive execution budget")
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
		got, err := c.Run(slots, 0, 64)
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
