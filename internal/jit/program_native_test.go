//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package jit

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
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

//go:noinline
func runProgramGrowingStack(c *Code, depth int) (uint64, error) {
	var padding [128]uint64
	for i := range padding {
		padding[i] = uint64(i + depth)
	}
	var steps uint64
	if depth != 0 {
		var err error
		steps, err = runProgramGrowingStack(c, depth-1)
		if err != nil {
			return 0, err
		}
	} else {
		exit, err := c.Run(nil, 0, MaxIterations)
		if err != nil || exit.Kind != ir.BudgetExit || exit.State.PC != 0 || exit.Steps != MaxIterations {
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
	p := &ir.Program{Code: []ir.Instruction{{Op: ir.Jump, Target: 0}}, Maps: []ir.StateMap{{PC: 0}}}
	c := newTestCode(t, p)
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
				got, err := runProgramGrowingStack(c, 64)
				if err != nil || got != want {
					t.Fatalf("native stack growth: %d, %v; want %d", got, err, want)
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
