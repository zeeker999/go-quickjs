//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package jit

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"runtime"
	"testing"
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

func dispatchTestProgram(locals int, instructions ...ir.Instruction) *ir.Program {
	p := &ir.Program{Locals: locals, Code: instructions}
	for pc := range instructions {
		p.Maps = append(p.Maps, ir.StateMap{PC: uint32(pc)})
	}
	return p
}

func dispatchTestCode(t testing.TB, p *ir.Program) *Code {
	t.Helper()
	c, err := CompileDispatch(p)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := c.Close(); err != nil {
			t.Error(err)
		}
	})
	return c
}

func TestNativeDispatchABI(t *testing.T) {
	var s dispatchState
	var p nativeDispatchProgram
	var f nativeDispatchFrame
	var call nativeDispatchSite
	if unsafe.Sizeof(s) != 128 || unsafe.Offsetof(s.depth) != 40 || unsafe.Offsetof(s.programCount) != 48 ||
		unsafe.Offsetof(s.ticks) != 56 || unsafe.Offsetof(s.stackUsed) != 64 || unsafe.Offsetof(s.stackLimit) != 72 ||
		unsafe.Offsetof(s.calls) != 80 || unsafe.Offsetof(s.frameLimit) != 88 || unsafe.Offsetof(s.programs) != 96 ||
		unsafe.Offsetof(s.frames) != 104 || unsafe.Offsetof(s.slots) != 112 || unsafe.Offsetof(s.sites) != 120 {
		t.Fatal("native dispatch ABI changed")
	}
	if unsafe.Sizeof(p) != 64 || unsafe.Offsetof(p.code) != 8 || unsafe.Offsetof(p.entries) != 16 ||
		unsafe.Offsetof(p.template) != 24 || unsafe.Offsetof(p.slots) != 32 || unsafe.Offsetof(p.params) != 40 ||
		unsafe.Offsetof(p.charge) != 48 || unsafe.Offsetof(p.sites) != 56 {
		t.Fatal("native program descriptor ABI changed")
	}
	if unsafe.Sizeof(f) != 128 || unsafe.Offsetof(f.pc) != 8 || unsafe.Offsetof(f.result) != 16 ||
		unsafe.Offsetof(f.argSlot) != 24 || unsafe.Offsetof(f.argc) != 32 || unsafe.Offsetof(f.charge) != 40 ||
		unsafe.Offsetof(f.steps) != 48 || unsafe.Offsetof(f.lastBudget) != 56 || unsafe.Offsetof(f.invocation) != 64 {
		t.Fatal("native frame ABI changed")
	}
	if unsafe.Sizeof(call) != 128 || unsafe.Offsetof(call.functionSlot) != 8 || unsafe.Offsetof(call.argSlot) != 16 ||
		unsafe.Offsetof(call.argc) != 24 || unsafe.Offsetof(call.result) != 32 || unsafe.Offsetof(call.receiverSlot) != 40 ||
		unsafe.Offsetof(call.function) != 48 || unsafe.Offsetof(call.receiver) != 64 {
		t.Fatal("native call-site ABI changed")
	}
}

type dispatchOracleFrame struct {
	program, pc, result, argSlot, argc int
	slots                              []ir.Value
	charge, steps                      uint64
}

// The evaluator supplies operation semantics. This independent coordinator
// models calls from the source IR and test grants, not native descriptor bytes.
func dispatchOracle(t *testing.T, programs []*ir.Program, graph []DispatchProgram, root []ir.Value, budget, ticks, stack uint64, limit int) (DispatchExit, []dispatchOracleFrame) {
	t.Helper()
	frames := []dispatchOracleFrame{{slots: append([]ir.Value(nil), root...)}}
	remaining, calls, used := budget, uint64(0), uint64(0)
	for {
		top := len(frames) - 1
		f := &frames[top]
		p := programs[f.program]
		exit, err := p.Evaluate(f.slots, f.pc, remaining)
		if err != nil {
			t.Fatal(err)
		}
		remaining -= exit.Steps
		f.steps += exit.Steps
		f.pc = int(exit.State.PC)
		if exit.Kind == ir.Returned && top != 0 {
			value, result, charge, steps := exit.Value, f.result, f.charge, f.steps
			frames = frames[:top]
			parent := &frames[top-1]
			parent.slots[result] = value
			parent.steps += steps
			used -= charge
			continue
		}
		if exit.Kind == ir.HostExit && p.Code[f.pc].Op == ir.Call {
			in := p.Code[f.pc]
			var grant *DispatchCall
			for i := range graph[f.program].Calls {
				candidate := &graph[f.program].Calls[i]
				if candidate.PC == f.pc {
					grant = candidate
					break
				}
			}
			if grant != nil && f.slots[in.Left.Slot] == grant.Function &&
				(in.Right.Slot < 0 || f.slots[in.Right.Slot] == grant.Receiver) &&
				ticks != 0 && len(frames) < limit && graph[grant.Target].StackCharge <= stack-used {
				target := graph[grant.Target]
				child := dispatchOracleFrame{program: grant.Target, slots: append([]ir.Value(nil), target.Template...), result: in.Dest, argSlot: in.Third.Slot, argc: in.Extra, charge: target.StackCharge}
				copy(child.slots[:min(target.Params, in.Extra)], f.slots[in.Third.Slot:in.Third.Slot+min(target.Params, in.Extra)])
				f.pc++
				f.steps++
				frames = append(frames, child)
				remaining--
				ticks--
				calls++
				used += target.StackCharge
				continue
			}
		}
		exit.Steps = budget - remaining
		return DispatchExit{Exit: exit, Program: f.program, Depth: top, Calls: calls, Ticks: ticks, StackUsed: used}, frames
	}
}

func TestNativeDispatchCalls(t *testing.T) {
	function := ir.Value{Kind: ir.Opaque, Bits: 17}
	receiver := ir.Value{Kind: ir.Opaque, Bits: 3}
	for _, method := range []bool{false, true} {
		this := ir.Literal(ir.Value{})
		if method {
			this = ir.Slot(0)
		}
		root := dispatchTestProgram(4,
			ir.Instruction{Op: ir.Call, Left: ir.Slot(1), Right: this, Third: ir.Slot(2), Extra: 1, Dest: 3},
			ir.Instruction{Op: ir.Binary, Operator: ir.Add, Left: ir.Slot(3), Right: ir.Literal(ir.Float(10)), Dest: 3},
			ir.Instruction{Op: ir.Return, Left: ir.Slot(3)},
		)
		callee := dispatchTestProgram(2,
			ir.Instruction{Op: ir.Binary, Operator: ir.Add, Left: ir.Slot(0), Right: ir.Literal(ir.Float(1)), Dest: 1},
			ir.Instruction{Op: ir.Return, Left: ir.Slot(1)},
		)
		programs := []*ir.Program{root, callee}
		graph := []DispatchProgram{
			{Code: dispatchTestCode(t, root), Template: make([]ir.Value, 4), Calls: []DispatchCall{{PC: 0, Target: 1, Function: function, Receiver: receiver}}},
			{Code: dispatchTestCode(t, callee), Template: make([]ir.Value, 2), Params: 1, StackCharge: 2},
		}
		var d Dispatch
		if err := d.Prepare(graph); err != nil {
			t.Fatal(err)
		}
		for _, wrong := range []string{"none", "function", "kind", "receiver"} {
			for budget := uint64(0); budget <= 7; budget++ {
				for _, ticks := range []uint64{0, 1, 8} {
					for _, limit := range []int{1, 2, 8} {
						for _, stack := range []uint64{0, 1, 2, 16} {
							slots := []ir.Value{receiver, function, ir.Float(7), ir.Float(-1)}
							switch wrong {
							case "function":
								slots[1].Bits++
							case "kind":
								slots[1].Kind = ir.Number
							case "receiver":
								slots[0].Bits++
							}
							want, frames := dispatchOracle(t, programs, graph, slots, budget, ticks, stack, limit)
							if err := d.Begin(0, 0, slots, limit, stack, ticks); err != nil {
								t.Fatal(err)
							}
							got, err := d.Run(nil, budget)
							if err != nil || got != want {
								t.Fatalf("method %v %s budget %d ticks %d stack %d limit %d: %+v/%+v error %v", method, wrong, budget, ticks, stack, limit, got, want, err)
							}
							for i, expected := range frames {
								f, values, err := d.Frame(i)
								if err != nil || f.Program != expected.program || f.PC != expected.pc || f.Steps != expected.steps || !reflect.DeepEqual(values, expected.slots) {
									t.Fatalf("frame %d: %+v values %v; want %+v; error %v", i, f, values, expected, err)
								}
							}
							if want.Calls != 0 && frames[0].steps != 0 && budget >= 5 && got.Exit.Kind == ir.Returned && got.Exit.Value != ir.Float(18) {
								t.Fatal(got)
							}
						}
					}
				}
			}
		}
		// A dispatch-compiled program retains the ordinary host-only interface.
		got, err := graph[0].Code.Run([]ir.Value{receiver, function, ir.Float(7), {}}, 0, 8)
		if err != nil || got.Kind != ir.HostExit || got.Steps != 0 || got.State.PC != 0 {
			t.Fatal(got, err)
		}
	}
}

func TestNativeDispatchHostAndResume(t *testing.T) {
	function := ir.Value{Kind: ir.Opaque, Bits: 17}
	root := dispatchTestProgram(3,
		ir.Instruction{Op: ir.Call, Left: ir.Slot(0), Right: ir.Literal(ir.Value{}), Third: ir.Slot(1), Extra: 1, Dest: 2},
		ir.Instruction{Op: ir.Return, Left: ir.Slot(2)},
	)
	callee := dispatchTestProgram(1,
		ir.Instruction{Op: ir.Host},
		ir.Instruction{Op: ir.Return, Left: ir.Slot(0)},
	)
	graph := []DispatchProgram{
		{Code: dispatchTestCode(t, root), Template: make([]ir.Value, 3), Calls: []DispatchCall{{PC: 0, Target: 1, Function: function}}},
		{Code: dispatchTestCode(t, callee), Template: make([]ir.Value, 1), Params: 1, StackCharge: 9},
	}
	var d Dispatch
	if err := d.Prepare(graph); err != nil {
		t.Fatal(err)
	}
	if err := d.Begin(0, 0, []ir.Value{function, ir.Value{Kind: ir.Opaque, Bits: 4}, {}}, 8, 9, 8); err != nil {
		t.Fatal(err)
	}
	exit, err := d.Run(nil, 8)
	if err != nil || exit.Depth != 1 || exit.Program != 1 || exit.Calls != 1 || exit.Exit.Kind != ir.HostExit || exit.Exit.Steps != 1 || exit.StackUsed != 9 {
		t.Fatal(exit, err)
	}
	f, slots, err := d.Frame(1)
	if err != nil || f.ArgSlot != 1 || f.ArgCount != 1 || f.ResultSlot != 2 || f.Invocation != 1 || slots[0].Bits != 4 {
		t.Fatal(f, slots, err)
	}
	runtime.GC()
	if err := d.Resume(1); err != nil {
		t.Fatal(err)
	}
	exit, err = d.Run(nil, 2)
	if err != nil || exit.Exit.Kind != ir.Returned || exit.Exit.Value != (ir.Value{Kind: ir.Opaque, Bits: 4}) || exit.Exit.Steps != 2 || exit.Depth != 0 || exit.StackUsed != 0 {
		t.Fatal(exit, err)
	}
	d.Reset()
	if _, _, err := d.Frame(0); !errors.Is(err, ir.ErrState) {
		t.Fatal(err)
	}
	if _, err := d.Run(nil, 1); !errors.Is(err, ir.ErrState) {
		t.Fatal(err)
	}
}

func TestNativeDispatchNestedAndRepeated(t *testing.T) {
	a, b := ir.Value{Kind: ir.Opaque, Bits: 17}, ir.Value{Kind: ir.Opaque, Bits: 19}
	root := dispatchTestProgram(4,
		ir.Instruction{Op: ir.Call, Left: ir.Slot(0), Right: ir.Literal(ir.Value{}), Third: ir.Slot(2), Extra: 1, Dest: 3},
		ir.Instruction{Op: ir.Call, Left: ir.Slot(1), Right: ir.Literal(ir.Value{}), Third: ir.Slot(3), Extra: 1, Dest: 2, Key: 1},
		ir.Instruction{Op: ir.Return, Left: ir.Slot(2)},
	)
	one := dispatchTestProgram(3,
		ir.Instruction{Op: ir.Call, Left: ir.Slot(1), Right: ir.Literal(ir.Value{}), Third: ir.Slot(0), Extra: 1, Dest: 2},
		ir.Instruction{Op: ir.Binary, Operator: ir.Add, Left: ir.Slot(2), Right: ir.Literal(ir.Float(1)), Dest: 2},
		ir.Instruction{Op: ir.Return, Left: ir.Slot(2)},
	)
	two := dispatchTestProgram(2,
		ir.Instruction{Op: ir.Binary, Operator: ir.Mul, Left: ir.Slot(0), Right: ir.Literal(ir.Float(2)), Dest: 1},
		ir.Instruction{Op: ir.Return, Left: ir.Slot(1)},
	)
	programs := []*ir.Program{root, one, two}
	graph := []DispatchProgram{
		{Code: dispatchTestCode(t, root), Template: make([]ir.Value, 4), Calls: []DispatchCall{{PC: 0, Target: 1, Function: a}, {PC: 1, Target: 2, Function: b}}},
		{Code: dispatchTestCode(t, one), Template: []ir.Value{{}, b, {}}, Params: 1, StackCharge: 3, Calls: []DispatchCall{{PC: 0, Target: 2, Function: b}}},
		{Code: dispatchTestCode(t, two), Template: make([]ir.Value, 2), Params: 1, StackCharge: 2},
	}
	var d Dispatch
	if err := d.Prepare(graph); err != nil {
		t.Fatal(err)
	}
	for budget := uint64(0); budget < 13; budget++ {
		for ticks := uint64(0); ticks < 5; ticks++ {
			for limit := 1; limit <= MaxDispatchFrames; limit++ {
				for stack := uint64(0); stack < 7; stack++ {
					slots := []ir.Value{a, b, ir.Float(7), {}}
					want, frames := dispatchOracle(t, programs, graph, slots, budget, ticks, stack, limit)
					if err := d.Begin(0, 0, slots, limit, stack, ticks); err != nil {
						t.Fatal(err)
					}
					got, err := d.Run(nil, budget)
					if err != nil || got != want {
						t.Fatalf("nested budget %d ticks %d limit %d stack %d: %+v/%+v, %v", budget, ticks, limit, stack, got, want, err)
					}
					for i, f := range frames {
						actual, values, err := d.Frame(i)
						if err != nil || actual.Program != f.program || actual.PC != f.pc || actual.Steps != f.steps || !reflect.DeepEqual(values, f.slots) {
							t.Fatalf("nested frame %d: %+v %v, want %+v, %v", i, actual, values, f, err)
						}
					}
				}
			}
		}
	}
}

func TestNativeDispatchRecursionLimit(t *testing.T) {
	fn := ir.Value{Kind: ir.Opaque, Bits: 1}
	p := dispatchTestProgram(2,
		ir.Instruction{Op: ir.Call, Left: ir.Slot(0), Right: ir.Literal(ir.Value{}), Third: ir.Slot(1), Dest: 1},
		ir.Instruction{Op: ir.Return, Left: ir.Slot(1)},
	)
	graph := []DispatchProgram{{Code: dispatchTestCode(t, p), Template: []ir.Value{fn, {}}, StackCharge: 2, Calls: []DispatchCall{{PC: 0, Target: 0, Function: fn}}}}
	var d Dispatch
	if err := d.Prepare(graph); err != nil {
		t.Fatal(err)
	}
	for limit := 1; limit <= MaxDispatchFrames; limit++ {
		if err := d.Begin(0, 0, []ir.Value{fn, {}}, limit, 100, 100); err != nil {
			t.Fatal(err)
		}
		exit, err := d.Run(nil, 100)
		if err != nil || exit.Exit.Kind != ir.HostExit || exit.Depth != limit-1 || exit.Calls != uint64(limit-1) || exit.Exit.Steps != uint64(limit-1) || exit.StackUsed != 2*uint64(limit-1) {
			t.Fatal(limit, exit, err)
		}
	}
}

func TestNativeDispatchDenialsBeforeEffects(t *testing.T) {
	p := dispatchTestProgram(1,
		ir.Instruction{Op: ir.Copy, Left: ir.Literal(ir.Float(9)), Dest: 0},
		ir.Instruction{Op: ir.Return, Left: ir.Slot(0)},
	)
	c := dispatchTestCode(t, p)
	var d Dispatch
	graph := []DispatchProgram{{Code: c, Template: []ir.Value{{}}}}
	if err := d.Prepare(graph); err != nil {
		t.Fatal(err)
	}
	if err := d.Begin(0, 0, []ir.Value{ir.Float(3)}, 8, 8, 8); err != nil {
		t.Fatal(err)
	}
	for _, invalid := range []DispatchProgram{
		{Code: c, Template: []ir.Value{{Kind: ir.Boolean, Bits: 2}}},
		{Code: c, Template: []ir.Value{{}}, Params: 2},
		{Code: c, Template: []ir.Value{{}}, Calls: []DispatchCall{{PC: 0, Target: 0, Function: ir.Value{Kind: ir.Opaque}}}},
	} {
		if err := d.Prepare([]DispatchProgram{invalid}); !errors.Is(err, ir.ErrState) {
			t.Fatal(err)
		}
	}
	for _, invalid := range []struct {
		program, pc, limit int
		slots              []ir.Value
	}{
		{0, 2, 8, []ir.Value{ir.Float(3)}}, {0, 0, 0, []ir.Value{ir.Float(3)}},
		{0, 0, 9, []ir.Value{ir.Float(3)}}, {1, 0, 8, []ir.Value{ir.Float(3)}},
		{0, 0, 8, nil}, {0, 0, 8, []ir.Value{{Kind: ir.Boolean, Bits: 2}}},
	} {
		if err := d.Begin(invalid.program, invalid.pc, invalid.slots, invalid.limit, 8, 8); !errors.Is(err, ir.ErrState) {
			t.Fatal(err)
		}
	}
	if _, err := d.Run(nil, MaxIterations+1); !errors.Is(err, ErrIterations) {
		t.Fatal(err)
	}
	if _, err := d.Run(make([]ir.ArrayView, 1), 8); !errors.Is(err, ir.ErrState) {
		t.Fatal(err)
	}
	_, values, err := d.Frame(0)
	if err != nil || values[0] != ir.Float(3) {
		t.Fatal(values, err)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := d.Run(nil, 8); !errors.Is(err, ErrClosed) || values[0] != ir.Float(3) {
		t.Fatal(values, err)
	}
}

func TestNativeDispatchCommittedArrays(t *testing.T) {
	function := ir.Value{Kind: ir.Opaque, Bits: 1}
	root := dispatchTestProgram(4,
		ir.Instruction{Op: ir.ArrayWrite, Left: ir.Slot(0), Right: ir.Literal(ir.Float(0)), Third: ir.Literal(ir.Float(41))},
		ir.Instruction{Op: ir.Call, Left: ir.Slot(1), Right: ir.Literal(ir.Value{}), Third: ir.Slot(0), Extra: 1, Dest: 3},
		ir.Instruction{Op: ir.Host},
		ir.Instruction{Op: ir.Return, Left: ir.Slot(3)},
	)
	for _, guard := range []bool{false, true} {
		callee := dispatchTestProgram(2,
			ir.Instruction{Op: ir.ArrayRead, Left: ir.Slot(0), Right: ir.Literal(ir.Float(0)), Dest: 1},
			ir.Instruction{Op: ir.Host},
			ir.Instruction{Op: ir.Return, Left: ir.Slot(1)},
		)
		if guard {
			callee.Code[1] = ir.Instruction{Op: ir.Binary, Operator: ir.Add, Left: ir.Slot(0), Right: ir.Literal(ir.Float(1)), Dest: 1}
		}
		graph := []DispatchProgram{
			{Code: dispatchTestCode(t, root), Template: make([]ir.Value, 4), Calls: []DispatchCall{{PC: 1, Target: 1, Function: function}}},
			{Code: dispatchTestCode(t, callee), Template: make([]ir.Value, 2), Params: 1},
		}
		for budget := uint64(1); budget <= 8; budget++ {
			var d Dispatch
			if err := d.Prepare(graph); err != nil {
				t.Fatal(err)
			}
			if err := d.Begin(0, 0, []ir.Value{{Kind: ir.Opaque}, function, {}, {}}, 8, 8, 8); err != nil {
				t.Fatal(err)
			}
			cell := struct {
				bits uint64
				ref  unsafe.Pointer
			}{bits: math.Float64bits(3)}
			views := make([]ir.ArrayView, ir.MaxSlots)
			views[0] = ir.ArrayView{Data: unsafe.Pointer(&cell), DenseLength: 1, Length: 1, NumberLimit: 0xfff8000000000000}
			var exit DispatchExit
			var steps uint64
			for batches := 0; batches < 8; batches++ {
				var err error
				exit, err = d.Run(views, budget)
				if err != nil {
					t.Fatal(err)
				}
				steps += exit.Exit.Steps
				if exit.Exit.Kind != ir.BudgetExit {
					break
				}
			}
			wantKind := ir.HostExit
			if guard {
				wantKind = ir.GuardExit
			}
			if exit.Exit.Kind != wantKind || exit.Depth != 1 || exit.Exit.State.PC != 1 || steps != 3 || cell.bits != math.Float64bits(41) {
				t.Fatal(guard, budget, exit, steps, cell)
			}
			cell.bits = math.Float64bits(99)
			if err := d.Resume(2); err != nil {
				t.Fatal(err)
			}
			exit, err := d.Run(views, 8)
			if err != nil || exit.Exit.Kind != ir.HostExit || exit.Depth != 0 || exit.Exit.State.PC != 2 || cell.bits != math.Float64bits(99) {
				t.Fatal(exit, err, cell)
			}
			if err := d.Resume(3); err != nil {
				t.Fatal(err)
			}
			exit, err = d.Run(views, 8)
			if err != nil || exit.Exit.Kind != ir.Returned || exit.Exit.Value != ir.Float(41) || cell.bits != math.Float64bits(99) {
				t.Fatal(exit, err, cell)
			}
		}
	}
}

func TestNativeDispatchRestore(t *testing.T) {
	function := ir.Value{Kind: ir.Opaque, Bits: 1}
	root := dispatchTestProgram(4,
		ir.Instruction{Op: ir.Call, Left: ir.Slot(0), Right: ir.Literal(ir.Value{}), Third: ir.Slot(1), Extra: 2, Dest: 3},
		ir.Instruction{Op: ir.Call, Left: ir.Slot(0), Right: ir.Literal(ir.Value{}), Third: ir.Slot(1), Extra: 1, Dest: 3, Key: 1},
		ir.Instruction{Op: ir.Return, Left: ir.Slot(3)},
	)
	callee := dispatchTestProgram(1, ir.Instruction{Op: ir.Host}, ir.Instruction{Op: ir.Return, Left: ir.Slot(0)})
	constant := dispatchTestProgram(0, ir.Instruction{Op: ir.Return, Left: ir.Literal(ir.Float(93))})
	graph := []DispatchProgram{
		{Code: dispatchTestCode(t, root), Template: make([]ir.Value, 4), Calls: []DispatchCall{{PC: 0, Target: 1, Function: function}, {PC: 1, Target: 1, Function: function}}},
		{Code: dispatchTestCode(t, callee), Template: []ir.Value{{}}, Params: 1, StackCharge: 5},
	}
	oldOwners := []*Code{graph[0].Code, graph[1].Code}
	var d Dispatch
	if err := d.Prepare(graph); err != nil {
		t.Fatal(err)
	}
	if err := d.Begin(0, 0, []ir.Value{function, ir.Float(7), ir.Float(99), {}}, 8, 9, 8); err != nil {
		t.Fatal(err)
	}
	exit, err := d.Run(nil, 8)
	if err != nil || exit.Exit.Kind != ir.HostExit || exit.Depth != 1 {
		t.Fatal(exit, err)
	}
	snapshots := make([]DispatchSnapshot, 2)
	for i := range snapshots {
		f, values, err := d.Frame(i)
		if err != nil {
			t.Fatal(err)
		}
		snapshots[i] = DispatchSnapshot{Frame: f, Values: values}
	}
	snapshots[1].Frame.PC = 1
	// Restore is transactional even when replacing closed owners. An already
	// committed child can finish under a changed grant for subsequent calls.
	graph[0].Code = dispatchTestCode(t, root)
	graph[1].Code = dispatchTestCode(t, callee)
	graph = append(graph, DispatchProgram{Code: dispatchTestCode(t, constant), StackCharge: 2})
	graph[0].Calls[1].Target = 2
	for _, mutate := range []func([]DispatchSnapshot){
		func(s []DispatchSnapshot) { s[0].Frame.PC = 2 },
		func(s []DispatchSnapshot) { s[1].Frame.ResultSlot++ },
		func(s []DispatchSnapshot) { s[1].Frame.ArgCount++ },
		func(s []DispatchSnapshot) { s[1].Frame.Invocation = 0 },
		func(s []DispatchSnapshot) { s[1].Frame.Program = 3 },
		func(s []DispatchSnapshot) { s[1].Values = nil },
	} {
		invalid := append([]DispatchSnapshot(nil), snapshots...)
		mutate(invalid)
		if err := d.Restore(graph, invalid, 8, 9, 8, exit.Calls); !errors.Is(err, ir.ErrState) {
			t.Fatal(err)
		}
	}
	if err := d.Restore(graph, snapshots, 8, 4, 8, exit.Calls); !errors.Is(err, ir.ErrState) {
		t.Fatal(err)
	}
	for _, owner := range oldOwners {
		if err := owner.Close(); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := d.Run(nil, 8); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	if err := d.Restore(graph, snapshots, 8, 9, 8, exit.Calls); err != nil {
		t.Fatal(err)
	}
	runtime.GC()
	exit, err = d.Run(nil, 8)
	if err != nil || exit.Exit.Kind != ir.Returned || exit.Exit.Value != ir.Float(93) || exit.Calls != 2 || exit.Exit.Steps != 4 || exit.StackUsed != 0 {
		t.Fatal(exit, err)
	}
	f, _, err := d.Frame(0)
	if err != nil || f.Steps != 5 {
		t.Fatal(f, err)
	}
}

func TestNativeDispatchFreshArguments(t *testing.T) {
	function := ir.Value{Kind: ir.Opaque, Bits: 1}
	root := dispatchTestProgram(4,
		ir.Instruction{Op: ir.Call, Left: ir.Slot(0), Right: ir.Literal(ir.Value{}), Third: ir.Slot(1), Extra: 2, Dest: 3},
		ir.Instruction{Op: ir.Call, Left: ir.Slot(0), Right: ir.Literal(ir.Value{}), Third: ir.Slot(1), Dest: 3, Key: 1},
		ir.Instruction{Op: ir.Return, Left: ir.Slot(3)},
	)
	callee := dispatchTestProgram(2,
		ir.Instruction{Op: ir.Copy, Left: ir.Slot(0), Dest: 1},
		ir.Instruction{Op: ir.Copy, Left: ir.Literal(ir.Float(5)), Dest: 0},
		ir.Instruction{Op: ir.Return, Left: ir.Slot(1)},
	)
	graph := []DispatchProgram{
		{Code: dispatchTestCode(t, root), Template: make([]ir.Value, 4), Calls: []DispatchCall{{PC: 0, Target: 1, Function: function}, {PC: 1, Target: 1, Function: function}}},
		{Code: dispatchTestCode(t, callee), Template: []ir.Value{{}, {}}, Params: 1},
	}
	var d Dispatch
	if err := d.Prepare(graph); err != nil {
		t.Fatal(err)
	}
	if err := d.Begin(0, 0, []ir.Value{function, ir.Float(7), ir.Float(99), {}}, 8, 8, 8); err != nil {
		t.Fatal(err)
	}
	exit, err := d.Run(nil, 99)
	if err != nil || exit.Exit.Kind != ir.Returned || exit.Exit.Value != (ir.Value{}) || exit.Calls != 2 || exit.Exit.Steps != 9 {
		t.Fatal(exit, err)
	}
}

// This measures transfer overhead only, not JavaScript or complete Crypto.
func BenchmarkNativeDispatch(b *testing.B) {
	for _, locals := range []int{8, 32, 128} {
		b.Run(fmt.Sprint(locals), func(b *testing.B) {
			function := ir.Value{Kind: ir.Opaque, Bits: 1}
			root := dispatchTestProgram(4,
				ir.Instruction{Op: ir.Call, Left: ir.Slot(0), Right: ir.Literal(ir.Value{}), Third: ir.Slot(1), Extra: 1, Dest: 1},
				ir.Instruction{Op: ir.Binary, Operator: ir.Sub, Left: ir.Slot(2), Right: ir.Literal(ir.Float(1)), Dest: 2},
				ir.Instruction{Op: ir.Branch, Operator: ir.Gt, Left: ir.Slot(2), Right: ir.Literal(ir.Float(0)), Target: 0, When: true},
				ir.Instruction{Op: ir.Return, Left: ir.Slot(1)},
			)
			callee := dispatchTestProgram(locals,
				ir.Instruction{Op: ir.Binary, Operator: ir.Add, Left: ir.Slot(0), Right: ir.Literal(ir.Float(1)), Dest: 1},
				ir.Instruction{Op: ir.Return, Left: ir.Slot(1)},
			)
			graph := []DispatchProgram{
				{Code: dispatchTestCode(b, root), Template: make([]ir.Value, 4), Calls: []DispatchCall{{PC: 0, Target: 1, Function: function}}},
				{Code: dispatchTestCode(b, callee), Template: make([]ir.Value, locals), Params: 1},
			}
			const calls = 256
			b.Run("native", func(b *testing.B) {
				var d Dispatch
				if err := d.Prepare(graph); err != nil {
					b.Fatal(err)
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if err := d.Begin(0, 0, []ir.Value{function, ir.Float(0), ir.Float(calls), {}}, 8, 8, calls); err != nil {
						b.Fatal(err)
					}
					exit, err := d.Run(nil, calls*6)
					if err != nil || exit.Exit.Kind != ir.Returned || exit.Exit.Value != ir.Float(calls) {
						b.Fatal(exit, err)
					}
				}
			})
			b.Run("Go", func(b *testing.B) {
				ordinary := make([]*Code, 2)
				for i, p := range []*ir.Program{root, callee} {
					var err error
					ordinary[i], err = Compile(p)
					if err != nil {
						b.Fatal(err)
					}
					b.Cleanup(func() { _ = ordinary[i].Close() })
				}
				values := make([]ir.Value, locals)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					slots := []ir.Value{function, ir.Float(0), ir.Float(calls), {}}
					pc := 0
					budget := uint64(calls * 6)
					for {
						exit, err := ordinary[0].RunEncodedArrays(slots, nil, pc, budget)
						if err != nil {
							b.Fatal(err)
						}
						if exit.Kind == ir.Returned {
							if exit.Value != ir.Float(calls) {
								b.Fatal(exit)
							}
							break
						}
						if exit.Kind != ir.HostExit || exit.State.PC != 0 {
							b.Fatal(exit)
						}
						budget -= exit.Steps + 1
						clear(values)
						values[0] = slots[1]
						exit, err = ordinary[1].RunEncodedArrays(values, nil, 0, budget)
						if err != nil || exit.Kind != ir.Returned {
							b.Fatal(exit, err)
						}
						budget -= exit.Steps
						slots[1], pc = exit.Value, 1
					}
				}
			})
		})
	}
}
