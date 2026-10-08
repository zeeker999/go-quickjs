package compile

import (
	"encoding/binary"
	"errors"
	"math"
	"strings"
	"sync"
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/compiler"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
	"github.com/go-quickjs/go-quickjs/internal/parser"
)

func compiledFunction(t *testing.T, source string) *bytecode.Function {
	t.Helper()
	ast, err := parser.Parse(source, parser.Options{})
	if err != nil {
		t.Fatal(err)
	}
	fn, err := compiler.Compile(ast, compiler.Options{Text: source})
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range fn.Constants {
		if c.Kind == bytecode.ConstFunction {
			return c.Fn
		}
	}
	t.Fatal("missing function")
	return nil
}

func fixture(code ...bytecode.Instr) *bytecode.Function {
	return &bytecode.Function{Code: code, LocalCount: 2, Locals: make([]bytecode.LocalDesc, 2), MaxStack: 4, HasSimpleParams: true}
}

func TestNumericPropertySelection(t *testing.T) {
	fn := compiledFunction(t, `function f(o){let n=o.n;for(let i=0;i<n;i++)o.array[i]=o.array[i]+o.step;return o.array[0]}`)
	p, err := Lower(fn)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for pc, in := range fn.Code {
		if in.Op != bytecode.OpGetProp {
			continue
		}
		name := fn.Names[in.A]
		seen[name] = true
		want := ir.PropertyRead
		if name == "array" {
			want = ir.ReferenceRead
		}
		if p.Code[pc].Op != want || p.Code[pc].Key != in.A {
			t.Fatalf("property %s lowered to %+v, want %v", name, p.Code[pc], want)
		}
	}
	if !seen["array"] || !seen["n"] || !seen["step"] {
		t.Fatal("missing numeric or reference property fixture")
	}
}

func TestCallFieldSelection(t *testing.T) {
	fn := compiledFunction(t, `function f(o,cb,n){let s=0;for(let i=0;i<n;i++){s+=o.array[0]+o.step;cb()}return s}`)
	for _, lower := range []struct {
		fn     func(*bytecode.Function) (*ir.Program, error)
		fields bool
	}{{Lower, false}, {LowerCalls, true}} {
		p, err := lower.fn(fn)
		if err != nil {
			t.Fatal(err)
		}
		if err := p.Validate(); err != nil {
			t.Fatal(err)
		}
		for pc, in := range fn.Code {
			if in.Op != bytecode.OpGetProp {
				continue
			}
			want := ir.Host
			if lower.fields && fn.Names[in.A] == "step" {
				want = ir.PropertyRead
			}
			if p.Code[pc].Op != want {
				t.Fatalf("call field %s: got %v, want %v", fn.Names[in.A], p.Code[pc].Op, want)
			}
		}
	}
}

func TestSmallCalleeSelection(t *testing.T) {
	fn := compiledFunction(t, `function f(){return this.x<0?0:this}`)
	if _, err := Lower(fn); err == nil {
		t.Fatal("small field callee entered the standalone policy")
	}
	p, err := LowerCallee(fn)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	if !p.This {
		t.Fatal("callee lost its receiver snapshot")
	}
	found := false
	for _, in := range p.Code {
		found = found || in.Op == ir.PropertyRead
	}
	if !found {
		t.Fatal("small callee lost its guarded numeric field")
	}
}

func TestReferencePropertySelection(t *testing.T) {
	for _, tc := range []struct {
		source string
		want   ir.Op
	}{
		{`function f(o,n){let a=o.array;for(let i=0;i<n;i++)a[i]+=1;return a[0]}`, ir.Host},
		{`function f(o,n){let s=0;for(let i=0;i<n;i++)s+=o.array[0];return s}`, ir.ReferenceRead},
		{`function f(o,n,cb){let s=0;for(let i=0;i<n;i++){s+=o.array[0];cb()}return s}`, ir.Host},
		{`function f(o,n){let s=0;for(let i=0;i<n;i++)s+=o.array[0]*scale;return s}`, ir.ReferenceRead},
	} {
		fn := compiledFunction(t, tc.source)
		p, err := Lower(fn)
		if err != nil {
			t.Fatal(err)
		}
		if err := p.Validate(); err != nil {
			t.Fatal(err)
		}
		found := false
		for pc, in := range fn.Code {
			if in.Op == bytecode.OpGetProp {
				found = true
				if p.Code[pc].Op != tc.want {
					t.Fatalf("%s: %+v, want %v", tc.source, p.Code[pc], tc.want)
				}
			}
		}
		if !found {
			t.Fatal("missing reference read")
		}
	}
}

func TestNumericGlobalSelection(t *testing.T) {
	fn := compiledFunction(t, `function f(a,n){for(let i=0;i<n;i++)a[i]=i*scale+offset;return a[n-1]}`)
	p, err := Lower(fn)
	if err != nil || len(p.Globals) != 2 || p.Locals != fn.LocalCount+len(fn.Upvalues)+2 {
		t.Fatalf("numeric binding layout: %+v, %v", p, err)
	}
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	for pc, in := range fn.Code {
		if in.Op == bytecode.OpGetGlobal {
			got := p.Code[pc]
			if got.Op != ir.BindingRead || got.Left.Slot < p.Locals-len(p.Globals) || got.Left.Slot >= p.Locals || got.Key != in.A {
				t.Fatalf("binding read lost its reserved view: %+v", got)
			}
		}
	}
}

func TestGlobalSlotBudget(t *testing.T) {
	fn := fixture(bytecode.Instr{Op: bytecode.OpGetGlobal, B: 1}, bytecode.Instr{Op: bytecode.OpPushInt, A: 2}, bytecode.Instr{Op: bytecode.OpMul}, bytecode.Instr{Op: bytecode.OpDrop}, bytecode.Instr{Op: bytecode.OpJump})
	fn.Names = []string{"scale"}
	fn.PropSites = 1
	fn.LocalCount, fn.MaxStack = MaxSlots-2, 2
	fn.Locals = make([]bytecode.LocalDesc, fn.LocalCount)
	p, err := Lower(fn)
	if err != nil || len(p.Globals) != 0 || p.Code[0].Op != ir.Host || p.Locals+p.StackSize != MaxSlots {
		t.Fatalf("binding view exceeded scratch capacity: %+v, %v", p, err)
	}
}

func TestShortCountdownSelection(t *testing.T) {
	for _, tc := range []struct {
		source  string
		counter uint16
	}{
		{`function f(n,a){while(--n>=0)a[n]=(a[n]+1)|0;return a[0]}`, 1},
		{`function f(a,n){while(--n>=0)a[n]=(a[n]+1)|0;return a[0]}`, 2},
		{`function f(n,a){n=7;while(--n>=0)a[n]=(a[n]+1)|0;return a[0]}`, 0},
		{`function f(n,a){while(--n>=0){a[n]=(a[n]+1)|0;n=7}return a[0]}`, 0},
		{`function f(n,a){while(n-->=0)a[n]=(a[n]+1)|0;return a[0]}`, 0},
		{`function f(n,a){while(--n>=0)a[n]=(a[n]+1)|0;while(--n>=0)a[n]+=1;return a[0]}`, 0},
	} {
		p, err := Lower(compiledFunction(t, tc.source))
		if err != nil || p.ShortCounter != tc.counter {
			t.Fatalf("countdown selection: %+v, %v; %s", p, err, tc.source)
		}
	}
}

func TestReceiverSlot(t *testing.T) {
	fn := compiledFunction(t, `function f(a){a[0]+=1;return this}`)
	p, err := Lower(fn)
	if err != nil || !p.This || p.Locals != fn.LocalCount+len(fn.Upvalues)+1 {
		t.Fatalf("receiver layout: %+v, %v", p, err)
	}
	for pc, in := range fn.Code {
		if in.Op == bytecode.OpPushThis {
			got := p.Code[pc]
			if got.Op != ir.Copy || got.Left.Slot != p.Locals-1 || !got.Check || got.CheckSlot != p.Locals-1 {
				t.Fatalf("receiver load lost its binding guard: %+v", got)
			}
			return
		}
	}
	t.Fatal("missing receiver load")
}

func TestReceiverSlotBudget(t *testing.T) {
	fn := fixture(bytecode.Instr{Op: bytecode.OpPushThis}, bytecode.Instr{Op: bytecode.OpDrop}, bytecode.Instr{Op: bytecode.OpPushInt}, bytecode.Instr{Op: bytecode.OpBitNot}, bytecode.Instr{Op: bytecode.OpDrop}, bytecode.Instr{Op: bytecode.OpJump})
	fn.LocalCount, fn.MaxStack = MaxSlots-1, 1
	fn.Locals = make([]bytecode.LocalDesc, fn.LocalCount)
	p, err := Lower(fn)
	if err != nil || p.This || p.Locals+p.StackSize != MaxSlots || p.Code[0].Op != ir.Host {
		t.Fatalf("receiver exceeded scratch storage instead of taking the host path: %+v, %v", p, err)
	}
}

func TestMixedPropertyLoopSelection(t *testing.T) {
	fn := compiledFunction(t, `function f(a,n,cb){for(let i=0;i<n;i++){a[i]=(this.x+i)|0;this.x+=1;cb()}return a[0]}`)
	p, err := Lower(fn)
	if err != nil {
		t.Fatal(err)
	}
	if p.This || p.Locals != fn.LocalCount+len(fn.Upvalues) {
		t.Fatal("mixed loop prepared an unprofitable receiver snapshot")
	}
	for _, in := range p.Code {
		if in.Op == ir.PropertyRead || in.Op == ir.PropertyWrite {
			t.Fatal("mixed loop prepared unprofitable numeric property views")
		}
	}
}

func TestRefusals(t *testing.T) {
	for _, tc := range []struct{ source, reason string }{
		{`function f(a) { return a.x }`, "host operations without native loop or array work"},
		{`function f(a,n,cb) { for(let i=0;i<n;i++){a.x++;cb()}return a.x }`, "property operations without native array or bitwise work"},
		{`function f(a) { return new a() }`, "unsupported opcode"},
		{`function f() { try { return 1 } catch(e) { return 2 } }`, "unsupported opcode push_catch"},
		{`function f() { eval('1') }`, "direct eval"},
		{`function f() { let a=1; return () => a }`, "captured local"},
		{`function f() { return arguments[0] }`, "arguments object"},
		{`function f(a=1) { return a }`, "non-simple parameters"},
		{`function* f() { yield 1 }`, "async or generator"},
		{`async function f() { return 1 }`, "async or generator"},
		{`function f() { with ({}) { return 1 } }`, "unsupported opcode"},
		{`function f(a) { return a ** 2 }`, "unsupported opcode pow"},
		{`function f() { const a=1; a=2; return a }`, "unsupported opcode"},
	} {
		t.Run(tc.reason+tc.source, func(t *testing.T) {
			_, err := Lower(compiledFunction(t, tc.source))
			if err == nil || !strings.Contains(err.Error(), tc.reason) {
				t.Fatalf("refusal = %v, want %q", err, tc.reason)
			}
		})
	}
}

func TestRemainderHostBoundaries(t *testing.T) {
	for _, code := range [][]bytecode.Instr{
		{{Op: bytecode.OpGetLocalIndex, B: 1}, {Op: bytecode.OpPushInt, A: 3}, {Op: bytecode.OpMod}, {Op: bytecode.OpReturn}},
		{{Op: bytecode.OpGetLocalIndex, B: 1}, {Op: bytecode.OpBinLocal, A: 1, B: uint32(bytecode.OpMod)}, {Op: bytecode.OpReturn}},
		{{Op: bytecode.OpGetLocalIndex, B: 1}, {Op: bytecode.OpBinImm, A: 3, B: uint32(bytecode.OpMod)}, {Op: bytecode.OpReturn}},
		{{Op: bytecode.OpGetLocalIndex, B: 1}, {Op: bytecode.OpDrop}, {Op: bytecode.OpLocalBinImm, A: uint32(bytecode.OpMod) << 24, B: 3}, {Op: bytecode.OpReturn}},
	} {
		p, err := Lower(fixture(code...))
		if err != nil {
			t.Fatal(err)
		}
		pc := len(code) - 2
		if p.Code[pc].Op != ir.Host {
			t.Fatalf("remainder at %d did not retain its host boundary", pc)
		}
		slots := make([]ir.Value, p.Locals+p.StackSize)
		exit, err := p.Evaluate(slots, pc, 1)
		if err != nil || exit.Kind != ir.HostExit || exit.State != p.Maps[pc] || exit.Steps != 0 {
			t.Fatalf("remainder exit %+v, %v", exit, err)
		}
	}
}

func TestMalformedBytecode(t *testing.T) {
	tests := []struct {
		name   string
		fn     *bytecode.Function
		pc     int
		reason string
	}{
		{"underflow", fixture(bytecode.Instr{Op: bytecode.OpAdd}), 0, "underflow"},
		{"local", fixture(bytecode.Instr{Op: bytecode.OpGetLocal, A: ^uint32(0)}), 0, "local index"},
		{"constant", fixture(bytecode.Instr{Op: bytecode.OpPushConst, A: ^uint32(0)}), 0, "constant"},
		{"target", fixture(bytecode.Instr{Op: bytecode.OpJump, A: ^uint32(0)}), 0, "branch target"},
		{"fallthrough", fixture(bytecode.Instr{Op: bytecode.OpNop}), 0, "fallthrough"},
		{"fused", fixture(bytecode.Instr{Op: bytecode.OpBinImm, B: 256 + uint32(bytecode.OpAdd)}), 0, "fused arithmetic"},
		{"comparison", fixture(bytecode.Instr{Op: bytecode.OpJumpIfCmpFalse, B: uint32(bytecode.OpAdd)}), 0, "fused comparison"},
		{"update", fixture(bytecode.Instr{Op: bytecode.OpUpdateLocal, B: 4}), 0, "update flags"},
		{"immutable", fixture(bytecode.Instr{Op: bytecode.OpSetLocalCheck}), 0, "immutable local"},
		{"join", fixture(
			bytecode.Instr{Op: bytecode.OpPushTrue},
			bytecode.Instr{Op: bytecode.OpJumpIfTrueKeep, A: 3},
			bytecode.Instr{Op: bytecode.OpJump, A: 3},
			bytecode.Instr{Op: bytecode.OpReturnUndef}), 3, "inconsistent stack depth"},
		{"unreachable", fixture(bytecode.Instr{Op: bytecode.OpReturnUndef}, bytecode.Instr{Op: bytecode.OpClosure}), 1, "unsupported opcode closure"},
	}
	tooDeep := fixture(bytecode.Instr{Op: bytecode.OpGetLocal2}, bytecode.Instr{Op: bytecode.OpReturn})
	tooDeep.MaxStack = 1
	tests = append(tests, struct {
		name   string
		fn     *bytecode.Function
		pc     int
		reason string
	}{"overflow", tooDeep, 0, "exceeds MaxStack"})
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Lower(tc.fn)
			var refusal *Refusal
			if !errors.As(err, &refusal) || refusal.PC != tc.pc || !strings.Contains(refusal.Reason, tc.reason) {
				t.Fatalf("refusal = %v, want pc %d %q", err, tc.pc, tc.reason)
			}
		})
	}
}

func TestCompilerBudgets(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(*bytecode.Function)
	}{
		{"instructions", func(f *bytecode.Function) { f.Code = make([]bytecode.Instr, MaxInstructions+1) }},
		{"slots", func(f *bytecode.Function) { f.MaxStack = MaxSlots }},
		{"negative", func(f *bytecode.Function) { f.LocalCount = -1 }},
		{"layout", func(f *bytecode.Function) { f.Locals = nil }},
		{"upvalues", func(f *bytecode.Function) { f.Upvalues = make([]bytecode.UpvalueDesc, MaxSlots) }},
		{"mapped arguments", func(f *bytecode.Function) { f.MappedArguments = true }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fn := fixture(bytecode.Instr{Op: bytecode.OpReturnUndef})
			tc.edit(fn)
			if _, err := Lower(fn); err == nil {
				t.Fatal("accepted function outside bounds")
			}
		})
	}
}

func TestLoopStateMaps(t *testing.T) {
	fn := compiledFunction(t, `function sum(n) { let s=0; for(let i=0; i<n; i++) s+=i; return s }`)
	p, err := Lower(fn)
	if err != nil {
		t.Fatal(err)
	}
	depths := []int{0, 1, 0, 1, 0, 1, 0, 2, 0, 2, 1, 0, 0, 0, 1, -1}
	if len(p.Maps) != len(depths) {
		t.Fatalf("unexpected fixture bytecode:\n%s", fn.Disassemble())
	}
	for pc, depth := range depths {
		if p.Maps[pc] != (ir.StateMap{PC: uint32(pc), Depth: depth}) {
			t.Fatalf("pc %d map = %+v, want depth %d", pc, p.Maps[pc], depth)
		}
	}
	slots := make([]ir.Value, p.Locals+p.StackSize)
	slots[0] = ir.Float(10)
	pc, exits := 0, 0
	for {
		exit, err := p.Evaluate(slots, pc, 3)
		if err != nil {
			t.Fatal(err)
		}
		if exit.Kind == ir.Returned {
			if exit.Value != ir.Float(45) || exits < 10 {
				t.Fatalf("result = %+v, budget exits = %d", exit, exits)
			}
			break
		}
		if exit.Kind != ir.BudgetExit || exit.Steps != 3 {
			t.Fatalf("exit = %+v", exit)
		}
		pc = int(exit.State.PC)
		exits++
		if exits > 100 {
			t.Fatal("did not finish bounded loop")
		}
	}
}

func TestKeepBranchDepths(t *testing.T) {
	for _, op := range []bytecode.Op{bytecode.OpJumpIfTrueKeep, bytecode.OpJumpIfFalseKeep} {
		fn := fixture(
			bytecode.Instr{Op: bytecode.OpGetLocal},
			bytecode.Instr{Op: op, A: 3},
			bytecode.Instr{Op: bytecode.OpPushInt, A: 7},
			bytecode.Instr{Op: bytecode.OpReturn})
		p, err := Lower(fn)
		if err != nil {
			t.Fatal(err)
		}
		if p.Maps[2].Depth != 0 || p.Maps[3].Depth != 1 {
			t.Fatalf("keep edge depths = %+v", p.Maps)
		}
		for _, v := range []ir.Value{ir.Bool(true), ir.Bool(false), ir.Float(math.NaN()), ir.Float(math.Copysign(0, -1)), {Kind: ir.Null}, {Kind: ir.Undefined}} {
			slots := make([]ir.Value, p.Locals+p.StackSize)
			slots[0] = v
			exit, err := p.Evaluate(slots, 0, 2)
			if err != nil || exit.Kind != ir.BudgetExit || exit.State.Depth != p.Maps[exit.State.PC].Depth {
				t.Fatalf("branch exit = %+v, %v", exit, err)
			}
			wantTaken := v == ir.Bool(true)
			if op == bytecode.OpJumpIfFalseKeep {
				wantTaken = !wantTaken
			}
			if (exit.State.PC == 3) != wantTaken {
				t.Fatalf("op %s value %+v: exit = %+v", op, v, exit)
			}
		}
	}
}

func TestSharedProgram(t *testing.T) {
	fn := compiledFunction(t, `function f(n) { let s=0; for(let i=0;i<n;i++) s+=i; return s }`)
	previous := fn.VMCode
	p, err := Lower(fn)
	if err != nil {
		t.Fatal(err)
	}
	if fn.VMCode != previous {
		t.Fatal("lowering changed the shared tree cache")
	}
	var wg sync.WaitGroup
	for n := 0; n < 8; n++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			for i := 0; i < 20; i++ {
				slots := make([]ir.Value, p.Locals+p.StackSize)
				slots[0] = ir.Float(float64(n))
				exit, err := p.Evaluate(slots, 0, 1000)
				if err != nil || exit.Kind != ir.Returned || exit.Value != ir.Float(float64(n*(n-1)/2)) {
					t.Errorf("shared program n=%d: %+v, %v", n, exit, err)
					return
				}
			}
		}(n)
	}
	wg.Wait()
}

// Accepted bytecode must be executable within the declared slots and bounded
// instruction count, even when its operands did not come from the compiler.
func FuzzLower(f *testing.F) {
	for _, code := range [][]bytecode.Instr{
		{{Op: bytecode.OpReturnUndef}},
		{{Op: bytecode.OpPushInt}, {Op: bytecode.OpReturn}},
		{{Op: bytecode.OpGetLocal2, B: 1}, {Op: bytecode.OpAdd}, {Op: bytecode.OpReturn}},
		{{Op: bytecode.OpJump}},
	} {
		data := make([]byte, 9*len(code))
		for pc, in := range code {
			data[pc*9] = byte(in.Op)
			binary.LittleEndian.PutUint32(data[pc*9+1:], in.A)
			binary.LittleEndian.PutUint32(data[pc*9+5:], in.B)
		}
		f.Add(data)
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 1024 {
			return
		}
		fn := fixture()
		fn.Constants = []bytecode.Constant{{Kind: bytecode.ConstNumber, Num: 1}}
		for i := 0; i+9 <= len(data); i += 9 {
			fn.Code = append(fn.Code, bytecode.Instr{Op: bytecode.Op(data[i]), A: binary.LittleEndian.Uint32(data[i+1:]), B: binary.LittleEndian.Uint32(data[i+5:])})
		}
		p, err := Lower(fn)
		if err != nil {
			return
		}
		if err := p.Validate(); err != nil {
			t.Fatal(err)
		}
		slots := make([]ir.Value, p.Locals+p.StackSize)
		slots[0], slots[1] = ir.Float(2), ir.Float(3)
		if _, err := p.Evaluate(slots, 0, 64); err != nil {
			t.Fatal(err)
		}
	})
}
