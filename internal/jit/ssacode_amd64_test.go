//go:build quickjs_jit && !android && !ios && (linux || windows)

package jit

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/compiler"
	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
	jitcompile "github.com/go-quickjs/go-quickjs/internal/jit/compile"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
	"github.com/go-quickjs/go-quickjs/internal/jit/mir"
	"github.com/go-quickjs/go-quickjs/internal/jit/ssa"
	"github.com/go-quickjs/go-quickjs/internal/parser"
)

// testValue is a VM value's layout: a number word and a pointer word.
type testValue struct {
	num uint64
	ref unsafe.Pointer
}

const testTagBase = 0xFFF8000000000000

// testEncoding mirrors the VM's tags (internal/vm/value.go).
var testEncoding = abi.Encoding{
	ValueSize: 16, NumOffset: 0, RefOffset: 8,
	Undefined: testTagBase | 1, Null: testTagBase | 2,
	True: testTagBase | 1<<8 | 3, False: testTagBase | 3,
	Uninitialized: testTagBase | 8, CanonicalNaN: 0x7FF8000000000000,
}

// testObjects are what references point to: a slot IR handle's.
var testObjects [8]int

// word encodes a slot IR value as the VM would hold it.
func word(v ir.Value) testValue {
	switch v.Kind {
	case ir.Number:
		if math.IsNaN(math.Float64frombits(v.Bits)) {
			return testValue{num: testEncoding.CanonicalNaN}
		}
		return testValue{num: v.Bits}
	case ir.Undefined:
		return testValue{num: testEncoding.Undefined}
	case ir.Null:
		return testValue{num: testEncoding.Null}
	case ir.Boolean:
		if v.Bits != 0 {
			return testValue{num: testEncoding.True}
		}
		return testValue{num: testEncoding.False}
	case ir.Uninitialized:
		return testValue{num: testEncoding.Uninitialized}
	}
	// A reference: its kind's tag, a payload, and a pointer. Two
	// references of one kind may share a word, as two objects do in the VM.
	kind := uint64(7)
	if v.Kind == ir.String {
		kind = 4
	}
	return testValue{num: testTagBase | v.Bits%2<<8 | kind, ref: unsafe.Pointer(&testObjects[v.Bits%8])}
}

// referenceExits counts the harness's exit records -- references copied,
// primitives stored over references, slots known only at run time -- and
// returned references, so that the test can tell it reaches them.
var referenceExits struct{ copies, scalars, maybes, returns int }

// applyRecords does what Go does with an exit's records (abi.Record),
// checking that each names a slot of the state once.
func applyRecords(ctx *abi.Context, frame []testValue, slots int) error {
	n := int(ctx.Records)
	if n > slots {
		return fmt.Errorf("%d records for %d slots", n, slots)
	}
	seen := map[uint64]bool{}
	src := make([]testValue, n)
	for i, r := range ctx.Record[:n] {
		slot := r.Slot &^ (abi.RecordScalar | abi.RecordMaybe)
		if slot >= uint64(slots) || seen[slot] {
			return fmt.Errorf("record %d: slot %d of %d, or twice", i, slot, slots)
		}
		seen[slot] = true
		switch {
		case r.Slot&abi.RecordScalar != 0:
			if frame[slot].ref == nil {
				return fmt.Errorf("record %d stores into slot %d, which holds no reference", i, slot)
			}
			src[i] = testValue{num: r.Word}
			referenceExits.scalars++
		case r.Slot&abi.RecordMaybe != 0:
			src[i] = testValue{num: r.Word}
			if from := int32(r.Arg); from >= 0 {
				if int(from) >= len(frame) {
					return fmt.Errorf("record %d reads slot %d", i, from)
				}
				if frame[from].ref != nil {
					src[i] = frame[from]
				}
			}
			referenceExits.maybes++
		default:
			if r.Arg >= uint64(len(frame)) || frame[r.Arg].ref == nil {
				return fmt.Errorf("record %d copies slot %d, which holds no reference", i, r.Arg)
			}
			src[i] = frame[r.Arg]
			referenceExits.copies++
		}
	}
	for i, r := range ctx.Record[:n] {
		frame[r.Slot&^(abi.RecordScalar|abi.RecordMaybe)] = src[i]
	}
	return nil
}

var nativeTestValues = []ir.Value{
	ir.Float(0), ir.Float(math.Copysign(0, -1)), ir.Float(1), ir.Float(-1), ir.Float(0.5), ir.Float(3),
	ir.Float(1 << 31), ir.Float(1<<32 + 1), ir.Float(-(1 << 31) - 1), ir.Float(math.NaN()),
	ir.Float(math.Inf(1)), ir.Float(math.Inf(-1)), ir.Float(5e-324), ir.Float(1e300), ir.Float(-1e300),
	ir.Float(9007199254740993), ir.Float(1 << 63), ir.Float(-(1 << 63)), ir.Float(1 << 62), ir.Float(4294967295.5),
	ir.Bool(true), ir.Bool(false), {Kind: ir.Undefined}, {Kind: ir.Null}, {Kind: ir.Uninitialized},
	{Kind: ir.Opaque, Bits: 3}, {Kind: ir.Opaque, Bits: 4}, {Kind: ir.String, Bits: 5},
}

func nativeTestValue(r *rand.Rand) ir.Value {
	if r.IntN(3) == 0 {
		return ir.Float(float64(r.IntN(20) - 5))
	}
	v := nativeTestValues[r.IntN(len(nativeTestValues))]
	if v.Kind == ir.Number && math.IsNaN(math.Float64frombits(v.Bits)) {
		v.Bits = testEncoding.CanonicalNaN
	}
	return v
}

func ssaTestProgram(r *rand.Rand) *ir.Program {
	locals := 1 + r.IntN(6)
	n := 2 + r.IntN(24)
	operand := func() ir.Operand {
		if r.IntN(4) == 0 {
			v := nativeTestValue(r)
			if v.Kind == ir.Opaque || v.Kind == ir.String {
				v = ir.Float(2)
			}
			return ir.Literal(v)
		}
		return ir.Slot(r.IntN(locals))
	}
	ops := []ir.Op{ir.Copy, ir.Binary, ir.Binary, ir.Binary, ir.Unary, ir.Update, ir.Update, ir.Branch, ir.Branch,
		ir.Jump, ir.Swap, ir.CopyPair, ir.StoreLoad, ir.Host, ir.Nop}
	operators := []ir.Operator{ir.Add, ir.Sub, ir.Mul, ir.Div, ir.Lt, ir.Le, ir.Gt, ir.Ge, ir.Eq, ir.Ne,
		ir.BitAnd, ir.BitOr, ir.BitXor, ir.Shl, ir.Shr, ir.UShr}
	p := &ir.Program{Locals: locals}
	for pc := 0; pc < n; pc++ {
		in := ir.Instruction{Op: ops[r.IntN(len(ops))], Left: operand(), Right: operand(),
			Dest: r.IntN(locals), Extra: r.IntN(locals), Target: r.IntN(n + 1), Postfix: r.IntN(2) == 0, When: r.IntN(2) == 0}
		switch in.Op {
		case ir.Binary:
			in.Operator = operators[r.IntN(len(operators))]
		case ir.Unary:
			in.Operator = []ir.Operator{ir.Neg, ir.Pos, ir.Not, ir.Int32, ir.BitNot}[r.IntN(5)]
		case ir.Update:
			in.Operator = []ir.Operator{ir.Add, ir.Sub}[r.IntN(2)]
			if r.IntN(2) == 0 {
				in.Extra = -1
			}
		case ir.Branch:
			in.Operator = []ir.Operator{ir.Truth, ir.Lt, ir.Le, ir.Gt, ir.Ge, ir.Eq, ir.Ne}[r.IntN(7)]
		}
		if r.IntN(8) == 0 {
			in.Check, in.CheckSlot = true, r.IntN(locals)
		}
		p.Code = append(p.Code, in)
	}
	p.Code = append(p.Code, ir.Instruction{Op: ir.Return, Left: operand()})
	p.Maps = make([]ir.StateMap, len(p.Code))
	for pc := range p.Maps {
		p.Maps[pc].PC = uint32(pc)
	}
	return p
}

var exitNames = map[uint64]ir.ExitKind{abi.ExitReturn: ir.Returned, abi.ExitDeopt: ir.GuardExit,
	abi.ExitHost: ir.HostExit, abi.ExitPoll: ir.BudgetExit}

// compiled is a program compiled by the new pipeline, for the harness.
type compiled struct {
	f    *ssa.Func
	mc   *mir.Code
	code *SSACode
}

// compileNative builds, optimizes and compiles p; it returns nil when the
// SSA builder does not support p.
func compileNative(p *ir.Program) (*compiled, error) {
	f, err := ssa.Build(p)
	if errors.Is(err, ssa.ErrUnsupported) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	ssa.Optimize(f)
	mc, err := mir.CompileAMD64(f, testEncoding)
	if err != nil {
		return nil, fmt.Errorf("CompileAMD64: %w\n%s", err, f)
	}
	code, err := NewSSACode(mc)
	if err != nil {
		return nil, err
	}
	return &compiled{f, mc, code}, nil
}

// nativeMismatch enters native code and the SSA evaluator at pc on the same
// slots, with polls every poll back-edges (0: none), and describes any
// difference in exit, return word or frame; "" if there is none.
func nativeMismatch(c *compiled, pc int, slots []ir.Value, poll int) string {
	f := c.f
	frame := make([]testValue, len(slots))
	for i, v := range slots {
		frame[i] = word(v)
	}
	want := append([]ir.Value(nil), slots...)
	wantExit, err := ssa.Evaluate(f, pc, want, poll)
	if err != nil {
		return err.Error()
	}
	counter := 1 << 62
	if poll > 0 {
		counter = poll
	}
	ctx := &abi.Context{Locals: unsafe.Pointer(&frame[0]), BackEdges: &counter}
	if f.StackSize > 0 {
		ctx.Stack = unsafe.Pointer(&frame[f.Locals])
	}
	if err := c.code.Run(pc, ctx); err != nil {
		return err.Error()
	}
	got := ir.Exit{Kind: exitNames[ctx.ExitKind], State: ir.StateMap{PC: uint32(ctx.ExitPC), Depth: int(ctx.ExitDepth)}}
	report := func(why string) string {
		return fmt.Sprintf("%s: from pc %d, poll %d, slots %v\nssa    %+v slots %v\nnative %+v ret %#x frame %v",
			why, pc, poll, slots, wantExit, want, got, ctx.Ret, frame)
	}
	if got.Kind != wantExit.Kind {
		return report("exit kind")
	}
	if got.Kind == ir.Returned {
		ret := testValue{num: ctx.Ret}
		if ctx.RetFrom != 0 && frame[ctx.RetFrom-1].ref != nil {
			ret = frame[ctx.RetFrom-1]
			referenceExits.returns++
		}
		if ret != word(wantExit.Value) {
			return report("return value")
		}
		return ""
	}
	if err := applyRecords(ctx, frame, f.Locals+int(ctx.ExitDepth)); err != nil {
		return report(err.Error())
	}
	if got.State != wantExit.State {
		return report("exit state")
	}
	for i := 0; i < f.Locals+wantExit.State.Depth; i++ {
		if w := word(want[i]); frame[i] != w {
			return report(fmt.Sprintf("slot %d", i))
		}
	}
	return ""
}

// minimize shrinks a failing program, turning instructions into Nop while
// it still validates, compiles and fails the same way from the same entry.
func minimize(p *ir.Program, pc int, slots []ir.Value, poll int) (*ir.Program, string) {
	fails := func(q *ir.Program) string {
		if q.Validate() != nil {
			return ""
		}
		// Removing an instruction can make a loop endless; without polls
		// neither side would come back.
		probe := append([]ir.Value(nil), slots...)
		if exit, err := q.Evaluate(probe, pc, 20000); err != nil || exit.Kind == ir.BudgetExit && poll == 0 {
			return ""
		}
		c, err := compileNative(q)
		if err != nil || c == nil {
			return ""
		}
		defer c.code.Close()
		if !c.code.HasEntry(pc) {
			return ""
		}
		return nativeMismatch(c, pc, append([]ir.Value(nil), slots...), poll)
	}
	best := fails(p)
	for changed := true; changed; {
		changed = false
		for i := range p.Code {
			if p.Code[i].Op == ir.Nop || p.Code[i].Op == ir.Return {
				continue
			}
			q := *p
			q.Code = append([]ir.Instruction(nil), p.Code...)
			q.Code[i] = ir.Instruction{Op: ir.Nop}
			if why := fails(&q); why != "" {
				p, best, changed = &q, why, true
			}
		}
	}
	return p, best
}

// checkNative compiles p and compares native code with the SSA evaluator
// from every entry on random slots. A failure is minimized and reported with
// the function, its allocation and the program.
func checkNative(t *testing.T, r *rand.Rand, p *ir.Program) (ok bool) {
	t.Helper()
	c, err := compileNative(p)
	if err != nil {
		t.Fatal(err)
	}
	if c == nil {
		return false
	}
	defer c.code.Close()
	for _, e := range c.f.Entries {
		for trial := 0; trial < 6; trial++ {
			slots := make([]ir.Value, p.Locals+p.StackSize)
			for i := range slots {
				slots[i] = nativeTestValue(r)
			}
			// Mostly primitives, so that code runs past the guards.
			if trial < 3 {
				for i, v := range slots {
					if v.Kind == ir.Opaque || v.Kind == ir.String {
						slots[i] = ir.Float(7)
					}
				}
			}
			for _, poll := range []int{0, 1, 3} {
				// The SSA evaluator runs forever where the program does; the
				// slot IR tells which do not finish.
				probe := append([]ir.Value(nil), slots...)
				if exit, _ := p.Evaluate(probe, e.PC, 20000); exit.Kind == ir.BudgetExit && poll == 0 {
					continue
				}
				if why := nativeMismatch(c, e.PC, slots, poll); why != "" {
					small, smallWhy := minimize(p, e.PC, slots, poll)
					mc, _ := compileNative(small)
					detail := ""
					if mc != nil {
						detail = mc.f.String() + "\n" + mc.mc.Locations
						mc.code.Close()
					}
					t.Fatalf("%s\n\nminimized: %s\n%s\nprogram: %+v", why, smallWhy, detail, small.Code)
				}
			}
		}
	}
	return true
}

func TestSSANativeMatchesEvaluator(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	compiled := 0
	for attempt := 0; attempt < 100000 && compiled < 2000; attempt++ {
		p := ssaTestProgram(r)
		if p.Validate() != nil {
			continue
		}
		if checkNative(t, r, p) {
			compiled++
		}
	}
	if compiled < 1000 {
		t.Fatalf("only %d programs compiled", compiled)
	}
	t.Logf("%d programs; exits with references: %+v", compiled, referenceExits)
	if referenceExits.copies == 0 || referenceExits.scalars == 0 || referenceExits.maybes == 0 || referenceExits.returns == 0 {
		t.Fatalf("some kind of record or reference return never happened: %+v", referenceExits)
	}
}

var ssaNativeCorpus = []string{
	`function f(n){let s=0;for(let i=0;i<n;i++)s+=i;return s}`,
	`function f(n){let s=0;for(let i=0;i<n;i++)s+=i*0.5-i/3;return s}`,
	`function f(n){var x=0.1;for(var i=0;i<n;i++)x=3.7*x*(1-x);return x}`,
	`function f(n){let h=0;for(let i=0;i<n;i++){h=(h<<5)-h+i|0;h^=h>>>13;h&=0x7fffffff;h|=1;h=~h;h>>=1}return h}`,
	`function f(n){let s=0;for(let i=0;i<n;i++){if(s>1e6)break;s+=i<10?i:-i}return s}`,
	`function f(a,b){let c=0;while(a>0){c+=a&b;a=a>>1;b=b<<1|0}return c}`,
	`function f(n){let a=1,b=1;for(let i=2;i<n;i++){let t=a+b;a=b;b=t}return b}`,
	`function f(n,m){let s=0;for(let i=0;i<n;i++)for(let j=0;j<m;j++)s+=i^j;return s}`,
	`function f(x){return !x?-x:+x}`,
	`function f(n){let s=0;for(let i=n;i--;)s+=i;return s}`,
	`function f(x){let n=0;do{x=x%2?3*x+1:x/2;n++}while(x!==1&&n<1000);return n}`,
}

func TestSSANativeFromJavaScript(t *testing.T) {
	r := rand.New(rand.NewPCG(13, 14))
	compiled := 0
	for _, src := range ssaNativeCorpus {
		prog, err := parser.Parse(src, parser.Options{})
		if err != nil {
			t.Fatal(err)
		}
		top, err := compiler.Compile(prog, compiler.Options{})
		if err != nil {
			t.Fatal(err)
		}
		for _, c := range top.Constants {
			if c.Kind != bytecode.ConstFunction {
				continue
			}
			p, err := jitcompile.Lower(c.Fn)
			if err != nil {
				continue
			}
			if checkNative(t, r, p) {
				compiled++
			}
		}
	}
	if compiled < len(ssaNativeCorpus)-2 {
		t.Fatalf("only %d of %d functions compiled", compiled, len(ssaNativeCorpus))
	}
}

// BenchmarkSSARoundTrip is the new pipeline's floor: Run, the bridge, an
// entry's checks, a host exit's record, and the return to Go. P2's gate is
// 15 ns or less.
func BenchmarkSSARoundTrip(b *testing.B) {
	for _, locals := range []int{1, 8} {
		b.Run(fmt.Sprintf("%d-locals", locals), func(b *testing.B) {
			p := &ir.Program{Locals: locals, Code: []ir.Instruction{{Op: ir.Host}, {Op: ir.Return, Left: ir.Slot(0)}},
				Maps: []ir.StateMap{{PC: 0}, {PC: 1}}}
			c, err := compileNative(p)
			if err != nil || c == nil {
				b.Fatal(err)
			}
			defer c.code.Close()
			frame := make([]testValue, locals)
			counter := 1 << 62
			ctx := &abi.Context{Locals: unsafe.Pointer(&frame[0]), BackEdges: &counter}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := c.code.Run(0, ctx); err != nil || ctx.ExitKind != abi.ExitHost {
					b.Fatal(err, ctx.ExitKind)
				}
			}
		})
	}
}
