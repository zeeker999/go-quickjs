//go:build quickjs_jit && !android && !ios && (linux || windows)

package jit

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
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

// testObject is an object's layout as far as native code reads it.
type testObject struct {
	class    uint8
	flags    uint8
	arrayLen uint32
	elems    []testValue
}

const (
	testClassArray = 2
	testFlagSparse = 4
)

// testEncoding mirrors the VM's tags (internal/vm/value.go), and its
// objects as testObject lays them out.
var testEncoding = abi.Encoding{
	ValueSize: 16, NumOffset: 0, RefOffset: 8,
	Undefined: testTagBase | 1, Null: testTagBase | 2,
	True: testTagBase | 1<<8 | 3, False: testTagBase | 3,
	Uninitialized: testTagBase | 8, CanonicalNaN: 0x7FF8000000000000,
	Object:      testTagBase | 7,
	ObjectClass: int32(unsafe.Offsetof(testObject{}.class)), ObjectFlags: int32(unsafe.Offsetof(testObject{}.flags)),
	ObjectArrayLen: int32(unsafe.Offsetof(testObject{}.arrayLen)), ObjectElems: int32(unsafe.Offsetof(testObject{}.elems)),
	ClassArray: testClassArray, FlagSparse: testFlagSparse,
}

// testHeap is what handles 0 to 3 name, as in package ssa's tests: an
// array of numbers, an array with holes and other tags among them, an
// object that is not an array, and an array whose length runs past its
// dense elements.
type testHeap []testArray

type testArray struct {
	cells  []uint64 // number words
	length uint64
	array  bool
}

func randomTestHeap(r *rand.Rand) testHeap {
	h := make(testHeap, 4)
	for i := range h {
		a := testArray{cells: make([]uint64, r.IntN(6)), array: i != 2}
		for j := range a.cells {
			switch k := r.IntN(6); {
			case i == 1 && k == 0:
				a.cells[j] = testEncoding.Uninitialized
			case i == 1 && k == 1:
				a.cells[j] = testEncoding.True
			default:
				a.cells[j] = math.Float64bits(float64(r.IntN(9) - 3))
			}
		}
		a.length = uint64(len(a.cells))
		if i == 3 {
			a.length += uint64(r.IntN(4))
		}
		h[i] = a
	}
	return h
}

// nativeHeap is a heap's objects. Native code reads them; the evaluators
// read and write the same cells through views. Every other handle refers to
// an object that is not an array, and every string to text, each its own,
// so that a reference copied from the wrong slot shows.
type nativeHeap struct {
	objects []testObject
	others  [8]testObject
	texts   [8]int
}

func (h testHeap) native() *nativeHeap {
	n := &nativeHeap{}
	for _, a := range h {
		o := testObject{elems: make([]testValue, len(a.cells))}
		for j, c := range a.cells {
			o.elems[j].num = c
		}
		if a.array {
			o.class = testClassArray
		}
		if a.length > uint64(len(a.cells)) {
			o.flags, o.arrayLen = testFlagSparse, uint32(a.length)
		}
		n.objects = append(n.objects, o)
	}
	return n
}

// views are the slot IR's views of the objects, as the VM grants them.
func (n *nativeHeap) views() []ir.ArrayView {
	vs := make([]ir.ArrayView, ir.MaxSlots)
	for i := range n.objects {
		o := &n.objects[i]
		v := ir.ArrayView{DenseLength: uint64(len(o.elems)), Length: uint64(len(o.elems))}
		if o.flags&testFlagSparse != 0 && uint64(o.arrayLen) > v.Length {
			v.Length = uint64(o.arrayLen)
		}
		if len(o.elems) > 0 {
			v.Data = unsafe.Pointer(&o.elems[0])
		}
		if o.class == testClassArray {
			v.NumberLimit = testTagBase
		}
		vs[i] = v
	}
	return vs
}

// same reports whether two instances' elements hold the same words.
func (n *nativeHeap) same(m *nativeHeap) bool {
	for i := range n.objects {
		if !slices.Equal(n.objects[i].elems, m.objects[i].elems) {
			return false
		}
	}
	return true
}

// word encodes a slot IR value as the VM would hold it.
func (n *nativeHeap) word(v ir.Value) testValue {
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
	// A reference: every object has one word, as in the VM, and so here does
	// every string; the pointer tells them apart.
	if v.Kind == ir.String {
		return testValue{num: testTagBase | 4, ref: unsafe.Pointer(&n.texts[v.Bits%8])}
	}
	if v.Bits < uint64(len(n.objects)) {
		return testValue{num: testEncoding.Object, ref: unsafe.Pointer(&n.objects[v.Bits])}
	}
	return testValue{num: testEncoding.Object, ref: unsafe.Pointer(&n.others[v.Bits%8])}
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
	{Kind: ir.Opaque, Bits: 0}, {Kind: ir.Opaque, Bits: 1}, {Kind: ir.Opaque, Bits: 2}, {Kind: ir.String, Bits: 6},
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
		ir.Jump, ir.Swap, ir.CopyPair, ir.StoreLoad, ir.Host, ir.Nop,
		ir.ArrayRead, ir.ArrayWrite, ir.ArrayLength, ir.ArrayKey, ir.ArrayUpdate}
	operators := []ir.Operator{ir.Add, ir.Sub, ir.Mul, ir.Div, ir.Lt, ir.Le, ir.Gt, ir.Ge, ir.Eq, ir.Ne,
		ir.BitAnd, ir.BitOr, ir.BitXor, ir.Shl, ir.Shr, ir.UShr}
	p := &ir.Program{Locals: locals}
	for pc := 0; pc < n; pc++ {
		in := ir.Instruction{Op: ops[r.IntN(len(ops))], Left: operand(), Right: operand(), Third: operand(),
			Dest: r.IntN(locals), Extra: r.IntN(locals), Target: r.IntN(n + 1), Postfix: r.IntN(2) == 0, When: r.IntN(2) == 0}
		switch in.Op {
		case ir.ArrayRead, ir.ArrayWrite, ir.ArrayLength, ir.ArrayKey:
			in.Left = ir.Slot(r.IntN(locals))
		case ir.ArrayUpdate:
			in.Left = ir.Slot(r.IntN(locals))
			in.Right = ir.Slot(in.Extra)
			in.Operator = []ir.Operator{ir.Add, ir.Sub}[r.IntN(2)]
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
func nativeMismatch(c *compiled, pc int, slots []ir.Value, poll int, heap testHeap) string {
	f := c.f
	nh, eh := heap.native(), heap.native()
	frame := make([]testValue, len(slots))
	for i, v := range slots {
		frame[i] = nh.word(v)
	}
	want := append([]ir.Value(nil), slots...)
	wantExit, err := ssa.EvaluateArrays(f, pc, want, eh.views(), poll)
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
		return fmt.Sprintf("%s: from pc %d, poll %d, slots %v, heap %v\nssa    %+v slots %v\nnative %+v ret %#x frame %v",
			why, pc, poll, slots, heap, wantExit, want, got, ctx.Ret, frame)
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
		if ret != nh.word(wantExit.Value) {
			return report("return value")
		}
		if !nh.same(eh) {
			return report("elements")
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
		if w := nh.word(want[i]); frame[i] != w {
			return report(fmt.Sprintf("slot %d", i))
		}
	}
	if !nh.same(eh) {
		return report("elements")
	}
	return ""
}

// minimize shrinks a failing program, turning instructions into Nop while
// it still validates, compiles and fails the same way from the same entry.
func minimize(p *ir.Program, pc int, slots []ir.Value, poll int, heap testHeap) (*ir.Program, string) {
	fails := func(q *ir.Program) string {
		if q.Validate() != nil {
			return ""
		}
		// Removing an instruction can make a loop endless; without polls
		// neither side would come back.
		probe := append([]ir.Value(nil), slots...)
		if exit, err := q.EvaluateArrays(probe, heap.native().views(), pc, 20000); err != nil || exit.Kind == ir.BudgetExit && poll == 0 {
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
		return nativeMismatch(c, pc, append([]ir.Value(nil), slots...), poll, heap)
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
			heap := randomTestHeap(r)
			// Mostly primitives, so that code runs past the guards.
			if trial < 2 {
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
				if exit, _ := p.EvaluateArrays(probe, heap.native().views(), e.PC, 20000); exit.Kind == ir.BudgetExit && poll == 0 {
					continue
				}
				if why := nativeMismatch(c, e.PC, slots, poll, heap); why != "" {
					small, smallWhy := minimize(p, e.PC, slots, poll, heap)
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
