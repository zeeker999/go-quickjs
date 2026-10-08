//go:build quickjs_jit && !android && !ios && (linux || windows)

package jit

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
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

// testUpvalue is a captured binding's cell, as far as native code reads it.
type testUpvalue struct{ slot *testValue }

// testPoison fills a frame's entries where captured bindings are numbered,
// which native code must never read or write: their values are in cells.
var testPoison = testValue{num: 0x0123456789abcdef}

// testString is a string's layout as far as native code reads it.
type testString struct {
	s           string
	left, right *testString
	length      int
	ascii       bool
	u16         *uint16
}

// testText is a string's spec: its code units, and how it is held -- flat
// ASCII, flat with its UTF-16 units cached, flat without them (which
// native code leaves to Go), or a rope (likewise).
type testText struct {
	units []uint16
	form  int
}

const (
	textASCII = iota
	textCached
	textUncached
	textRope
)

func randomText(r *rand.Rand) testText {
	t := testText{form: r.IntN(4)}
	for range r.IntN(5) {
		u := uint16('a' + r.IntN(26))
		if t.form == textCached || t.form == textUncached || r.IntN(2) == 0 {
			u = uint16(0x100 + r.IntN(0x2000))
		}
		t.units = append(t.units, u)
	}
	return t
}

// testCharCodeAt is the handle of the object that is charCodeAt, the
// intrinsic, in a heap whose context has it.
const testCharCodeAt = 6

// testObject is an object's layout as far as native code reads it.
type testObject struct {
	class    uint8
	flags    uint8
	arrayLen uint32
	elems    []testValue
	shape    uintptr
	props    []testProperty
}

// testProperty is a property table's entry.
type testProperty struct {
	key   uint32
	flags uint8
	value testValue
}

// testShapes are the shapes test objects have, each with its table's keys
// and flags: native code only compares a shape's address, so these are just
// distinct numbers. A site's feedback names one, or one no object has.
var testShapes = []struct {
	shape uintptr
	keys  []uint32
	flags []uint8
}{
	{0x1000, []uint32{10, 11, 12}, []uint8{testWritable, testWritable, 0}},
	{0x2000, []uint32{11}, []uint8{testWritable}},
	{0x3000, []uint32{12, 10}, []uint8{testWritable, testAccessor}},
}

const (
	testClassObject = 0
	testClassArray  = 2
	testFlagSparse  = 4

	// A property's flags.
	testWritable = 1 << iota
	testAccessor
	testDeleted
	testPrivate
	testUninit
)

// testFlags are the flags a shapeless object's properties have.
var testFlags = []uint8{testWritable, testWritable, 0, testAccessor, testDeleted, testPrivate | testWritable, testUninit | testWritable}

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
	UpvalueSlot: int32(unsafe.Offsetof(testUpvalue{}.slot)),
	String:      testTagBase | 4, StringData: int32(unsafe.Offsetof(testString{}.s)),
	StringLeft: int32(unsafe.Offsetof(testString{}.left)), StringLength: int32(unsafe.Offsetof(testString{}.length)),
	StringASCII: int32(unsafe.Offsetof(testString{}.ascii)), StringU16: int32(unsafe.Offsetof(testString{}.u16)),
	ObjectShape: int32(unsafe.Offsetof(testObject{}.shape)), ObjectProps: int32(unsafe.Offsetof(testObject{}.props)),
	PropertySize: int32(unsafe.Sizeof(testProperty{})), PropertyValue: int32(unsafe.Offsetof(testProperty{}.value)),
	PropertyKey: int32(unsafe.Offsetof(testProperty{}.key)), PropertyFlags: int32(unsafe.Offsetof(testProperty{}.flags)),
	ClassObject: testClassObject, PropNotData: testAccessor | testDeleted | testPrivate,
	PropNotWritable: testAccessor | testDeleted | testPrivate | testUninit | testWritable, PropWritable: testWritable,
	PropUninit: testUninit,
}

// testHeap is what handles 0 to 4 name, as in package ssa's tests: an
// array of numbers, an array with holes and other tags among them, an
// object that is not an array, an array whose length runs past its dense
// elements, and the global object.
type testHeap []testArray

type testArray struct {
	cells  []uint64 // number words
	length uint64
	array  bool
	shape  int // into testShapes, or -1 for none
	// texts are the strings handles 5 and 6 name, and intrinsic says
	// whether the context has charCodeAt: the global object's spec has them.
	texts     []testText
	intrinsic bool
	keys      []uint32 // the table's keys, flags and values' number words
	flags     []uint8
	props     []ir.Value
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
		a.shape = r.IntN(len(testShapes)+1) - 1
		if i == 2 && r.IntN(2) == 0 {
			// The ordinary object: often shapeless, so that it is searched.
			a.shape = -1
		}
		if a.shape >= 0 {
			a.keys, a.flags = testShapes[a.shape].keys, testShapes[a.shape].flags
		} else {
			for range r.IntN(4) {
				a.keys = append(a.keys, 10+uint32(r.IntN(4)))
				a.flags = append(a.flags, testFlags[r.IntN(len(testFlags))])
			}
		}
		a.props = make([]ir.Value, len(a.keys))
		for j := range a.props {
			switch r.IntN(8) {
			case 0:
				a.props[j] = ir.Bool(true)
			case 1, 2:
				// A reference: to one of the heap's objects, another, or text.
				a.props[j] = ir.Value{Kind: ir.Opaque, Bits: uint64(r.IntN(6))}
			case 3:
				a.props[j] = ir.Value{Kind: ir.String, Bits: 5}
			default:
				a.props[j] = ir.Float(float64(r.IntN(9) - 3))
			}
		}
		h[i] = a
	}
	// The global object, handle 4: an ordinary object whose bindings may
	// be in their temporal dead zone; its length's bits are which of the
	// keys 10 to 13 script-level lexical bindings shadow.
	g := testArray{shape: -1, length: uint64(r.IntN(16)) & uint64(r.IntN(16)),
		texts: []testText{randomText(r), randomText(r)}, intrinsic: r.IntN(4) != 0}
	for range 4 {
		g.keys = append(g.keys, 10+uint32(r.IntN(4)))
		g.flags = append(g.flags, testFlags[r.IntN(len(testFlags))])
		g.props = append(g.props, ir.Float(float64(r.IntN(9)-3)))
		if r.IntN(4) == 0 {
			g.props[len(g.props)-1] = ir.Value{Kind: ir.Opaque, Bits: uint64(r.IntN(4))}
		}
	}
	h = append(h, g)
	return h
}

// nativeHeap is a heap's objects. Native code reads them; the evaluators
// read and write the same cells through views. Every other handle refers to
// an object that is not an array, and every string to text, each its own,
// so that a reference copied from the wrong slot shows.
type nativeHeap struct {
	objects []testObject
	others  [8]testObject
	texts   [8]testString
	specs   []testText // handles 5 and 6's
	// intrinsic is the context's charCodeAt cell.
	intrinsic testValue
	// evaluated is the objects' property values for the SSA evaluator.
	evaluated [][]ir.Value
	// lexNames is the names script-level lexical bindings have, a bit for
	// each key: those of the global object's spec's length's bits.
	lexNames []uint64
}

func (h testHeap) native() *nativeHeap {
	// The objects are made first, at the addresses they keep, so that
	// properties can refer to them.
	n := &nativeHeap{objects: make([]testObject, len(h))}
	for i, a := range h {
		o := &n.objects[i]
		o.elems = make([]testValue, len(a.cells))
		for j, c := range a.cells {
			o.elems[j].num = c
		}
		if a.array {
			o.class = testClassArray
		}
		if a.length > uint64(len(a.cells)) {
			o.flags, o.arrayLen = testFlagSparse, uint32(a.length)
		}
		if a.shape >= 0 {
			o.shape = testShapes[a.shape].shape
		}
		o.props = make([]testProperty, len(a.props))
		for j, v := range a.props {
			o.props[j] = testProperty{key: a.keys[j], flags: a.flags[j], value: n.word(v)}
		}
		n.evaluated = append(n.evaluated, slices.Clone(a.props))
	}
	if len(h) > 4 {
		n.lexNames = []uint64{h[4].length << 10}
		n.specs = h[4].texts
		for i, t := range h[4].texts {
			s := &n.texts[5+i]
			s.length = len(t.units)
			switch t.form {
			case textASCII:
				b := make([]byte, len(t.units))
				for j, u := range t.units {
					b[j] = byte(u)
				}
				s.s, s.ascii = string(b), true
			case textCached:
				s.s = "not read"
				if len(t.units) > 0 {
					s.u16 = &slices.Clone(t.units)[0]
				}
			case textUncached:
				s.s = "not read"
			case textRope:
				// A rope's flags are valid, as the VM's are; its UTF-8 is not.
				s.left, s.right = &testString{}, &testString{}
				s.ascii = !slices.ContainsFunc(t.units, func(u uint16) bool { return u >= 0x80 })
			}
		}
		if h[4].intrinsic {
			n.intrinsic = n.word(ir.Value{Kind: ir.Opaque, Bits: testCharCodeAt})
		}
	}
	return n
}

// heap is what the SSA evaluator reads: views of the arrays' cells, and the
// objects' shapes with its own copy of their property words.
func (n *nativeHeap) heap() ssa.Heap {
	h := ssa.Heap{Arrays: n.views()}
	for i := range n.objects {
		o := &n.objects[i]
		e := ssa.Object{Shape: o.shape, Ordinary: o.class == testClassObject, Props: n.evaluated[i]}
		for _, p := range o.props {
			e.Keys = append(e.Keys, p.key)
			e.Data = append(e.Data, p.flags&testEncoding.PropNotData == 0)
			e.Writable = append(e.Writable, p.flags&testEncoding.PropNotWritable == testWritable)
			e.Uninit = append(e.Uninit, p.flags&testUninit != 0)
		}
		h.Objects = append(h.Objects, e)
	}
	if len(h.Objects) > 4 {
		h.Strings = map[uint64]ssa.String{}
		for i, t := range n.specs {
			// Read natively when flat and at hand: ASCII, or cached.
			flat := t.form == textASCII || t.form == textCached || len(t.units) == 0 && t.form != textRope
			h.Strings[uint64(5+i)] = ssa.String{Units: t.units, Flat: flat}
		}
		if n.intrinsic.ref != nil {
			h.CharCodeAt = &ir.Value{Kind: ir.Opaque, Bits: testCharCodeAt}
		}
		h.Global, h.Lexical = &h.Objects[4], map[uint32]bool{}
		for k := uint32(10); k < 14; k++ {
			if n.lexNames[0]&(1<<k) != 0 {
				h.Lexical[k] = true
			}
		}
	}
	return h
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
		if o.class != testClassArray {
			// The VM grants no view of an object that is not an array.
			continue
		}
		v.NumberLimit = testTagBase
		vs[i] = v
	}
	return vs
}

// same reports whether native code's instance holds the same words as the
// evaluator's: elements, and properties.
func (n *nativeHeap) same(m *nativeHeap) bool {
	for i := range n.objects {
		if !slices.Equal(n.objects[i].elems, m.objects[i].elems) {
			return false
		}
		for j, p := range n.objects[i].props {
			if p.value != n.word(m.evaluated[i][j]) {
				return false
			}
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
		return testValue{num: testEncoding.String, ref: unsafe.Pointer(&n.texts[v.Bits%8])}
	}
	if v.Bits < uint64(len(n.objects)) {
		return testValue{num: testEncoding.Object, ref: unsafe.Pointer(&n.objects[v.Bits])}
	}
	return testValue{num: testEncoding.Object, ref: unsafe.Pointer(&n.others[v.Bits%8])}
}

// referenceExits counts the harness's exit records -- references copied,
// primitives stored over references, sources known only at run time, and
// among them heap cells -- and returned references, so that the test can
// tell it reaches them.
var referenceExits struct{ copies, scalars, maybes, cells, returns int }

// source is the value a run-time source names (origin.go): a slot's, or,
// at or above abi.MaxRecords, the value at a heap cell's address; nil for a
// primitive's -1. word holds the source, as native code wrote it.
func source(word *uint64, at func(int) *testValue) *testValue {
	switch from := int64(*word); {
	case from < 0:
		return nil
	case from < abi.MaxRecords:
		return at(int(from))
	}
	referenceExits.cells++
	return *(**testValue)(unsafe.Pointer(word))
}

// applyRecords does what Go does with an exit's records (abi.Record),
// checking that each names a slot of the state once.
func applyRecords(ctx *abi.Context, at func(int) *testValue, slots int) error {
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
			if at(int(slot)).ref == nil {
				return fmt.Errorf("record %d stores into slot %d, which holds no reference", i, slot)
			}
			src[i] = testValue{num: r.Word}
			referenceExits.scalars++
		case r.Slot&abi.RecordMaybe != 0:
			src[i] = testValue{num: r.Word}
			if from := int64(r.Arg); from >= int64(slots) && from < abi.MaxRecords {
				return fmt.Errorf("record %d reads slot %d", i, from)
			}
			if v := source(&ctx.Record[i].Arg, at); v != nil && v.ref != nil {
				src[i] = *v
			}
			referenceExits.maybes++
		default:
			if r.Arg >= uint64(slots) || at(int(r.Arg)).ref == nil {
				return fmt.Errorf("record %d copies slot %d, which holds no reference", i, r.Arg)
			}
			src[i] = *at(int(r.Arg))
			referenceExits.copies++
		}
	}
	for i, r := range ctx.Record[:n] {
		*at(int(r.Slot &^ (abi.RecordScalar | abi.RecordMaybe))) = src[i]
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
	{Kind: ir.Opaque, Bits: testCharCodeAt}, {Kind: ir.Opaque, Bits: testCharCodeAt},
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

// layout is where a program's locals are: the first frame of them are the
// frame's, and the rest captured bindings, or the receiver at slot this
// (-1 if none), which the program reads and never writes. It also holds
// what the VM knows of the program's property sites, by PC.
type layout struct {
	frame, this int
	sites       map[int]site
	globals     map[int]ssa.GlobalSite
}

// Global is ssa.Feedback's.
func (l layout) Global(pc int) (ssa.GlobalSite, bool) {
	s, ok := l.globals[pc]
	return s, ok
}

// site is a property site's feedback.
type site = ssa.PropertySite

// Property is ssa.Feedback's.
func (l layout) Property(pc int) (ssa.PropertySite, bool) {
	s, ok := l.sites[pc]
	return s, ok
}

// ssaTestProgram makes a program and says where its locals are.
func ssaTestProgram(r *rand.Rand) (*ir.Program, layout) {
	frameLocals := 1 + r.IntN(6)
	locals := frameLocals + r.IntN(3)
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
		ir.ArrayRead, ir.ArrayWrite, ir.ArrayLength, ir.ArrayKey, ir.ArrayUpdate, ir.PropertyRead, ir.PropertyWrite,
		ir.ReferenceRead, ir.ReferenceRead, ir.BindingRead, ir.BindingRead, ir.StringMethod, ir.StringCode, ir.StringCode}
	operators := []ir.Operator{ir.Add, ir.Sub, ir.Mul, ir.Div, ir.Lt, ir.Le, ir.Gt, ir.Ge, ir.Eq, ir.Ne,
		ir.BitAnd, ir.BitOr, ir.BitXor, ir.Shl, ir.Shr, ir.UShr, ir.Mod, ir.Mod}
	p := &ir.Program{Locals: locals}
	sites, globals := map[int]site{}, map[int]ssa.GlobalSite{}
	for pc := 0; pc < n; pc++ {
		in := ir.Instruction{Op: ops[r.IntN(len(ops))], Left: operand(), Right: operand(), Third: operand(),
			Dest: r.IntN(frameLocals), Extra: r.IntN(frameLocals), Target: r.IntN(n + 1), Postfix: r.IntN(2) == 0, When: r.IntN(2) == 0}
		switch in.Op {
		case ir.ArrayRead, ir.ArrayWrite, ir.ArrayLength, ir.ArrayKey:
			in.Left = ir.Slot(r.IntN(locals))
		case ir.StringMethod:
			in.Left = ir.Slot(r.IntN(locals))
		case ir.StringCode:
			in.Left, in.Right = ir.Slot(r.IntN(locals)), ir.Slot(r.IntN(locals))
		case ir.BindingRead:
			in.Left = ir.Literal(ir.Value{Kind: ir.Undefined})
			if r.IntN(4) != 0 {
				globals[pc] = ssa.GlobalSite{Key: 10 + uint32(r.IntN(4)), Index: int32(r.IntN(5)) - 1}
			}
		case ir.PropertyRead, ir.PropertyWrite, ir.ReferenceRead:
			in.Left = ir.Slot(r.IntN(locals))
			// Mostly a site the VM knows: a key, and sometimes a shape with the
			// property's index -- plain data, and writable for a write, as the
			// VM's caches remember -- or a shape no object has.
			if r.IntN(4) != 0 {
				s := site{Key: 10 + uint32(r.IntN(4))}
				switch r.IntN(3) {
				case 0:
					sh := testShapes[r.IntN(len(testShapes))]
					i := r.IntN(len(sh.keys))
					f := sh.flags[i]
					if f&testEncoding.PropNotData == 0 && (in.Op == ir.PropertyRead || f&testEncoding.PropNotWritable == testWritable) {
						s.Key, s.Shape, s.Index = sh.keys[i], sh.shape, int32(i)
					}
				case 1:
					s.Shape = 0x9000
				}
				sites[pc] = s
			}
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
	this := -1
	if locals > frameLocals && r.IntN(2) == 0 {
		this = locals - 1
	}
	return p, layout{frameLocals, this, sites, globals}
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
func compileNative(p *ir.Program, l layout) (*compiled, error) {
	f, err := ssa.BuildWith(p, l)
	if errors.Is(err, ssa.ErrUnsupported) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	f.FrameLocals, f.ThisSlot = l.frame, l.this
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
	// Captured bindings live in cells, and the receiver in the context;
	// their numbers' entries in the frame are poison.
	ctx := &abi.Context{}
	this := (*testValue)(unsafe.Pointer(&ctx.This))
	captured := make([]testValue, f.Locals-f.FrameLocals)
	cells := make([]*testUpvalue, len(captured))
	for i, v := range slots {
		frame[i] = nh.word(v)
		if k := i - f.FrameLocals; k >= 0 && i < f.Locals {
			captured[k], frame[i] = frame[i], testPoison
			cells[k] = &testUpvalue{slot: &captured[k]}
			if i == f.ThisSlot {
				*this = captured[k]
				captured[k] = testPoison
			}
		}
	}
	at := func(i int) *testValue {
		if i == f.ThisSlot {
			return this
		}
		if k := i - f.FrameLocals; k >= 0 && i < f.Locals {
			return &captured[k]
		}
		return &frame[i]
	}
	want := append([]ir.Value(nil), slots...)
	wantExit, err := ssa.EvaluateHeap(f, pc, want, eh.heap(), poll)
	if err != nil {
		return err.Error()
	}
	// Without polls, a program finishes within 2,000 back-edges (finishes);
	// native code that runs on far past that has gone wrong, and its poll
	// says so.
	counter := 1 << 20
	if poll > 0 {
		counter = poll
	}
	ctx.Locals, ctx.BackEdges = unsafe.Pointer(&frame[0]), &counter
	if f.StackSize > 0 {
		ctx.Stack = unsafe.Pointer(&frame[f.Locals])
	}
	if len(cells) > 0 {
		ctx.Upvalues = unsafe.Pointer(&cells[0])
	}
	if len(nh.objects) > 4 {
		ctx.Global, ctx.LexNames = unsafe.Pointer(&nh.objects[4]), unsafe.Pointer(&nh.lexNames)
		*(*testValue)(unsafe.Pointer(&ctx.CharCodeAt)) = nh.intrinsic
	}
	if err := c.code.Run(pc, ctx); err != nil {
		return err.Error()
	}
	got := ir.Exit{Kind: exitNames[ctx.ExitKind], State: ir.StateMap{PC: uint32(ctx.ExitPC), Depth: int(ctx.ExitDepth)}}
	report := func(why string) string {
		return fmt.Sprintf("%s: from pc %d, poll %d, slots %v, heap %v, %d frame locals, receiver %d\nssa    %+v slots %v\nnative %+v ret %#x frame %v captured %v receiver %v",
			why, pc, poll, slots, heap, f.FrameLocals, f.ThisSlot, wantExit, want, got, ctx.Ret, frame, captured, *this)
	}
	if got.Kind != wantExit.Kind {
		return report("exit kind")
	}
	if got.Kind == ir.Returned {
		ret := testValue{num: ctx.Ret}
		from := ctx.RetFrom - 1
		if v := source(&from, at); v != nil && v.ref != nil {
			ret = *v
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
	if err := applyRecords(ctx, at, f.Locals+int(ctx.ExitDepth)); err != nil {
		return report(err.Error())
	}
	if got.State != wantExit.State {
		return report("exit state")
	}
	for i := 0; i < f.Locals+wantExit.State.Depth; i++ {
		if w := nh.word(want[i]); *at(i) != w {
			return report(fmt.Sprintf("slot %d", i))
		}
	}
	for i := f.FrameLocals; i < f.Locals; i++ {
		if frame[i] != testPoison {
			return report(fmt.Sprintf("frame entry %d, a captured binding's number", i))
		}
	}
	if !nh.same(eh) {
		return report("elements")
	}
	return ""
}

// finishes reports whether f, entered at pc, exits within 2,000 back-edges.
// Feedback lets native code go further than the slot IR, which leaves every
// property operation to Go, so the SSA evaluator itself is asked.
func finishes(f *ssa.Func, pc int, slots []ir.Value, heap testHeap) bool {
	exit, err := ssa.EvaluateHeap(f, pc, append([]ir.Value(nil), slots...), heap.native().heap(), 2000)
	return err == nil && exit.Kind != ir.BudgetExit
}

// minimize shrinks a failing program, turning instructions into Nop while
// it still validates, compiles and fails the same way from the same entry.
func minimize(p *ir.Program, l layout, pc int, slots []ir.Value, poll int, heap testHeap) (*ir.Program, string) {
	fails := func(q *ir.Program) string {
		if q.Validate() != nil {
			return ""
		}
		c, err := compileNative(q, l)
		if err != nil || c == nil {
			return ""
		}
		defer c.code.Close()
		// Removing an instruction can make a loop endless; without polls
		// neither side would come back.
		if !c.code.HasEntry(pc) || poll == 0 && !finishes(c.f, pc, slots, heap) {
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
func checkNative(t *testing.T, r *rand.Rand, p *ir.Program, l layout) (ok bool) {
	t.Helper()
	c, err := compileNative(p, l)
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
				// Without polls, a program that loops forever would never
				// come back from either side.
				if poll == 0 && !finishes(c.f, e.PC, slots, heap) {
					continue
				}
				if why := nativeMismatch(c, e.PC, slots, poll, heap); why != "" {
					small, smallWhy := minimize(p, l, e.PC, slots, poll, heap)
					mc, _ := compileNative(small, l)
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
		p, l := ssaTestProgram(r)
		if p.Validate() != nil {
			continue
		}
		if checkNative(t, r, p, l) {
			compiled++
		}
	}
	if compiled < 1000 {
		t.Fatalf("only %d programs compiled", compiled)
	}
	t.Logf("%d programs; exits with references: %+v", compiled, referenceExits)
	if referenceExits.copies == 0 || referenceExits.scalars == 0 || referenceExits.maybes == 0 || referenceExits.cells == 0 || referenceExits.returns == 0 {
		t.Fatalf("some kind of record or reference return never happened: %+v", referenceExits)
	}
}

// TestSSANativeCapturedShadow reads an array through a shadow that names a
// captured binding: x=c; loop { x.length; x=y } merges the captured c, x as
// entered at the loop, and y, so ArrayOf finds x's array at run time, in a
// binding's cell on the first iteration. Random programs seldom do.
func TestSSANativeCapturedShadow(t *testing.T) {
	// Slots: x, y, the length, then the captured c.
	p := &ir.Program{Locals: 4, Code: []ir.Instruction{
		{Op: ir.Copy, Dest: 0, Left: ir.Slot(3)},
		{Op: ir.ArrayLength, Dest: 2, Left: ir.Slot(0)},
		{Op: ir.Copy, Dest: 0, Left: ir.Slot(1)},
		{Op: ir.Jump, Target: 1},
	}}
	p.Maps = make([]ir.StateMap, len(p.Code))
	for pc := range p.Maps {
		p.Maps[pc].PC = uint32(pc)
	}
	c, err := compileNative(p, layout{3, -1, nil, nil})
	if err != nil || c == nil {
		t.Fatalf("compile: %v", err)
	}
	defer c.code.Close()
	if !strings.Contains(c.f.String(), "consts") {
		t.Fatalf("x has no shadow:\n%s", c.f)
	}
	r := rand.New(rand.NewPCG(5, 6))
	for trial := 0; trial < 20; trial++ {
		heap := randomTestHeap(r)
		for _, slots := range [][]ir.Value{
			{ir.Float(0), {Kind: ir.Opaque, Bits: 1}, ir.Float(0), {Kind: ir.Opaque, Bits: 0}},
			{{Kind: ir.Opaque, Bits: 3}, {Kind: ir.Opaque, Bits: 0}, ir.Float(0), {Kind: ir.Opaque, Bits: 1}},
			{ir.Float(0), {Kind: ir.Opaque, Bits: 2}, ir.Float(0), {Kind: ir.Opaque, Bits: 0}},
		} {
			for _, e := range c.f.Entries {
				for _, poll := range []int{1, 2, 3} {
					if why := nativeMismatch(c, e.PC, slots, poll, heap); why != "" {
						t.Fatal(why)
					}
				}
			}
		}
	}
}

// TestSSANativeStrings drives charCodeAt through every form a string has,
// indexes in range and out, with the intrinsic in the context and without,
// and with another callee: random programs seldom line a string, its method
// and an index up. Each run must match the evaluator, and the code units
// must be read natively for ASCII and cached strings alike.
func TestSSANativeStrings(t *testing.T) {
	// Slots: the string, the index, the method, the code.
	p := &ir.Program{Locals: 4, Code: []ir.Instruction{
		{Op: ir.StringMethod, Left: ir.Slot(0), Dest: 2},
		{Op: ir.StringCode, Left: ir.Slot(2), Right: ir.Slot(0), Third: ir.Slot(1), Dest: 3},
		{Op: ir.Return, Left: ir.Slot(3)},
	}}
	p.Maps = make([]ir.StateMap, len(p.Code))
	for pc := range p.Maps {
		p.Maps[pc].PC = uint32(pc)
	}
	c, err := compileNative(p, layout{4, -1, nil, nil})
	if err != nil || c == nil {
		t.Fatalf("compile: %v", err)
	}
	defer c.code.Close()
	r := rand.New(rand.NewPCG(7, 8))
	read := map[int]int{}
	for form := textASCII; form <= textRope; form++ {
		for _, intrinsic := range []bool{true, false} {
			heap := randomTestHeap(r)
			heap[4].texts = []testText{{units: []uint16{'h', 'i', '!'}, form: form}, {units: []uint16{0x3b1, 0x3b2}, form: form}}
			if form == textASCII {
				heap[4].texts[1].form = textCached
			}
			heap[4].intrinsic = intrinsic
			for _, s := range []uint64{5, 6} {
				for _, i := range []float64{0, 1, 2, 3, -1, 0.5, math.Copysign(0, -1)} {
					for _, callee := range []ir.Value{{Kind: ir.Opaque, Bits: testCharCodeAt}, {Kind: ir.Opaque, Bits: 0}, ir.Float(1)} {
						for _, pc := range []int{0, 1} {
							slots := []ir.Value{{Kind: ir.String, Bits: s}, ir.Float(i), callee, ir.Float(0)}
							if why := nativeMismatch(c, pc, slots, 0, heap); why != "" {
								t.Fatalf("form %d, intrinsic %v, string %d, index %v, callee %v, pc %d: %s", form, intrinsic, s, i, callee, pc, why)
							}
							want := slices.Clone(slots)
							if exit, _ := ssa.EvaluateHeap(c.f, pc, want, heap.native().heap(), 0); exit.Kind == ir.Returned {
								read[heap[4].texts[s-5].form]++
							}
						}
					}
				}
			}
		}
	}
	if read[textASCII] == 0 || read[textCached] == 0 || read[textUncached]+read[textRope] != 0 {
		t.Fatalf("code units read by form: %v", read)
	}
}

// TestSSANativeRemainder pins %'s fast path: integers divide natively, with
// a zero remainder carrying the dividend's sign and -2**63 % -1 not
// faulting; everything else exits to Go, where math.Mod answers.
func TestSSANativeRemainder(t *testing.T) {
	p := &ir.Program{Locals: 3, Code: []ir.Instruction{
		{Op: ir.Binary, Operator: ir.Mod, Dest: 2, Left: ir.Slot(0), Right: ir.Slot(1)},
		{Op: ir.Return, Left: ir.Slot(2)},
	}}
	p.Maps = make([]ir.StateMap, len(p.Code))
	for pc := range p.Maps {
		p.Maps[pc].PC = uint32(pc)
	}
	c, err := compileNative(p, layout{3, -1, nil, nil})
	if err != nil || c == nil {
		t.Fatalf("compile: %v", err)
	}
	defer c.code.Close()
	negZero := math.Copysign(0, -1)
	for _, tc := range []struct {
		x, y   float64
		native bool
	}{
		{7, 3, true}, {-7, 3, true}, {7, -3, true}, {-7, -3, true},
		{-4, 2, true}, {4, -2, true}, {negZero, 5, true}, {0, -5, true},
		{-1 << 63, -1, true}, {-1 << 63, 7, true}, {1 << 62, 3, true}, {9007199254740992, 10, true},
		{5, 0, false}, {5.5, 2, false}, {5, 2.5, false}, {math.NaN(), 1, false}, {1, math.NaN(), false},
		{math.Inf(1), 1, false}, {1, math.Inf(-1), false}, {1 << 63, 3, false},
	} {
		heap := randomTestHeap(rand.New(rand.NewPCG(1, 2)))
		slots := []ir.Value{ir.Float(tc.x), ir.Float(tc.y), ir.Float(0)}
		if why := nativeMismatch(c, 0, slots, 0, heap); why != "" {
			t.Fatalf("%v %% %v: %s", tc.x, tc.y, why)
		}
		exit, _ := ssa.EvaluateHeap(c.f, 0, slices.Clone(slots), heap.native().heap(), 0)
		if native := exit.Kind == ir.Returned; native != tc.native {
			t.Fatalf("%v %% %v: native %v, want %v", tc.x, tc.y, native, tc.native)
		}
		if tc.native {
			want := math.Mod(tc.x, tc.y)
			if math.Float64bits(math.Float64frombits(exit.Value.Bits)) != math.Float64bits(want) {
				t.Fatalf("%v %% %v = %v, want %v", tc.x, tc.y, math.Float64frombits(exit.Value.Bits), want)
			}
		}
	}
	// The division takes RDX, the operands' base: a remainder on the
	// operand stack, stored there by a host exit's stub, must land in it.
	q := &ir.Program{Locals: 2, StackSize: 1, Code: []ir.Instruction{
		{Op: ir.Binary, Operator: ir.Mod, Dest: 2, Left: ir.Slot(0), Right: ir.Slot(1)},
		{Op: ir.Host},
		{Op: ir.Return, Left: ir.Slot(2)},
	}, Maps: []ir.StateMap{{PC: 0}, {PC: 1, Depth: 1}, {PC: 2, Depth: 1}}}
	d, err := compileNative(q, layout{2, -1, nil, nil})
	if err != nil || d == nil {
		t.Fatalf("compile: %v", err)
	}
	defer d.code.Close()
	heap := randomTestHeap(rand.New(rand.NewPCG(1, 2)))
	for _, x := range []float64{17, -17, 18} {
		if why := nativeMismatch(d, 0, []ir.Value{ir.Float(x), ir.Float(6), ir.Float(0)}, 0, heap); why != "" {
			t.Fatalf("%v %% 6 on the stack: %s", x, why)
		}
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
			if checkNative(t, r, p, layout{c.Fn.LocalCount, -1, nil, nil}) {
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
			c, err := compileNative(p, layout{locals, -1, nil, nil})
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
