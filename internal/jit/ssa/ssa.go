// Package ssa is the JIT's optimizing IR: a typed SSA form built from the
// slot IR (internal/jit/ir), whose state maps become the frame states that
// every guard and exit carries. See docs/jit-phase2-design.md.
//
// A function has several entries: the function's start, each loop header
// (where a poll returns and on-stack replacement enters), and the instruction
// after each host exit (where Go resumes it after running that instruction).
// Every exit names the exact interpreter state to resume, so deoptimization
// stays exact by construction, as it is in the slot IR.
//
// Values are typed. A slot holds a Tagged value -- an ir.Value, a kind and
// its bits -- and arithmetic works on unboxed Float64, Int32 and Bool values,
// reached through guards that exit when a value is not what the operation
// needs.
package ssa

import (
	"fmt"
	"strings"

	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

// Type is a value's representation.
type Type uint8

const (
	// Tagged is a slot value: an ir.Value of any kind.
	Tagged Type = iota
	Float64
	// Int32 also carries ToUint32's results, as their bits.
	Int32
	Bool
	// None is a value used only for its effect or control: a guard, a check.
	None
	// Ptr is an object's address, read from the frame slot that holds it.
	// It never reaches a slot or crosses an exit.
	Ptr
	// Source is where a tagged value came from, at run time: a slot's index,
	// -1 for a primitive, or the address of the heap cell it was loaded
	// from, which is at or above abi.MaxRecords (origin.go).
	Source
)

func (t Type) String() string {
	return [...]string{"tagged", "f64", "i32", "bool", "none", "ptr", "source"}[t]
}

// Op is a value's operation.
type Op uint8

const (
	OpInvalid Op = iota

	// Values.
	OpLoadSlot    // Aux: the slot, read from the frame at an entry.
	OpConst       // Const: a tagged constant.
	OpConstF64    // Const.Bits: the number's bits.
	OpConstSource // Aux: a slot's index, or -1: a source (origin.go)
	OpPhi         // Args: one per predecessor, in Block.Preds order.

	// Guards and checks: each has a frame state and exits to it when it
	// fails. Aux is the exit kind (ir.GuardExit or ir.HostExit).
	OpUnboxF64  // tagged -> f64, if a number
	OpTruth     // tagged -> bool, if undefined, null, a boolean or a number
	OpCheckInit // tagged -> none, if not uninitialized

	// Arrays, read in place (D8). The guards' semantics are the slot IR's
	// array views' (ir.ArrayView), less writes into holes: an array of class
	// Array, elements that hold numbers, and an index that is an integer
	// below 2**32.
	OpArrayOf   // tagged -> ptr, if an array
	OpElemKey   // f64 -> none, if an index
	OpElemRead  // ptr, f64 -> f64: the element, if the index's is a number
	OpElemWrite // ptr, f64, f64 -> none: stores, if the index's is a number
	// OpElemCell is an element read as a reference: the element's cell, if
	// the index is an integer within the dense elements and the element is
	// not a hole, whatever it holds. OpLoadCell reads it, with the cell as
	// its shadow (origin.go).
	OpElemCell // ptr, f64 -> source
	// OpLength is x.length of an array, as the slot IR's views have it, or
	// of a string, whose length is always at hand, rope or not.
	OpLength // tagged -> f64

	// Properties, read in place (D8). Where the VM's cache for the site knows
	// a shape (Const.Bits, the shape's address, which the VM keeps alive; 0
	// if none), an object of it has the property at Index of its table, a
	// writable one for a write. Any other ordinary object whose table is
	// small is searched for Key, as the VM searches it: the property must be
	// its own plain data, and writable for a write. Values are numbers, as
	// in the slot IR; anything else exits to Go.
	OpObjectOf  // tagged -> ptr, if an object
	OpPropRead  // ptr -> f64: the property, if the shape's and a number
	OpPropWrite // ptr, f64 -> none: stores, if the shape's and a number
	// A reference read from an object: the property's cell, found as a read
	// finds it, and the value there, whatever it is. The value's Shadow is
	// the cell, which an exit has Go copy from (origin.go).
	OpPropCell // ptr -> source: the property's value's address
	OpLoadCell // source -> tagged: the value at a cell
	// A global name's cell (abi.Context.Global): the binding at Index of
	// the global object's table, if it is Key's and plain, initialized
	// data, and no script-level lexical binding of the name shadows it.
	OpGlobalCell // -> source
	// charCodeAt, as the slot IR's string kernels call it. OpStringMethod is
	// a string's charCodeAt: the context's cell (abi.Context.CharCodeAt),
	// if the receiver is a string and the cell holds the intrinsic.
	// OpStringCode is a call of it: the code unit at an index of a flat
	// string, if the callee is the intrinsic.
	OpStringMethod // tagged -> source
	OpStringCode   // tagged callee, tagged string, f64 index -> f64

	// Boxing: a typed value as a slot value.
	OpBoxF64
	OpBoxBool

	// Arithmetic on numbers.
	OpAddF64
	OpSubF64
	OpMulF64
	OpDivF64
	// OpModF64 is JavaScript's % of two numbers that are integers below
	// 2**63 in magnitude, the divisor not zero; for anything else it exits
	// to Go (Aux), which computes it as math.Mod.
	OpModF64
	OpNegF64 // flips the sign bit, as the slot IR does, NaN included
	OpCmpF64 // Aux: ir.Lt, ir.Le, ir.Gt, ir.Ge, ir.Eq or ir.Ne -> bool
	OpNot    // bool -> bool
	// Equality with null or undefined, whatever the other operand is, as
	// JavaScript defines it. OpStrictNullish is x === null (Aux 0) or
	// x === undefined (Aux 1): its word is that one. OpLooseNullish is
	// x == null, as == undefined: the word is either, or an object's with
	// Annex B's [[IsHTMLDDA]] (abi.Encoding.FlagHTMLDDA), which it reads
	// through the object's origin, as OpObjectOf finds it, exiting to Go
	// (Aux) where that cannot name it.
	OpStrictNullish // tagged -> bool
	OpLooseNullish  // tagged -> bool

	// Integer conversions and bitwise operations, as JavaScript defines them.
	OpToInt32  // f64 -> i32: ToUint32's bits
	OpAndI32   // i32, i32 -> i32
	OpOrI32    //
	OpXorI32   //
	OpShlI32   // shift counts are masked to five bits
	OpSarI32   //
	OpShrU32   // i32, i32 -> i32 holding the unsigned result's bits
	OpNotI32   //
	OpI32ToF64 // signed
	OpU32ToF64 // unsigned

)

var opNames = [...]string{
	OpInvalid: "invalid", OpLoadSlot: "load", OpConst: "const", OpConstF64: "constf", OpConstSource: "consts", OpPhi: "phi",
	OpUnboxF64: "unbox", OpTruth: "truth", OpCheckInit: "checkinit", OpBoxF64: "boxf", OpBoxBool: "boxb",
	OpArrayOf: "arrayof", OpElemKey: "elemkey", OpElemRead: "elemread", OpElemWrite: "elemwrite", OpElemCell: "elemcell", OpLength: "length",
	OpObjectOf: "objectof", OpPropRead: "propread", OpPropWrite: "propwrite", OpPropCell: "propcell", OpLoadCell: "loadcell", OpGlobalCell: "globalcell",
	OpStringMethod: "stringmethod", OpStringCode: "stringcode",
	OpAddF64: "addf", OpSubF64: "subf", OpMulF64: "mulf", OpDivF64: "divf", OpModF64: "modf", OpNegF64: "negf", OpCmpF64: "cmpf",
	OpNot: "not", OpStrictNullish: "strictnullish", OpLooseNullish: "loosenullish", OpToInt32: "toi32", OpAndI32: "and", OpOrI32: "or", OpXorI32: "xor", OpShlI32: "shl",
	OpSarI32: "sar", OpShrU32: "shr", OpNotI32: "noti", OpI32ToF64: "i2f", OpU32ToF64: "u2f",
}

func (o Op) String() string {
	if int(o) < len(opNames) {
		return opNames[o]
	}
	return fmt.Sprintf("op%d", o)
}

// isGuard reports whether an op exits when its operand is not what it needs.
func (o Op) isGuard() bool {
	switch o {
	case OpUnboxF64, OpTruth, OpCheckInit, OpModF64, OpArrayOf, OpElemKey, OpElemRead, OpElemWrite, OpElemCell, OpLength,
		OpObjectOf, OpPropRead, OpPropWrite, OpPropCell, OpGlobalCell, OpStringMethod, OpStringCode, OpLooseNullish:
		return true
	}
	return false
}

// readsMemory reports whether a guard's result or exit depends on elements
// or properties, which writes change: two of them are not the same guard.
// (An array's length and an object's shape change only in Go.)
func (o Op) readsMemory() bool {
	return o == OpElemRead || o == OpElemWrite || o == OpElemCell || o == OpPropRead || o == OpPropWrite || o == OpPropCell ||
		o == OpGlobalCell
}

// Value is one SSA value.
type Value struct {
	ID    int
	Op    Op
	Type  Type
	Args  []*Value
	Aux   int
	Const ir.Value
	// Index is a property's index in tables of the shape a property
	// operation knows, and Key the property's atom; their Aux is their exit
	// kind, as every guard's is.
	Index int
	Key   uint32
	// State is the frame to exit to, for guards.
	State *FrameState
	// Shadow is a tagged value's origin at run time, where no compiler can
	// name it: an ambiguous phi's is a Source phi; a value loaded from a cell
	// has the cell (origin.go).
	Shadow *Value
	Block  *Block
	// Uses counts the values, frame states and controls that use this one.
	Uses int
}

func (v *Value) String() string { return fmt.Sprintf("v%d", v.ID) }

// FrameState is the interpreter's state at a bytecode PC: the PC to resume
// and the value of every live slot (locals, then Depth operands).
type FrameState struct {
	PC    uint32
	Depth int
	Slots []*Value
	// Site is the slot IR PC of the operation whose guards and exits use
	// this state, or -1 for a block's entry or loop header (abi.Context's
	// ExitSite).
	Site int
}

// Kind is how a block ends.
type Kind uint8

const (
	BlockPlain  Kind = iota // goes to Succs[0]
	BlockIf                 // Control is a bool: Succs[0] if true, Succs[1] if false
	BlockReturn             // Control is the tagged result
	BlockExit               // leaves to State with ExitKind (a host exit or a guard)
)

// Block is a basic block.
type Block struct {
	ID      int
	Kind    Kind
	Values  []*Value
	Control *Value
	Succs   []*Block
	Preds   []*Block
	// State and ExitKind describe a BlockExit's exit.
	State    *FrameState
	ExitKind ir.ExitKind
	// Header is a loop header's state on entry, where a poll exits to, and
	// an entry block's state, where its guards exit to.
	Header *FrameState
	// PC is the bytecode PC the block starts at, or -1 for an entry block.
	PC int
	// LoopHeader marks a block a backward branch reaches; Backedge marks the
	// edges into it from inside the loop (by predecessor index).
	LoopHeader bool
	Backedge   []bool
}

// Entry is a place native code can be entered: an entry block, which loads
// the live slots, for a bytecode PC.
type Entry struct {
	PC    int
	Depth int
	Block *Block
}

// Func is a function in SSA form.
type Func struct {
	Blocks  []*Block
	Entries []Entry
	// Locals and StackSize are the slot IR program's: the frame's shape.
	Locals, StackSize int
	// FrameLocals is how many of the Locals are the frame's own; the rest,
	// up to Locals, are captured bindings, which native code reads through
	// their cells (abi.Context.Upvalues) and never writes. Build sets it to
	// Locals; a VM whose program has captured bindings lowers it.
	FrameLocals int
	// ThisSlot is the slot the receiver is read from, one of those past
	// FrameLocals, which native code finds in the context
	// (abi.Context.This); -1 if none.
	ThisSlot int
	// written marks the slots some instruction writes (Written).
	written []bool
	nextID  int
	// values and refs are slabs values, their arguments and frame states'
	// slots come from, a chunk at a time: a compile at run time makes
	// hundreds of each (BenchmarkJITCompile in internal/vm). Chunks double,
	// up to a limit, so a small function takes little.
	values     []Value
	refs       []*Value
	states     []FrameState
	valueChunk int
	refChunk   int
	// scratch is unboxPhis's tables, kept from one round of Optimize to
	// the next.
	scratch struct {
		flags []bool
		vals  []*Value
	}
}

// newState returns a new frame state like s, from a slab.
func (f *Func) newState(s FrameState) *FrameState {
	if len(f.states) == 0 {
		f.states = make([]FrameState, 32)
	}
	p := &f.states[0]
	f.states = f.states[1:]
	*p = s
	return p
}

// alloc returns a new value like v, numbered next.
func (f *Func) alloc(v Value) *Value {
	if len(f.values) == 0 {
		f.valueChunk = min(max(2*f.valueChunk, 16), 256)
		f.values = make([]Value, f.valueChunk)
	}
	p := &f.values[0]
	f.values = f.values[1:]
	*p = v
	p.ID = f.nextID
	f.nextID++
	return p
}

// refsOf returns a slice of n values, nil for none. Its capacity is its
// length, so an append copies it rather than write over a neighbour's.
func (f *Func) refsOf(n int) []*Value {
	if n == 0 {
		return nil
	}
	if len(f.refs) < n {
		f.refChunk = min(max(2*f.refChunk, 64), 1024)
		f.refs = make([]*Value, max(f.refChunk, n))
	}
	s := f.refs[:n:n]
	f.refs = f.refs[n:]
	return s
}

func (f *Func) newBlock(pc int) *Block {
	b := &Block{ID: len(f.Blocks), PC: pc}
	f.Blocks = append(f.Blocks, b)
	return b
}

func (f *Func) newValue(b *Block, op Op, t Type, args ...*Value) *Value {
	v := f.alloc(Value{Op: op, Type: t, Args: f.refsOf(len(args)), Block: b})
	copy(v.Args, args)
	for _, a := range args {
		a.Uses++
	}
	b.Values = append(b.Values, v)
	return v
}

// NumValues bounds the function's value IDs, for tables indexed by them.
func (f *Func) NumValues() int { return f.nextID }

// Written reports whether any instruction writes a slot: a slot no
// instruction writes holds its value from entry in every frame state.
func (f *Func) Written(slot int) bool { return f.written[slot] }

// EntryFor returns the entry for a bytecode PC.
func (f *Func) EntryFor(pc int) (Entry, bool) {
	for _, e := range f.Entries {
		if e.PC == pc {
			return e, true
		}
	}
	return Entry{}, false
}

// String prints the function, for tests and debugging.
func (f *Func) String() string {
	var b strings.Builder
	for _, e := range f.Entries {
		fmt.Fprintf(&b, "entry pc %d depth %d: b%d\n", e.PC, e.Depth, e.Block.ID)
	}
	for _, blk := range f.Blocks {
		fmt.Fprintf(&b, "b%d (pc %d)", blk.ID, blk.PC)
		if len(blk.Preds) > 0 {
			b.WriteString(" <-")
			for _, p := range blk.Preds {
				fmt.Fprintf(&b, " b%d", p.ID)
			}
		}
		if blk.LoopHeader {
			b.WriteString(" loop")
		}
		if blk.Header != nil {
			fmt.Fprintf(&b, " state %v", blk.Header.Slots)
		}
		b.WriteString("\n")
		for _, v := range blk.Values {
			fmt.Fprintf(&b, "  %v = %v %v", v, v.Op, v.Type)
			for _, a := range v.Args {
				fmt.Fprintf(&b, " %v", a)
			}
			if v.Op == OpConst || v.Op == OpConstF64 {
				fmt.Fprintf(&b, " [%d:%#x]", v.Const.Kind, v.Const.Bits)
			}
			if v.Op == OpLoadSlot || v.Op == OpCmpF64 {
				fmt.Fprintf(&b, " aux=%d", v.Aux)
			}
			if v.State != nil {
				fmt.Fprintf(&b, " @pc%d", v.State.PC)
			}
			b.WriteString("\n")
		}
		switch blk.Kind {
		case BlockPlain:
			fmt.Fprintf(&b, "  goto b%d\n", blk.Succs[0].ID)
		case BlockIf:
			fmt.Fprintf(&b, "  if %v b%d b%d\n", blk.Control, blk.Succs[0].ID, blk.Succs[1].ID)
		case BlockReturn:
			fmt.Fprintf(&b, "  return %v\n", blk.Control)
		case BlockExit:
			fmt.Fprintf(&b, "  exit %d @pc%d\n", blk.ExitKind, blk.State.PC)
		}
	}
	return b.String()
}
