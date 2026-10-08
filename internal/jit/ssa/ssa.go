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
)

func (t Type) String() string {
	return [...]string{"tagged", "f64", "i32", "bool", "none"}[t]
}

// Op is a value's operation.
type Op uint8

const (
	OpInvalid Op = iota

	// Values.
	OpLoadSlot // Aux: the slot, read from the frame at an entry.
	OpConst    // Const: a tagged constant.
	OpConstF64 // Const.Bits: the number's bits.
	OpPhi      // Args: one per predecessor, in Block.Preds order.

	// Guards and checks: each has a frame state and exits to it when it
	// fails. Aux is the exit kind (ir.GuardExit or ir.HostExit).
	OpUnboxF64  // tagged -> f64, if a number
	OpTruth     // tagged -> bool, if undefined, null, a boolean or a number
	OpCheckInit // tagged -> none, if not uninitialized
	// OpCheckScalar exits unless a slot holds a primitive: an entry checks
	// every live slot, so that native code never holds a reference (see
	// docs/jit-phase2-design.md, the walking skeleton).
	OpCheckScalar

	// Boxing: a typed value as a slot value.
	OpBoxF64
	OpBoxBool

	// Arithmetic on numbers.
	OpAddF64
	OpSubF64
	OpMulF64
	OpDivF64
	OpNegF64 // flips the sign bit, as the slot IR does, NaN included
	OpCmpF64 // Aux: ir.Lt, ir.Le, ir.Gt, ir.Ge, ir.Eq or ir.Ne -> bool
	OpNot    // bool -> bool

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
	OpInvalid: "invalid", OpLoadSlot: "load", OpConst: "const", OpConstF64: "constf", OpPhi: "phi",
	OpUnboxF64: "unbox", OpTruth: "truth", OpCheckInit: "checkinit", OpCheckScalar: "checkscalar", OpBoxF64: "boxf", OpBoxBool: "boxb",
	OpAddF64: "addf", OpSubF64: "subf", OpMulF64: "mulf", OpDivF64: "divf", OpNegF64: "negf", OpCmpF64: "cmpf",
	OpNot: "not", OpToInt32: "toi32", OpAndI32: "and", OpOrI32: "or", OpXorI32: "xor", OpShlI32: "shl",
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
	return o == OpUnboxF64 || o == OpTruth || o == OpCheckInit || o == OpCheckScalar
}

// Value is one SSA value.
type Value struct {
	ID    int
	Op    Op
	Type  Type
	Args  []*Value
	Aux   int
	Const ir.Value
	// State is the frame to exit to, for guards.
	State *FrameState
	Block *Block
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
	nextID            int
}

func (f *Func) newBlock(pc int) *Block {
	b := &Block{ID: len(f.Blocks), PC: pc}
	f.Blocks = append(f.Blocks, b)
	return b
}

func (f *Func) newValue(b *Block, op Op, t Type, args ...*Value) *Value {
	v := &Value{ID: f.nextID, Op: op, Type: t, Args: args, Block: b}
	f.nextID++
	for _, a := range args {
		a.Uses++
	}
	b.Values = append(b.Values, v)
	return v
}

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
