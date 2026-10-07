// Package ir defines the pointer-free slot IR used to develop the optional
// native executor. Its Go evaluator is a correctness oracle, not a VM tier.
package ir

import "math"

// Kind identifies a scalar or a handle into Go-owned reference storage.
type Kind uint64

const (
	Undefined Kind = iota
	Number
	Boolean
	Null
	Uninitialized
	Opaque
)

// Value is pointer-free. Opaque Bits is an index into a caller-owned root
// table; it is never a Go address. The caller keeps that table alive.
type Value struct {
	Bits uint64
	Kind Kind
}

// Float preserves IEEE bits, including signed zero. The VM adapter must use
// the VM's numeric constructor when publishing it, to canonicalize NaNs.
func Float(n float64) Value { return Value{Bits: math.Float64bits(n), Kind: Number} }

// Bool returns the scalar representation of a JavaScript boolean.
func Bool(b bool) Value {
	if b {
		return Value{Bits: 1, Kind: Boolean}
	}
	return Value{Kind: Boolean}
}

// Operand selects a slot, or Literal when Slot is -1.
type Operand struct {
	Slot    int
	Literal Value
}

// Slot returns an operand that reads slot n.
func Slot(n int) Operand { return Operand{Slot: n} }

// Literal returns an operand that reads a scalar constant.
func Literal(v Value) Operand { return Operand{Slot: -1, Literal: v} }

// Op identifies a slot instruction. Each instruction commits atomically with
// respect to guards: a guard exit retains the entire pre-instruction state.
type Op uint8

const (
	Nop Op = iota
	Copy
	CopyPair
	StoreLoad
	Swap
	Binary
	Unary
	Update
	Jump
	Branch
	Return
)

// Operator selects an arithmetic, comparison, or truthiness operation.
type Operator uint8

const (
	Add Operator = iota
	Sub
	Mul
	Div
	Lt
	Le
	Gt
	Ge
	Eq
	Ne
	Neg
	Pos
	Not
	Truth
)

// Instruction reads Left and Right and writes Dest (and Extra for paired
// operations). StoreLoad stores Left into Dest before reading Right into
// Extra, so aliased locals match bytecode's sequential store/load behavior.
// Update stores the increment/decrement in Dest and, when Extra >= 0, pushes
// either the old or new value there. Branch takes Target when its condition
// equals When; otherwise execution falls through. Check guards CheckSlot
// against the temporal dead zone before any writes.
type Instruction struct {
	Op        Op
	Operator  Operator
	Left      Operand
	Right     Operand
	Dest      int
	Extra     int
	Target    int
	When      bool
	Postfix   bool
	Check     bool
	CheckSlot int
}

// StateMap describes state immediately before a bytecode instruction. PC is
// the instruction to resume, not the VM's PC after fetching it. Depth is the
// live operand count; -1 marks unreachable code. Slots [0, Locals) are locals,
// and the next Depth slots are the operand stack, in interpreter order.
type StateMap struct {
	PC    uint32
	Depth int
}

// Program is immutable after lowering. Code and Maps have one entry per
// bytecode instruction, including unreachable instructions. Construction is
// the responsibility of a validated bytecode adapter.
type Program struct {
	Locals    int
	StackSize int
	Code      []Instruction
	Maps      []StateMap
}

// ExitKind identifies a completed return or a resumable exit.
type ExitKind uint8

const (
	Returned ExitKind = iota
	GuardExit
	BudgetExit
)

// Exit records a result or the exact interpreter state to resume. Steps counts
// committed IR instructions; a failing guard does not consume a step.
type Exit struct {
	Kind  ExitKind
	State StateMap
	Value Value
	Steps uint64
}
