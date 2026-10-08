// Package abi fixes the contract between the JIT's generated code
// (internal/jit/mir) and the Go that runs it (the VM, through internal/jit):
// the context block native code is handed, and the value encoding the VM
// uses, which the JIT is given as data rather than by importing the VM.
// See docs/jit-phase2-design.md.
package abi

import "unsafe"

// SpillSlots bounds the spill area: values the register allocator cannot
// keep in registers live there, 8 bytes each.
const SpillSlots = 256

// Context is the block native code reaches through its context register.
// Go sets the frame's addresses before every entry; native code writes the
// exit record and spills. Native code only reads its pointers.
type Context struct {
	// Locals and Stack are the frame's first local and first operand: each
	// slot is an Encoding.ValueSize-byte VM value.
	Locals unsafe.Pointer
	Stack  unsafe.Pointer
	// BackEdges is the interpreter's back-edge counter, which every native
	// back-edge decrements; at zero or below native code polls.
	BackEdges *int
	// The exit record.
	ExitKind  uint64
	ExitPC    uint64
	ExitDepth uint64
	// Ret is a return's number word.
	Ret uint64
	// Spill holds what the allocator could not keep in registers.
	Spill [SpillSlots]uint64
}

// Offsets of Context's fields, which generated code addresses.
var (
	OffLocals    = int32(unsafe.Offsetof(Context{}.Locals))
	OffStack     = int32(unsafe.Offsetof(Context{}.Stack))
	OffBackEdges = int32(unsafe.Offsetof(Context{}.BackEdges))
	OffExitKind  = int32(unsafe.Offsetof(Context{}.ExitKind))
	OffExitPC    = int32(unsafe.Offsetof(Context{}.ExitPC))
	OffExitDepth = int32(unsafe.Offsetof(Context{}.ExitDepth))
	OffRet       = int32(unsafe.Offsetof(Context{}.Ret))
	OffSpill     = int32(unsafe.Offsetof(Context{}.Spill))
)

// Exit kinds, written to Context.ExitKind.
const (
	// ExitReturn: Ret holds the result's number word.
	ExitReturn uint64 = iota
	// ExitDeopt: a guard failed. The frame holds the state at ExitPC with
	// ExitDepth operands; the interpreter runs from there.
	ExitDeopt
	// ExitHost: the instruction at ExitPC is for Go to run. The frame holds
	// the state before it; native code can be entered again after it.
	ExitHost
	// ExitPoll: BackEdges ran out at a loop header. The frame holds the
	// state at ExitPC, the header, where native code can be entered again.
	ExitPoll
)

// Encoding is how the VM represents a value in memory: a number word (a
// float64, or a NaN-boxed tag) and a pointer word.
type Encoding struct {
	// ValueSize is the bytes per slot; NumOffset and RefOffset locate the
	// number and pointer words in a slot.
	ValueSize, NumOffset, RefOffset int32
	// Number words: a word whose top 13 bits are not all set is a number.
	// These are the words for the primitives that are not numbers.
	Undefined, Null, True, False, Uninitialized uint64
	// CanonicalNaN is the one NaN a number word may hold.
	CanonicalNaN uint64
}

// IsNumber reports whether a number word holds a number.
func IsNumber(word uint64) bool { return word>>51 != 0x1FFF }
