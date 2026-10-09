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

// MaxRecords bounds an exit's records, one per slot at most, and so the
// slots of a function the JIT compiles.
const MaxRecords = 256

// Record is an exit's instruction to Go for one slot, which native code
// cannot write. Slot is the slot, and its flags say what to write there:
//   - none: the value of slot Arg, a reference;
//   - RecordScalar: the primitive whose number word is Word, over a
//     reference, whose pointer word only Go may clear;
//   - RecordMaybe: the value of slot int32(Arg) if that holds a reference,
//     and otherwise -- or if int32(Arg) is negative -- the primitive Word.
//
// Every slot a record reads holds its value from entry still, so Go reads
// them all before it writes any.
type Record struct {
	Slot, Arg, Word, _ uint64
}

// Record flags.
const (
	RecordScalar = 1 << 63
	RecordMaybe  = 1 << 62
)

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
	// Upvalues is the function's first captured binding's cell: an array of
	// pointers to cells, each holding at Encoding.UpvalueSlot a pointer to
	// the binding's value. Native code only reads them.
	Upvalues unsafe.Pointer
	// Global is the object a function's global names are read from, its
	// closure's scope, and LexNames the names script-level lexical bindings
	// have, which shadow them: a slice of uint64, a bit for each atom. Go
	// sets both before every entry; native code only reads them.
	Global, LexNames unsafe.Pointer
	// CharCodeAt is String.prototype.charCodeAt, the intrinsic, when Go has
	// found it still there before an entry of a function that reads it, and
	// otherwise nothing (a nil pointer word).
	CharCodeAt Slot
	// This is the receiver, laid out as a VM value, which Go sets before
	// every entry of a function that reads it; uninitialized in a derived
	// constructor before super() returns.
	This Slot
	// The exit record. ExitSite is the slot IR PC of the operation whose
	// guard or exit this was, or -1 (all ones) for a block's entry or loop
	// header state -- an entry's speculation, or a poll: what a policy that
	// stops speculating where it failed needs, since ExitPC is where the
	// interpreter resumes, often the start of the statement.
	ExitKind  uint64
	ExitPC    uint64
	ExitDepth uint64
	ExitSite  uint64
	// Ret is a return's number word. RetFrom is 0 when that is the result,
	// or 1 plus the slot whose value, a reference, is.
	Ret, RetFrom uint64
	// Records counts the Record entries an exit filled. Every other slot of
	// its state is in the frame already.
	Records uint64
	Record  [MaxRecords]Record
	// Spill holds what the allocator could not keep in registers.
	Spill [SpillSlots]uint64
}

// Slot is a VM value's layout -- a number word, then a pointer word -- held
// where the garbage collector sees the pointer.
type Slot struct {
	Num uint64
	Ref unsafe.Pointer
}

// Offsets of Context's fields, which generated code addresses.
var (
	OffLocals    = int32(unsafe.Offsetof(Context{}.Locals))
	OffStack     = int32(unsafe.Offsetof(Context{}.Stack))
	OffBackEdges = int32(unsafe.Offsetof(Context{}.BackEdges))
	OffUpvalues  = int32(unsafe.Offsetof(Context{}.Upvalues))
	OffThis      = int32(unsafe.Offsetof(Context{}.This))
	OffGlobal    = int32(unsafe.Offsetof(Context{}.Global))
	OffLexNames  = int32(unsafe.Offsetof(Context{}.LexNames))
	OffCharCode  = int32(unsafe.Offsetof(Context{}.CharCodeAt))
	OffExitKind  = int32(unsafe.Offsetof(Context{}.ExitKind))
	OffExitPC    = int32(unsafe.Offsetof(Context{}.ExitPC))
	OffExitDepth = int32(unsafe.Offsetof(Context{}.ExitDepth))
	OffExitSite  = int32(unsafe.Offsetof(Context{}.ExitSite))
	OffRet       = int32(unsafe.Offsetof(Context{}.Ret))
	OffRetFrom   = int32(unsafe.Offsetof(Context{}.RetFrom))
	OffRecords   = int32(unsafe.Offsetof(Context{}.Records))
	OffRecord    = int32(unsafe.Offsetof(Context{}.Record))
	OffSpill     = int32(unsafe.Offsetof(Context{}.Spill))
)

// Exit kinds, written to Context.ExitKind.
const (
	// ExitReturn: Ret holds the result's number word, or RetFrom names
	// the slot that holds it.
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
	// Object is every object's number word: its pointer word tells objects
	// apart.
	Object uint64

	// What native code reads of an object, at these offsets from the
	// pointer a slot holds (docs/jit-production-plan.md, D8). It writes
	// nothing there but the number words of elements that hold numbers.
	//
	// ObjectClass is the class byte, ClassArray an array's. ObjectFlags is
	// a flags byte, in which FlagSparse marks an array with elements kept as
	// properties past its dense ones, whose length is then the uint32 at
	// ObjectArrayLen when that is larger. ObjectElems is the dense elements'
	// slice: a pointer to the first value, then the count.
	ObjectClass, ObjectFlags, ObjectArrayLen, ObjectElems int32
	ClassArray, FlagSparse                                uint8
	// FlagHTMLDDA is the flags byte's bit for Annex B's [[IsHTMLDDA]],
	// which makes an object == null and == undefined.
	FlagHTMLDDA uint8

	// String is every string's number word. A string's UTF-8 form's data
	// pointer is at StringData, valid only when the pointer at StringLeft is
	// nil (a rope's halves); StringLength is its length in UTF-16 code
	// units, an int; a true byte at StringASCII says each byte is a code
	// unit; and StringU16, when not nil, points to its code units.
	String                                                       uint64
	StringData, StringLeft, StringLength, StringASCII, StringU16 int32

	// UpvalueSlot is the offset, in a captured binding's cell, of the
	// pointer to its value (Context.Upvalues).
	UpvalueSlot int32

	// An object's properties (D8): ObjectShape is the offset of its shape
	// pointer, which says where each property is and what it is, and
	// ObjectProps of its property table's slice. Each entry is PropertySize
	// bytes: its key, a uint32 atom, at PropertyKey, a flags byte at
	// PropertyFlags, and its value at PropertyValue. ObjectProto is the
	// offset of its prototype's pointer, which the shape does not settle.
	ObjectShape, ObjectProps, PropertySize, PropertyKey, PropertyFlags, PropertyValue int32
	ObjectProto                                                                       int32
	// An ordinary object of class ClassObject, whose table has at most
	// MaxScan entries, may be searched for a key, as the VM's own small
	// objects are. A property is plain data when its flags have none of
	// PropNotData, and a plain writable one when they have PropWritable of
	// PropNotWritable. PropUninit marks a binding in its temporal dead zone.
	ClassObject, PropNotData, PropNotWritable, PropWritable, PropUninit uint8
}

// MaxScan bounds the table native code searches for a key.
const MaxScan = 8

// MaxEqualUnits bounds the strings, in UTF-16 code units, native code
// compares by their bytes: it does not poll, and a longer comparison is
// Go's, which the collector can stop.
const MaxEqualUnits = 256

// NumberLimit bounds number words: a word below it holds a number, and
// every tag is at or above it.
const NumberLimit = 0xFFF8000000000000

// IsNumber reports whether a number word holds a number.
func IsNumber(word uint64) bool { return word < NumberLimit }
