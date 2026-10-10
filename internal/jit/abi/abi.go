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

// MaxKeeps is how many references a function's code keeps (Context.Keep).
const MaxKeeps = 64

// Record is an exit's instruction to Go for one slot, which native code
// cannot write. Slot is the slot, and its flags say what to write there:
//   - none: the value of slot Arg, a reference;
//   - RecordScalar: the primitive whose number word is Word, over a
//     reference, whose pointer word only Go may clear;
//   - RecordMaybe: the value at address Arg, a source (ssa's origin.go), if
//     that holds a reference, and otherwise -- or if Arg is 0 -- the
//     primitive Word.
//   - RecordDirect: the value whose number word is Word and whose pointer
//     word is the context's RecordRef at the record's index, as a native
//     call read them before the callee ran (ssa's OpCall).
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
	RecordDirect = 1 << 61
)

// Context is the block native code reaches through its context register.
// Go sets the frame's addresses before every entry; native code writes the
// exit record and spills, and, for a native call, the callee's context.
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
	// or the address of the value, a reference, that is: a source (ssa's
	// origin.go).
	Ret, RetFrom uint64
	// A native call (mir's): a caller's code sets up its callee's frame in
	// the VM's stack, and its context, the next in the runtime's array of
	// them (ContextSize apart), and jumps to its code. ReturnTo is where a
	// return goes in its caller's code, or 0 for Run's caller in Go;
	// RetValue is the value it returns there, both words. Live marks a
	// context a native call runs in; Closure, Base and Level say whose
	// frame it is -- the VM's closure, the frame's index in the VM's stack,
	// and the context's in the array -- for Go to make the VM's frames from
	// when a callee leaves native code; Level stays below LevelLimit.
	// StackBase and StackEnd are the VM's stack's first value and length,
	// StackTop and StackHigh its next free and highest used indices, which
	// a call moves. TailReturn is where the records' tail goes on, in a
	// caller's code, instead of returning to Go: 0 but during a call.
	ReturnTo   uintptr
	RetValue   Slot
	Live       uint64
	Closure    unsafe.Pointer
	Base       uint64
	Level      uint64
	LevelLimit uint64
	StackBase  unsafe.Pointer
	StackEnd   uint64
	StackTop   *int
	StackHigh  *int
	TailReturn uintptr
	// An inlined callee's frame, which an exit inside it wrote past its
	// caller's operands (ssa's InlineState), is in the next context, whose
	// Live is LiveInline: InlineClosure is the callee's closure's address,
	// which the caller's code keeps alive and Go finds among those it
	// inlined; InlineLocals its program's locals, past which its operands
	// are; InlineThis 1 plus its receiver's slot, or 0. Base is where the
	// frame is; the exit fields say where it left.
	InlineClosure, InlineLocals, InlineThis uint64
	// Records counts the Record entries an exit filled. Every other slot of
	// its state is in the frame already.
	Records uint64
	Record  [MaxRecords]Record
	// RecordRef are direct records' pointer words (RecordDirect), where
	// the collector sees them; RecordHigh bounds those written, which Go
	// clears when native code returns.
	RecordRef  [MaxRecords]unsafe.Pointer
	RecordHigh uint64
	// Spill holds what the allocator could not keep in registers.
	Spill [SpillSlots]uint64
	// Keep holds references copied out of cells a store then overwrites
	// while slots still hold them (ssa's OpKeep), read from here since: the
	// collector sees them, as it does the frame's; they are written only
	// while it does not mark. Go clears them when the code returns.
	Keep [MaxKeeps]Slot
	// ExitDesc is an ExitTable exit's description, an *ExitDescriptor,
	// which the code holds; Regs and XRegs are the registers the exit
	// saved, by number, whose words the description says where to find:
	// Go writes the frame from them (jit.ApplyExit), as V8's deoptimizer
	// reads a frame state's translation.
	ExitDesc uintptr
	Regs     [32]uint64
	XRegs    [32]uint64
}

// ExitDescriptor is an exit's frame state as data (an ExitTable exit): the
// slots to write and where each one's value is, the inlined callees'
// frames to describe, and the exit's own kind, PC, depth and site, which
// Go puts in the context once it has written the frame -- the frame as an
// exit's code would have written it, records and all.
type ExitDescriptor struct {
	Kind, PC, Depth uint64
	Site            int64
	Slots           []ExitSlot
	// Inline are the inlined callees' frames, the outermost first, each in
	// a context of its own past the code's.
	Inline []ExitInline
	// The frame's layout: its locals, from LocalsReg's address; the
	// receiver's slot, in the context; the captured bindings, through
	// their cells; and the operands, from StackReg's address. Enc is the
	// values' encoding, one for all the code's descriptions.
	FrameLocals, ThisSlot, Locals int32
	LocalsReg, StackReg           uint8
	Enc                           *Encoding
	// Direct marks a description whose slots may be written one by one:
	// none takes a reference from another it writes, by its origin or a
	// source that may be one's address.
	Direct bool
}

// ExitSlot is a slot an exit writes: its value's number word, from Value;
// and its pointer word -- from Shadow, the source the value's reference
// was read from, if it has one; from slot Origin's as the code was entered,
// if that held the reference (Load: the value is that slot's load, so its
// word need not match); none otherwise.
type ExitSlot struct {
	Slot   int32
	Origin int32
	Load   bool
	// At and OriginAt are where slots Slot and Origin are.
	At, OriginAt ExitAddr
	Value        ExitLoc
	Shadow       ExitLoc
}

// ExitAddr is where a slot is: Off bytes into the locals or the operands,
// the receiver in the context, or the cell of captured binding Off.
type ExitAddr struct {
	Base uint8
	Off  int32
}

// ExitAddr's bases.
const (
	ExitInLocals uint8 = iota
	ExitInStack
	ExitInThis
	ExitInCell
)

// ExitLoc is where an exit finds a word: in a register or a spill slot as
// it is, or boxed from a float or a boolean there; a constant; a slot's
// address; or none (ExitNone).
type ExitLoc struct {
	Kind uint8
	N    int32
	Word uint64
}

// ExitLoc's kinds.
const (
	ExitNone uint8 = iota
	ExitReg
	ExitSpill
	ExitConst
	ExitF64Reg
	ExitF64Spill
	ExitBoolReg
	ExitBoolSpill
	ExitSlotAddr
)

// ExitInline is an inlined callee's frame an exit describes in a context
// (LiveInline): what an exit's code writes there.
type ExitInline struct {
	Closure, Locals, ThisSlot uint64
	Kind, PC, Depth           uint64
	Site, Base                int64
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
	OffRecordRef = int32(unsafe.Offsetof(Context{}.RecordRef))
	OffRecordHi  = int32(unsafe.Offsetof(Context{}.RecordHigh))
	OffSpill     = int32(unsafe.Offsetof(Context{}.Spill))
	OffKeep      = int32(unsafe.Offsetof(Context{}.Keep))

	OffReturnTo   = int32(unsafe.Offsetof(Context{}.ReturnTo))
	OffRetValue   = int32(unsafe.Offsetof(Context{}.RetValue))
	OffLive       = int32(unsafe.Offsetof(Context{}.Live))
	OffClosure    = int32(unsafe.Offsetof(Context{}.Closure))
	OffBase       = int32(unsafe.Offsetof(Context{}.Base))
	OffLevel      = int32(unsafe.Offsetof(Context{}.Level))
	OffLevelLimit = int32(unsafe.Offsetof(Context{}.LevelLimit))
	OffStackBase  = int32(unsafe.Offsetof(Context{}.StackBase))
	OffStackEnd   = int32(unsafe.Offsetof(Context{}.StackEnd))
	OffStackTop   = int32(unsafe.Offsetof(Context{}.StackTop))
	OffStackHigh  = int32(unsafe.Offsetof(Context{}.StackHigh))
	OffTailReturn = int32(unsafe.Offsetof(Context{}.TailReturn))

	OffExitDesc = int32(unsafe.Offsetof(Context{}.ExitDesc))
	OffRegs     = int32(unsafe.Offsetof(Context{}.Regs))
	OffXRegs    = int32(unsafe.Offsetof(Context{}.XRegs))

	OffInlineClosure = int32(unsafe.Offsetof(Context{}.InlineClosure))
	OffInlineLocals  = int32(unsafe.Offsetof(Context{}.InlineLocals))
	OffInlineThis    = int32(unsafe.Offsetof(Context{}.InlineThis))

	// ContextSize is a Context's size, and so the distance from one to the
	// next in an array of them.
	ContextSize = int32(unsafe.Sizeof(Context{}))
)

// ObjectPool is the objects a construction site native code makes, `new
// C(...)`, takes for C's receiver: made by Go as the VM makes them for a
// construction -- C's prototype, its root shape, room for what its body
// adds -- the last Count of Objects, each cleared as native code takes it;
// Proto is the prototype they were made with, which native code compares
// C's own with first. Go fills it when the site leaves native code for
// want of one.
type ObjectPool struct {
	Objects [PoolSize]unsafe.Pointer
	Count   uint64
	Proto   unsafe.Pointer
}

// PoolSize is how many objects an ObjectPool holds.
const PoolSize = 16

// ObjectPool's offsets.
const (
	OffPoolObjects = int32(unsafe.Offsetof(ObjectPool{}.Objects))
	OffPoolCount   = int32(unsafe.Offsetof(ObjectPool{}.Count))
	OffPoolProto   = int32(unsafe.Offsetof(ObjectPool{}.Proto))
)

// What a context's Live says: a native call's callee runs in it, or an
// inlined callee's frame is described in it.
const (
	LiveCall   = 1
	LiveInline = 2
)

// Exit kinds, written to Context.ExitKind.
const (
	// ExitReturn: Ret holds the result's number word, or RetFrom is where
	// the result is.
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
	// ExitTable: an exit that left its frame to Go to write, from
	// ExitDesc and the registers it saved; Go then puts the exit's own
	// kind here. No one but the code that runs native code sees it.
	ExitTable
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
	// FlagExtensible is its bit for an object properties may be added to,
	// FlagLengthWritable an array's whose length may change.
	FlagExtensible, FlagLengthWritable uint8

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

	// WriteBarrier is the address of the byte compiled Go tests before it
	// stores a pointer: the runtime's write-barrier flag, set while the
	// collector marks. A store that changes a pointer word exits to Go
	// while it is set, and otherwise stores, as compiled Go does. It
	// cannot change between the test and the store: Go sets it only with
	// the world stopped, which native code, never a preemption point, is
	// not (docs/jit-progress.md, decided 2026-10-09).
	WriteBarrier uint64
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
