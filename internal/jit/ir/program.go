// Package ir defines the pointer-free slot IR used to develop the optional
// native executor. Its Go evaluator is a correctness oracle, not a VM tier.
package ir

import (
	"math"
	"unsafe"
)

// ArrayView borrows dense storage for one bounded native entry. Each cell is
// sixteen bytes: numeric bits followed by a Go reference that native code must
// never change. Bits below NumberLimit identify existing numeric data properties.
// WritableHole, when nonzero, permits replacing that pointer-free hole marker
// with a number. The caller proves ordinary writable/extensible array storage
// and absence of inherited indexed properties before granting this permission.
// The caller owns and roots the storage, and rebuilds views after every callback.
// With NumberLimit zero, Data instead borrows an ordinary object's property
// table, DenseLength counts at most MaxProperties cells, and WritableHole is
// the numeric tag boundary. The two permissions are mutually exclusive.
// With both NumberLimit and WritableHole zero, a nonzero Length grants a
// ReferenceCell table instead: Length is its numeric tag boundary. Numeric
// operations follow Cell to live storage; reference reads additionally check
// the snapshot identity before copying its rooted handle. These permissions
// preserve the same 40-byte view and numeric native-entry ABI.
// Missing or incompatible fields exit to the host.
type ArrayView struct {
	Data         unsafe.Pointer
	DenseLength  uint64
	Length       uint64
	NumberLimit  uint64
	WritableHole uint64
}

// String handles reuse the view ABI: Data borrows flat bytes or UTF-16 units,
// DenseLength counts code units, and Length is the unit width (one or two).
// WritableHole, when nonzero, is the rooted intrinsic charCodeAt handle plus
// one, granted only after an ordinary live prototype lookup. String's distinct
// kind prevents numeric array/property operations from using this storage.
// The intrinsic's Opaque view has DenseLength=CharCodeAtBuiltin and no other
// permissions. The adapter retains all owners and refreshes after callbacks.
const CharCodeAtBuiltin = uint64(1 << 61)

// MaxProperties bounds the native linear search within one IR instruction.
const MaxProperties = 8

// PropertyCell describes borrowed own data. Only Flags' low three bits are
// ordinary attributes; bit zero grants writes. Native stores change Bits only
// after proving the existing value numeric, leaving Reference untouched.
type PropertyCell struct {
	Key       uint32
	Flags     uint8
	_         [3]byte
	Bits      uint64
	Reference unsafe.Pointer
}

// ReferenceCell grants a read of one selected own field. Cell borrows
// the live property; Bits and Reference must still match before Handle is used.
// Handle equal to MaxSlots denies a reference read. Numeric reads and writes
// use Cell's live value. Native code never writes or publishes the Go pointers.
type ReferenceCell struct {
	Key       uint32
	_         [4]byte
	Bits      uint64
	Reference unsafe.Pointer
	Cell      *PropertyCell
	Handle    uint64
}

// Kind identifies a scalar or a handle into Go-owned reference storage.
type Kind uint64

const (
	Undefined Kind = iota
	Number
	Boolean
	Null
	Uninitialized
	Opaque
	String
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
	ArrayRead
	ArrayWrite
	ArrayLength
	ArrayKey
	ArrayUpdate
	Insert3
	Host
	Insert2
	PropertyRead
	PropertyWrite
	BindingRead
	ReferenceRead
	Call
	StringMethod
	StringCode
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
	Int32
	BitAnd
	BitOr
	BitXor
	Shl
	Shr
	UShr
	BitNot
	// Mod is JavaScript's % on numbers, C's fmod: only the new pipeline's
	// lowering makes it, and a non-number exits to Go, as Eq does.
	Mod
)

// Instruction reads Left and Right and writes Dest (and Extra for paired
// operations). StoreLoad stores Left into Dest before reading Right into
// Extra, so aliased locals match bytecode's sequential store/load behavior.
// Update stores the increment/decrement in Dest and, when Extra >= 0, pushes
// either the old or new value there. Branch takes Target when its condition
// equals When; otherwise execution falls through. Check guards CheckSlot
// against the temporal dead zone before any writes.
// ArrayRead/ArrayWrite use Left as the array handle and Right as the numeric
// index; Third is the stored number. ArrayUpdate commits an updated index in
// Extra only after the read succeeds. ArrayKey guards without converting a key.
// Host exits before executing the corresponding VM instruction.
// PropertyRead/PropertyWrite and BindingRead search Left's table for Key. A
// BindingRead borrows one resolved numeric binding cell. Right supplies
// the stored number; any failed permission or type check takes a host exit.
// Call reads its function from Left, its optional receiver from Right, and
// Extra arguments starting at Third.Slot. Dest receives the result. Key is
// the sequential call-site index. Ordinary evaluation exits before the call;
// a prepared native dispatch graph may transfer to a guarded callee instead.
// StringMethod reads Left's intrinsic permission into Dest. StringCode reads
// the callee from Left, the string receiver from Right and the index from Third.
// Failed string permissions take an uncommitted Host exit.
type Instruction struct {
	Op        Op
	Operator  Operator
	Left      Operand
	Right     Operand
	Third     Operand
	Dest      int
	Extra     int
	Target    int
	When      bool
	Postfix   bool
	Check     bool
	CheckSlot int
	Key       uint32
	// Strict marks an Eq or Ne that is === or !==. On numbers the two
	// agree; the new pipeline compares with null and undefined natively,
	// where they do not.
	Strict bool
}

// StateMap describes state immediately before a bytecode instruction. PC is
// the instruction to resume, not the VM's PC after fetching it. Depth is the
// live operand count; -1 marks unreachable code. Slots [0, Locals) hold locals
// and read-only captured bindings, receiver snapshots and binding-view handles;
// the next Depth slots hold operands.
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
	// This reserves a read-only receiver snapshot after captured bindings.
	This bool
	// Globals contains source-name indices for reserved binding-view slots,
	// following captured bindings and the optional receiver snapshot.
	Globals []uint32
	Code    []Instruction
	Maps    []StateMap
}

// ExitKind identifies a completed return or a resumable exit.
type ExitKind uint8

const (
	Returned ExitKind = iota
	GuardExit
	BudgetExit
	HostExit
)

// MaxCallSites bounds the call instructions in one program; each has its
// index in Key.
const MaxCallSites = 16

// Exit records a result or the exact interpreter state to resume. Steps counts
// committed IR instructions; a failing guard does not consume a step.
type Exit struct {
	Kind  ExitKind
	State StateMap
	Value Value
	Steps uint64
}
