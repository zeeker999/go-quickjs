package bytecode

import (
	"fmt"
	"strings"
)

// The register form of a function's code: what the register interpreter
// runs (docs/register-interpreter.md).
//
// A frame is a window of registers in the runtime's value stack. The first
// LocalCount are the function's locals, in the slots the stack form gives
// them, so that an upvalue, a stack trace and the debugger find a local
// where they always did; the temporaries the stack form keeps on its operand
// stack follow, one register for each depth that stack reaches, and then the
// few registers the register form adds of its own.
//
// An instruction is one 64-bit word, read as a struct of an operation and
// four operands:
//
//	bits  0- 7  Op  the operation
//	bits  8-15  X   a small operand: an argument count, a comparison, a flag
//	bits 16-31  A   a register, most often the destination
//	bits 32-47  B   a register, an index or a small immediate
//	bits 48-63  C   a register, an index or a small immediate
//
// B and C read together are BC, a 32-bit index or a signed jump offset. A
// register is 16 bits, so a function has up to 65,536 of them without any
// form of its own for more; a function that needs more registers runs in the
// stack form until the far moves of the design's later stage exist.

// RInstr is one register instruction.
type RInstr uint64

// ROp is a register operation.
type ROp uint8

// MakeR builds an instruction from its fields.
func MakeR(op ROp, x uint8, a, b, c uint16) RInstr {
	return RInstr(uint64(op) | uint64(x)<<8 | uint64(a)<<16 | uint64(b)<<32 | uint64(c)<<48)
}

// MakeRBC builds an instruction whose B and C are one 32-bit operand.
func MakeRBC(op ROp, x uint8, a uint16, bc uint32) RInstr {
	return RInstr(uint64(op) | uint64(x)<<8 | uint64(a)<<16 | uint64(bc)<<32)
}

// Op is the instruction's operation.
func (w RInstr) Op() ROp { return ROp(w) }

// X is the instruction's 8-bit operand.
func (w RInstr) X() uint8 { return uint8(w >> 8) }

// A is the instruction's first 16-bit operand.
func (w RInstr) A() uint16 { return uint16(w >> 16) }

// B is the instruction's second 16-bit operand.
func (w RInstr) B() uint16 { return uint16(w >> 32) }

// C is the instruction's third 16-bit operand.
func (w RInstr) C() uint16 { return uint16(w >> 48) }

// BC is B and C read as one 32-bit operand.
func (w RInstr) BC() uint32 { return uint32(w >> 32) }

// SC is C read as a signed 16-bit operand: an immediate or a jump offset.
func (w RInstr) SC() int32 { return int32(int16(w >> 48)) }

// SB is B read as a signed 16-bit immediate.
func (w RInstr) SB() int32 { return int32(int16(w >> 32)) }

// WithSC is w with C replaced by the signed offset off, which the caller
// has checked fits in 16 bits.
func (w RInstr) WithSC(off int32) RInstr {
	return w&^(0xffff<<48) | RInstr(uint64(uint16(int16(off)))<<48)
}

// WithBC is w with BC replaced by bc.
func (w RInstr) WithBC(bc uint32) RInstr {
	return w&^(0xffffffff<<32) | RInstr(uint64(bc)<<32)
}

// WithA is w with A replaced by a.
func (w RInstr) WithA(a uint16) RInstr {
	return w&^(0xffff<<16) | RInstr(uint64(a)<<16)
}

// The register operations. R[n] is register n of the frame; an offset is
// relative to the instruction after the jump. Each operation that reads a
// property, a global or a name keeps the stack form's cache site, so that
// the two forms of a function share its caches.
const (
	RNop ROp = iota
	// RMove: R[A] = R[B].
	RMove
	// RLoadInt: R[A] = int32(BC).
	RLoadInt
	// RLoadConst: R[A] = Constants[BC].
	RLoadConst
	// RLoadUndef, RLoadNull, RLoadTrue, RLoadFalse, RLoadEmpty and
	// RLoadUninit load what they name into R[A]: RLoadEmpty the empty
	// string, RLoadUninit the dead zone's marker.
	RLoadUndef
	RLoadNull
	RLoadTrue
	RLoadFalse
	RLoadEmpty
	RLoadUninit
	// RLoadThis: R[A] = this, which a frame the register form runs has
	// bound from its start.
	RLoadThis
	// RCheckRef throws the ReferenceError of a binding in its dead zone if
	// R[A] is the dead zone's marker; the message names Locals[B].
	RCheckRef
	// RGetUpval: R[A] = upvalue BC. RSetUpval: upvalue BC = R[A].
	RGetUpval
	RSetUpval
	// RGetGlobal: R[A] = the global Names[B], site C.
	RGetGlobal
	// RSetGlobal: the global Names[B], site C, = R[A], as set_global.
	RSetGlobal
	// RGetProp: R[A] = R[B].name, site C, as get_prop; the site's name is
	// RegCode.SiteName[C].
	RGetProp
	// RGetMethod: R[A] = R[B].name, site C, as get_prop_this, the read of a
	// method a call is about to make.
	RGetMethod
	// RSetProp: R[A].name = R[B], site C, as set_prop.
	RSetProp
	// RGetIndex: R[A] = R[B][R[C]].
	RGetIndex
	// RSetIndex: R[A][R[B]] = R[C].
	RSetIndex
	// RDefineField: defines R[A].Names[C] = R[B], as define_field.
	RDefineField
	// RNewObject: R[A] = a new object literal of B properties, site C.
	RNewObject
	// RNewArray: R[A] = an array of the C values from R[B].
	RNewArray
	// RClosure: R[A] = a closure of Constants[BC].
	RClosure
	// RAdd, RSub, RMul: R[A] = R[B] op R[C].
	RAdd
	RSub
	RMul
	// RBin: R[A] = R[B] op R[C], where X is the stack form's Op of the
	// operator: a division, a remainder, a power, a bitwise operator or a
	// shift, an equality or a relational comparison.
	RBin
	// RBinI: R[A] = R[B] op SC, where X is the stack form's Op of the
	// operator: add, sub, mul, a bitwise operator or a shift.
	RBinI
	// RInc and RDec: R[A] = ToNumeric(R[B]) + 1 or - 1.
	RInc
	RDec
	// RUnary: R[A] = op R[B], where X is the stack form's Op: neg, pos,
	// not, bit_not or typeof.
	RUnary
	// RJmp: jump by int32(BC).
	RJmp
	// RJmpF and RJmpT: jump by int32(BC) if R[A] is falsy or truthy.
	RJmpF
	RJmpT
	// RJmpCmp: jump by SC if R[A] op R[B] is X's RCmpTrue bit, where X's low
	// bits are the stack form's Op of the comparison.
	RJmpCmp
	// RJmpCmpI is RJmpCmp with the right operand the immediate SB.
	RJmpCmpI
	// RRet returns R[A]; RRetUndef returns undefined.
	RRet
	RRetUndef
	// RCall: R[B] = R[A](R[A+1] ... R[A+X]), with this undefined.
	RCall
	// RCallMethod: R[B] = R[A+1](R[A+2] ... R[A+1+X]), with this R[A].
	RCallMethod
	// RNew: R[B] = new R[A](R[A+1] ... R[A+X]).
	RNew
	// RCloseUpvals closes the upvalues of the locals from A up, as
	// close_upvalues does when a block that captured bindings ends.
	RCloseUpvals

	// RInvalid is the last operation there can be, which no code holds:
	// the interpreter's switch has a case for it so that its jump table
	// spans every value of an ROp and needs no range check.
	RInvalid ROp = 255
)

// RCmpTrue is the bit of RJmpCmp's X that says it jumps when the
// comparison holds, rather than when it does not.
const RCmpTrue = 0x80

// RegCode is a function's code in the register form.
type RegCode struct {
	Code []RInstr
	// NumRegs is how many registers a frame of the code uses: the
	// function's locals, its temporaries, and the register form's own.
	NumRegs int
	// SrcPC is, for each instruction, the stack form's instruction it was
	// made from: what a frame's pc holds while the instruction runs, so
	// that positions, stack traces and the debugger read the stack form's
	// tables as they always did.
	SrcPC []uint32
	// SiteName is the name of each property site an instruction names,
	// indexed by its cache site.
	SiteName []uint32
	// This is the register the function's this is loaded into as it
	// starts, or -1 if the code reads no this.
	This int
}

var rOpNames = [...]string{
	RNop: "nop", RMove: "move", RLoadInt: "load_int", RLoadConst: "load_const",
	RLoadUndef: "load_undef", RLoadNull: "load_null", RLoadTrue: "load_true",
	RLoadFalse: "load_false", RLoadEmpty: "load_empty", RLoadUninit: "load_uninit",
	RLoadThis: "load_this", RCheckRef: "check_ref",
	RGetUpval: "get_upval", RSetUpval: "set_upval",
	RGetGlobal: "get_global", RSetGlobal: "set_global",
	RGetProp: "get_prop", RGetMethod: "get_method", RSetProp: "set_prop",
	RGetIndex: "get_index", RSetIndex: "set_index", RDefineField: "define_field",
	RNewObject: "new_object", RNewArray: "new_array", RClosure: "closure",
	RAdd: "add", RSub: "sub", RMul: "mul", RBin: "bin", RBinI: "bin_imm",
	RInc: "inc", RDec: "dec", RUnary: "unary",
	RJmp: "jmp", RJmpF: "jmp_false", RJmpT: "jmp_true", RJmpCmp: "jmp_cmp",
	RJmpCmpI: "jmp_cmp_imm", RRet: "ret", RRetUndef: "ret_undef",
	RCall: "call", RCallMethod: "call_method", RNew: "new",
	RCloseUpvals: "close_upvals",
}

func (op ROp) String() string {
	if int(op) < len(rOpNames) && rOpNames[op] != "" {
		return rOpNames[op]
	}
	return fmt.Sprintf("rop(%d)", uint8(op))
}

// Format renders the instruction at pc, for tests and disassembly.
func (rc *RegCode) Format(pc int) string {
	w := rc.Code[pc]
	r := func(n uint16) string { return fmt.Sprintf("r%d", n) }
	target := func(off int32) string { return fmt.Sprintf("-> %d", pc+1+int(off)) }
	cmp := func() string {
		s := Op(w.X() &^ RCmpTrue).String()
		if w.X()&RCmpTrue == 0 {
			return "!" + s
		}
		return s
	}
	switch op := w.Op(); op {
	case RMove, RInc, RDec:
		return fmt.Sprintf("%s %s, %s", op, r(w.A()), r(w.B()))
	case RLoadInt:
		return fmt.Sprintf("%s %s, %d", op, r(w.A()), int32(w.BC()))
	case RLoadConst, RClosure:
		return fmt.Sprintf("%s %s, k%d", op, r(w.A()), w.BC())
	case RLoadUndef, RLoadNull, RLoadTrue, RLoadFalse, RLoadEmpty, RLoadUninit, RLoadThis, RRet, RCloseUpvals:
		return fmt.Sprintf("%s %s", op, r(w.A()))
	case RCheckRef:
		return fmt.Sprintf("%s %s, local %d", op, r(w.A()), w.B())
	case RGetUpval:
		return fmt.Sprintf("%s %s, u%d", op, r(w.A()), w.BC())
	case RSetUpval:
		return fmt.Sprintf("%s u%d, %s", op, w.BC(), r(w.A()))
	case RGetGlobal:
		return fmt.Sprintf("%s %s, name%d @%d", op, r(w.A()), w.B(), w.C())
	case RSetGlobal:
		return fmt.Sprintf("%s name%d @%d, %s", op, w.B(), w.C(), r(w.A()))
	case RGetProp, RGetMethod:
		return fmt.Sprintf("%s %s, %s @%d", op, r(w.A()), r(w.B()), w.C())
	case RSetProp:
		return fmt.Sprintf("%s %s @%d, %s", op, r(w.A()), w.C(), r(w.B()))
	case RDefineField:
		return fmt.Sprintf("%s %s name%d, %s", op, r(w.A()), w.C(), r(w.B()))
	case RGetIndex, RAdd, RSub, RMul:
		return fmt.Sprintf("%s %s, %s, %s", op, r(w.A()), r(w.B()), r(w.C()))
	case RSetIndex:
		return fmt.Sprintf("%s %s[%s], %s", op, r(w.A()), r(w.B()), r(w.C()))
	case RNewObject:
		return fmt.Sprintf("%s %s, %d @%d", op, r(w.A()), w.B(), w.C())
	case RNewArray:
		return fmt.Sprintf("%s %s, %s x%d", op, r(w.A()), r(w.B()), w.C())
	case RBin:
		return fmt.Sprintf("%s.%s %s, %s, %s", op, Op(w.X()), r(w.A()), r(w.B()), r(w.C()))
	case RBinI:
		return fmt.Sprintf("%s.%s %s, %s, %d", op, Op(w.X()), r(w.A()), r(w.B()), w.SC())
	case RUnary:
		return fmt.Sprintf("%s.%s %s, %s", op, Op(w.X()), r(w.A()), r(w.B()))
	case RJmp:
		return fmt.Sprintf("%s %s", op, target(int32(w.BC())))
	case RJmpF, RJmpT:
		return fmt.Sprintf("%s %s %s", op, r(w.A()), target(int32(w.BC())))
	case RJmpCmp:
		return fmt.Sprintf("%s %s %s, %s %s", op, cmp(), r(w.A()), r(w.B()), target(w.SC()))
	case RJmpCmpI:
		return fmt.Sprintf("%s %s %s, %d %s", op, cmp(), r(w.A()), w.SB(), target(w.SC()))
	case RCall, RNew:
		return fmt.Sprintf("%s %s = %s(%d)", op, r(w.B()), r(w.A()), w.X())
	case RCallMethod:
		return fmt.Sprintf("%s %s = %s.%s(%d)", op, r(w.B()), r(w.A()), r(w.A()+1), w.X())
	default:
		return op.String()
	}
}

// String renders the whole code, one instruction a line.
func (rc *RegCode) String() string {
	var sb strings.Builder
	for pc := range rc.Code {
		fmt.Fprintf(&sb, "%4d  %s\n", pc, rc.Format(pc))
	}
	return sb.String()
}
