//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"math"
	"slices"
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
	"github.com/go-quickjs/go-quickjs/internal/jit/ssa"
)

// jitRegFeedback is the new pipeline's feedback for a function's register
// code, the program compile.LowerReg makes of it (docs/register-
// interpreter.md, Stage 3, section 3, item 3): asked by the program's
// instructions, as the builder asks, and read through each one's origin,
// the register pc, from the caches the register interpreter fills -- a
// property site by its instruction's C, a global's by its C and name, a
// call's by its C (regCalls).
//
// It answers what register code records today. What the register
// interpreter does not record yet -- what a fast path did not expect
// (Generic), call counts at a site (Frequency), the JIT's own tables
// learned at its exits, which the stack form's code keys by its stack pcs
// -- it answers with nothing, and the builder then speculates nothing
// there. It is the stack form's jitFeedback's other face, of its own type
// while the stack form's JIT stays the default: no method of jitFeedback
// asks which form it serves, and the snapshot of a background compile
// (zzjit_snapshot_gen.go) is of jitFeedback alone.
type jitRegFeedback struct {
	fb *jitFeedback
	rc *bytecode.RegCode
	p  *ir.Program
}

// jitRegFeedbackFor is cl's register code's feedback for p, the program
// LowerReg made of it.
func (r *Runtime) jitRegFeedbackFor(cl *closure, p *ir.Program) *jitRegFeedback {
	if cl == nil || cl.fn == nil || cl.fn.RegCode() == nil || p == nil || p.Origin == nil {
		return nil
	}
	return &jitRegFeedback{fb: &jitFeedback{r: r, fn: cl.fn, cl: cl}, rc: cl.fn.RegCode(), p: p}
}

// at is the register instruction of the program's instruction pc, and what
// that instruction is.
func (f *jitRegFeedback) at(pc int) (bytecode.RInstr, ir.Op, bool) {
	if pc < 0 || pc >= len(f.p.Code) || int(f.p.Origin[pc]) >= len(f.rc.Code) {
		return 0, 0, false
	}
	return f.rc.Code[f.p.Origin[pc]], f.p.Code[pc].Op, true
}

// name is the atom of the register instruction w's name operand n, at
// register pc rpc (bytecode.RNameFar).
func (f *jitRegFeedback) name(rpc int, n uint16) (Atom, bool) {
	i := uint32(n)
	if n == bytecode.RNameFar {
		if rpc >= len(f.rc.FarNames) {
			return 0, false
		}
		i = f.rc.FarNames[rpc]
	}
	if int(i) >= len(f.fb.cl.names) {
		return 0, false
	}
	return f.fb.cl.names[i], true
}

// Property is ssa.Feedback's: the property site of a read or a write the
// program's instruction pc makes -- get_prop's, get_method's,
// get_method_w's, set_prop's, or add_prop's read or store, which share the
// store's site -- as its cache, by the register instruction's C, knows it.
// A cache that gave up, or holds dictionary objects or a chain's cell, is
// megamorphic; a read whose cache never filled searches the receiver's own
// table (Scan), as the stack form's.
func (f *jitRegFeedback) Property(pc int) (ssa.PropertySite, bool) {
	w, op, ok := f.at(pc)
	if !ok || f.fb.cl == nil {
		return ssa.PropertySite{}, false
	}
	switch w.Op() {
	case bytecode.RGetProp, bytecode.RGetMethod, bytecode.RGetMethodW, bytecode.RSetProp, bytecode.RAddProp, bytecode.RAddPropI:
	default:
		return ssa.PropertySite{}, false
	}
	if op != ir.PropertyRead && op != ir.ReferenceRead && op != ir.PropertyWrite {
		return ssa.PropertySite{}, false
	}
	c := int(w.C())
	if c >= len(f.rc.SiteName) || int(f.rc.SiteName[c]) >= len(f.fb.cl.names) {
		return ssa.PropertySite{}, false
	}
	key := f.fb.cl.names[f.rc.SiteName[c]]
	if c >= len(f.fb.cl.ic) {
		return ssa.PropertySite{Key: uint32(key)}, true
	}
	cache := &f.fb.cl.ic[c]
	write := op == ir.PropertyWrite
	site, ok := f.fb.siteOf(key, write, cache)
	if !ok {
		return site, false
	}
	site.Megamorphic = cache.fills >= maxCacheFills || cache.shape == dictShape || cache.cell != nil
	if !write && !site.Megamorphic && w.Op() == bytecode.RGetProp && cache.fills == 0 {
		site.Scan = true
	}
	return site, true
}

// Global is ssa.Feedback's: the global a read, a write or a strict
// assignment's check at the program's instruction pc names, found where
// the global object or a script's lexical bindings have it, starting from
// the slot its cache, by the register instruction's C, remembers.
func (f *jitRegFeedback) Global(pc int) (ssa.GlobalSite, bool) {
	w, op, ok := f.at(pc)
	if !ok || f.fb.cl == nil {
		return ssa.GlobalSite{}, false
	}
	rpc := int(f.p.Origin[pc])
	site, write, check := int(w.C()), false, false
	switch w.Op() {
	case bytecode.RGetGlobal:
	case bytecode.RSetGlobal:
		write = true
	case bytecode.RGetGlobalOpt:
		// typeof's read, which has no site of its own.
		site = -1
	case bytecode.RCheckGlobalRef:
		check = true
	default:
		return ssa.GlobalSite{}, false
	}
	if op != ir.BindingRead && op != ir.BindingWrite && op != ir.BindingCheck {
		return ssa.GlobalSite{}, false
	}
	name, ok := f.name(rpc, w.B())
	if !ok {
		return ssa.GlobalSite{}, false
	}
	return f.fb.globalSite(name, site, write, check)
}

// CallShape is ssa.CalledFeedback's: the argument count of the call at the
// program's instruction pc, and whether it is a method's -- call_w's and
// call_method_w's, not one in tail position.
func (f *jitRegFeedback) CallShape(pc int) (int, bool, bool) {
	w, _, ok := f.at(pc)
	if !ok {
		return 0, false, false
	}
	rpc := int(f.p.Origin[pc])
	if rpc < len(f.rc.Marks) && f.rc.Marks[rpc]&bytecode.RMarkTail != 0 {
		return 0, false, false
	}
	switch w.Op() {
	case bytecode.RCallW:
		return int(w.X()), false, true
	case bytecode.RCallMW:
		return int(w.X()), true, true
	}
	return 0, false, false
}

// CalledClosure is ssa.CalledClosureFeedback's: nothing yet -- the closure
// made at pc inlined where a call calls it takes its program from its
// register code, which Stage 3a-3's inliner lowers.
func (f *jitRegFeedback) CalledClosure(pc int) (ssa.MadeCallback, ssa.InlineSite, bool) {
	return ssa.MadeCallback{}, ssa.InlineSite{}, false
}

// callTarget is the function the call at the program's instruction pc
// last called, as its site's cache (regCalls, by the register
// instruction's C) holds it: a monomorphic target from the first compile,
// which the stack form's JIT learns at its own exits. Its closure is nil
// for a built-in.
func (f *jitRegFeedback) callTarget(pc int) (*Object, *closure, bool) {
	w, _, ok := f.at(pc)
	if !ok || f.fb.cl == nil {
		return nil, nil, false
	}
	switch w.Op() {
	case bytecode.RCallW, bytecode.RCallMW, bytecode.RNewW:
	default:
		return nil, nil, false
	}
	calls := f.fb.cl.regCalls
	if int(w.C()) >= len(calls) || calls[w.C()].callee == nil {
		return nil, nil, false
	}
	k := &calls[w.C()]
	return k.callee, k.cl, true
}

// constant is the constant a StringConst or a BigIntConst at the
// program's instruction pc reads -- load_const's, a switch's case's, or
// load_empty's empty string -- and whether it is one of op's.
func (f *jitRegFeedback) constant(pc int, op ir.Op) (Value, bool) {
	w, iop, ok := f.at(pc)
	if !ok || iop != op || f.fb.cl == nil {
		return Value{}, false
	}
	if w.Op() == bytecode.RLoadEmpty {
		return Str(emptyString), op == ir.StringConst
	}
	k := f.p.Code[pc].Key
	if w.Op() != bytecode.RLoadConst && w.Op() != bytecode.RSwitch || int(k) >= len(f.fb.cl.consts) {
		return Value{}, false
	}
	v := f.fb.cl.consts[k]
	if op == ir.StringConst && !v.IsString() || op == ir.BigIntConst && !v.IsBigInt() {
		return Value{}, false
	}
	return v, true
}

// StringCell is ssa.ConstantFeedback's: a cell holding the string constant
// the program's instruction pc reads, as jitFeedback's.
func (f *jitRegFeedback) StringCell(pc int) (uintptr, bool) {
	v, ok := f.constant(pc, ir.StringConst)
	if !ok {
		return 0, false
	}
	cell := new(Value)
	*cell = v
	k := f.fb.keep()
	k.cells = append(k.cells, cell)
	return uintptr(unsafe.Pointer(cell)), true
}

// StringContent is ssa.StringContentFeedback's: the string the constant at
// the program's instruction pc is, flat.
func (f *jitRegFeedback) StringContent(pc int) (string, bool) {
	v, ok := f.constant(pc, ir.StringConst)
	if !ok {
		return "", false
	}
	if s := v.String(); s.left == nil {
		return s.s, true
	}
	return "", false
}

// BigIntCell is ssa.BigIntConstFeedback's: the cell a copy of the BigInt
// constant at the program's instruction pc is in, as jitFeedback's.
func (f *jitRegFeedback) BigIntCell(pc int) (uintptr, uint64, int, bool) {
	v, ok := f.constant(pc, ir.BigIntConst)
	if !ok {
		return 0, 0, ssa.BigConstHeap, false
	}
	cell := new(Value)
	*cell = v
	k := f.fb.keep()
	k.cells = append(k.cells, cell)
	if cell.isInt64Big() {
		return uintptr(unsafe.Pointer(cell)), uint64(cell.int64Big()), ssa.BigConstInt64, true
	}
	if cell.isShortBig() {
		if neg, mag := cell.shortBigParts(); !neg {
			return uintptr(unsafe.Pointer(cell)), mag, ssa.BigConstUint64, true
		}
	}
	return uintptr(unsafe.Pointer(cell)), 0, ssa.BigConstHeap, true
}

// Generic is ssa.Feedback's: the register instruction of the program's
// instruction pc is one a guard of the function's code failed at before
// (jitState's regFailed), as the stack form's code learns at its exits:
// built generic, what it did not expect is Go's to make.
func (f *jitRegFeedback) Generic(pc int) bool {
	s := f.fb.r.jit
	if s == nil || s.regFailed == nil || pc < 0 || pc >= len(f.p.Origin) {
		return false
	}
	return slices.Contains(s.regFailed[f.fb.r.jitKeyOf(f.fb.fn)], int32(f.p.Origin[pc]))
}

// EntryGeneric is ssa.Feedback's: nothing, as Generic.
func (f *jitRegFeedback) EntryGeneric(pc int) bool { return false }

// Inline is ssa.Feedback's: nothing yet. The call's target is known
// (callTarget); making it an InlineSite takes the callee's program, which
// the stack form's JIT lowers from its stack code (Stage 3a-3).
func (f *jitRegFeedback) Inline(pc int) (ssa.InlineSite, bool) { return ssa.InlineSite{}, false }

// GoCalls is ssa.GoFeedback's: native code made from register code calls
// Go, which makes what needs no frame of the VM's from what the code
// passes it, and a call on the register frame (jitRegGoCall); what needs
// the stack form's frame, a call of Go refuses, and the code leaves there.
func (f *jitRegFeedback) GoCalls() bool { return true }

// NativeCalls is ssa.Feedback's: a call -- call_w's, call_method_w's or
// new_w's, its operands no more than a call of Go takes -- is a call of
// Go (ssa.GoCall), which makes it on the register frame without leaving
// native code (jitRegGoCall). A call in tail position is not, which the
// interpreter makes in constant space at a host exit.
func (f *jitRegFeedback) NativeCalls(pc int) []ssa.CallSite {
	w, op, ok := f.at(pc)
	if !ok || op != ir.Host || !f.p.Code[pc].Window {
		return nil
	}
	switch w.Op() {
	case bytecode.RTypeof:
		// Its name, which Go makes from native code.
		return []ssa.CallSite{{Go: ssa.GoCall, TypeOf: true, ThisSlot: -1}}
	case bytecode.RInstanceOf:
		// Go's, from native code, by the site's cache, as the interpreter
		// makes it: it may run code, a Symbol.hasInstance method or a
		// proxy's trap.
		return []ssa.CallSite{{Go: ssa.GoCall, InstanceOf: true, Argc: 1, ThisSlot: -1}}
	}
	rpc := int(f.p.Origin[pc])
	if rpc < len(f.rc.Marks) && f.rc.Marks[rpc]&bytecode.RMarkTail != 0 {
		return nil
	}
	argc, method := int(w.X()), w.Op() == bytecode.RCallMW
	n := argc + 1
	if method {
		n++
	}
	if n > abi.MaxGoArgs {
		return nil
	}
	return []ssa.CallSite{{Go: ssa.GoCall, Argc: argc, Method: method, New: w.Op() == bytecode.RNewW, ThisSlot: -1}}
}

// The runtime's addresses the builder asks once a function, the same for
// either form's code: jitFeedback's answers.

// ArrayProtos is ssa.AppendFeedback's, jitFeedback's.
func (f *jitRegFeedback) ArrayProtos() ([2]ssa.Holder, bool) { return f.fb.ArrayProtos() }

// GenNext is ssa.GenStepFeedback's half that is the realm's, jitFeedback's.
func (f *jitRegFeedback) GenNext() uintptr { return f.fb.GenNext() }

// GenSteps is ssa.GenStepFeedback's other half: no generator is known to
// a register loop's step yet (jitEntry's genSteps are the stack form's).
func (f *jitRegFeedback) GenSteps(pc int) []ssa.GenStep { return nil }

// HasCache is ssa.HasCacheFeedback's, jitFeedback's.
func (f *jitRegFeedback) HasCache() uintptr { return f.fb.HasCache() }

// KeyAtoms is ssa.KeyAtomsFeedback's, jitFeedback's.
func (f *jitRegFeedback) KeyAtoms() (uintptr, uintptr) { return f.fb.KeyAtoms() }

// MapNexts is ssa.MapStepFeedback's, jitFeedback's.
func (f *jitRegFeedback) MapNexts() [2]uintptr { return f.fb.MapNexts() }

// TypeofCells is ssa.TypeofFeedback's, jitFeedback's.
func (f *jitRegFeedback) TypeofCells() (uintptr, bool) { return f.fb.TypeofCells() }

// WalkedCloseCell is ssa.WalkedCloseFeedback's, jitFeedback's.
func (f *jitRegFeedback) WalkedCloseCell() uintptr { return f.fb.WalkedCloseCell() }

// TypedKind is ssa.TypedKindFeedback's: the cell the element kinds native
// code met at the program's instruction pc gather in, as jitFeedback's,
// by the register instruction's pc -- kept in the function's lowering
// beside the stack form's cells, under keys of their own (-1-pc).
func (f *jitRegFeedback) TypedKind(pc int) (ssa.TypedSite, bool) {
	s := f.fb.r.jit
	if s == nil || pc < 0 || pc >= len(f.p.Origin) {
		return ssa.TypedSite{}, false
	}
	l := s.lowering(f.fb.fn)
	key := -1 - int32(f.p.Origin[pc])
	cell := l.typedCells[key]
	if cell == nil {
		if l.typedCells == nil {
			l.typedCells = map[int32]*Value{}
		}
		cell = new(Value)
		l.typedCells[key] = cell
		l.bytes += jitTypedCellBytes
		s.lowerBytes += jitTypedCellBytes
	}
	k := f.fb.keep()
	k.cells = append(k.cells, cell)
	return ssa.TypedSite{Kinds: uint16(math.Float64bits(cell.num)), Cell: uintptr(unsafe.Pointer(cell))}, true
}
