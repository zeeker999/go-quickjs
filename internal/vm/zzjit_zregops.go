//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
	"github.com/go-quickjs/go-quickjs/internal/jit/ssa"
)

// Register code's operations native code makes from what the stack form's
// feedback answers of them, by register pc.

// regDefined is the value the register instruction just before register
// pc rpc leaves in register reg, where that instruction reads a binding --
// a global's (get_global), a captured one's (get_upval, get_upval_check)
// -- as it holds it now, as the stack form's jitChainValue finds what the
// code reads before an operation: what the code compiled for it checks it
// still reads.
func (f *jitRegFeedback) regDefined(rpc, reg int) (Value, bool) {
	cl := f.fb.cl
	if cl == nil || rpc < 1 || rpc > len(f.rc.Code) {
		return Undefined, false
	}
	w := f.rc.Code[rpc-1]
	if int(w.A()) != reg {
		return Undefined, false
	}
	switch w.Op() {
	case bytecode.RGetGlobal:
		name, ok := f.name(rpc-1, w.B())
		if !ok {
			return Undefined, false
		}
		return f.fb.r.jitGlobalValue(cl, name)
	case bytecode.RGetUpval, bytecode.RGetUpvalCheck:
		k := int(w.BC())
		if k >= len(cl.upvalues) || cl.upvalues[k] == nil {
			return Undefined, false
		}
		if v := cl.upvalues[k].get(); !v.IsUninitialized() {
			return v, true
		}
	}
	return Undefined, false
}

// InstanceOf is ssa.InstanceOfFeedback's: the instanceof at the program's
// instruction pc (instanceof's Window Host), of the constructor the
// register instruction before it reads (regDefined), made natively along
// the value's prototypes, as the stack form's (jitFeedback's InstanceOf):
// checked to have that constructor, its Symbol.hasInstance the realm's.
func (f *jitRegFeedback) InstanceOf(pc int) (ssa.InstanceOfSite, bool) {
	w, op, ok := f.at(pc)
	if !ok || op != ir.Host || w.Op() != bytecode.RInstanceOf || f.fb.r.hasInstanceFn == nil {
		return ssa.InstanceOfSite{}, false
	}
	v, ok := f.regDefined(int(f.p.Origin[pc]), int(w.C()))
	if !ok || !v.IsObject() {
		return ssa.InstanceOfSite{}, false
	}
	return f.fb.instanceOfSite(v.Object())
}

// IndexOf is ssa.IndexOfFeedback's: the search -- String.prototype's or
// Array.prototype's indexOf, lastIndexOf or includes, the realm's
// (jitSearchKind) -- the method call at the program's instruction pc last
// called, as its site's cache holds it (methodCall), which the stack form's
// code learns at its calls of Go (jitIndexOfs), and its kind. Go makes it
// from native code (ssa.GoIndexOf), and leaves what would run script to
// the call after an exit.
func (f *jitRegFeedback) IndexOf(pc int) (uintptr, int, bool) {
	o, n, ok := f.methodCall(pc, true)
	if !ok || n != 1 && n != 2 {
		return 0, 0, false
	}
	k := f.fb.r.jitSearchKind(o)
	if k < 0 {
		return 0, 0, false
	}
	f.fb.keep().holders = append(f.fb.keep().holders, o)
	return uintptr(unsafe.Pointer(o)), k, true
}

// ParseInt is ssa.ParseIntFeedback's: the realm's parseInt, which the plain
// call of one argument at the program's instruction pc last called
// (methodCall), made of a number natively, as the stack form's
// (jitParseInts).
func (f *jitRegFeedback) ParseInt(pc int) (uintptr, bool) {
	o, n, ok := f.methodCall(pc, false)
	if !ok || n != 1 || o == nil || o != f.fb.r.jitParseInt {
		return 0, false
	}
	f.fb.keep().holders = append(f.fb.keep().holders, o)
	return uintptr(unsafe.Pointer(o)), true
}

// BigString is ssa.BigStringFeedback's: BigInt.prototype.toString, the
// realm's own, which the method call at the program's instruction pc last
// called (methodCall), Go's from native code, as the stack form's
// (jitBigStrings).
func (f *jitRegFeedback) BigString(pc int) (uintptr, bool) {
	o, n, ok := f.methodCall(pc, true)
	if !ok || n > 1 || o == nil || o != f.fb.r.jitBigToString {
		return 0, false
	}
	f.fb.keep().holders = append(f.fb.keep().holders, o)
	return uintptr(unsafe.Pointer(o)), true
}

var (
	_ ssa.InstanceOfFeedback = (*jitRegFeedback)(nil)
	_ ssa.IndexOfFeedback    = (*jitRegFeedback)(nil)
	_ ssa.ParseIntFeedback   = (*jitRegFeedback)(nil)
	_ ssa.BigStringFeedback  = (*jitRegFeedback)(nil)
)

// jitRegCodeIn reports whether the code making the call of Go in ctx, in
// context idx, is register code: a run of it (jitState's regRuns), or a
// native callee of one -- which calls only register code natively. Its
// exit PC is then a register pc, which no stack form's instruction is at.
func (r *Runtime) jitRegCodeIn(ctx *abi.Context, idx int) bool {
	s := r.jit
	switch ctx.Live {
	case 0:
		return idx < len(s.regRuns) && s.regRuns[idx].f != nil
	case abi.LiveCall:
		return r.jitRegRootOf(idx) >= 0
	}
	return false
}

// jitRegPlainSet is, for a call of Go register code makes in ctx, in
// context idx (reg), a set_prop's write jitGoCallPlain may make without a
// frame (jitPlainWrite): the code's closure -- the run's, or a native
// callee's -- and the write as the stack form's instruction names it, its
// site's name and its cache, of the register instruction at its register
// pc; a nil closure where it is none, or is an inlined callee's.
func (r *Runtime) jitRegPlainSet(ctx *abi.Context, idx int) (*closure, bytecode.Instr, bool) {
	if !r.jitRegCodeIn(ctx, idx) {
		return nil, bytecode.Instr{}, false
	}
	if st := jitGoCallState(ctx); st != nil && st.Inline != 0 {
		return nil, bytecode.Instr{}, true
	}
	var cl *closure
	var c *regCode
	if ctx.Live == 0 {
		run := r.jit.regRuns[idx]
		cl, c = run.f.cl, run.c
	} else {
		cl = (*closure)(ctx.Closure)
		if cl != nil {
			c = cl.reg
		}
	}
	rpc := int(ctx.ExitPC)
	if cl == nil || c == nil || rpc >= len(c.rc.Code) {
		return nil, bytecode.Instr{}, true
	}
	w := c.rc.Code[rpc]
	if w.Op() != bytecode.RSetProp || int(w.C()) >= len(c.rc.SiteName) {
		return nil, bytecode.Instr{}, true
	}
	return cl, bytecode.Instr{Op: bytecode.OpSetProp, A: c.rc.SiteName[w.C()], B: uint32(w.C())}, true
}
