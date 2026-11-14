//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

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
	_ ssa.ClosureFeedback    = (*jitRegFeedback)(nil)
	_ ssa.CellsFeedback      = (*jitRegFeedback)(nil)
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

// jitRegAllocSeen learns, after Go made new_w w at register pc rpc of f's
// code e, a construction with o, the realm's Array, of no argument or of
// a length native code makes (jitArrayRoom), which needs nothing run: the
// next compile takes its arrays from a pool (regAllocSite), as the stack
// form's code learns it (jitAllocates), the code stale from here. It
// reports whether it learned it.
func (r *Runtime) jitRegAllocSeen(f *frame, c *regCode, e *jitRegEntry, rpc int, w bytecode.RInstr, o *Object) bool {
	argc := int(w.X())
	if argc > 1 {
		return false
	}
	room := 0
	if argc == 1 {
		// The length, in the window's register past the receiver's, the
		// call having written its result elsewhere.
		a := f.base - int(c.locals) + int(w.A()) + 2
		if int(w.B()) == int(w.A())+2 || a < 0 || a >= len(r.stack) {
			return false
		}
		n := r.stack[a]
		if !n.IsNumber() || n.num < 0 || n.num != math.Trunc(n.num) {
			return false
		}
		if room = jitArrayRoom(n.num); room == 0 {
			return false
		}
	}
	l := r.jit.regLearnedOf(r.jitKeyOf(f.cl.fn))
	for _, x := range l.nativeCalls {
		if int(x.pc) == rpc && x.cl == nil && !x.forEach && x.pool != nil {
			return false
		}
	}
	pool := r.jitNewPool(o)
	if argc == 1 {
		pool = r.jitNewArrayPool(o, room)
	}
	l.nativeCalls = append(l.nativeCalls, jitInline{pc: int32(rpc), obj: o, pool: pool})
	r.jitRegReopt(f.cl, e)
	return true
}

// regAllocSite is the construction new_w w makes with the realm's Array
// natively from a pool (jitRegAllocSeen), as the stack form's NativeCalls
// makes one (ssa.CallSite's Alloc): new Array(), or new Array(n) of a
// length its pool's arrays have room for; any other, and any other
// function, Go's after an exit.
func (f *jitRegFeedback) regAllocSite(rpc int, w bytecode.RInstr, argc int) (ssa.CallSite, bool) {
	l, r := f.fb.e, f.fb.r
	if w.Op() != bytecode.RNewW || argc > 1 || l == nil {
		return ssa.CallSite{}, false
	}
	for _, x := range l.nativeCalls {
		if int(x.pc) != rpc || x.cl != nil || x.forEach || x.pool == nil || x.obj != r.proto.arrayCtor {
			continue
		}
		if argc == 1 && x.pool.Room == 0 {
			return ssa.CallSite{}, false
		}
		k := f.fb.keep()
		k.holders, k.pools = append(k.holders, x.obj), append(k.pools, x.pool)
		return ssa.CallSite{Callee: uintptr(unsafe.Pointer(x.obj)), ThisSlot: -1,
			Pool: uintptr(unsafe.Pointer(x.pool)), Alloc: true, Argc: argc}, true
	}
	return ssa.CallSite{}, false
}

// Closure is ssa.ClosureFeedback's: closure_w's closure at the program's
// instruction pc made natively from a pool of closures of its template, as
// the stack form's (jitFeedback's Closure): its captured registers' cells
// those of the code's (ir.Program's Cells), by their place among the
// captured registers (jitRegCapturedRegs), its maker's captured bindings'
// from the closure's own. Not an arrow that reads its this, which native
// code would write from its context's This, which register code's frame
// does not hold (its window's register 1 does); nor one that reads its
// maker's new.target or super; nor in a derived constructor's code, nor an
// inlined callee's.
func (f *jitRegFeedback) Closure(pc int) (ssa.ClosureSite, bool) {
	fb := f.fb
	w, op, ok := f.at(pc)
	if !ok || op != ir.Closure || w.Op() != bytecode.RClosureW || fb.e == nil || fb.root != nil || fb.cl == nil ||
		fb.fn.Kind == bytecode.KindDerivedConstructor {
		return ssa.ClosureSite{}, false
	}
	k := int(w.BC())
	if k >= len(fb.cl.consts) || k >= len(f.rc.Caps) || jitNeedsMaker(fb.cl.consts[k]) {
		return ssa.ClosureSite{}, false
	}
	t := fb.cl.consts[k].template()
	if t == nil || t.fn.Kind == bytecode.KindArrow && t.fn.UsesThis {
		return ssa.ClosureSite{}, false
	}
	regs := jitRegCapturedRegs(f.p.Captured)
	caps := make([]int32, len(t.fn.Upvalues))
	for i, desc := range t.fn.Upvalues {
		if !desc.FromParent {
			caps[i] = int32(desc.Index) << 1
			continue
		}
		if i >= len(f.rc.Caps[k]) {
			return ssa.ClosureSite{}, false
		}
		at := slices.Index(regs, int(f.rc.Caps[k][i]))
		if at < 0 {
			return ssa.ClosureSite{}, false
		}
		caps[i] = int32(at)<<1 | 1
	}
	l := fb.r.jitRegClosureLiteral(fb.e, fb.cl, int(f.p.Origin[pc]), k)
	if l == nil {
		return ssa.ClosureSite{}, false
	}
	fb.keep().pools = append(fb.keep().pools, l.pool)
	return ssa.ClosureSite{Pool: uintptr(unsafe.Pointer(l.pool)), Captures: caps}, true
}

// jitRegClosureLiteral is the pool of closures of cl's constant k, its
// template, closure_w's at register pc rpc of cl's register code, in e,
// made if e has none, as the stack form's jitLiteralAt makes a closure's.
func (r *Runtime) jitRegClosureLiteral(e *jitEntry, cl *closure, rpc, k int) *jitLiteral {
	for i := range e.literals {
		if int(e.literals[i].pc) == rpc && e.literals[i].fn == cl.fn {
			return &e.literals[i]
		}
	}
	tmpl := cl.consts[k]
	e.literals = append(e.literals, jitLiteral{pc: int32(rpc), pool: new(abi.ObjectPool), fn: cl.fn, tmpl: tmpl, env: cl.env})
	l := &e.literals[len(e.literals)-1]
	l.pool.Source = unsafe.Pointer(&jitPoolSource{tmpl: tmpl, fn: cl.fn, env: cl.env})
	r.jitFillLiteral(l)
	return l
}

// CellsPool is ssa.CellsFeedback's: the pool the code takes its captured
// registers' cells from (ir.Program's Cells), at its entries and scopes'
// ends, as the stack form's (jitFeedback's CellsPool): the captured
// registers in their cells' order, each one's cell made holding the
// register's value at the entry -- an argument, or undefined past them.
func (f *jitRegFeedback) CellsPool() (uintptr, []int32, bool) {
	fb := f.fb
	if fb.e == nil || fb.root != nil || fb.cl == nil || !f.p.Cells {
		return 0, nil, false
	}
	regs := jitRegCapturedRegs(f.p.Captured)
	if n := len(regs); n == 0 || n > 4 {
		return 0, nil, false
	}
	cells := make([]int32, len(regs))
	for i, reg := range regs {
		cells[i] = int32(reg)<<1 | 1
	}
	e := fb.e
	if e.cellsPool == nil {
		e.cellsPool = &abi.ObjectPool{Size: abi.PoolSize, Cells: uint64(len(regs))}
		jitFillCells(e.cellsPool)
	}
	fb.keep().pools = append(fb.keep().pools, e.cellsPool)
	return uintptr(unsafe.Pointer(e.cellsPool)), cells, true
}

// jitRegFrameless makes register instruction op, whole, for native code
// in ctx without its frame where that runs nothing and throws nothing, as
// the stack form's operations are made (jitOpPlain, jitOpFrameless): an
// object's spread -- learning the layouts' copy native code makes next
// (jitCopyLearn) -- an array's spread of a dense array's elements, an
// element appended. Its operands are the instruction's registers (compile's
// RegAccess), the target first; it writes none. It reports whether it
// made it; what it does not make, the instruction run whole on the frame
// (jitRegWhole) makes.
func (r *Runtime) jitRegFrameless(ctx *abi.Context, op bytecode.ROp) bool {
	var in bytecode.Instr
	switch op {
	case bytecode.RCopyDataProps:
		in.Op = bytecode.OpCopyDataProps
	case bytecode.RArraySpread:
		in.Op = bytecode.OpArraySpread
	case bytecode.RArrayPush:
		in.Op = bytecode.OpArrayPush
	default:
		return false
	}
	if int(ctx.GoOp>>abi.GoOpArgcShift&0xff) != 2 {
		return false
	}
	args := unsafe.Slice((*Value)(unsafe.Pointer(&ctx.GoArgs[0])), 2)
	v, ok := r.jitOpFrameless(nil, nil, in, args)
	if !ok {
		return false
	}
	*(*Value)(unsafe.Pointer(&ctx.Keep[ctx.GoKeep])) = v
	return true
}

// regWholeSite is the call of Go a register instruction run whole makes
// (ssa.CallSite's Whole), with what native code makes of it itself first,
// as the stack form's sites of the same operations have it: an array
// spread by its elements (ArrayIter), an element appended (Append), an
// object spread by the layouts Go has copied (Copies), a literal's field
// defined along its shape's transition (DefineNext).
func (f *jitRegFeedback) regWholeSite(pc int, w bytecode.RInstr) []ssa.CallSite {
	_, argc, _, ok := f.p.Code[pc].WholeOperands(f.p.Regs)
	site := ssa.CallSite{Go: ssa.GoCall, Whole: true, Argc: argc, ThisSlot: -1}
	r := f.fb.r
	switch {
	case !ok || argc != 2:
		return nil
	case w.Op() == bytecode.RDefineField:
		// A literal's field along its shape's first transition by the
		// key (DefineNext), as the stack form's define_field's.
		key, ok := f.name(int(f.p.Origin[pc]), w.C())
		if !ok || key >= 1<<24 {
			return nil
		}
		site.Key, site.DefineNext = uint32(key), true
	case w.Op() == bytecode.RArraySpread:
		site.ArrayIter = uintptr(unsafe.Pointer(&r.jit.arrayIter))
	case w.Op() == bytecode.RArrayPush:
		site.Append = true
	case w.Op() == bytecode.RCopyDataProps:
		if r.jit.copies == nil {
			r.jit.copies = new([jitCopyEntries]jitCopyEntry)
		}
		site.Copies = uintptr(unsafe.Pointer(r.jit.copies))
	default:
		return nil
	}
	return []ssa.CallSite{site}
}
