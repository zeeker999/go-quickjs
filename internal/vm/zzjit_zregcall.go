//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"slices"
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/jit"
	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
	jitcompile "github.com/go-quickjs/go-quickjs/internal/jit/compile"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
	"github.com/go-quickjs/go-quickjs/internal/jit/mir"
	"github.com/go-quickjs/go-quickjs/internal/jit/ssa"
)

// Native calls between register functions (docs/register-interpreter.md,
// Stage 3, A5): native code made from register code calls a register
// function's native code directly, as the stack form's native calls do --
// through the function's link (jitLink), which holds its code's call entry
// while it has code native callers may call, the callee's window made at
// the VM's stack's top past the caller's frame (ssa.CallSite's Window), its
// context the caller's next. No frame of the VM's is made for a native
// call: where Go runs while native callees are under way -- a call of Go
// one of them makes (jitRegGoCallNested), a callee that left native code
// (jitRegUnwind) -- it makes each level's frame as regEnterW would, and
// pins each level's code (jitRegEntry's levels), which the cache then
// neither drops nor compiles again until the level is done: the code each
// level runs is its function's in the cache for as long as it runs.

// jitRegFrameOps are the calls of Go on the frame (ssa.GoHost,
// GoArgument, GoCells, GoClosure, GoClose) that Go makes for native code
// of a register frame (jitRegFrameOp); it leaves the others to an exit.
var jitRegFrameOps = map[ssa.GoOp]bool{ssa.GoHost: true, ssa.GoArgument: true, ssa.GoCells: true, ssa.GoClosure: true, ssa.GoClose: true}

// regLinkOf is the link native callers call cl's function's register code
// c through (ssa.CallSite's Entry): its code's call entry while it has
// code they may call (regSetLink), none otherwise -- a call of it then
// Go's, from native code.
func (s *jitState) regLinkOf(cl *closure, c *regCode) *jitLink {
	key := s.rt.jitRegKeyOf(cl)
	if l := s.regLinks[key]; l != nil {
		s.regHint(cl, l)
		return l
	}
	if s.regLinks == nil {
		s.regLinks = map[jitKey]*jitLink{}
	}
	l := &jitLink{key: key, params: 2 + uint64(c.params), locals: uint64(c.numRegs), frame: uint64(c.numRegs),
		flags: s.rt.jitRegLinkFlags(cl, c)}
	if e := s.regCache[key]; e != nil && e.code != nil && !e.invalid && (e.life == nil || !e.life.notNative) {
		l.regSet(e)
	}
	s.regLinks[key] = l
	s.regHint(cl, l)
	return l
}

// jitRegLinkFlags are the flags of the link of cl's function, whose
// register code is c (abi.LinkCoerce...), as jitLinkFlags are the stack
// form's: what a generic call reads of a function it has no site for
// (ssa.CallSite's Generic). LinkParams counts the window's function and
// receiver. A function a generic call may not make is one native callers
// do not call natively at a call compiled for it either (jitRegCallable):
// a class's constructor, a getter's, a setter's; one whose call is not its
// body, a generator's or an async function's; one that keeps its
// arguments, or runs a direct eval; top-level code; and, under
// WithNodeQuirks, a sloppy function, whose f.arguments reads arguments
// past its parameters a native call does not keep.
func (r *Runtime) jitRegLinkFlags(cl *closure, c *regCode) uint64 {
	fn := cl.fn
	var f uint64
	if c.coerceThis {
		f |= abi.LinkCoerce
	}
	if fn.HasTailCall {
		f |= abi.LinkTail
	}
	if fn.Kind == bytecode.KindArrow {
		f |= abi.LinkArrow
	}
	if fn.Kind != bytecode.KindNormal && fn.Kind != bytecode.KindArrow && fn.Kind != bytecode.KindMethod ||
		!jitRegCalled(cl, c) || c.args || c.newTarget || fn.HasDirectEval || len(fn.EvalScopes) != 0 || r.nodeQuirks && !fn.Strict {
		f |= abi.LinkNotGeneric
	}
	return f
}

// regHint gives cl's template a slot of the hint table holding l, its
// function's link, as the stack form's entries have one (remember): what
// a generic call finds a function's link by (mir's genericCheck). A link
// keeps its slot until its function is gone (regBytes, regUnhint).
func (s *jitState) regHint(cl *closure, l *jitLink) {
	if l.hint == 0 {
		var i uint32
		if n := len(s.hintFree); n != 0 {
			i, s.hintFree = s.hintFree[n-1], s.hintFree[:n-1]
		} else {
			if n := s.hintsCap(); n != cap(s.hints) {
				s.hints = append(make([]jitHintSlot, 0, n), s.hints...)
			}
			if len(s.hints) == 0 {
				// Slot 0 is no hint.
				s.hints = append(s.hints, jitHintSlot{})
			}
			i = uint32(len(s.hints))
			s.hints = append(s.hints, jitHintSlot{})
		}
		s.hints[i] = jitHintSlot{fn: uintptr(unsafe.Pointer(cl.fn)), link: l}
		s.hintTable = abi.HintTable{Slots: unsafe.Pointer(unsafe.SliceData(s.hints)), Len: uint64(len(s.hints))}
		l.hint = i
	}
	if cl.hint() != l.hint {
		cl.setHint(l.hint)
	}
}

// regUnhint frees l's slot of the hint table, its function gone.
func (s *jitState) regUnhint(l *jitLink) {
	if l.hint != 0 {
		s.hints[l.hint] = jitHintSlot{}
		s.hintFree = append(s.hintFree, l.hint)
		l.hint = 0
	}
}

// regSet has native callers call e's code through l: its call entry, and
// the room its frame takes past the VM's stack's top, which a call checks
// (abi.Encoding's LinkFrame) -- never less than any code it had took.
func (l *jitLink) regSet(e *jitRegEntry) {
	l.setCode(e.code)
	l.frame = max(l.frame, uint64(e.p.Regs+e.scratch+e.frame))
}

// regLinkUpdate sets the link of the function whose key is key, if it has
// one, to call e's code, or none where e has none or is invalid.
func (s *jitState) regLinkUpdate(key jitKey, e *jitRegEntry) {
	l := s.regLinks[key]
	if l == nil {
		return
	}
	if e == nil || e.code == nil || e.invalid || e.life != nil && e.life.notNative {
		l.clearCode()
		return
	}
	l.regSet(e)
}

// regCallSites are the native calls jitRegFeedback's NativeCalls answers
// for register instruction w, a call, at the program's instruction pc:
// of the function its site's cache holds, where native callers may call
// it, and of those the code learned the call makes at its calls of Go and
// its exits (jitRegCallSeen), up to jitCallTargets of them, as the stack
// form's code calls natively what jitCallSeen saw its calls call. What the
// cache holds is learned too, so that a call of it Go makes, its code not
// compiled yet, is not taken for news.
func (f *jitRegFeedback) regCallSites(pc int, w bytecode.RInstr, argc int, method bool) []ssa.CallSite {
	sites, _, _ := f.regCallTargets(pc, w, argc, method)
	return sites
}

// regCallTargets is regCallSites with each site's function object and
// closure, by index.
func (f *jitRegFeedback) regCallTargets(pc int, w bytecode.RInstr, argc int, method bool) ([]ssa.CallSite, []*Object, []*closure) {
	l := f.fb.e
	rpc := int32(f.p.Origin[pc])
	newW := w.Op() == bytecode.RNewW
	var sites []ssa.CallSite
	var objs []*Object
	var closures []*closure
	add := func(o *Object, cl *closure, rc *regCode, pool *abi.ObjectPool, parent *Object, byCode bool) {
		for i, s := range sites {
			if s.ByCode != 0 && jitRegSameCode(s, cl) || s.Callee == uintptr(unsafe.Pointer(o)) && !byCode {
				return
			}
			if byCode && jitRegSameClosureCode(closures[i], cl) {
				// The same function's -- the same object, which the site's
				// cache holds, among them -- checked by its code from here.
				sites, objs, closures = slices.Delete(sites, i, i+1), slices.Delete(objs, i, i+1), slices.Delete(closures, i, i+1)
				break
			}
		}
		if len(sites) < jitCallTargets {
			if site, ok := f.regCallSite(pc, o, cl, rc, argc, method, pool, parent, byCode); ok {
				sites, objs, closures = append(sites, site), append(objs, o), append(closures, cl)
			}
		}
	}
	if calls := f.fb.cl.regCalls; !newW && w.Op() != bytecode.RApplyArgs && uint(w.C()) < uint(len(calls)) {
		if e := &calls[w.C()]; jitRegCallKind(e) {
			add(e.callee, e.cl, e.rc, nil, nil, false)
			if len(sites) != 0 && l != nil && !jitRegCallKnown(l, rpc, e.callee) {
				l.nativeCalls = append(l.nativeCalls, jitInline{pc: rpc, cl: e.cl, obj: e.callee})
			}
		}
	}
	if l != nil {
		for _, x := range l.nativeCalls {
			if x.pc == rpc && (x.pool != nil) == newW && !x.forEach && x.cl != nil && x.active == nil {
				add(x.obj, x.cl, x.cl.reg, x.pool, x.parent, x.byCode)
			}
		}
	}
	return sites, objs, closures
}

// jitRegSameClosureCode reports whether closures a and b run the same
// function in the same environment and realm.
func jitRegSameClosureCode(a, b *closure) bool {
	return a.fn == b.fn && a.env == b.env && a.realm == b.realm
}

// jitRegSameCode reports whether site, checked by its function's code,
// calls cl's.
func jitRegSameCode(site ssa.CallSite, cl *closure) bool {
	return site.ByCode == uintptr(unsafe.Pointer(cl.fn)) && site.Env == uintptr(unsafe.Pointer(cl.env)) && site.Realm == uintptr(unsafe.Pointer(cl.realm))
}

// jitRegCallKind reports whether e, a call site's cache, holds a callee the
// call path enters as a frame of its own where it does (regCallEnter,
// regCallThis, regCallObjThis): a leaf's too, which its calls of the
// interpreter answer without one.
func jitRegCallKind(e *regCallCache) bool {
	switch e.kind &^ regCallMarks {
	case regCallEnter, regCallThis, regCallObjThis:
		return e.callee != nil && e.cl != nil
	}
	return false
}

// jitRegCallKnown reports whether l, what a function's register code
// learned, knows that its call at register pc rpc calls o.
func jitRegCallKnown(l *jitEntry, rpc int32, o *Object) bool {
	for _, x := range l.nativeCalls {
		if x.pc == rpc && x.obj == o {
			return true
		}
	}
	return false
}

// jitRegCallSeen learns, after Go made the call at register pc rpc of c
// -- a call of Go native code made (jitRegGoCall), or one a host exit
// left to it -- the function its site's cache holds now, as the stack
// form's code learns a call's targets at its calls of Go (jitCallSeen):
// one native callers may call that the call was not known to make is the
// next compile's to call natively (regCallSites), the code to be compiled
// again for it (jitRegReopt); up to jitCallTargets functions a call. It reports whether it
// learned one.
func (r *Runtime) jitRegCallSeen(f *frame, c *regCode, e *jitRegEntry, rpc int, callee Value) bool {
	if e.invalid || rpc >= len(c.rc.Code) {
		return false
	}
	w := regAt(c.code, rpc)
	var o, parent *Object
	var cl *closure
	switch w.Op() {
	case bytecode.RCallW, bytecode.RCallMW, bytecode.RCallWT, bytecode.RCallMWT:
		calls := f.cl.regCalls
		if uint(w.C()) >= uint(len(calls)) {
			return false
		}
		ce := &calls[w.C()]
		if !jitRegCallKind(ce) || ce.rc != ce.cl.reg || !r.jitRegCallable(ce.callee, ce.cl, ce.rc, int(w.X())) {
			return false
		}
		o, cl = ce.callee, ce.cl
	case bytecode.RApplyArgs:
		// The function f.apply(x, arguments) called, which no cache holds:
		// callee, if apply was the realm's.
		if !callee.IsObject() || r.applyFn == nil || !r.stack[jitRegWindowOf(f, c)+int(w.B())+1].IsObject() ||
			r.stack[jitRegWindowOf(f, c)+int(w.B())+1].Object() != r.applyFn {
			return false
		}
		o = callee.Object()
		fd := o.fn()
		if fd == nil || fd.closure == nil || !r.jitRegCallable(o, fd.closure, fd.closure.reg, 0) {
			return false
		}
		cl = fd.closure
	case bytecode.RNewW:
		// A construction's function, which no cache of the call's holds:
		// the one it constructed with, callee.
		if !callee.IsObject() {
			return false
		}
		o = callee.Object()
		if o == r.proto.arrayCtor {
			// new Array() or new Array(n): made from a pool (jitRegAllocSeen).
			return r.jitRegAllocSeen(f, c, e, rpc, w, o)
		}
		var ok bool
		if cl, parent, ok = r.jitRegConstructTarget(o, int(w.X())); !ok {
			return false
		}
	default:
		return false
	}
	l := r.jitRegLearnedFor(f, e)
	n := 0
	for i := range l.nativeCalls {
		x := &l.nativeCalls[i]
		if int(x.pc) == rpc && x.pool == nil && x.obj != o && jitRegSameClosureCode(x.cl, cl) {
			// Another object of a function the call has called, a closure
			// its maker makes anew: called by its code from here
			// (ssa.CallSite's ByCode), as the stack form's call learns it
			// (jitSameCode).
			if x.byCode {
				r.jitRegNativeCallee(cl, cl.reg)
				return false
			}
			x.byCode = true
			r.jitRegReopt(f.cl, e)
			return true
		}
	}
	for _, x := range l.nativeCalls {
		if int(x.pc) == rpc {
			if x.obj == o {
				// Known: a call of it Go made for want of its code, which
				// native callers call once it has some -- or one known
				// past the functions the code calls natively, which
				// regCallTargets fills with the site's cache's first: a
				// generic call calls it through its link, which it finds
				// by its hint (regHint), given it here as below.
				if w.Op() != bytecode.RNewW {
					r.jitState().regLinkOf(cl, cl.reg)
				}
				r.jitRegNativeCallee(cl, cl.reg)
				return false
			}
			n++
		}
	}
	if n >= jitCallTargets {
		if w.Op() != bytecode.RNewW {
			// Past the functions it calls natively, one a generic call
			// calls through its link (ssa.CallSite's Generic), which it
			// finds by its hint (regHint): given one, and code native
			// callers may call, as the stack form's generic calls have
			// their functions compiled (jitCallTarget). Go had made every
			// call of one no site was compiled for.
			r.jitState().regLinkOf(cl, cl.reg)
			r.jitRegNativeCallee(cl, cl.reg)
		}
		return false
	}
	x := jitInline{pc: int32(rpc), cl: cl, obj: o, parent: parent}
	if w.Op() == bytecode.RNewW {
		// Its objects, made as the construction makes them, from a pool
		// (ssa.CallSite's Pool).
		if x.pool = r.jitNewPool(o); x.pool.Count == 0 {
			return false
		}
	}
	l.nativeCalls = append(l.nativeCalls, x)
	r.jitRegReopt(f.cl, e)
	return true
}

// jitRegConstructable reports whether native callers may construct with o
// natively, of argc arguments, its own code running (regCallSite's pool):
// a base constructor -- a plain function's object, or a class's of no
// field to initialize, as the stack form's constructions are made
// (jitConstructs) -- or a derived class's own constructor, its instance
// from the pool, its super() native or Go's, which gives the class's
// fields (jitConstructsDerived) -- of register code native callers may
// call, that does not read new.target -- one that hands its arguments on,
// this.initialize.apply(this, arguments), among them, whose forwarding
// call its code makes natively (ssa.CallSite's Forward) -- its prototype
// its own property, as regConstruct runs a construction in a register
// frame. A class's had been constructed by Go at every construction,
// AWFY's CD, Json and Havlak hundreds of thousands of times a run, and a
// derived class's at every construction until now: AWFY's Json and
// DeltaBlue.
func (r *Runtime) jitRegConstructable(o *Object, argc int) bool {
	fd := o.fn()
	if fd == nil || fd.closure == nil || fd.bound || fd.extra != nil || proxyOf(o) != nil {
		return false
	}
	cl := fd.closure
	switch fd.ctorKind {
	case ctorBase:
		if fd.fieldInit() != nil || cl.fn.Kind != bytecode.KindNormal && cl.fn.Kind != bytecode.KindConstructor {
			return false
		}
	case ctorDerived:
		// A class without a constructor of its own whose parent native
		// code constructs is constructed with that parent's code
		// (jitRegDerivedBase).
		if cl.fn.Kind != bytecode.KindDerivedConstructor || fd.superCtor() != o || o.class != ClassFunction ||
			jitRegForwardsSuper(cl.reg) && r.jitRegDerivedBase(o, argc) != nil {
			return false
		}
	default:
		return false
	}
	if cl.reg == nil || jitRegReadsNewTarget(cl, cl.reg) || o.findOwn(atomPrototype) < 0 {
		return false
	}
	return r.jitRegTarget(o, cl, cl.reg, argc)
}

// jitRegReadsNewTarget reports whether cl's register code c reads
// new.target, which a frame Go makes of a native callee's level has not --
// but a class's constructor's, only ever constructed, whose construction,
// or super(), writes its new.target to its context (abi.Context's
// NewTarget), where the frames Go makes of its level, and the stand-in
// its new_target is made on, find it (jitRegPushLevel, jitRegFrameRun).
func jitRegReadsNewTarget(cl *closure, c *regCode) bool {
	return c.newTarget && !isClassConstructorKind(cl.fn.Kind)
}

// jitRegConstructTarget is the closure native code constructs o with, of
// argc arguments: o's own (jitRegConstructable), or, where o's
// constructor is the one a class without one has, the closure of the
// class it extends, parent (jitRegDerivedBase); ok false for a
// construction Go makes.
func (r *Runtime) jitRegConstructTarget(o *Object, argc int) (cl *closure, parent *Object, ok bool) {
	if r.jitRegConstructable(o, argc) {
		return o.fn().closure, nil, true
	}
	if p := r.jitRegDerivedBase(o, argc); p != nil {
		return p.fn().closure, p, true
	}
	return nil, nil, false
}

// jitRegDerivedBase is the class a class o extends, if o's constructor is
// the one a class without one has, constructor(...args) { super(...args) }
// (jitRegForwardsSuper), o has no fields to give its instances, and native
// code constructs with that class (jitRegConstructable); or nil. new o(...)
// then does what a construction with that class does, its object made as
// o makes one -- o's prototype, o's root shape and room, as
// constructWithTarget makes it for super() to hand on -- and its result
// o's, as the stack form's (jitDerivedBase) and V8, which skips a default
// derived constructor (FindNonDefaultConstructorOrConstruct). AWFY's Havlak
// constructs a DictIdEntry, a DictEntry's subclass, for each entry of its
// dictionaries, and DeltaBlue an EditConstraint, a UnaryConstraint's.
func (r *Runtime) jitRegDerivedBase(o *Object, argc int) *Object {
	fd := o.fn()
	if fd == nil || fd.closure == nil || fd.native != nil || fd.bound || fd.ctorKind != ctorDerived || fd.fieldInit() != nil ||
		fd.closure.realm != r.Realm || o.class != ClassFunction || o.findOwn(atomPrototype) < 0 || !jitRegForwardsSuper(fd.closure.reg) {
		return nil
	}
	if p := o.proto(); p != nil && r.jitRegConstructable(p, argc) {
		return p
	}
	return nil
}

// jitRegForwardsSuper reports whether c is the register code of the
// constructor a derived class without one has: super(...args), then the
// this it bound returned.
func jitRegForwardsSuper(c *regCode) bool {
	if c == nil || len(c.rc.Code) == 0 || c.rc.Code[0].Op() != bytecode.RSuperForward {
		return false
	}
	for _, w := range c.rc.Code[1:] {
		switch w.Op() {
		case bytecode.RLoadUndef, bytecode.RMove, bytecode.RDerivedRet, bytecode.RRet, bytecode.RNop:
		default:
			return false
		}
	}
	return true
}

// jitRegCallable reports whether native callers may call o, cl's function
// object, whose register code is rc, with argc arguments (regCallSite): a
// function a call does not throw for, which a class's constructor is.
func (r *Runtime) jitRegCallable(o *Object, cl *closure, rc *regCode, argc int) bool {
	return !isClassConstructorKind(cl.fn.Kind) && r.jitRegTarget(o, cl, rc, argc)
}

// jitRegTarget reports whether native callers may call or construct with
// o, cl's function object, whose register code is rc, with argc arguments
// (jitRegCallable, jitRegConstructable).
func (r *Runtime) jitRegTarget(o *Object, cl *closure, rc *regCode, argc int) bool {
	fd := o.fn()
	if fd == nil || fd.closure != cl || fd.bound || cl.realm != r.Realm || rc == nil || rc != cl.reg || !jitRegCalled(cl, rc) ||
		cl.fn.HasDirectEval {
		return false
	}
	if rc.args && (argc > abi.MaxGoArgs || cl.fn.MappedArguments && cl.fn.ParamCount > 0) {
		// One that keeps its arguments is given them in its context
		// (ssa.CallSite's FrameArgs), as many as it holds; a sloppy one's
		// arguments object aliasing its parameters is Go's to make.
		return false
	}
	if cc := cl.closureCode; cc == nil || cc.jitRegRefused {
		return false
	}
	if r.nodeQuirks && argc > int(rc.keep) {
		return false
	}
	if x := fd.extra; x != nil && (!fd.arrow || x.lexThisRef != nil || x.rare != nil) {
		return false
	}
	return true
}

// regCallSite is jitRegFeedback's native call of o, cl's function object,
// whose register code is rc, at the call of the program's instruction pc,
// of argc arguments, a method's where method says (NativeCalls): through
// its link, its window made from the call's (ssa.CallSite's Window) -- or
// false for one Go makes: a callee of another realm, of a frame the call
// path makes otherwise (one that keeps its arguments, a leaf's, a class's
// constructor's, an arrow's sharing a derived constructor's this), one the
// JIT does not compile, or past the arguments a sloppy callee's frame keeps
// for f.arguments.
func (f *jitRegFeedback) regCallSite(pc int, o *Object, cl *closure, rc *regCode, argc int, method bool, pool *abi.ObjectPool, parent *Object, byCode bool) (ssa.CallSite, bool) {
	fb := f.fb
	r := fb.r
	forward := false
	if w, _, ok := f.at(pc); ok && w.Op() == bytecode.RApplyArgs {
		// f.apply(x, arguments) (ssa.CallSite's Forward): f's receiver x,
		// as a method call's; its arguments the frame's, as many as its
		// context holds (mir's checks), none known here.
		forward, method, argc = true, true, 0
	}
	if pool == nil && !r.jitRegCallable(o, cl, rc, argc) {
		return ssa.CallSite{}, false
	}
	if pool != nil {
		if tc, tp, ok := r.jitRegConstructTarget(o, argc); !ok || tc != cl || tp != parent || rc != cl.reg || pool.Count == 0 {
			return ssa.CallSite{}, false
		}
	}
	arrow, this := o.fn().arrow, cl.fn.UsesThis
	l := r.jitState().regLinkOf(cl, rc)
	k := fb.keep()
	k.links = append(k.links, l)
	k.holders = append(k.holders, o)
	site := ssa.CallSite{
		Callee: uintptr(unsafe.Pointer(o)), Closure: uintptr(unsafe.Pointer(cl)),
		Entry: uintptr(unsafe.Pointer(&l.entry)), Count: uintptr(unsafe.Pointer(&l.in)),
		Argc: argc, Method: method, Params: 2 + int(rc.params), LocalCount: int(rc.numRegs), ThisSlot: -1,
		Window: f.p.Code[pc].Left.Slot + 1,
		// A callee that keeps its arguments -- reads arguments, has a rest
		// parameter -- is given all of them and their count in its context
		// (abi.Context's Args), as the stack form's native call gives them
		// (jitFrameArgs), which the frames Go makes of its level have
		// (jitRegPushLevel) and its arguments' reads answer from
		// (jitRegFrameOp's GoArgument).
		FrameArgs: rc.args,
	}
	if byCode && pool == nil {
		// Checked by its code, its environment and its realm, not its
		// object: the closure is the callee's at run time (mir's byCode).
		site.ByCode, site.Env, site.Realm = uintptr(unsafe.Pointer(cl.fn)), uintptr(unsafe.Pointer(cl.env)), uintptr(unsafe.Pointer(cl.realm))
	}
	if forward {
		site.Argc, site.Forward, site.Apply = 2, true, uintptr(unsafe.Pointer(r.applyFn))
		k.holders = append(k.holders, r.applyFn)
	}
	if cl.fn.HasTailCall {
		// Whether its calls of itself in tail position may give its frame
		// up (ssa.OpTailSelf), which the call writes to its context at
		// every call, as the stack form's (abi.Context's TailFrame): a
		// call's frame, not a construction's, as the interpreter gives up
		// one (regTailTo) -- an arrow's sharing a derived constructor's
		// this is no callee of a native call (jitRegTarget).
		site.Tail, site.TailFrame = true, pool == nil
	}
	if pool != nil {
		// A construction: its receiver from the pool, made as regConstruct
		// makes one, while the function's prototype is still the object
		// the pool's are of (mir's constructGuards).
		k.pools = append(k.pools, pool)
		site.Pool, site.ProtoIndex, site.ProtoKey = uintptr(unsafe.Pointer(pool)), int(o.findOwn(atomPrototype)), uint32(atomPrototype)
		// A derived constructor's code: the pool's object its instance,
		// its this unbound until its super() binds it (ssa.CallSite's
		// Derived); a class's with no constructor of its own, run with its
		// parent's code, while o's prototype is still that parent
		// (Parent), as the stack form's.
		site.Derived = cl.fn.Kind == bytecode.KindDerivedConstructor
		if parent != nil {
			site.Parent = uintptr(unsafe.Pointer(parent))
			k.holders = append(k.holders, parent)
		}
		return site, true
	}
	// A receiver, for a callee that reads one, as the call path gives it --
	// the window's register 1 is undefined otherwise, but for a method
	// call's (mir's windowSetup).
	switch {
	case !this:
	case arrow:
		// The this it captured (regCallThis).
		site.ArrowThis = true
	case !method:
		// Undefined, or, sloppy, the global object (regCallSlow).
		site.PlainThis = true
		if rc.coerceThis {
			if !r.globalThis.IsObject() {
				return ssa.CallSite{}, false
			}
			g := r.globalThis.Object()
			site.Global = uintptr(unsafe.Pointer(g))
			k.holders = append(k.holders, g)
		}
	case rc.coerceThis:
		// A receiver to coerce is Go's (regCallObjThis).
		site.Coerce = true
	}
	if forward {
		// Its receiver the third operand (mir's windowForward), no
		// method call's.
		site.Method = false
	}
	return site, true
}

// regSuperSites are the calls jitRegFeedback's NativeCalls answers for
// register instruction w, super_call's of B arguments, at the program's
// instruction pc, of a derived constructor's code: a native call of the
// parent its code learned the super() constructs with (jitRegSuperSeen) --
// called as the stack form's is (ssa.CallSite's Super), its window made
// past the frame's, the instance its receiver -- or Go's call
// (ssa.GoCall's Super), on the frame, which binds its this, of a class that
// may make an arrow before a super() (jitRegArrowBeforeSuper), whose
// binding the arrow shares, or of a parent it learned none of.
func (f *jitRegFeedback) regSuperSites(pc int, w bytecode.RInstr) []ssa.CallSite {
	fb := f.fb
	argc := int(w.B())
	if fb.fn.Kind != bytecode.KindDerivedConstructor || !f.p.This {
		return nil
	}
	if l, rpc := fb.e, int32(f.p.Origin[pc]); l != nil && !jitRegArrowBeforeSuper(f.rc, fb.fn) {
		for _, x := range l.nativeCalls {
			if x.pc != rpc || x.active == nil {
				continue
			}
			if site, ok := f.regSuperSite(pc, x, argc); ok {
				return []ssa.CallSite{site}
			}
			break
		}
	}
	if argc+1 > abi.MaxGoArgs {
		return nil
	}
	return []ssa.CallSite{{Go: ssa.GoCall, Super: true, Argc: argc, ThisSlot: -1}}
}

// regSuperSite is the native super() at the program's instruction pc of x,
// a parent the code learned it constructs with (jitRegSuperSeen), of argc
// arguments: the code of the class whose prototype it is still -- x.obj,
// or past one with no constructor of its own, x.parent -- called through
// its link, while the function running is x.active, the one the code was
// learned for (mir's superGuards); or false where that code is no longer
// one native callers construct with.
func (f *jitRegFeedback) regSuperSite(pc int, x jitInline, argc int) (ssa.CallSite, bool) {
	fb := f.fb
	r := fb.r
	p, skip, cl := r.jitRegSuperTarget(x.active, argc)
	if p != x.obj || skip != x.parent || cl != x.cl || cl == nil {
		return ssa.CallSite{}, false
	}
	rc := cl.reg
	l := r.jitState().regLinkOf(cl, rc)
	k := fb.keep()
	k.links = append(k.links, l)
	k.holders = append(k.holders, x.obj, x.active)
	site := ssa.CallSite{
		Callee: uintptr(unsafe.Pointer(x.obj)), Closure: uintptr(unsafe.Pointer(cl)),
		Entry: uintptr(unsafe.Pointer(&l.entry)), Count: uintptr(unsafe.Pointer(&l.in)),
		Argc: argc, Params: 2 + int(rc.params), LocalCount: int(rc.numRegs), ThisSlot: -1,
		Window: f.p.Code[pc].Left.Slot + 1,
		Super:  true, Active: uintptr(unsafe.Pointer(x.active)), Derived: cl.fn.Kind == bytecode.KindDerivedConstructor,
		// A parent that keeps its arguments is given them in its context,
		// as any native callee (FrameArgs).
		FrameArgs: rc.args,
	}
	if x.parent != nil {
		site.Parent = uintptr(unsafe.Pointer(x.parent))
		k.holders = append(k.holders, x.parent)
	}
	return site, true
}

// jitRegSuperTarget is the parent d's super() of argc arguments constructs
// with natively (ssa.CallSite's Super), and the closure whose code it runs,
// as the stack form's (jitSuperTarget): d a derived class of no fields --
// which super() would give the instance -- its prototype p a constructor
// native callers construct with -- a base one of no fields, or a derived
// one's own, neither reading new.target, which the frames Go makes of a
// native callee's level do not have -- or, p a class with no constructor
// of its own, the code of the one it extends, skip, as V8 skips a default
// derived constructor. nil where there is none.
func (r *Runtime) jitRegSuperTarget(d *Object, argc int) (p, skip *Object, cl *closure) {
	dfd := d.fn()
	if dfd == nil || dfd.closure == nil || dfd.ctorKind != ctorDerived || dfd.superCtor() != d || dfd.fieldInit() != nil {
		return nil, nil, nil
	}
	code := func(o *Object) *closure {
		fd := o.fn()
		if fd == nil || fd.native != nil || o.class != ClassFunction || !r.jitRegConstructable(o, argc) {
			return nil
		}
		return fd.closure
	}
	if p = d.proto(); p == nil {
		return nil, nil, nil
	}
	if cl = code(p); cl != nil {
		return p, nil, cl
	}
	if q := r.jitRegDerivedBase(p, argc); q != nil {
		return p, q, q.fn().closure
	}
	return nil, nil, nil
}

// jitRegArrowBeforeSuper reports whether fn, a derived constructor whose
// register code is rc, may make an arrow before a super() of its runs, as
// the stack form's (jitArrowBeforeSuper): an arrow shares the
// constructor's this binding, which a super() native code makes binds in
// the context (its This), not in a binding of the VM's an arrow made
// before shares. No arrow at a pc before a super()'s, and no jump back from
// one to a super(): every arrow fn makes is made once any super() it runs
// has run, Go making it then on a binding bound (jitBindThis), or, on a
// native callee's level, of the this its context holds (jitRegFrameOp).
func jitRegArrowBeforeSuper(rc *bytecode.RegCode, fn *bytecode.Function) bool {
	lastSuper, firstArrow := -1, len(rc.Code)
	consts := fn.Constants
	if rc.Constants != nil {
		consts = rc.Constants
	}
	for pc, w := range rc.Code {
		switch w.Op() {
		case bytecode.RSuperCall, bytecode.RSuperForward, bytecode.RSuperSpread:
			lastSuper = pc
		case bytecode.RClosureW:
			if k := int(w.BC()); k < len(consts) {
				if c := consts[k]; c.Kind == bytecode.ConstFunction && c.Fn != nil && c.Fn.Kind == bytecode.KindArrow {
					firstArrow = min(firstArrow, pc)
				}
			}
		}
	}
	if firstArrow == len(rc.Code) || lastSuper < 0 {
		return false
	}
	if firstArrow < lastSuper {
		return true
	}
	for pc, w := range rc.Code {
		for _, t := range jitcompile.RegSuccessors(rc, pc, w) {
			if pc >= firstArrow && t <= lastSuper {
				return true
			}
		}
	}
	return false
}

// jitRegSuperSeen notes the super() at register pc rpc of f's code, e's,
// that Go made -- at a host exit, or a call of Go native code made -- the
// parent it constructs with, which native code calls from the next
// compile on (regSuperSites), as the stack form's (jitSuperSeen).
func (r *Runtime) jitRegSuperSeen(f *frame, c *regCode, e *jitRegEntry, rpc int) {
	if e == nil || e.invalid || f.callee == nil || rpc >= len(c.rc.Code) || regAt(c.code, rpc).Op() != bytecode.RSuperCall {
		return
	}
	l := r.jit.regLearnedOf(r.jitRegKeyOf(f.cl))
	for _, x := range l.nativeCalls {
		if int(x.pc) == rpc && x.active != nil {
			// Known: a super() Go made for want of the parent's code,
			// which native callers call once it has some.
			r.jitRegNativeCallee(x.cl, x.cl.reg)
			return
		}
	}
	p, skip, cl := r.jitRegSuperTarget(f.callee, int(regAt(c.code, rpc).B()))
	if p == nil || cl.reg == nil {
		return
	}
	r.jitRegNativeCallee(cl, cl.reg)
	l.nativeCalls = append(l.nativeCalls, jitInline{pc: int32(rpc), cl: cl, obj: p, parent: skip, active: f.callee})
	r.jitRegReopt(f.cl, e)
}

// jitRegLevel is a native call under way whose callee runs register code,
// in context ctx: its closure cl, the callee's register code c and its
// code's entry e, its window at wb; pc the register pc it is at -- the
// call it makes, or, the innermost, where it left -- and back where it
// returns to in its caller's code, base its context's Base.
type jitRegLevel struct {
	ctx  int
	cl   *closure
	c    *regCode
	e    *jitRegEntry
	wb   int
	pc   int
	back uintptr
	base uint64
	// construct marks a construction's callee (its caller's new_w, or a
	// super()'s parent, super_call's), this its receiver, which the
	// construction's result is if not an object -- a derived
	// constructor's, an object always (jitDerivedResult).
	construct bool
	this      Value
	// gen is, for the resumption of a generator a loop's step or a call of
	// next made natively, the generator, and cursor the step's, nil for a
	// call of next's (jitRegGenLevel).
	gen    *generator
	cursor *iterState
}

// jitRegConstructs marks l, the level whose caller's code c makes the call
// at register pc rpc, a construction's where that is new_w or super_call:
// its receiver the one the call gave its context (abi.Context's This).
func (r *Runtime) jitRegConstructs(l *jitRegLevel, c *regCode, rpc int) {
	if rpc >= len(c.rc.Code) {
		return
	}
	switch regAt(c.code, rpc).Op() {
	case bytecode.RNewW, bytecode.RSuperCall:
		l.construct, l.this = true, *(*Value)(unsafe.Pointer(&r.jit.ssaCtxs[l.ctx].This))
	}
}

// jitRegLevelAt is the level of the native call running in context i,
// its code pinned (jitRegEntry's levels), or false where it is none of
// register code's.
func (r *Runtime) jitRegLevelAt(i int) (jitRegLevel, bool) {
	s := r.jit
	c := s.ssaCtxs[i]
	cl := (*closure)(c.Closure)
	if cl == nil || cl.reg == nil {
		return jitRegLevel{}, false
	}
	// Its closure's code, while the cache holds it for this realm
	// (jitRegEntry's cached): a key made and a map searched at each level
	// of each call of Go from deep native recursion had cost EarleyBoyer
	// most of its time once its recursion ran natively, and been a tenth
	// of AWFY's Json.
	var e *jitRegEntry
	if cc := cl.closureCode; cc != nil {
		e = cc.jitReg
	}
	if e == nil || !e.cached || e.key.realm != r.jitRealmKey() {
		e = s.regCache[r.jitRegKeyOf(cl)]
	}
	if e == nil || e.code == nil || e.p == nil {
		return jitRegLevel{}, false
	}
	e.levels++
	l := jitRegLevel{ctx: i, cl: cl, c: cl.reg, e: e, wb: int(c.Base), back: c.ReturnTo, base: c.Base}
	if cl.fn.Generator {
		r.jitRegGenLevel(&l, c)
	}
	return l, true
}

// jitRegPushLevel makes the VM's frame of a level, as regEnterW makes a
// call's: its function, its window, its pc, its new.target undefined.
func (r *Runtime) jitRegPushLevel(l *jitRegLevel, pc uint32) *frame {
	if l.gen != nil {
		// A generator's resumption: the frame of the realm's next that
		// resumed it below its own, as regGenStep makes them.
		r.jitPushGenNext(l.gen)
	}
	off := r.frameDepth - r.frameBase
	if uint(off) >= uint(len(r.cur)) {
		r.seekFrameBlock()
		off = r.frameDepth - r.frameBase
	}
	r.frameDepth++
	if r.frameDepth > r.frameHigh {
		r.frameHigh = r.frameDepth
	}
	f := &r.cur[off]
	f.cl = l.cl
	f.callee = nil
	if v := r.stack[l.wb]; v.IsObject() {
		f.callee = v.Object()
	}
	f.base = l.wb + int(l.c.locals)
	f.pc = pc
	if len(f.openUpvalues) != 0 {
		f.openUpvalues = f.openUpvalues[:0]
	}
	if f.thisRef != nil || f.withScopes != nil || f.evalVars != nil || f.native != "" || f.args != nil {
		regClearFrame(f)
	}
	f.newTarget = Undefined
	if l.construct {
		// Its new.target, which the construction wrote to its context --
		// a super()'s, the one it was given (mir's windowSuper) -- and a
		// derived constructor's frame as runFD makes it: its function,
		// its instance and its this binding, bound if its super() bound
		// it (jitDerivedFrame).
		c := r.jit.ssaCtxs[l.ctx]
		if c.NewTarget != 0 {
			f.newTarget = Obj(*(**Object)(unsafe.Pointer(&c.NewTarget)))
		} else if f.callee != nil {
			f.newTarget = Obj(f.callee)
		}
		if l.cl.fn.Kind == bytecode.KindDerivedConstructor {
			jitDerivedFrame(f, c)
		}
	}
	if l.c.args {
		// The arguments its native call gave its context (ssa.CallSite's
		// FrameArgs), as regEnterW keeps a call's.
		f.args = jitCtxArgs(r.jit.ssaCtxs[l.ctx])
		p := int(l.c.params)
		f.locals = r.stack[l.wb+2 : l.wb+2+p : l.wb+2+p]
	}
	f.regKind = regFrameEnter
	if l.gen != nil {
		// Its this, function and arguments the generator's, as regGenRun
		// makes a resumption's frame.
		r.jitRegGenFill(f, l)
	}
	return f
}

// jitCtxArgs is a copy of the arguments context c holds (abi.Context's
// Argc and Args), which a native call of a function that keeps its
// arguments wrote there.
func jitCtxArgs(c *abi.Context) []Value {
	n := min(int(c.Argc), len(c.Args))
	args := make([]Value, n)
	for i := range n {
		args[i] = *(*Value)(unsafe.Pointer(&c.Args[i]))
	}
	return args
}

// jitRegPopLevel ends a level's frame f, as regEnterW ends a call's: what
// closures captured of it closed, the VM's stack's top back at its window
// -- a generator's resumption's, the frame of the next below it too.
func (r *Runtime) jitRegPopLevel(f *frame, wb int, gen bool) {
	f.regKind = 0
	if len(f.openUpvalues) != 0 {
		regCloseUpvalues(f)
	}
	r.frameDepth--
	if gen {
		r.frameDepth--
	}
	r.stackTop = wb
}

// jitRegRootOf is the context of the run of register code whose native
// calls run in the contexts up to idx, or -1 where there is none.
func (r *Runtime) jitRegRootOf(idx int) int {
	s := r.jit
	root := idx - 1
	for root >= 0 && s.ssaCtxs[root].Live == abi.LiveCall {
		root--
	}
	if root < 0 || s.ssaCtxs[root].Live != 0 || root >= len(s.regRuns) || s.regRuns[root].f == nil {
		return -1
	}
	return root
}

// jitRegGoCallNested is jitRegGoCall for a native callee's code, in
// context idx, of the run in context root: each level from the run's on
// is given the VM's frame while the call runs -- its function, its window,
// past the call it makes -- as the interpreter's frames are during a
// call, for a stack trace, and the depth of calls, to show; the call is
// made on the calling level's frame; and every level's code is pinned
// meanwhile. A call that threw, or after which the level's code is
// invalid, Go finishes at the exit native code then takes (regPending),
// once the callee levels have left native code (jitRegUnwind).
func (r *Runtime) jitRegGoCallNested(ctx *abi.Context, idx, root int) bool {
	s := r.jit
	if idx >= len(s.ssaCtxs) || s.ssaCtxs[idx] != ctx {
		return false
	}
	// Its levels and their frames on the state's stacks of them, past an
	// enclosing call's: a call of Go deep in a recursion of native calls,
	// past eight levels, had allocated both.
	lbase, fbase := len(s.regNestLevels), len(s.regNestFrames)
	ok := true
	for i := root + 1; i <= idx; i++ {
		l, found := r.jitRegLevelAt(i)
		if !found {
			ok = false
			break
		}
		s.regNestLevels = append(s.regNestLevels, l)
	}
	levels := s.regNestLevels[lbase:]
	if ok {
		run := s.regRuns[root]
		rpc, _ := jit.CallPC(s.ssaCtxs[root])
		run.f.pc = regSrc(run.c, int(rpc)) + 1
		caller, at := run.c, int(rpc)
		for k := range levels {
			l := &levels[k]
			r.jitRegConstructs(l, caller, at)
			if k < len(levels)-1 {
				p, _ := jit.CallPC(s.ssaCtxs[l.ctx])
				caller, at = l.c, int(p)
			}
		}
		for k := range levels {
			l := &levels[k]
			pc := uint32(0)
			if k < len(levels)-1 {
				rpc, _ := jit.CallPC(s.ssaCtxs[l.ctx])
				pc = regSrc(l.c, int(rpc)) + 1
			}
			s.regNestFrames = append(s.regNestFrames, r.jitRegPushLevel(l, pc))
		}
		frames := s.regNestFrames[fbase:]
		ctxTop := s.ctxTop
		s.ctxTop = max(ctxTop, idx+1)
		last := &levels[len(levels)-1]
		top := r.stackTop
		own := 0
		for k := range levels {
			if levels[k].e == last.e {
				own++
			}
		}
		// A native callee's call of Go, judged as the stack form's
		// (jitCalleeProfit), the next one's work counted from it.
		cpc := int(ctx.ExitPC)
		r.jitRegCalleeProfit(last.e, cpc)
		ok = r.jitRegGoCallOn(ctx, idx, jitRegRun{f: frames[len(frames)-1], c: last.c, e: last.e, wb: last.wb, own: own})
		s.regGoMark = jitRegGoMark{e: last.e, stretch: s.entries, edges: r.backEdges - s.pollCredit, rpc: cpc}
		s.ctxTop = ctxTop
		for k := len(frames) - 1; k >= 0; k-- {
			r.jitRegPopLevel(frames[k], levels[k].wb, levels[k].gen != nil)
		}
		r.stackTop = top
	}
	for k := range levels {
		levels[k].e.levels--
		s.regClose(levels[k].e)
	}
	clear(s.regNestLevels[lbase:])
	clear(s.regNestFrames[fbase:])
	s.regNestLevels, s.regNestFrames = s.regNestLevels[:lbase], s.regNestFrames[:fbase]
	return ok
}

// jitRegUnwound counts a native call of e's code that left native code
// (jitEntry's nativeOut), as the stack form's jitUnwound does: code that
// leaves on more than one in jitUnwindShare of its native callers' calls
// is compiled again, if it is learning, up to jitUnwindReopts times, its
// native callers calling it still; past that, or learning nothing, native
// callers no longer call it (notNative) -- they call Go at the call, which
// makes it -- until Go has made jitRetryNative calls of it, more each time
// (jitRegRetryNative). Before, a callee that left at every call was
// unwound at every call, its callers' levels finished by Go each time:
// AWFY Havlak's size, at and hashFn, a million times each.
func (r *Runtime) jitRegUnwound(e *jitRegEntry) {
	life := e.life
	if life == nil || e.invalid {
		return
	}
	l := r.jit.regLinks[e.key]
	if l == nil {
		return
	}
	if life.nativeOut++; life.nativeOut%jitUnwindProbe != 0 || life.nativeOut*jitUnwindShare <= l.in {
		return
	}
	if life.unwindReopts < jitUnwindReopts && jitLearning(life) {
		life.unwindReopts++
		l.in, life.nativeOut = 0, 0
		return
	}
	life.notNative, life.nativeRetry = true, 0
	l.clearCode()
	life.nativeBackoff = min(life.nativeBackoff+1, jitNativeBackoffs)
}

// jitRegRetryNative counts a call Go makes of e's code, which native
// callers no longer call (notNative), and has them call it again after
// jitNativeRetry such calls, doubled for each time it was found to leave
// too often, its leaving counted afresh, as jitRetryNative does.
func (r *Runtime) jitRegRetryNative(e *jitRegEntry) {
	life := e.life
	if life.nativeRetry++; life.nativeRetry < jitNativeRetry<<(life.nativeBackoff-1) {
		return
	}
	life.notNative, life.nativeOut = false, 0
	if l := r.jit.regLinks[e.key]; l != nil {
		l.in = 0
		if r.jit.regCache[e.key] == e && e.code != nil && !e.invalid {
			l.regSet(e)
		}
	}
}

// jitRegUnwind finishes the native calls the code running in context idx
// made, the innermost of which left native code, and returns what the
// outermost returned or threw, as jitUnwindNative finishes the stack
// form's: each level is given the VM's frame, as regEnterW would have made
// it, from the outermost; the innermost is finished from where it left, as
// the run of a frame's code is (jitRegLeft); then each level from the call
// it made, with what its callee answered: in its native code, where the
// call returns to, if its code is still its function's and valid -- its
// context, spills and keeps as it left them -- or, its call's state written
// to its window (jit.ExpandCall), by Go, the answer to the call's register
// or the exception to the level's handlers (regPending), and on from there.
// Each level's code is pinned until the level is done.
func (r *Runtime) jitRegUnwind(idx int) (Value, error) {
	s := r.jit
	var levels []jitRegLevel
	for i := idx + 1; i < len(s.ssaCtxs) && s.ssaCtxs[i].Live == abi.LiveCall; i++ {
		l, ok := r.jitRegLevelAt(i)
		if !ok {
			// Native code calls only register code natively from register
			// code, and its levels' code stays in the cache while they run.
			panic("vm: a native call of register code whose callee has no code")
		}
		levels = append(levels, l)
	}
	n := len(levels)
	for k := range levels {
		l := &levels[k]
		c := s.ssaCtxs[l.ctx]
		if k < n-1 {
			rpc, _ := jit.CallPC(c)
			l.pc = int(rpc)
		} else {
			if c.ExitKind == abi.ExitTable {
				// An exit that left its frame and its fields to Go, which
				// writes them before they are read.
				s.ssaRecords += uint64(s.exitScratch.Apply(c))
			}
			l.pc = int(c.ExitPC)
		}
		c.Live, c.ReturnTo = 0, 0
	}
	if n != 0 {
		rpc, _ := jit.CallPC(s.ssaCtxs[idx])
		caller, at := s.regRuns[idx].c, int(rpc)
		for k := range levels {
			r.jitRegConstructs(&levels[k], caller, at)
			caller, at = levels[k].c, levels[k].pc
		}
	}
	s.unwound += uint64(n)
	if n != 0 {
		// The innermost left native code: counted against its native
		// callers' calling it (jitRegUnwound).
		r.jitRegUnwound(levels[n-1].e)
		// A call from the last context, which the contexts, not the
		// frames the VM allows, limited: more of them, as the stack form's
		// unwinding has them made (jitUnwindNative).
		if l := &levels[n-1]; l.ctx == len(s.ssaCtxs)-1 && s.bounds.LevelLimit == uint64(len(s.ssaCtxs)) && s.ssaCtxs[l.ctx].ExitKind == abi.ExitHost {
			r.jitGrowContexts()
		}
	}
	// What Go runs to finish a level runs in contexts past every level's:
	// a level's native code, resumed after its call, finds its spills and
	// keeps where it left them.
	ctxTop := s.ctxTop
	s.ctxTop = max(ctxTop, idx+n+1)
	frames := make([]*frame, n)
	for k := range levels {
		l := &levels[k]
		pc := regSrc(l.c, l.pc)
		if k < n-1 {
			// Past the call it makes, as the interpreter's frame is during
			// one, which a stack trace shows.
			pc++
		}
		frames[k] = r.jitRegPushLevel(l, pc)
		// Its captured registers' cells, closed while it ran natively, open
		// on its window as the interpreter's are.
		r.jitRegReopenCells(frames[k], l.e, l.wb, s.ssaCtxs[l.ctx], false)
	}
	var v Value
	var err error
	for k := n - 1; k >= 0; k-- {
		l, f := &levels[k], frames[k]
		c := s.ssaCtxs[l.ctx]
		switch {
		case k == n-1:
			// Its records first, which the objects its code did not make
			// are materialized from with its inlined callees' (jitRegInlined);
			// applied, none is left for jitRegFinishLevel.
			r.jitRegApplyRecords(f, l.e.p, l.wb, c)
			if iv, ierr, ok := r.jitRegInlined(l.ctx, f, l.e, l.pc, l.wb); ok {
				// It left inside a callee its code inlined, at the call
				// it is inlined at: Go finishes the callee, and the call
				// goes on with what it answered (regPending).
				s.regPending = jitGoPending{set: true, pc: l.pc, ctx: l.ctx, v: iv, err: ierr}
			}
			v, err = r.jitRegFinishLevel(f, l, c)
		case err == nil && r.stopped == nil && !l.e.invalid && l.e.code != nil && levels[k+1].back != 0 && l.e.code.Contains(levels[k+1].back):
			// On in its native code where the call returns to.
			callee := s.ssaCtxs[l.ctx+1]
			callee.Base, callee.Live, callee.ReturnTo = levels[k+1].base, 0, 0
			*(*Value)(unsafe.Pointer(&callee.RetValue)) = v
			s.resumed++
			s.resumedLevels++
			var pc int
			var done bool
			pc, v, done, err = r.jitRegGoOnAt(f, l.c, l.e, l.wb, 0, l.ctx, levels[k+1].back)
			if err == nil && !done {
				v, err = r.runReg(f, l.c, pc)
			}
		default:
			jit.ExpandCall(c)
			s.regPending = jitGoPending{set: true, pc: l.pc, ctx: l.ctx, v: v, err: err}
			v, err = r.jitRegFinishLevel(f, l, c)
		}
		if err == errRegRestart {
			// Its frame given to a callee in place by a call in tail
			// position (regRestartRun).
			v, err = r.regRestartRun(f)
		}
		if l.gen != nil {
			// A generator's resumption: its suspension or its end saved,
			// and the step's result (jitRegGenDone).
			v, err = r.jitRegGenDone(f, l, v, err)
		}
		r.jitRegPopLevel(f, l.wb, l.gen != nil)
		l.e.levels--
		s.regClose(l.e)
		if l.e.life != nil && l.e.life.reopt && l.cl.reg == l.c && r.jitRegLearnedDue(l.cl, l.e) {
			// Due to be compiled again for what it learned -- a call of Go
			// left for it (jitRegCompileDueNow) -- now that no native call
			// runs it: compiled here, as the stack form's code is once its
			// run has left (jitRecompileInRun).
			r.jitRegReoptimize(l.cl, l.c, l.e)
		}
		if err == errTailCall {
			// Ended in a call in tail position, its frame given up: the
			// call made here.
			v, err = r.regTailRest()
		}
		switch {
		case l.construct && l.cl.fn.Kind == bytecode.KindDerivedConstructor:
			// A derived construction's: an object, or what it throws
			// (jitDerivedResult).
			v, err = r.jitDerivedResult(v, err)
		case l.construct && err == nil && !v.IsObject():
			// A construction's result is its receiver, unless an object.
			v = l.this
		}
	}
	s.ctxTop = ctxTop
	return v, err
}

// jitRegFinishLevel finishes level l, whose frame is f, from where its code
// left in its context c: its records written, then on as a run of the
// frame's code goes on from an exit (jitRegLeft) -- in native code, at the
// next entry, or in the interpreter -- to its end.
func (r *Runtime) jitRegFinishLevel(f *frame, l *jitRegLevel, c *abi.Context) (Value, error) {
	s := r.jit
	if c.ExitKind == abi.ExitTable {
		s.ssaRecords += uint64(s.exitScratch.Apply(c))
	}
	r.jitRegApplyRecords(f, l.e.p, l.wb, c)
	next, pc, v, done, err, native := r.jitRegLeft(f, l.c, l.e, l.wb, c, Undefined)
	if native {
		pc, v, done, err = r.jitRegGoOn(f, l.c, l.e, l.wb, next)
	}
	if err == nil && !done {
		v, err = r.runReg(f, l.c, pc)
	}
	return v, err
}

// InlineExit is ssa.InlineExitFeedback's: of an inlined callee's
// feedback, whether native code may leave the inlined frame at the
// program's instruction pc, as the stack form's (jitInlineExit): a throw,
// which leaves its frame anyway, and the call, or the construction, whose
// result it throws -- throw new Error(...) -- as seldom run as the throw,
// as V8's code inlines a function whose cold paths call the runtime to
// throw. One taken after all counts toward the call's inlining's exits
// (jitInlineLeft), which stop it, the callee then called natively; a
// callee with such a path had been called natively, never inlined.
func (f *jitRegFeedback) InlineExit(pc int) bool {
	w, _, ok := f.at(pc)
	if !ok || f.fb.root == nil {
		return false
	}
	switch w.Op() {
	case bytecode.RThrow:
		return true
	case bytecode.RCallW, bytecode.RCallMW, bytecode.RNewW:
		next := int(f.p.Origin[pc]) + 1
		return next < len(f.rc.Code) && f.rc.Code[next].Op() == bytecode.RThrow && f.rc.Code[next].A() == w.B()
	}
	return false
}

// KeepLimit is ssa.KeepFeedback's: the build's keep cells' limit, 0 for
// the default (ssa.VirtualKeeps), as the stack form's: abi.MaxKeeps for a
// build again of code whose keep cells live at once were more than the
// context has (mir.ErrKeeps, jitRegBuild).
func (f *jitRegFeedback) KeepLimit() int { return f.fb.keepLimit }

// jitRegBuild builds cl's register code c lowered to p, from its feedback
// asked now, with its keep cells limited to keepLimit (KeepLimit), its
// entry kept at frameAt (ssa.Func's FrameAt), into w's workspaces: its
// feedback, its SSA, its loops' headers and the backend's code, or why it
// refused.
func (r *Runtime) jitRegBuild(w *jit.Workspace, cl *closure, c *regCode, p *ir.Program, keepLimit, frameAt int) (*jitRegFeedback, *ssa.Func, map[uint32]bool, *mir.Code, error) {
	rf := r.jitRegFeedbackFor(cl, p)
	rf.fb.keepLimit = keepLimit
	f, err := ssa.BuildIn(&w.SSA, p, rf)
	if err != nil {
		return rf, nil, nil, nil, err
	}
	f.FrameAt = frameAt
	looped := jitRegShape(f, p, cl, c)
	ssa.Optimize(f)
	mc, err := mir.CompileWith(&w.MIR, f, jitEncoding, mir.Settings())
	return rf, f, looped, mc, err
}

// jitRegWindowOf is where the window of f, running c, is in the VM's stack.
func jitRegWindowOf(f *frame, c *regCode) int { return f.base - int(c.locals) }
