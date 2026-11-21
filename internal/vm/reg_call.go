package vm

import (
	"sync/atomic"
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
)

// The register interpreter's calls and slow paths: what its loop does out
// of line, so that the loop's cases stay small (see reg_run.go).

// regCall calls callee with this and args from the register interpreter. A
// function of this realm that has register code, and that a call runs with
// nothing to establish but its frame, is run here directly: its frame made
// here, its registers run by runReg, with no recover and no other Go call
// between the two loops. Anything else is called by callDirect. A frame
// that ended in a tail
// call (errTailCall) has its callee called in its place, in this same call
// of regCall, as run does: a chain of tail calls runs in constant space.
func (r *Runtime) regCall(callee, this Value, args []Value) (Value, error) {
	if !callee.IsObject() {
		return r.callDirect(callee, this, args)
	}
	o := callee.Object()
	fd := o.fn()
	if fd == nil || fd.closure == nil || fd.bound {
		return r.callDirect(callee, this, args)
	}
	cl := fd.closure
	c := cl.reg
	if c == nil {
		c = r.regOf(cl)
	}
	if c == nil || c.wide || cl.realm != r.Realm {
		return r.callDirect(callee, this, args)
	}
	if fd.extra != nil {
		// An arrow whose surroundings are its this alone; anything else a
		// function carries of its surroundings is runFD's to establish.
		x := fd.extra
		if !fd.arrow || x.lexThisRef != nil || x.rare != nil {
			return r.callDirect(callee, this, args)
		}
		this = x.lexThis
	}
	if c.coerceThis && !this.IsObject() {
		if !this.IsNullish() {
			return r.callDirect(callee, this, args)
		}
		this = r.globalThis
	}
	if err := r.tick(); err != nil {
		return Undefined, err
	}
	if r.frameDepth >= r.maxFrames {
		return Undefined, r.throwRangeError("maximum call stack size exceeded")
	}
	base := r.stackTop
	top := base + int(c.numRegs)
	if top > len(r.stack) {
		return Undefined, r.throwRangeError("maximum call stack size exceeded")
	}
	r.stackTop = top
	if top > r.stackHigh {
		r.stackHigh = top
	}
	lc := int(c.locals)
	locals := r.stack[base : base+lc : base+lc]
	m := min(int(c.params), len(locals), len(args))
	//lint:ignore S1001 a loop, not copy
	for i, a := range args[:m] {
		locals[i] = a
	}
	for i := m; i < len(locals); i++ {
		locals[i] = Undefined
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
	f.cl = cl
	f.locals = locals
	f.base = base + lc
	f.pc = 0
	f.this = this
	f.newTarget = Undefined
	f.callee = o
	f.args = args
	if len(f.openUpvalues) != 0 {
		f.openUpvalues = f.openUpvalues[:0]
	}
	if f.thisRef != nil {
		f.thisRef = nil
	}
	if f.withScopes != nil {
		f.withScopes = nil
	}
	if f.evalVars != nil {
		f.evalVars = nil
	}
	if len(f.handlers) != 0 {
		f.handlers = f.handlers[:0]
	}
	if f.native != "" {
		f.native = ""
	}
	if f.savedSP != 0 {
		f.savedSP = 0
	}
	v, err := r.runReg(f, c, 0)
	r.popFrameOf(f, base)
	if err != nil {
		return r.regCallEnd(v, err)
	}
	return v, err
}

// regCallMiss is a call rhCallW's cache does not answer: a site's first
// call or a new callee, a body answered without a frame, a frame that keeps
// its arguments, a this to coerce, or a callee of no register code.
//
//go:noinline
func (r *Runtime) regCallMiss(f *frame, c *regCode, w bytecode.RInstr, wb, pc int) error {
	v, err := r.regCallSlow(f, c, w, wb+int(w.A()), pc)
	if err != nil {
		return r.regMissEnd(err, wb+int(w.B()))
	}
	r.stack[wb+int(w.B())] = v
	return nil
}

// regGenCall is a call of o, a generator function or an async function of
// this realm with closure cl, whose window is at base: its object made as
// callClosure makes it, the receiver coerced as there, with no frame, since
// the body does not run until the object is stepped; an async function's
// first step is made at once (runAsync), an error binding its parameters
// rejecting its promise.
func (r *Runtime) regGenCall(o *Object, cl *closure, method bool, base, argc int) (Value, error) {
	if err := r.tick(); err != nil {
		return Undefined, err
	}
	this := Undefined
	if method {
		this = r.stack[base+1]
	}
	if fn := cl.fn; fn.UsesThis && !fn.Strict && !this.IsObject() {
		if !this.IsNullish() {
			w, err := r.toObject(this)
			if err != nil {
				return Undefined, err
			}
			this = Obj(w)
		} else {
			this = r.globalThis
		}
	}
	async := cl.fn.Async
	if async && !cl.fn.Generator && !cl.fn.HasDirectEval {
		// Its first run made as a call, the generator made only where it
		// suspends (regAsyncRun).
		if c := r.regGenOf(cl, o); c != nil && !c.args {
			if fd := o.fn(); fd != nil && fd.extra == nil {
				return r.regAsyncRun(o, cl, c, this, base, argc), nil
			}
		}
	}
	gen, err := r.newGenerator(cl, this, r.stack[base+2:base+2+argc], o, async, Undefined)
	if err != nil {
		if async {
			p := r.newPromise()
			r.rejectPromise(p, thrownValue(err))
			return Obj(p), nil
		}
		return Undefined, err
	}
	if async {
		return r.runAsync(genOf(gen)), nil
	}
	return Obj(gen), nil
}

// regMarkedRest is regMarkedCall's call of o on recv with args: a
// built-in's as callRest makes it, without callDirect's search; anything
// else's as register code calls (regCall).
func (r *Runtime) regMarkedRest(o *Object, recv Value, args []Value) (Value, error) {
	if fd := o.fn(); fd != nil && fd.native != nil {
		return r.callRest(o, fd, recv, args)
	}
	return r.regCall(Obj(o), recv, args)
}

// regMarked reports whether call_method_w pc's Aux, aux, marks a call of o
// that regMarkedCall makes: of anything where the call makes its literal
// argument; of the realm's exec or test on a literal; of its exec or
// replace whose value is dead, or exec's reused at a site execReuseOf has
// (regReuseSites). What regMarkedCall does otherwise is the call
// regNative or rhCallW makes, slower.
func (r *Runtime) regMarked(c *regCode, o *Object, pc int, m uint8) bool {
	switch {
	case m&bytecode.RMarkLitArg != 0:
		return true
	case m&bytecode.RMarkLitRecv != 0:
		return o == r.regexpExecFn || o == r.regexpTestFn
	case m&bytecode.RMarkReuse != 0:
		return o == r.regexpExecFn && regReuseAt(c, pc) != nil
	case m&bytecode.RMarkDead != 0:
		return o == r.regexpExecFn || o == r.strReplaceFn
	}
	return false
}

// regAuxAt is the Aux of the instruction at pc, or 0.
func regAuxAt(c *regCode, pc int) uint32 {
	if pc < len(c.rc.Aux) {
		return c.rc.Aux[pc]
	}
	return 0
}

// regNameIdx is the index in the function's names of op, the name
// operand of the instruction at pc: op itself, or, where it is RNameFar,
// the index past 16 bits FarNames has for the instruction. Only cold
// operations and miss paths read a name.
func regNameIdx(c *regCode, pc int, op uint16) uint32 {
	if op == bytecode.RNameFar {
		return c.rc.FarNames[pc]
	}
	return uint32(op)
}

// regName is the name regNameIdx finds, of closure cl's names.
func regName(cl *closure, c *regCode, pc int, op uint16) Atom {
	return cl.names[regNameIdx(c, pc, op)]
}

// regReuseAt is the reuse site of call_method_w pc, or nil.
func regReuseAt(c *regCode, pc int) *execReuseSite {
	if s := regAt(c.code, pc).C(); int(s) < len(c.reuse) {
		return c.reuse[s]
	}
	return nil
}

// regCallKind is how a call site of register code enters fd's closure, a
// method's call or not: of the form's own layout, not suspending, of this
// realm, neither a class's constructor nor with a direct eval (enter),
// the kind its cache gives it and the this that kind writes -- 0 where
// its this is not one a cache can give, an arrow's of a derived
// constructor, say. rc is the code a call enters: the closure's typed
// code where it has some (regTypedOf), for a site of typed code or for
// one of ordinary code where the callee is no leaf, answered without a
// frame; its ordinary code otherwise. A site enters code of its own
// code's kind and heap alone (regCacheCallee). caller is the site's code,
// nil for none.
func (r *Runtime) regCallKind(fd *funcData, method bool, caller *regCode) (cl *closure, rc *regCode, kind uint8, this Value, enter bool) {
	cl = fd.closure
	if rc = cl.reg; rc == nil {
		rc = r.regOf(cl)
		if rc != nil && rc.rc.Leaf != bytecode.LeafNone && !fd.planned {
			// A leaf's code, compiled at its first call, is entered at
			// once from then on (planCall), as any other closure's
			// register code makes.
			planCall(fd)
		}
	}
	if rc == nil || !rc.wide || rc.suspends || cl.realm != r.Realm || isClassConstructorKind(cl.fn.Kind) || cl.fn.HasDirectEval {
		return cl, rc, 0, Undefined, false
	}
	if x := fd.extra; x == nil {
		switch {
		case !method:
			kind, this = regCallThis, Undefined
			if rc.coerceThis {
				this = r.globalThis
			}
		case rc.coerceThis:
			kind = regCallObjThis
		default:
			kind = regCallEnter
		}
	} else if fd.arrow && x.lexThisRef == nil && x.rare == nil {
		kind, this = regCallThis, x.lexThis
	}
	if cl.regTyped != nil || !cl.asmTried {
		if t := r.regTypedOf(cl); t != nil && (caller != nil && caller.typed || rc.leaf == regLeafNone) {
			rc = t
			if tw := t.typedExt().twin; tw != nil {
				// A runtime with the JIT's: the closure whose code is the
				// typed code (asmTwin), its frames' and its native code's.
				cl = tw
			}
		}
	}
	return cl, rc, kind, this, true
}

// regCacheCallee makes site e's cache callee o's, of closure cl and code
// rc, entered as kind with this, marked where rhCallW leaves it to
// regCallMiss: a leaf, a frame that keeps its arguments, and code a code
// entry of its own runs (regCallCross, regTypedCross). caller is the site's
// code, nil for none.
func regCacheCallee(e *regCallCache, o *Object, cl *closure, rc *regCode, this Value, kind uint8, caller *regCode) {
	e.callee, e.cl, e.rc, e.this, e.kind = o, cl, rc, this, kind
	if rc.leaf != regLeafNone {
		e.kind |= regCallOut | regCallLeaf
	} else if rc.args {
		e.kind |= regCallOut
	}
	if regTypedCross(rc, caller) {
		e.kind |= regCallCross
	}
}

// regTypedCross reports whether code rc, called from code caller (nil for
// none), is entered in a code entry of its own: code of another kind,
// typed or ordinary, whose table the entry's context does not hold, or
// typed code of another heap than the caller's, which the entry's context
// does not hold either. A typed function with no heap is entered in its
// caller's entry, whatever its heap; its own callees with one are then
// entered in entries of their own.
func regTypedCross(rc, caller *regCode) bool {
	typed := caller != nil && caller.typed
	if rc.typed != typed {
		return true
	}
	if !typed {
		return false
	}
	h := rc.typedExt().heap
	return h != nil && h != caller.typedExt().heap
}

func (r *Runtime) regCallSlow(f *frame, c *regCode, w bytecode.RInstr, base, pc int) (Value, error) {
	argc := int(w.X())
	method := w.Op().MethodCall()
	callee := r.stack[base]
	if callee.IsObject() {
		o := callee.Object()
		if f.cl.regCalls == nil && c.rc.CallSites > 0 {
			r.regCallsMake(f.cl, c)
		}
		var e *regCallCache
		if calls := f.cl.regCalls; uint(w.C()) < uint(len(calls)) {
			e = &calls[w.C()]
			if !f.cl.regCallsHeld {
				r.regCallsHold(f.cl.closureCode)
			}
		}
		var (
			cl   *closure
			rc   *regCode
			kind uint8
			this Value
		)
		if e != nil && e.callee == o && e.kind == regCallGen {
			return r.regGenCall(o, e.cl, method, base, argc)
		}
		if method && pc < len(c.rc.Marks) {
			// A call its marks say regMarkedCall makes where they are the
			// callee's, and the site then sends it there.
			if m := c.rc.Marks[pc]; m != 0 && r.regMarked(c, o, pc, m) {
				if e != nil {
					*e = regCallCache{callee: o, kind: regCallMarked}
				}
				return r.regMarkedCall(f, c, base, argc, pc, m, regAuxAt(c, pc))
			}
		}
		if e != nil && e.callee == o && e.kind != 0 && e.kind != regCallNative {
			// A callee the site knows, whose call is not rhCallW's alone.
			cl, rc, kind, this = e.cl, e.rc, e.kind&^regCallMarks, e.this
		} else if fd := o.fn(); fd != nil && fd.closure != nil && !fd.bound {
			var enter bool
			cl, rc, kind, this, enter = r.regCallKind(fd, method, c)
			if fn := cl.fn; !enter && e != nil && fn.Generator != fn.Async && fd.extra == nil && cl.realm == r.Realm {
				// A generator or an async function of this realm: the site
				// makes its object as callClosure makes one (regCallGen),
				// without the generic call's search.
				e.callee, e.cl, e.rc, e.this, e.kind = o, cl, nil, Undefined, regCallGen
				return r.regGenCall(o, cl, method, base, argc)
			}
			if kind != 0 && e != nil {
				regCacheCallee(e, o, cl, rc, this, kind, c)
			}
		}
		switch kind {
		case regCallThis:
			r.stack[base+1] = this
		case regCallObjThis:
			if !r.stack[base+1].IsObject() {
				// A this to coerce: the generic call's.
				kind = 0
			}
		}
		if kind != 0 {
			// A body that only reads or writes a property of its this, or
			// computes a value calling nothing, is answered without a frame,
			// as callDirect answers it (leafCall, pureCall): its frame would
			// cost more than its body. A pure body that calls is run as
			// register code, faster than the frameless evaluator's calls.
			if leaf := rc.leaf; leaf != regLeafNone {
				if leaf == regLeafPure {
					if cl.pureMiss < pureMissLimit {
						if v, ok := r.pureCall(o.fn(), r.stack[base+1], r.stack[base+2:base+2+argc]); ok {
							return v, nil
						}
					}
					if cl.pureMiss >= pureMissLimit && e != nil && e.callee == o && !rc.args {
						// A body the frameless evaluator has given up on,
						// for good: its calls are entered as frames.
						e.kind &^= regCallOut | regCallLeaf
					}
				} else if this := r.stack[base+1]; leaf == regLeafThis && this.IsObject() {
					if v, ok := r.leafCall(cl, this.Object(), r.stack[base+2:base+2+argc]); ok {
						return v, nil
					}
				}
			}
			if argc > int(rc.keep) && !rc.args {
				// The arguments past the parameters, which the bindings
				// overwrite: f.arguments reads them. A frame that keeps
				// its arguments has them in f.args.
				r.regKeepWindow(base, int(rc.keep), argc)
			}
			if rc.typed {
				// Typed code, which the typed table runs: a site of typed
				// code, or a crossing from ordinary code (regCallCross).
				return r.regEnterWTyped(o, cl, rc, base, argc)
			}
			return r.regEnterW(o, cl, rc, base, argc, r.stack[base+2:base+2+argc], Undefined)
		}
	}
	if method && callee.IsObject() && callee.Object() == r.callFn {
		// f.call(t, ...): f called with this t in call's frame, without
		// the generic call's search for call and then for f.
		if v, err, ok := r.regCallThrough(f, c, w, base, argc); ok {
			return v, err
		}
		return r.callThrough(r.callFn, r.stack[base+1], r.stack[base+2:base+2+argc])
	}
	if calls := f.cl.regCalls; callee.IsObject() && uint(w.C()) < uint(len(calls)) && !c.typed {
		// A built-in of this realm: the site calls it as callDirect does,
		// without the generic call's search (regNative). Not at a site of
		// typed code, whose native code would call it with no exit, after
		// which it would read its heap as it was before the call.
		o := callee.Object()
		if fd := o.fn(); fd != nil && fd.native != nil && !fd.bound && (fd.realm == nil || fd.realm == r.Realm) && o != r.callFn {
			calls[w.C()] = regCallCache{callee: o, kind: regCallNative}
		}
	}
	if regReportOn {
		regCrossOf(regCrossOut, callee)
	}
	this := Undefined
	if method {
		this = r.stack[base+1]
	}
	return r.regCall(callee, this, r.stack[base+2:base+2+argc])
}

// regCallsMake makes the call caches of cl's register code c, at its
// first call or construction.
//
//go:noinline
func (r *Runtime) regCallsMake(cl *closure, c *regCode) {
	cl.regCalls = r.newRegCalls(cl.closureCode, c.rc.CallSites)
	if jitBuilt {
		r.jitRegCountsMake(cl.closureCode, c.rc.CallSites)
	}
}

// regCallThrough is call_method_w's call of the realm's
// Function.prototype.call, f.call(t, ...), its window at base: f, a
// function the site's cache holds marked regCallThrough, entered in the
// window one register on -- f its callee, t its this, the arguments after
// -- as call_w enters the callee its cache holds (regEnterW), without
// call's frame, which a stack trace and a caller passes over (V8 has
// none); and the cache made f's where f is one the call handler enters
// so, the native handler's to enter from then on -- a leaf with a frame,
// as the handler enters one Go answers frameless otherwise, and one that
// keeps its arguments as regEnterW enters it. A built-in of this realm is
// called as the site would call it (regNative), one register on. ok is
// false where the call is callThrough's: no this given, a callee of
// another kind, a this to coerce.
func (r *Runtime) regCallThrough(f *frame, c *regCode, w bytecode.RInstr, base, argc int) (Value, error, bool) {
	calls := f.cl.regCalls
	if argc < 1 || uint(w.C()) >= uint(len(calls)) || !regIsObject(&r.stack[base+1]) {
		return Undefined, nil, false
	}
	e, o := &calls[w.C()], (*Object)(r.stack[base+1].ref)
	if e.callee != o || e.kind&regCallThrough == 0 {
		fd := o.fn()
		switch {
		case fd == nil || fd.bound:
			return Undefined, nil, false
		case fd.native != nil:
			// A built-in of this realm: called as the site would call it
			// (regNative), one register on.
			if fd.realm != nil && fd.realm != r.Realm || o == r.callFn {
				return Undefined, nil, false
			}
			*e = regCallCache{callee: o, kind: regCallNative | regCallThrough}
		case fd.closure == nil || r.nodeQuirks:
			// With Node's quirks, f.arguments reads the arguments as call
			// passed them (regDirectLegacy): callThrough's.
			return Undefined, nil, false
		default:
			cl, rc, kind, this, enter := r.regCallKind(fd, true, c)
			if !enter || kind == 0 || regTypedCross(rc, c) {
				return Undefined, nil, false
			}
			e.callee, e.cl, e.rc, e.this, e.kind = o, cl, rc, this, kind|regCallThrough
		}
	}
	if e.kind == regCallNative|regCallThrough {
		// The built-in's call of the instruction one register on, the
		// arguments one fewer (regNative), its value R[B].
		wb := base - int(w.A())
		ws := bytecode.MakeR(w.Op(), w.X()-1, w.A()+1, w.B(), w.C())
		if err := r.regNative(o, ws, wb); err != nil {
			return Undefined, err, true
		}
		return r.stack[wb+int(w.B())], nil, true
	}
	nb := base + 1
	switch e.kind &^ regCallThrough {
	case regCallThis:
		r.stack[nb+1] = e.this
	case regCallObjThis:
		if !r.stack[nb+1].IsObject() {
			return Undefined, nil, false
		}
	}
	n := argc - 1
	if n > int(e.rc.keep) {
		// The arguments past the parameters, which the bindings
		// overwrite: f.arguments reads them.
		r.regKeepWindow(nb, int(e.rc.keep), n)
	}
	v, err := r.regEnterW(o, e.cl, e.rc, nb, n, r.stack[nb+2:nb+2+n], Undefined)
	return v, err, true
}

// regCallCache is a call site's cache: the function it called last, its
// closure and register code, how it is entered, and the this a call that
// is not a method's gives it.
type regCallCache struct {
	callee *Object
	cl     *closure
	rc     *regCode
	// this is a call's receiver; for regCallConstruct its unused words
	// cache the constructor's prototype slot (regCtorThis).
	this Value
	kind uint8
}

// The kinds of regCallCache: what entering the callee takes of its this.
// regCallOut marks one rhCallW leaves to regCallMiss: a body answered
// without a frame, or a frame that keeps its arguments.
const (
	// regCallEnter is a method's call of a function whose this is the
	// method's receiver as it is.
	regCallEnter = 1 + iota
	// regCallThis writes the cache's this: a plain call's undefined or
	// global object, or an arrow's this.
	regCallThis
	// regCallObjThis is a method's call of a function that coerces its
	// this, entered as it is for an object receiver.
	regCallObjThis

	// regCallNative is a built-in of this realm, which regNative calls.
	regCallNative = 4
	// regCallGen is a generator function of this realm, whose object
	// regGenCall makes.
	regCallGen = 5
	// regCallMarked is the realm's exec or replace at a call_method_w
	// whose Aux marks its value dead or reused, which regMarkedCall
	// calls: the mark costs the site's other callees nothing.
	regCallMarked = 6
	// regCallConstruct is new_w's: a base constructor of this realm with
	// register code, whose objects' room is settled, and that gives its
	// objects nothing but what its body stores (regConstruct), its object
	// made as regCtorThis makes it.
	regCallConstruct = 7

	regCallOut = 8
	// regCallLeaf marks a callee whose body rhCallW may answer without a
	// frame (leafCall, pureCall).
	regCallLeaf = 16
	// regCallThrough marks the cache of a call of the realm's
	// Function.prototype.call, f.call(t, ...), whose callee, closure, code
	// and kind are f's, entered in the window one register on
	// (regCallThrough): no kind's, so that neither rhCallW nor the call
	// handler's hit takes it for a call of f, nor the JIT's feedback, and
	// none a turn's end restores (regCallsRestored).
	regCallThrough = 32
	// regCallCross marks a callee whose code is of another kind than the
	// site's, typed or ordinary: entered in a code entry of its own, by
	// the table of its kind, from the miss (regCallSlow), which none of
	// the hit paths' compares of the kind lets through.
	regCallCross = 64
)

// regCallMarks are the marks of a call cache's kind that say how a call
// enters its callee, not what it gives it as this.
const regCallMarks = regCallOut | regCallLeaf | regCallCross

// regIsObject reports whether *v is an object, its tag read as a word.
func regIsObject(v *Value) bool {
	return *(*uint64)(unsafe.Pointer(v)) == objectBits
}

// regClearFrame clears what a frame's last user left that a register
// frame has none of.
//
//go:noinline
func regClearFrame(f *frame) {
	f.thisRef = nil
	f.withScopes = nil
	f.evalVars = nil
	f.native = ""
	f.args = nil
}

// regClosure is closure_w: a closure of the nested function constant BC,
// each upvalue it takes from this frame open on the register the code's
// capture table names for it.
func (r *Runtime) regClosure(f *frame, c *regCode, w bytecode.RInstr, wb int) *Object {
	k := w.BC()
	tmpl := f.cl.consts[k]
	t := tmpl.template()
	caps := c.rc.Caps[k]
	if c.wide {
		// What an arrow captures of this is the frame's register 1, which
		// the frame record does not keep (regEnterW).
		f.this = r.stack[wb+1]
	}
	if t.fn.CapturesByValue {
		// What it takes by value copied from its registers, as
		// newClosureIn's newClosureByValue copies it from the slots.
		return regMadeClosure(newClosureByValue[struct{}](r, f, t, caps, wb))
	}
	// Fill the closure's capture list directly from the register mapping.
	// Only bindings first opened here need cells of their own.
	ups := t.fn.Upvalues
	var o *Object
	var fd *funcData
	var child *closure
	var captured []*upvalue
	var extra *funcExtra
	var room []upvalue
	proto := r.funcProtoFor(t.fn)
	if t.captureObject != 0 && len(f.openUpvalues) == 0 {
		var fo *scriptFuncObject
		fo, captured, room = newCapturedFuncObject(proto, len(ups), t.captureObject)
		o, fd, child = &fo.Object, &fo.fn, &fo.cl
	} else {
		o, fd, child, captured, extra = newScriptFuncObject(proto, ClassFunction, len(ups), t.fn.Kind == bytecode.KindArrow)
	}
	for i, u := range ups {
		if !u.FromParent {
			captured[i] = f.cl.upvalues[u.Index]
			continue
		}
		slot := &r.stack[wb+int(caps[i])]
		var up *upvalue
		for _, o := range f.openUpvalues {
			if o.slot == slot {
				up = o
				break
			}
		}
		if up == nil {
			if len(room) == 0 {
				room = make([]upvalue, cellsWanted[struct{}](r, f, ups, i, caps, wb))
			}
			up, room = &room[0], room[1:]
			up.slot = slot
			f.openUpvalues = append(f.openUpvalues, up)
		}
		captured[i] = up
	}
	return regMadeClosure(r.finishClosure(f, t, o, fd, child, captured, extra))
}

// regMadeClosure is closure_w's closure o, its function's code planned
// where it has been built, as makeClosure plans: a function's code is built
// at its first call, which plans the closure called (planCall), so that one
// never called is never compiled (bytecode.RegLater).
func regMadeClosure(o *Object) *Object {
	if fd := o.fn(); fd != nil {
		if atomic.LoadPointer(&fd.closure.fn.VMCode) != nil {
			planCall(fd)
			// A pure body's too, which planCall leaves undecided for
			// pureCall: a closure register code makes enters its code at
			// once (regPlan), whose callers -- a built-in's callback above
			// all -- then run the code the JIT learns from and compiles.
			regPlan(fd)
		}
	}
	return o
}
