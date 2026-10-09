//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"math"
	"os"
	"slices"
	"unsafe"
	"weak"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/jit"
	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
	jitcompile "github.com/go-quickjs/go-quickjs/internal/jit/compile"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
	"github.com/go-quickjs/go-quickjs/internal/jit/mir"
	"github.com/go-quickjs/go-quickjs/internal/jit/ssa"
)

// The new pipeline (docs/jit-phase2-design.md): slot IR, then typed SSA
// (internal/jit/ssa), then machine code (internal/jit/mir). Runtimes choose
// it with QJS_JIT_PIPELINE=ssa until it reaches parity with the slot IR
// emitters; a function it does not compile goes to those.

var jitSSADefault = os.Getenv("QJS_JIT_PIPELINE") == "ssa"

// jitSSABackend reports whether this architecture has the new pipeline.
const jitSSABackend = true

// jitEncoding is how a Value lies in memory, for generated code, and what
// of an Object it reads (D8): an array's class, flags, length and elements,
// which no Go code changes while native code runs, since nothing else runs
// on a runtime's goroutine then and no other goroutine may use the runtime.
// TestJITObjectLayout holds the fields' widths to what native code loads.
var jitEncoding = abi.Encoding{
	ValueSize:     int32(unsafe.Sizeof(Value{})),
	NumOffset:     int32(unsafe.Offsetof(Value{}.num)),
	RefOffset:     int32(unsafe.Offsetof(Value{}.ref)),
	Undefined:     math.Float64bits(Undefined.num),
	Null:          math.Float64bits(Null.num),
	True:          math.Float64bits(True.num),
	False:         math.Float64bits(False.num),
	Uninitialized: math.Float64bits(uninitialized.num),
	CanonicalNaN:  canonicalNaN,
	Object:        objectBits,
	UpvalueSlot:   int32(unsafe.Offsetof(upvalue{}.slot)),
	ObjectShape:   int32(unsafe.Offsetof(Object{}.shape)),
	ObjectProps:   int32(unsafe.Offsetof(Object{}.props)),
	ObjectProto:   int32(unsafe.Offsetof(Object{}.proto)),
	WriteBarrier:  jit.WriteBarrier(),
	PropertySize:  int32(unsafe.Sizeof(Property{})),
	PropertyKey:   int32(unsafe.Offsetof(Property{}.key)),
	PropertyFlags: int32(unsafe.Offsetof(Property{}.flags)),
	PropertyValue: int32(unsafe.Offsetof(Property{}.value)),

	ClassObject:     uint8(ClassObject),
	PropNotData:     uint8(propAccessor | propPrivate | propDeleted),
	PropNotWritable: uint8(propAccessor | propPrivate | propDeleted | propUninit | propWritable),
	PropWritable:    uint8(propWritable),
	PropUninit:      uint8(propUninit),

	String:       math.Float64bits(mkTag(KindString, 0)),
	StringData:   int32(unsafe.Offsetof(String{}.s)),
	StringLeft:   int32(unsafe.Offsetof(String{}.left)),
	StringLength: int32(unsafe.Offsetof(String{}.length)),
	StringASCII:  int32(unsafe.Offsetof(String{}.ascii)),
	StringU16:    int32(unsafe.Offsetof(String{}.u16)),

	ObjectClass:        int32(unsafe.Offsetof(Object{}.class)),
	ObjectFlags:        int32(unsafe.Offsetof(Object{}.flags)),
	ObjectArrayLen:     int32(unsafe.Offsetof(Object{}.arrayLen)),
	ObjectElems:        int32(unsafe.Offsetof(Object{}.elems)),
	ClassArray:         uint8(ClassArray),
	FlagSparse:         uint8(objHasSparseElements),
	FlagHTMLDDA:        uint8(objHTMLDDA),
	FlagExtensible:     uint8(objExtensible),
	FlagLengthWritable: uint8(objArrayLengthWritable),
}

// jitReoptimizations is how many times a function's code is compiled again
// for speculations that failed (jitDeoptimized); after that its guards
// deoptimize, as any do, and jitSSAProfit decides whether it pays.
const jitReoptimizations = 4

// jitDeoptimized notes a guard that failed in e's code, at the site
// ExitSite names, or, for -1, an entry's speculation at the entry the
// stretch started at: the code is compiled again at its next entry, with
// the site, or the entry, generic (jitFeedback.Generic). A site the
// builder has no generic form for is built as before, and its next failure
// is not new.
func jitDeoptimized(e *jitEntry, ctx *abi.Context, start int) {
	if e.reopts >= jitReoptimizations {
		return
	}
	if site := int32(int64(ctx.ExitSite)); site >= 0 {
		if !slices.Contains(e.failed, site) {
			e.failed = append(e.failed, site)
			e.reopt = true
		}
	} else if !slices.Contains(e.failedEntries, uint32(start)) {
		e.failedEntries = append(e.failedEntries, uint32(start))
		e.reopt = true
	}
}

// jitFedSite is a property site, by PC, and the shape its cache knew when
// the code was compiled, by address, or 0 for none; and how often it has
// left native code since its cache learned another.
type jitFedSite struct {
	pc     uint32
	shape  uintptr
	missed uint32
	// seen is the shape its cache knew at the last of those exits: they are
	// counted while it stays the one.
	seen uintptr
}

// jitPolySite is a read that met objects of shapes the code was not
// compiled for (jitEntry.poly): where each had the property, as a cache
// filled for it has it (fillPropCache).
type jitPolySite struct {
	pc    uint32
	cases []propCache
}

// jitPropertyCases is how many shapes a read is compiled for, as V8's
// polymorphic inline caches keep: one more leaves for Go.
const jitPropertyCases = 4

// jitPolySeen notes a read native code left to Go at pc, its receiver on top
// of the frame's operands, up to sp: one found on a prototype, of a shape
// the read has not met, joins those it has, up to jitPropertyCases, and
// the code is compiled again for them once they have settled
// (jitPolySettled). A read of a receiver's own property needs none: native
// code scans for it (mir's property).
func (r *Runtime) jitPolySeen(f *frame, e *jitEntry, pc, sp int, in bytecode.Instr) {
	recv := r.stack[sp-1]
	if !recv.IsObject() || int(in.A) >= len(f.cl.names) {
		return
	}
	var c propCache
	r.fillPropCache(&c, recv.Object(), f.cl.names[in.A], false)
	if c.shape == nil || c.shape == noShape || c.p1 == nil {
		return
	}
	i := slices.IndexFunc(e.poly, func(p jitPolySite) bool { return int(p.pc) == pc })
	if i < 0 {
		e.poly = append(e.poly, jitPolySite{pc: uint32(pc)})
		i = len(e.poly) - 1
	}
	p := &e.poly[i]
	if len(p.cases) >= jitPropertyCases || slices.ContainsFunc(p.cases, func(k propCache) bool { return k.shape == c.shape }) {
		return
	}
	p.cases = append(p.cases, c)
	e.polyPending, e.polySettled = true, 0
}

// jitPolySettle is how many exits in a row native code makes, once a read
// met a shape it was not compiled for, that find no other at any read
// before the code is compiled again for them: a method whose reads meet a
// class's subclasses one by one, as DeltaBlue's constraints are met, is
// compiled again once for them all, not at each -- as V8 optimizes again
// once a function's feedback has settled rather than at each new map.
const jitPolySettle = 4

// jitPolySettled counts an exit from e's code toward compiling it again for
// the shapes its reads have met (jitPolySeen).
func jitPolySettled(e *jitEntry) {
	if e.polyPending {
		if e.polySettled++; e.polySettled >= jitPolySettle {
			e.polyPending, e.polySettled, e.polyReopt = false, 0, true
		}
	}
}

// jitWrongShapeExits is how often a site compiled for one shape leaves
// native code in a row, its cache knowing one other, before the code is
// compiled again: compiling is not free, and a site whose objects are of
// more shapes than one would have it compiled for each in turn.
const jitWrongShapeExits = 32

// jitFed notes an exit to Go at pc. A property site whose cache has met
// objects of a shape the code was not compiled for -- none, for code
// compiled on its first call or a branch not taken until then; another,
// when its receivers' shape changed after -- leaves for Go every time;
// once its cache knows a shape, the code is compiled again for it, as V8
// deoptimizes for insufficient feedback, at once, or for a wrong map, after
// jitWrongShapeExits in a row with one shape.
func jitFed(cl *closure, e *jitEntry, pc uint32) {
	i := slices.IndexFunc(e.fed, func(s jitFedSite) bool { return s.pc == pc })
	if i < 0 || e.reopts >= jitReoptimizations || int(pc) >= len(cl.fn.Code) {
		return
	}
	in := cl.fn.Code[pc]
	if int(in.B) >= len(cl.ic) {
		return
	}
	c, site := &cl.ic[in.B], &e.fed[i]
	if c.shape == noShape || uintptr(unsafe.Pointer(c.shape)) == site.shape {
		return
	}
	if now := uintptr(unsafe.Pointer(c.shape)); now != site.seen {
		site.seen, site.missed = now, 0
	}
	if site.missed++; site.shape == 0 || site.missed >= jitWrongShapeExits {
		e.fed = slices.Delete(e.fed, i, i+1)
		e.reopt = true
	}
}

// jitReoptimize compiles cl's function again for e, its entry, with what
// failed in its code generic, and replaces its code, which runs nowhere:
// native code leaves for Go to do anything else. If the function no longer
// compiles, the code it has is kept.
func (r *Runtime) jitReoptimize(cl *closure, e *jitEntry) {
	switch {
	case e.reopt:
		e.reopts++
	case e.inlineReopt:
		e.inlineReopts++
	case e.upgradeReopt:
		e.upgradeReopts++
	}
	e.reopt, e.inlineReopt, e.polyReopt, e.upgradeReopt = false, false, false, false
	// What it compiles for, the reads' shapes met since included.
	e.polyPending, e.polySettled = false, 0
	fn := cl.fn
	r.jitInlineNativeCalls(e)
	lower := jitcompile.LowerSSA
	if e.ssaCallee {
		lower = jitcompile.LowerSSAInline
	}
	p, err := lower(fn)
	if err != nil {
		return
	}
	code, fb := r.compileSSA(fn, cl, p, r.jitAllowance(fn)+e.ssa.Size(), e)
	if code == nil {
		return
	}
	old := e.ssa
	if len(fb.inlined) > len(e.ssaInlined) {
		// It inlines more: its native callers may inline it now, and
		// theirs, with it.
		r.jitCallersReopt(e, jitInlineDepth)
	}
	e.ssa, e.ssaShapes, e.ssaHolders, e.fed, e.ssaStats = code, fb.shapes, fb.holders, fb.fed, jitSSAStats{}
	e.ssaStrings, e.ssaCallees, e.ssaInlined, e.ssaPools = e.ssaStrings || fb.strings, fb.callees, fb.inlined, fb.pools
	if !e.notNative {
		e.nativeEntry = code.EntryAddress(0)
	}
	// What it leaves on is counted afresh: it may have left on what it was
	// compiled for now.
	e.nativeIn, e.nativeOut = 0, 0
	old.Close()
	r.jit.reoptimized++
}

// compileSSA compiles a lowered function with the new pipeline, or returns
// nil. It takes functions whose slots are the frame's locals, its captured
// bindings, read through their cells, the receiver, which Go puts in the
// context, and its operands: no global slots yet.
//
// The closure's caches say where its sites' properties are (jitFeedback),
// which it returns: the shapes the code compares objects with, and the
// prototypes it compares receivers' with, to be kept alive, and the sites
// whose caches were empty.
func (r *Runtime) compileSSA(fn *bytecode.Function, cl *closure, p *ir.Program, limit int, e *jitEntry) (*jit.SSACode, *jitFeedback) {
	this := 0
	if p.This {
		this = 1
	}
	if len(p.Globals) != 0 || p.Locals != fn.LocalCount+len(fn.Upvalues)+this {
		return nil, nil
	}
	fb := &jitFeedback{r: r, fn: fn, cl: cl, e: e}
	// The compile's memory comes from the runtime's workspaces, taken back
	// once its code is placed.
	s := r.jit
	defer s.ssaWork.Rewind()
	defer s.mirWork.Rewind()
	f, err := ssa.BuildIn(&s.ssaWork, p, fb)
	if err != nil {
		return nil, nil
	}
	f.FrameLocals = fn.LocalCount
	if p.This {
		f.ThisSlot = fn.LocalCount + len(fn.Upvalues)
	}
	ssa.Optimize(f)
	mc, err := mir.CompileIn(&s.mirWork, f, jitEncoding)
	if err != nil || len(mc.Bytes) > limit {
		return nil, nil
	}
	code, err := jit.NewSSACode(r.jit.arena, mc)
	if err != nil {
		return nil, nil
	}
	return code, fb
}

// jitFeedback is ssa.Feedback from a closure: its names' atoms, and its
// property caches. A site that has met objects of one shape, the property
// their own plain data -- and writable, for a write -- knows where it is in
// any object of that shape: the shape settles the table's layout and the
// attributes. A shape a cache remembers is marked seen, so the VM replaces
// it rather than change it when an object's layout changes; the code holds
// it by address, and the entry keeps it alive. Objects of other shapes, and
// of none -- the VM gives a small object a shape only when a cache asks --
// are searched for the key.
type jitFeedback struct {
	r      *Runtime
	fn     *bytecode.Function
	cl     *closure
	shapes []*shape
	// holders are the prototypes the code compares receivers' with
	// (ssa.Holder), held by address, which the entry keeps alive.
	holders []*Object
	// fed are the property sites, each with the shape its cache knew, or
	// none (jitFed).
	fed []jitFedSite
	// root is the function's feedback for an inlined callee's, which keeps
	// what the callee's sites hold for the code; nil for the function's
	// own. strings marks an inlined callee that calls charCodeAt. callees
	// are the entries of the functions the code calls natively.
	root    *jitFeedback
	strings bool
	callees []*jitEntry
	// inlined are the closures of the callees the code inlines; pools the
	// object pools its constructions take from.
	inlined []*closure
	pools   []*abi.ObjectPool
	// e is the entry being compiled again, whose failed speculations
	// (jitEntry.failed) are built generic; nil for a first compile.
	e *jitEntry
}

// keep is the feedback that keeps what the code holds by address: the
// function's own, for an inlined callee's too.
func (fb *jitFeedback) keep() *jitFeedback {
	if fb.root != nil {
		return fb.root
	}
	return fb
}

// Inline is ssa.Feedback's: a call jitCallSeen has seen call one function
// it may inline, unless that stopped. A callee inlines nothing itself.
func (fb *jitFeedback) Inline(pc int) (ssa.InlineSite, bool) {
	if pc >= len(fb.fn.Code) {
		return ssa.InlineSite{}, false
	}
	var targets []jitInline
	if fb.root != nil {
		// An inlined callee's calls, inlined in it as its own code calls
		// them (jitInlineSite).
		if in, ok := fb.r.jitInlineSite(fb.fn, pc, jitInlineDepth-1); ok {
			targets = []jitInline{in}
		}
	} else if fb.e != nil && !slices.Contains(fb.e.notInline, int32(pc)) {
		targets = fb.e.inlines
	}
	k := fb.keep()
	for _, in := range targets {
		if int(in.pc) != pc {
			continue
		}
		p, err := jitcompile.LowerSSAInline(in.cl.fn)
		if err != nil {
			return ssa.InlineSite{}, false
		}
		call := fb.fn.Code[pc]
		this := -1
		if p.This {
			this = in.cl.fn.LocalCount + len(in.cl.fn.Upvalues)
		}
		for _, x := range p.Code {
			k.strings = k.strings || x.Op == ir.StringMethod || x.Op == ir.StringCode
		}
		k.holders = append(k.holders, in.obj)
		k.inlined = append(k.inlined, in.cl)
		return ssa.InlineSite{
			Program: p, Feedback: &jitFeedback{r: fb.r, fn: in.cl.fn, cl: in.cl, root: k},
			Callee: uintptr(unsafe.Pointer(in.obj)), Closure: uintptr(unsafe.Pointer(in.cl)),
			Argc: int(call.A), Method: call.Op == bytecode.OpCallMethod,
			Params: in.cl.fn.ParamCount, ThisSlot: this, Coerce: in.cl.fn.CoerceThis,
		}, true
	}
	return ssa.InlineSite{}, false
}

// NativeCalls is ssa.Feedback's: the functions jitCallSeen has seen a call
// call whose native code a caller's may call -- compiled, and not reading
// what a frame made natively does not have: captured bindings,
// charCodeAt's intrinsic. The callees' entries are kept by the code.
func (fb *jitFeedback) NativeCalls(pc int) []ssa.CallSite {
	if fb.e == nil || fb.root != nil || pc >= len(fb.fn.Code) {
		return nil
	}
	var sites []ssa.CallSite
	for _, in := range fb.e.nativeCalls {
		if int(in.pc) != pc {
			continue
		}
		if in.cl == nil && in.pop {
			// Array.prototype.pop, whose fast path the call's checks keep
			// to: the array's own last element, of the realm's prototype.
			fb.holders = append(fb.holders, in.obj)
			sites = append(sites, ssa.CallSite{Callee: uintptr(unsafe.Pointer(in.obj)), ThisSlot: -1, Method: true, Pop: true,
				Protos: [2]ssa.Holder{{Object: uintptr(unsafe.Pointer(fb.r.proto.array))}}})
			continue
		}
		if in.cl == nil && in.push {
			// Array.prototype.push, whose fast path the call's checks
			// keep to: the prototypes as they are now.
			ap, op := fb.r.proto.array, fb.r.proto.object
			if ap.shape == nil || op.shape == nil || ap.proto != op || op.proto != nil {
				continue
			}
			k := fb.keep()
			k.shapes = append(k.shapes, remember(ap.shape), remember(op.shape))
			k.holders = append(k.holders, in.obj, ap, op)
			sites = append(sites, ssa.CallSite{Callee: uintptr(unsafe.Pointer(in.obj)), ThisSlot: -1, Argc: 1, Method: true, Push: true,
				Protos: [2]ssa.Holder{{Object: uintptr(unsafe.Pointer(ap)), Shape: uintptr(unsafe.Pointer(ap.shape))},
					{Object: uintptr(unsafe.Pointer(op)), Shape: uintptr(unsafe.Pointer(op.shape))}}})
			continue
		}
		if in.cl == nil {
			// A built-in's construction with nothing to run.
			fb.holders, fb.pools = append(fb.holders, in.obj), append(fb.pools, in.pool)
			sites = append(sites, ssa.CallSite{Callee: uintptr(unsafe.Pointer(in.obj)), ThisSlot: -1,
				Pool: uintptr(unsafe.Pointer(in.pool)), Alloc: true})
			continue
		}
		// Only looked up: a compile here would share the workspaces of the
		// one asking. jitCallSeen compiled it.
		ce := fb.r.jit.cache[weak.Make(in.cl.fn)]
		if ce == nil || ce.ssa == nil || ce.ssaStrings || len(in.cl.fn.Upvalues) != 0 {
			continue
		}
		call, fn := fb.fn.Code[pc], in.cl.fn
		this := -1
		if ce.this {
			this = fn.LocalCount
		}
		fb.callees = append(fb.callees, ce)
		fb.holders = append(fb.holders, in.obj)
		site := ssa.CallSite{
			Callee: uintptr(unsafe.Pointer(in.obj)), Closure: uintptr(unsafe.Pointer(in.cl)),
			Entry: uintptr(unsafe.Pointer(&ce.nativeEntry)), Count: uintptr(unsafe.Pointer(&ce.nativeIn)),
			Argc: int(call.A), Method: call.Op == bytecode.OpCallMethod,
			Params: fn.ParamCount, LocalCount: fn.LocalCount, MaxStack: fn.MaxStack, ThisSlot: this, Coerce: fn.CoerceThis,
		}
		if in.via {
			site.Via = uintptr(unsafe.Pointer(fb.r.callFn))
			fb.holders = append(fb.holders, fb.r.callFn)
		}
		if call.Op == bytecode.OpNew {
			i := in.obj.findOwn(atomPrototype)
			if in.pool == nil || i < 0 {
				continue
			}
			site.Pool, site.ProtoIndex, site.ProtoKey = uintptr(unsafe.Pointer(in.pool)), int(i), uint32(atomPrototype)
			site.Coerce = false
			fb.pools = append(fb.pools, in.pool)
		}
		sites = append(sites, site)
	}
	return sites
}

// jitInline is a call jitCallSeen has seen call a function it may inline:
// the call's PC, the function, and how often native code has left the
// inlined callee for Go there.
type jitInline struct {
	pc    int32
	cl    *closure
	obj   *Object
	exits uint32
	// pool, for a construction made natively, is where its objects come
	// from (abi.ObjectPool); via marks a call of Function.prototype.call
	// that calls cl's function (jitCalledVia).
	pool *abi.ObjectPool
	via  bool
	// push marks a call of Array.prototype.push (jitPushes), pop one of
	// Array.prototype.pop (jitPops).
	push, pop bool
}

// jitCallsToInline is how often a call leaves native code before it is
// looked at for inlining: a call seldom made is not worth compiling for.
// jitInlineReoptimizations is how many times a function's code is
// compiled again for its calls, apart from jitReoptimizations: a method
// whose calls are on different branches, as Richards' tasks' are, learns
// them at different times.
// jitCallsHot is how often a call leaves native code, its function past
// those compiles, before it is looked at all the same: up to twice as many.
const (
	jitCallsToInline         = 4
	jitInlineReoptimizations = 4
	jitCallsHot              = 128
)

// What jitCallSeen knows of a call (jitEntry.callSites), past its count: that
// it is made natively, that it is neither that nor inlined, or that it is
// inlined.
const (
	jitCallNative  = 253
	jitCallDone    = 254
	jitCallInlined = 255
)

// jitCallTargets is how many functions a call may call natively: one it
// calls that is not among them leaves for Go, which makes the call.
const jitCallTargets = 4

// jitInlineExits is how often native code may leave an inlined callee for
// Go before the call stops being inlined: each such exit has Go make the
// call over again.
const jitInlineExits = 16

// jitCallSeen notes a call native code left to Go at pc, with the frame's
// operands up to sp: from inside the callee, where it is inlined, which
// after jitInlineExits stops it being inlined; otherwise, the first time,
// whether it calls a function that may be inlined (jitInlinable). The code
// is compiled again for the calls found to inline when a call seen before
// leaves native code again -- once the calls around them, a loop's, have
// been seen too -- or at its next entry. A call seen before costs a look
// at the lists.
func (r *Runtime) jitCallSeen(f *frame, e *jitEntry, pc, sp int, in bytecode.Instr) {
	if in.Op == bytecode.OpNew {
		r.jitRefillPools(f, e, pc, sp, in)
	}
	if e.callSites != nil && e.callSites[pc] == jitCallInlined {
		if o, cl := r.jitCalled(f, sp, in); cl != nil && !slices.ContainsFunc(e.inlines, func(x jitInline) bool { return int(x.pc) == pc && x.obj == o }) {
			r.jitInlinedCalls(e, pc, o, cl)
			return
		}
	}
	if e.callSites != nil && e.callSites[pc] == jitCallNative {
		r.jitCallTarget(f, e, pc, sp, in)
		if e.inlinePending && e.inlineReopts < 2*jitInlineReoptimizations {
			e.inlinePending, e.inlineReopt = false, true
		}
		return
	}
	if e.inlineReopts >= 2*jitInlineReoptimizations {
		return
	}
	// Past its compiles for its calls, a call that keeps leaving earns one
	// more, which every call left to Go costs less than.
	settle := uint8(jitCallsToInline)
	if e.inlineReopts >= jitInlineReoptimizations {
		settle = jitCallsHot
	}
	if e.callSites == nil {
		e.callSites = make([]uint8, len(f.cl.fn.Code))
	}
	switch n := e.callSites[pc]; {
	case n == jitCallInlined:
		r.jitInlineLeft(e, pc)
		fallthrough
	case n == jitCallDone:
		if e.inlinePending {
			e.inlinePending, e.inlineReopt = false, true
		}
		return
	case n+1 < settle:
		e.callSites[pc]++
		if e.inlinePending && n > 0 {
			e.inlinePending, e.inlineReopt = false, true
		}
		return
	}
	e.callSites[pc] = jitCallDone
	o, cl := r.jitCalled(f, sp, in)
	if cl == nil {
		if o, cl := r.jitCalledVia(f, sp, in); cl != nil {
			// G.call(this, ...): G called natively, never inlined.
			if ce := r.jitNativeCallee(cl); ce != nil {
				e.nativeCalls = append(e.nativeCalls, jitInline{pc: int32(pc), cl: cl, obj: o, via: true})
				e.callSites[pc] = jitCallNative
				e.inlinePending = true
			}
			return
		}
		if o := r.jitPops(sp, in); o != nil {
			// a.pop(), made natively where its fast path makes it.
			e.nativeCalls = append(e.nativeCalls, jitInline{pc: int32(pc), obj: o, pop: true})
			e.callSites[pc] = jitCallNative
			e.inlinePending = true
			return
		}
		if o := r.jitPushes(sp, in); o != nil {
			// a.push(v), made natively where its fast path makes it.
			e.nativeCalls = append(e.nativeCalls, jitInline{pc: int32(pc), obj: o, push: true})
			e.callSites[pc] = jitCallNative
			e.inlinePending = true
			return
		}
		if o := r.jitAllocates(sp, in); o != nil {
			// A built-in's construction with nothing to run: made from a
			// pool.
			pool := new(abi.ObjectPool)
			r.jitFillPool(pool, o)
			e.nativeCalls = append(e.nativeCalls, jitInline{pc: int32(pc), obj: o, pool: pool})
			e.callSites[pc] = jitCallNative
			e.inlinePending = true
		}
		return
	}
	if in.Op != bytecode.OpNew && cl.fn != f.cl.fn && (r.jitInlinable(cl.fn) || r.jitInlinesCalls(cl.fn, jitInlineDepth-1)) {
		e.inlines = append(e.inlines, jitInline{pc: int32(pc), cl: cl, obj: o})
		e.callSites[pc] = jitCallInlined
		e.inlinePending = true
		return
	}
	// One that is not inlined is called natively, if its code allows it
	// (NativeCalls), the call itself for a recursive one.
	if in.Op == bytecode.OpNew && !r.jitConstructs(o) {
		return
	}
	if ce := r.jitNativeCallee(cl); ce != nil {
		e.nativeCalls = append(e.nativeCalls, r.jitNativeTarget(int32(pc), cl, o, in))
		e.callSites[pc] = jitCallNative
		e.inlinePending = true
		if ce != e && len(ce.nativeCallers) < jitNativeCallers && !slices.Contains(ce.nativeCallers, e) {
			ce.nativeCallers = append(ce.nativeCallers, e)
		}
	}
}

// jitNativeCallers is how many callers' entries an entry tells when it
// inlines more (jitEntry.nativeCallers).
const jitNativeCallers = 8

// jitCallersReopt has the code of e's native callers, and theirs, depth
// levels up, compiled again where it would inline a call it makes
// natively now (jitInlineNativeCalls): one entered from Go is, at its next
// entry or exit, up to jitInlineReoptimizations times, apart from those
// its own calls have.
func (r *Runtime) jitCallersReopt(e *jitEntry, depth int) {
	if depth == 0 {
		return
	}
	for _, c := range e.nativeCallers {
		if c.upgradeReopts < jitInlineReoptimizations && r.jitNativeCallsInline(c, nil) {
			c.upgradeReopt = true
		}
		r.jitCallersReopt(c, depth-1)
	}
}

// jitInlineNativeCalls has e's code, compiled again, inline the calls
// it made natively to one function that may now be inlined with the calls
// it makes, which its own code has come to inline (jitInlinesCalls), as V8
// inlines a callee's callees when it optimizes again.
func (r *Runtime) jitInlineNativeCalls(e *jitEntry) {
	r.jitNativeCallsInline(e, func(i int) {
		x := e.nativeCalls[i]
		e.nativeCalls = slices.Delete(e.nativeCalls, i, i+1)
		e.inlines = append(e.inlines, jitInline{pc: x.pc, cl: x.cl, obj: x.obj})
		e.callSites[x.pc] = jitCallInlined
	})
}

// jitNativeCallsInline reports whether e's code calls natively, at a call
// that calls it alone, a function that may be inlined with its calls
// now, calling inline with the index of each in e.nativeCalls, which it
// may delete, if it is not nil.
func (r *Runtime) jitNativeCallsInline(e *jitEntry, inline func(int)) bool {
	found := false
	for i := 0; i < len(e.nativeCalls); i++ {
		x := e.nativeCalls[i]
		pc := int(x.pc)
		one := !slices.ContainsFunc(e.nativeCalls, func(y jitInline) bool { return y.pc == x.pc && y.cl != x.cl })
		if !one || x.cl == nil || r.jit.cache[weak.Make(x.cl.fn)] == e || e.callSites == nil || pc >= len(e.callSites) ||
			e.callSites[pc] != jitCallNative || slices.Contains(e.notInline, x.pc) || !r.jitInlinesCalls(x.cl.fn, jitInlineDepth-1) {
			continue
		}
		found = true
		if inline == nil {
			return true
		}
		inline(i)
		i--
	}
	return found
}

// jitInlineLeft counts an exit from the callee inlined at pc, or from its
// call's checks: after jitInlineExits the call is not inlined but made
// natively, if the callee's code allows it, and the code is compiled
// again.
func (r *Runtime) jitInlineLeft(e *jitEntry, pc int) {
	for i := range e.inlines {
		if x := &e.inlines[i]; int(x.pc) == pc && !slices.Contains(e.notInline, x.pc) {
			if x.exits++; x.exits >= jitInlineExits {
				e.notInline = append(e.notInline, x.pc)
				if e.callSites != nil {
					e.callSites[pc] = jitCallDone
					if r.jitNativeCallee(x.cl) != nil {
						e.nativeCalls = append(e.nativeCalls, jitInline{pc: x.pc, cl: x.cl, obj: x.obj})
						e.callSites[pc] = jitCallNative
					}
				}
				e.inlineReopt = true
			}
		}
	}
}

// jitInlineDepth is how many levels of calls are inlined: a callee, the
// calls it makes, theirs (ssa's maxInlineDepth).
const jitInlineDepth = 3

// jitInlinesCalls reports whether a function that makes calls may be
// inlined with them, as V8 inlines a callee's callees: it may be but for
// its calls (ssa.InlineCalls), and its own code inlines every one, each a
// function that may be inlined, with its calls if depth allows.
func (r *Runtime) jitInlinesCalls(fn *bytecode.Function, depth int) bool {
	if depth <= 0 || len(fn.Upvalues) != 0 || fn.TopLevel || fn.IsModule || fn.UsesArguments || fn.HasDirectEval {
		return false
	}
	e := r.jit.cache[weak.Make(fn)]
	if e == nil || e.callSites == nil {
		return false
	}
	p, err := jitcompile.LowerSSAInline(fn)
	if err != nil {
		return false
	}
	calls, ok := ssa.InlineCalls(p)
	if !ok || len(calls) == 0 {
		return false
	}
	for _, pc := range calls {
		if _, ok := r.jitInlineSite(fn, pc, depth); !ok {
			return false
		}
	}
	return true
}

// jitInlineSite is the function fn's call at pc is inlined with, inside a
// callee inlined with fn, depth levels allowing: the one fn's own code
// inlines there, or the one it calls natively there, and no other, which
// may be inlined, with its own calls if depth allows (jitInlinesCalls).
func (r *Runtime) jitInlineSite(fn *bytecode.Function, pc, depth int) (jitInline, bool) {
	e := r.jit.cache[weak.Make(fn)]
	if e == nil || pc >= len(e.callSites) || slices.Contains(e.notInline, int32(pc)) {
		return jitInline{}, false
	}
	var list []jitInline
	switch e.callSites[pc] {
	case jitCallInlined:
		list = e.inlines
	case jitCallNative:
		list = e.nativeCalls
	}
	var in jitInline
	n := 0
	for _, x := range list {
		if int(x.pc) == pc {
			in, n = x, n+1
		}
	}
	if n != 1 || in.cl == nil || in.cl.fn == fn || !r.jitInlinable(in.cl.fn) && !r.jitInlinesCalls(in.cl.fn, depth-1) {
		return jitInline{}, false
	}
	return in, true
}

// jitSeedCalls decides, before e's code for cl's function is compiled, the
// method calls whose target the existing tiers have met (jitMethodAt): each
// is inlined, or called natively if its target has code, at once -- as
// jitCallSeen would decide it once the call had left native code, which
// costs a compile for each call learned so, and as V8's TurboFan takes a
// call's target from its method load's feedback. A call whose target has
// no code yet, or that is not seen so, is left to jitCallSeen. A target
// that turns out wrong costs an exit: a call checks its callee.
func (r *Runtime) jitSeedCalls(cl *closure, e *jitEntry, p *ir.Program) {
	fn := cl.fn
	if len(p.Maps) != len(fn.Code) {
		return
	}
	for pc, in := range fn.Code {
		if in.Op != bytecode.OpCallMethod || e.callSites != nil && e.callSites[pc] != 0 {
			continue
		}
		o, callee := r.jitMethodAt(cl, p, pc)
		if callee == nil {
			continue
		}
		if e.callSites == nil {
			e.callSites = make([]uint8, len(fn.Code))
		}
		if callee.fn != fn && (r.jitInlinable(callee.fn) || r.jitInlinesCalls(callee.fn, jitInlineDepth-1)) {
			e.inlines = append(e.inlines, jitInline{pc: int32(pc), cl: callee, obj: o})
			e.callSites[pc] = jitCallInlined
			continue
		}
		cf := callee.fn
		ce := r.jit.cache[weak.Make(cf)]
		if ce == nil || ce.ssa == nil || cf.TopLevel || cf.IsModule || cf.UsesArguments || cf.HasDirectEval || len(cf.Upvalues) != 0 {
			continue
		}
		e.nativeCalls = append(e.nativeCalls, jitInline{pc: int32(pc), cl: callee, obj: o})
		e.callSites[pc] = jitCallNative
		if ce != e && len(ce.nativeCallers) < jitNativeCallers && !slices.Contains(ce.nativeCallers, e) {
			ce.nativeCallers = append(ce.nativeCallers, e)
		}
	}
}

// jitMethodAt is the function the method call at pc calls as the existing
// tiers have met it, and its closure: the get_prop_this that read the
// call's callee -- the nearest before it at the callee's depth, with
// nothing between but the arguments, above it -- whose cache found one
// plain data property on a prototype, a function jitCalled would take.
// Or nil.
func (r *Runtime) jitMethodAt(cl *closure, p *ir.Program, pc int) (*Object, *closure) {
	fn := cl.fn
	depth := p.Maps[pc].Depth
	slot := depth - int(fn.Code[pc].A) - 1
	if depth < 0 || slot < 1 {
		return nil, nil
	}
	read := -1
	for q := pc - 1; q >= 0; q-- {
		d := p.Maps[q].Depth
		if d == slot && fn.Code[q].Op == bytecode.OpGetPropThis {
			read = q
			break
		}
		// Only the arguments' instructions, which leave the callee be: the
		// first pushes on it, and the rest work above.
		if d < slot+1 || d == slot+1 && p.Maps[q+1].Depth < slot+2 {
			return nil, nil
		}
	}
	if read < 0 {
		return nil, nil
	}
	in := fn.Code[read]
	if int(in.B) >= len(cl.ic) {
		return nil, nil
	}
	c := &cl.ic[in.B]
	h := c.p1
	if c.p2 != nil {
		h = c.p2
	}
	if c.shape == nil || c.shape == noShape || c.getter || h == nil || c.idx < 0 || int(c.idx) >= len(h.props) {
		return nil, nil
	}
	prop := &h.props[c.idx]
	if prop.flags&(propAccessor|propPrivate|propDeleted|propUninit) != 0 || !prop.value.IsObject() {
		return nil, nil
	}
	o := prop.value.Object()
	fd := o.fn()
	if fd == nil || fd.native != nil || fd.bound || fd.closure == nil || fd.closure.realm != r.Realm ||
		fd.closure.scope() != cl.scope() {
		return nil, nil
	}
	return o, fd.closure
}

// jitInlinedCalls has a call inlined for one function that calls another,
// o, cl's, call them natively instead, and others, up to jitCallTargets
// (jitCallTarget): the code is compiled again for it at once.
func (r *Runtime) jitInlinedCalls(e *jitEntry, pc int, o *Object, cl *closure) {
	e.notInline = append(e.notInline, int32(pc))
	e.callSites[pc] = jitCallNative
	e.inlineReopt = true
	for _, x := range e.inlines {
		if int(x.pc) == pc && r.jitNativeCallee(x.cl) != nil {
			e.nativeCalls = append(e.nativeCalls, jitInline{pc: x.pc, cl: x.cl, obj: x.obj})
		}
	}
	if r.jitNativeCallee(cl) != nil {
		e.nativeCalls = append(e.nativeCalls, jitInline{pc: int32(pc), cl: cl, obj: o})
	}
}

// jitCalled is the function object a call at the top of the frame's
// operands, up to sp, calls, and its closure, if it is one native code may
// call or inline: one the script made, in the frame's realm and scope.
func (r *Runtime) jitCalled(f *frame, sp int, in bytecode.Instr) (*Object, *closure) {
	callee := r.stack[sp-int(in.A)-1]
	if !callee.IsObject() {
		return nil, nil
	}
	o := callee.Object()
	fd := o.fn()
	if fd == nil || fd.native != nil || fd.bound || fd.closure == nil || fd.closure.realm != r.Realm ||
		fd.closure.scope() != f.cl.scope() {
		return nil, nil
	}
	return o, fd.closure
}

// jitCallTarget notes a call made natively that left for Go at pc: a
// function it calls that it was not compiled to call joins those it calls
// natively, up to jitCallTargets, and the code is compiled again for it at
// once -- as V8's call feedback goes polymorphic.
func (r *Runtime) jitCallTarget(f *frame, e *jitEntry, pc, sp int, in bytecode.Instr) {
	o, cl := r.jitCalled(f, sp, in)
	if cl == nil {
		return
	}
	n := 0
	for _, x := range e.nativeCalls {
		if int(x.pc) == pc {
			if x.obj == o || x.cl == nil {
				// One it calls, or a built-in's construction, which a site
				// makes alone. One it calls whose code native callers do not
				// call for now is called again after so many such calls.
				if ce := r.jit.hint(cl.hint()); x.obj == o && ce != nil && ce.notNative {
					jitRetryNative(ce)
				}
				return
			}
			n++
		}
	}
	if n >= jitCallTargets || in.Op == bytecode.OpNew && !r.jitConstructs(o) {
		return
	}
	if ce := r.jitNativeCallee(cl); ce != nil {
		e.nativeCalls = append(e.nativeCalls, r.jitNativeTarget(int32(pc), cl, o, in))
		e.inlineReopt = true
	}
}

// jitNativeTarget is a function a call at pc calls natively: for a
// construction, with the pool its objects come from, filled.
func (r *Runtime) jitNativeTarget(pc int32, cl *closure, o *Object, in bytecode.Instr) jitInline {
	x := jitInline{pc: pc, cl: cl, obj: o}
	if in.Op == bytecode.OpNew {
		x.pool = new(abi.ObjectPool)
		r.jitFillPool(x.pool, o)
	}
	return x
}

// jitCalledVia is the function a method call at the top of the stack
// calls through Function.prototype.call, the realm's own, G.call(this,
// ...): G, its receiver, and its closure, if native code may call it, as
// jitCalled has it; or nil.
func (r *Runtime) jitCalledVia(f *frame, sp int, in bytecode.Instr) (*Object, *closure) {
	if in.Op != bytecode.OpCallMethod || in.A == 0 || r.callFn == nil {
		return nil, nil
	}
	callee, recv := r.stack[sp-int(in.A)-1], r.stack[sp-int(in.A)-2]
	if !callee.IsObject() || callee.Object() != r.callFn || !recv.IsObject() {
		return nil, nil
	}
	o := recv.Object()
	fd := o.fn()
	if fd == nil || fd.native != nil || fd.bound || fd.closure == nil || fd.closure.realm != r.Realm ||
		fd.closure.scope() != f.cl.scope() {
		return nil, nil
	}
	return o, fd.closure
}

// jitPops is Array.prototype.pop, the realm's own, if a method call at the
// top of the stack calls it with no argument on an array its fast path
// pops -- dense, of the realm's prototype -- or nil.
func (r *Runtime) jitPops(sp int, in bytecode.Instr) *Object {
	if in.Op != bytecode.OpCallMethod || in.A != 0 {
		return nil
	}
	callee, recv := r.stack[sp-1], r.stack[sp-2]
	if !callee.IsObject() {
		return nil
	}
	fd := callee.Object().fn()
	if fd == nil || fd.elemOp != elemPop || fd.realm != r.Realm || r.plainArray(recv) == nil {
		return nil
	}
	return callee.Object()
}

// jitPushes is Array.prototype.push, the realm's own, if a method call at
// the top of the stack calls it with one argument on an array its fast
// path appends to -- dense, of the realm's prototype, which with its own
// has no element or indexed property (noInheritedIndices) -- or nil.
func (r *Runtime) jitPushes(sp int, in bytecode.Instr) *Object {
	if in.Op != bytecode.OpCallMethod || in.A != 1 {
		return nil
	}
	callee, recv := r.stack[sp-2], r.stack[sp-3]
	if !callee.IsObject() {
		return nil
	}
	fd := callee.Object().fn()
	if fd == nil || fd.elemOp != elemPush || fd.realm != r.Realm {
		return nil
	}
	if o := r.plainArray(recv); o == nil || !r.noInheritedIndices(o) {
		return nil
	}
	return callee.Object()
}

// jitAllocates is the built-in a construction at the top of the stack
// makes with nothing to run, which native code makes from a pool: `new
// Array()`, the realm's own; or nil.
func (r *Runtime) jitAllocates(sp int, in bytecode.Instr) *Object {
	if in.Op != bytecode.OpNew || in.A != 0 {
		return nil
	}
	if c := r.stack[sp-1]; c.IsObject() && c.Object() == r.proto.arrayCtor {
		return c.Object()
	}
	return nil
}

// jitConstructs reports whether native code may construct with o, `new
// o(...)`, as constructWithTarget would: a plain function of this realm,
// compiled, a base constructor with no fields to give its instances, its
// prototype its own data property.
func (r *Runtime) jitConstructs(o *Object) bool {
	fd := o.fn()
	return fd != nil && fd.closure != nil && fd.native == nil && !fd.bound && fd.ctorKind == ctorBase && fd.fieldInit == nil &&
		fd.closure.realm == r.Realm && o.class == ClassFunction && o.findOwn(atomPrototype) >= 0
}

// jitFillPool makes pool's objects for constructions with o, as
// constructWithTarget makes one -- o's prototype, its constructor's root
// shape, room for what its body adds -- while o's prototype is its own
// data property holding an object; the pool is left empty otherwise, and
// native code leaves every construction to Go.
func (r *Runtime) jitFillPool(pool *abi.ObjectPool, o *Object) {
	clear(pool.Objects[:])
	pool.Count, pool.Proto = 0, nil
	if o == r.proto.arrayCtor {
		// `new Array()`: an empty array, of the realm's prototype, with the
		// layout a property cache gives an array it meets (ensureShape),
		// which native code compares receivers' with: a method read on one
		// it made, a.push, would leave for Go to give it.
		for i := range pool.Objects {
			a := r.newArrayFrom(nil)
			r.ensureShape(a)
			pool.Objects[i] = unsafe.Pointer(a)
		}
		pool.Count, pool.Proto = abi.PoolSize, unsafe.Pointer(r.proto.array)
		return
	}
	p := o.getOwnVisible(atomPrototype)
	fd := o.fn()
	if p == nil || p.flags&propAccessor != 0 || !p.value.IsObject() || fd == nil || fd.closure == nil {
		return
	}
	proto := p.value.Object()
	props := int(fd.closure.fn.ThisProps)
	root := r.shapes.ctorRoot(fd)
	if root != nil {
		props = max(props, int(root.slack))
	}
	for i := range pool.Objects {
		obj := newLiteralObject(proto, ClassObject, props)
		obj.shape = root
		pool.Objects[i] = unsafe.Pointer(obj)
	}
	pool.Count, pool.Proto = abi.PoolSize, unsafe.Pointer(proto)
}

// jitRefillPools fills, at a construction that left native code at pc,
// the pool of the function it constructs with, if native code constructs
// with it there and the pool is empty, or was made for another prototype.
func (r *Runtime) jitRefillPools(f *frame, e *jitEntry, pc, sp int, in bytecode.Instr) {
	c := r.stack[sp-int(in.A)-1]
	if !c.IsObject() {
		return
	}
	o := c.Object()
	for _, x := range e.nativeCalls {
		if int(x.pc) != pc || x.obj != o || x.pool == nil {
			continue
		}
		if x.pool.Count == 0 {
			r.jitFillPool(x.pool, o)
		} else if x.cl == nil {
			// A built-in's, whose prototype the realm fixed.
		} else if p := o.getOwnVisible(atomPrototype); p == nil || !p.value.IsObject() || unsafe.Pointer(p.value.Object()) != x.pool.Proto {
			r.jitFillPool(x.pool, o)
		}
	}
}

// jitNativeCallee is the entry of cl's function, compiled now if it is not,
// if a call may make it natively (NativeCalls), or nil: the new pipeline
// compiles it, with a frame of its own, as a call makes it. One LowerSSA
// refuses for leaving native code with no loop to pay for Go's entries to
// it is compiled for native callers alone, who pay none of that, from
// LowerSSAInline, and Go does not enter it (entrySlow). It is compiled
// here, not when a caller's code asks for it: that compile would share the
// caller's workspaces. Code native callers do not call for now, for
// leaving too often (notNative), is the entry all the same: a call's code
// reads where to call at each call, and leaves for Go while there is
// nowhere, until the callee is called natively again (jitRetryNative).
func (r *Runtime) jitNativeCallee(cl *closure) *jitEntry {
	fn := cl.fn
	if fn.TopLevel || fn.IsModule || fn.UsesArguments || fn.HasDirectEval || len(fn.Upvalues) != 0 {
		return nil
	}
	e := r.jitFor(cl)
	if e == nil || e.ssa != nil || e.code != nil || e.deferred || !r.jitSSA {
		if e != nil && e.ssa != nil {
			return e
		}
		return nil
	}
	p, err := jitcompile.LowerSSAInline(fn)
	if err != nil {
		return nil
	}
	r.jitSeedCalls(cl, e, p)
	code, fb := r.compileSSA(fn, cl, p, r.jitAllowance(fn), e)
	if code == nil {
		return nil
	}
	e.setSSA(fn, p, code, fb)
	e.ssaCallee, e.entrySlow = true, true
	r.jit.compiled++
	return e
}

// setSSA gives e the code the new pipeline compiled for fn from p.
func (e *jitEntry) setSSA(fn *bytecode.Function, p *ir.Program, code *jit.SSACode, fb *jitFeedback) {
	e.ssa, e.this, e.ssaShapes, e.ssaHolders, e.fed = code, p.This, fb.shapes, fb.holders, fb.fed
	e.ssaStrings, e.ssaCallees, e.ssaInlined, e.ssaPools = fb.strings, fb.callees, fb.inlined, fb.pools
	e.nativeEntry = code.EntryAddress(0)
	e.ssaLoop = jitLoopLength(fn)
	for _, in := range p.Code {
		e.ssaStrings = e.ssaStrings || in.Op == ir.StringMethod || in.Op == ir.StringCode
	}
}

// jitInlinable reports whether fn may be inlined into a caller: it
// captures nothing, and ssa.Inlinable takes its program. Each function is
// asked once.
func (r *Runtime) jitInlinable(fn *bytecode.Function) bool {
	s := r.jit
	key := weak.Make(fn)
	if ok, seen := s.inlinable[key]; seen {
		return ok
	}
	ok := false
	if len(fn.Upvalues) == 0 && !fn.TopLevel && !fn.IsModule && !fn.UsesArguments && !fn.HasDirectEval {
		p, err := jitcompile.LowerSSAInline(fn)
		ok = err == nil && ssa.Inlinable(p)
	}
	if s.inlinable == nil {
		s.inlinable = map[weak.Pointer[bytecode.Function]]bool{}
	}
	if len(s.inlinable) >= jitCacheEntries*4 {
		clear(s.inlinable)
	}
	s.inlinable[key] = ok
	return ok
}

// Generic and EntryGeneric are ssa.Feedback's: the sites and entries whose
// guards failed in code compiled before (jitDeoptimized).
func (fb *jitFeedback) Generic(pc int) bool {
	return fb.e != nil && slices.Contains(fb.e.failed, int32(pc))
}

func (fb *jitFeedback) EntryGeneric(pc int) bool {
	return fb.e != nil && slices.Contains(fb.e.failedEntries, uint32(pc))
}

// Global is where the global a site reads was last found in the closure's
// scope (the site's idx, which the interpreter keeps there), or, before the
// interpreter has run the site, where the scope has the name now. Native
// code checks the key at that index, and, as the interpreter does, that no
// script-level lexical binding of the name shadows it (lexShadows).
func (fb *jitFeedback) Global(pc int) (ssa.GlobalSite, bool) {
	if fb.cl == nil || pc >= len(fb.fn.Code) {
		return ssa.GlobalSite{}, false
	}
	in := fb.fn.Code[pc]
	if in.Op != bytecode.OpGetGlobal || int(in.A) >= len(fb.cl.names) {
		return ssa.GlobalSite{}, false
	}
	name, env := fb.cl.names[in.A], fb.cl.scope()
	if env == nil {
		return ssa.GlobalSite{}, false
	}
	i := int32(-1)
	if int(in.B) < len(fb.cl.ic) {
		i = fb.cl.ic[in.B].idx
	}
	if i < 0 || int(i) >= len(env.props) || env.props[i].key != name {
		i = env.findOwn(name)
	}
	if i < 0 {
		return ssa.GlobalSite{}, false
	}
	site := ssa.GlobalSite{Key: uint32(name), Index: i}
	if p := &env.props[i]; !p.isAccessor() && p.flags&(propWritable|propConfigurable) == 0 {
		switch {
		case p.value.IsUndefined():
			site.Fixed, site.Constant = true, ir.Value{Kind: ir.Undefined}
		case p.value.IsNumber():
			site.Fixed, site.Constant = true, ir.Float(p.value.Number())
		}
	}
	return site, true
}

func (fb *jitFeedback) Property(pc int) (ssa.PropertySite, bool) {
	site, ok := fb.property(pc)
	if ok && site.Shape != 0 && fb.e != nil && fb.root == nil {
		site.Cases = fb.cases(pc, site.Shape)
	}
	if ok && fb.root == nil && int(fb.fn.Code[pc].B) < len(fb.cl.ic) && fb.cl.ic[fb.fn.Code[pc].B].fills < maxCacheFills &&
		!slices.ContainsFunc(fb.fed, func(s jitFedSite) bool { return s.pc == uint32(pc) }) {
		// A site whose cache may yet learn a shape (jitFed).
		// The cache's shape, though the code may not use it (a getter's,
		// a write that adds the property): what changing it says.
		known := uintptr(0)
		if c := fb.cl.ic[fb.fn.Code[pc].B].shape; c != noShape {
			known = uintptr(unsafe.Pointer(c))
		}
		fb.fed = append(fb.fed, jitFedSite{pc: uint32(pc), shape: known})
	}
	return site, ok
}

// cases are the shapes other than first the read at pc met (jitPolySeen),
// as ssa.PropertyCase has them: the code keeps them and their holders.
func (fb *jitFeedback) cases(pc int, first uintptr) []ssa.PropertyCase {
	var cases []ssa.PropertyCase
	for _, p := range fb.e.poly {
		if int(p.pc) != pc {
			continue
		}
		for _, c := range p.cases {
			if uintptr(unsafe.Pointer(c.shape)) == first {
				continue
			}
			k := ssa.PropertyCase{Shape: uintptr(unsafe.Pointer(c.shape)), Index: c.idx}
			fb.shapes = append(fb.shapes, c.shape)
			for i, h := range [2]struct {
				p *Object
				s *shape
			}{{c.p1, c.s1}, {c.p2, c.s2}} {
				if h.p == nil {
					break
				}
				k.Holders[i] = ssa.Holder{Object: uintptr(unsafe.Pointer(h.p)), Shape: uintptr(unsafe.Pointer(h.s))}
				fb.shapes, fb.holders = append(fb.shapes, h.s), append(fb.holders, h.p)
			}
			cases = append(cases, k)
		}
	}
	return cases
}

func (fb *jitFeedback) property(pc int) (ssa.PropertySite, bool) {
	if fb.cl == nil || pc >= len(fb.fn.Code) {
		return ssa.PropertySite{}, false
	}
	in := fb.fn.Code[pc]
	if in.Op != bytecode.OpGetProp && in.Op != bytecode.OpGetPropThis && in.Op != bytecode.OpSetProp || int(in.A) >= len(fb.cl.names) {
		return ssa.PropertySite{}, false
	}
	site := ssa.PropertySite{Key: uint32(fb.cl.names[in.A])}
	if int(in.B) >= len(fb.cl.ic) {
		return site, true
	}
	c := &fb.cl.ic[in.B]
	k := fb.keep()
	if in.Op == bytecode.OpSetProp && c.next != nil && c.next != setterNext && !c.getter {
		site.Add = fb.add(c)
		return site, true
	}
	if c.shape == nil || c.shape == noShape || c.getter || c.next != nil || c.idx < 0 {
		return site, true
	}
	if c.p1 == nil {
		if c.idx < c.shape.n {
			k.shapes = append(k.shapes, remember(c.shape))
			site.Shape, site.Index = uintptr(unsafe.Pointer(c.shape)), c.idx
		}
		return site, true
	}
	// Found on a prototype (propCache.holder): the receiver's shape, then
	// each prototype's, which the code compares as the cache does, and
	// the index in the last one's table. A read only: a write makes a
	// property of the receiver's own.
	last := c.s1
	if c.p2 != nil {
		last = c.s2
	}
	if in.Op == bytecode.OpSetProp || c.s1 == nil || c.s1 == noShape || c.p2 != nil && (c.s2 == nil || c.s2 == noShape) || c.idx >= last.n {
		return site, true
	}
	k.shapes = append(k.shapes, remember(c.shape), remember(c.s1))
	k.holders = append(k.holders, c.p1)
	site.Holders[0] = ssa.Holder{Object: uintptr(unsafe.Pointer(c.p1)), Shape: uintptr(unsafe.Pointer(c.s1))}
	if c.p2 != nil {
		k.shapes = append(k.shapes, remember(c.s2))
		k.holders = append(k.holders, c.p2)
		site.Holders[1] = ssa.Holder{Object: uintptr(unsafe.Pointer(c.p2)), Shape: uintptr(unsafe.Pointer(c.s2))}
	}
	site.Shape, site.Index = uintptr(unsafe.Pointer(c.shape)), c.idx
	return site, true
}

// add is what a write whose cache adds its property adds (propCache.adds),
// as ssa.PropertyAdd has it, or nil for one native code leaves to Go: one
// whose next shape wants its table's index built (appendTransition). The
// code keeps the shapes and prototypes it compares.
func (fb *jitFeedback) add(c *propCache) *ssa.PropertyAdd {
	next := c.next
	if c.shape == nil || c.shape == noShape || next.index == nil && int(next.n) > linearScanLimit {
		return nil
	}
	add := &ssa.PropertyAdd{From: uintptr(unsafe.Pointer(c.shape)), Next: uintptr(unsafe.Pointer(next)), Flags: uint8(next.flags)}
	for i, h := range [2]struct {
		p *Object
		s *shape
	}{{c.p1, c.s1}, {c.p2, c.s2}} {
		if h.p == nil {
			break
		}
		if h.s == nil || h.s == noShape {
			return nil
		}
		add.Protos[i] = ssa.Holder{Object: uintptr(unsafe.Pointer(h.p)), Shape: uintptr(unsafe.Pointer(h.s))}
	}
	k := fb.keep()
	k.shapes = append(k.shapes, remember(c.shape), remember(next))
	for i, h := range [2]struct {
		p *Object
		s *shape
	}{{c.p1, c.s1}, {c.p2, c.s2}} {
		if add.Protos[i].Object != 0 {
			k.shapes, k.holders = append(k.shapes, remember(h.s)), append(k.holders, h.p)
		}
	}
	return add
}

// jitSlot is a frame's slot as the JIT numbers them: its locals, its
// captured bindings, the receiver if the code reads it (from the context,
// where Go put it), then its operands.
func (r *Runtime) jitSlot(f *frame, e *jitEntry, i int) *Value {
	n, u := f.cl.fn.LocalCount, len(f.cl.upvalues)
	switch {
	case i < n:
		return &f.locals[i]
	case i < n+u:
		return f.cl.upvalues[i-n].slot
	case e.this && i == n+u:
		return (*Value)(unsafe.Pointer(&r.jit.ssaCtx.This))
	}
	if e.this {
		u++
	}
	return &r.stack[f.base+i-n-u]
}

// jitStringMethod is String.prototype.charCodeAt for native code: the
// intrinsic, while the prototype still has it as it was made, which is
// what the old pipeline's string views asked too; and otherwise nothing.
func (r *Runtime) jitStringMethod() Value {
	if r.jitCharCodeAt.ref == nil {
		return Value{}
	}
	m := r.proto.str.getOwn(r.atoms.intern("charCodeAt"))
	if m == nil || m.flags&^propDefault != 0 || !m.value.StrictEquals(r.jitCharCodeAt) {
		return Value{}
	}
	return r.jitCharCodeAt
}

// jitSource is the value a run-time source names (internal/jit/ssa,
// origin.go), which word holds as native code wrote it: the value's
// address -- a slot's in the frame, the receiver's in a context, a captured
// binding's, or a heap cell's native code loaded a reference from -- or nil
// for a primitive's 0. It still holds the reference: native code stores no
// pointer, and nothing else has run since it read it. Reading the address
// back as a pointer is outside unsafe.Pointer's documented rules, and sound
// while Go's heap does not move (docs/jit-progress.md, D8, decided
// 2026-10-08).
func jitSource(word *uint64) *Value {
	if *word == 0 {
		return nil
	}
	return *(**Value)(unsafe.Pointer(word))
}

// jitApplyRecords writes the slots an exit left to Go (abi.Record): the
// references it moved, and the primitives it put where a reference was.
// The slots records read hold their values from entry until the first
// write, so every one is read first; one record, the usual case, reads its
// value and writes it.
func (r *Runtime) jitApplyRecords(f *frame, e *jitEntry, ctx *abi.Context) {
	n := int(ctx.Records)
	if n == 0 {
		return
	}
	r.jit.ssaRecords += uint64(n)
	const flags = abi.RecordScalar | abi.RecordMaybe | abi.RecordDirect
	if n == 1 {
		rec := &ctx.Record[0]
		*r.jitSlot(f, e, int(rec.Slot&^flags)) = r.jitRecordValue(f, e, ctx, 0)
		return
	}
	var buf [8]Value
	src := buf[:0]
	for i := range ctx.Record[:n] {
		src = append(src, r.jitRecordValue(f, e, ctx, i))
	}
	for i, rec := range ctx.Record[:n] {
		*r.jitSlot(f, e, int(rec.Slot&^flags)) = src[i]
	}
}

// jitRecordValue is the value a record writes, read from the frame as the
// exit left it.
func (r *Runtime) jitRecordValue(f *frame, e *jitEntry, ctx *abi.Context, i int) Value {
	switch rec := &ctx.Record[i]; {
	case rec.Slot&abi.RecordDirect != 0:
		slot := abi.Slot{Num: rec.Word, Ref: ctx.RecordRef[i]}
		return *(*Value)(unsafe.Pointer(&slot))
	case rec.Slot&abi.RecordScalar != 0:
		return Value{num: math.Float64frombits(rec.Word)}
	case rec.Slot&abi.RecordMaybe != 0:
		if s := jitSource(&rec.Arg); s != nil && s.ref != nil {
			return *s
		}
		return Value{num: math.Float64frombits(rec.Word)}
	default:
		return *r.jitSlot(f, e, int(rec.Arg))
	}
}

// jitSSAStats counts what one function's new-pipeline code does. Each
// entry starts a native stretch, which ends in a return, an exit to Go or
// a failed guard (ended counts those two), or a poll; work estimates, in
// slot IR instructions, what the stretches did natively.
type jitSSAStats struct {
	entries, hosts, guards, polls, records uint64
	ended, work                            uint64
	// guardsAt counts failed guards by their site (abi.Context.ExitSite:
	// the slot IR PC of the operation, or -1 for an entry's speculation);
	// nil until one fails.
	guardsAt map[int32]uint64
}

// jitSSAProbe is how many native stretches a function runs between looks
// at whether its native code pays (jitSSAProfit), and jitSSAMinWork the
// work a stretch must do, on average, for it to: an entry, and the exit or
// return that ends the stretch, cost what the tree tier takes for about
// that many instructions (the V8 suite's medians of five: 10 left Richards'
// task methods, called from the tree tier and calling out again, native at
// a loss; 20 is best overall, 1,374.8 ms against 1,413.6).
const (
	jitSSAProbe   = 64
	jitSSAMinWork = 20
)

// jitSSAProfit accounts for a native stretch that started at pc, with the
// back-edge counter at edges, and ended as ctx says; every jitSSAProbe
// stretches it decides whether the function's native code pays. Code whose
// stretches do little work before they end -- leaving for Go, a method
// call at every iteration, or returning, a small method the tree tier calls
// that calls out again -- costs more there than it saves, and the tree tier
// runs it from then on. The work is estimated: the distance from the entry
// to the exit or return (the return's block), in slot IR instructions, and
// a loop's mean length for each back-edge taken natively.
func (r *Runtime) jitSSAProfit(e *jitEntry, ctx *abi.Context, start, edges int) {
	st := &e.ssaStats
	work := uint64(0)
	if back := edges - r.backEdges; back > 0 {
		work = uint64(back) * uint64(e.ssaLoop)
	}
	if ctx.ExitKind == abi.ExitHost || ctx.ExitKind == abi.ExitDeopt || ctx.ExitKind == abi.ExitReturn {
		if ctx.ExitKind != abi.ExitReturn {
			st.ended++
		}
		if site := int64(ctx.ExitSite); site > int64(start) {
			work += uint64(site - int64(start))
		}
	}
	st.work += work
	if st.entries%jitSSAProbe == 0 && st.work < st.entries*jitSSAMinWork {
		// Go stops entering it; native callers, which pay neither, go on
		// calling it (NativeCalls).
		e.entrySlow = true
	}
}

// jitLoopLength is the mean length of fn's loops, in instructions: the
// distance each backward branch jumps; 1 for a function with none.
func jitLoopLength(fn *bytecode.Function) uint32 {
	total, loops := 0, 0
	for pc, in := range fn.Code {
		switch in.Op {
		case bytecode.OpJump, bytecode.OpJumpIfFalse, bytecode.OpJumpIfTrue, bytecode.OpJumpIfCmpFalse:
			if int(in.A) <= pc {
				total += pc - int(in.A) + 1
				loops++
			}
		}
	}
	if loops == 0 {
		return 1
	}
	return uint32(total / loops)
}

// runSSA runs a function compiled by the new pipeline from pc, where the
// frame has depth operands, until it returns or leaves native code for the
// rest of the invocation.
func (r *Runtime) runSSA(f *frame, e *jitEntry, pc, depth int) (Value, error, bool) {
	if !e.ssa.HasEntry(pc) {
		return Undefined, nil, false
	}
	s := r.jit
	if s.ssaCtxs == nil {
		s.ssaCtxs = new([jitContexts]abi.Context)
	}
	idx := s.ctxTop
	if idx >= jitContexts-1 {
		// As deep in native code, through Go, as there are contexts.
		return Undefined, nil, false
	}
	s.ctxTop++
	defer func() {
		s.ctxTop--
		// What native code kept (abi.Context.Keep), and the pointer words
		// its native calls recorded (RecordRef), its callees' too, are not
		// kept past it.
		clear(s.ssaCtxs[idx].Keep[:])
		for i := idx; i < jitContexts; i++ {
			if c := &s.ssaCtxs[i]; c.RecordHigh != 0 {
				clear(c.RecordRef[:c.RecordHigh])
				c.RecordHigh = 0
			}
		}
	}()
	ctx := &s.ssaCtxs[idx]
	// resume, when not 0, is where native code goes on after a native
	// call whose callee Go finished, in place of an entry.
	var resume uintptr
	for {
		// What Go has run since the last entry may have changed the frame, how
		// deep calls are, or the scope.
		s.ssaCtx = ctx
		ctx.ReturnTo, ctx.Live, ctx.TailReturn = 0, 0, 0
		r.jitShareContexts(idx, f)
		ctx.Locals = unsafe.Pointer(unsafe.SliceData(f.locals))
		ctx.Stack = unsafe.Pointer(&r.stack[f.base])
		ctx.Upvalues = unsafe.Pointer(unsafe.SliceData(f.cl.upvalues))
		if e.ssaStrings {
			*(*Value)(unsafe.Pointer(&ctx.CharCodeAt)) = r.jitStringMethod()
		}
		if e.this {
			this, bound := f.thisValue()
			if !bound {
				this = uninitialized
			}
			*(*Value)(unsafe.Pointer(&ctx.This)) = this
		}
		s.entries++
		s.ssaEntries++
		e.ssaStats.entries++
		start, edges := pc, r.backEdges
		var err error
		if resume != 0 {
			err, resume = e.ssa.Resume(resume, &s.ssaCtxs[idx+1]), 0
		} else {
			err = e.ssa.Run(pc, ctx)
		}
		if err != nil {
			return r.jitInterpret(f, f.base+depth, nil)
		}
		r.jitSSAProfit(e, ctx, start, edges)
		if c := &s.ssaCtxs[idx+1]; c.Live == abi.LiveCall {
			// A native call's callee left native code: Go finishes it
			// (jitUnwindNative), then this code goes on natively where the
			// callee would have returned to, its state where it left it --
			// as V8's lazy deoptimization leaves a caller's optimized frame
			// alone. Not while the collector marks, as native code takes
			// pointers there; nor into code compiled again meanwhile, nor
			// after a throw: then as below.
			s.hosts++
			e.ssaStats.hosts++
			back, base, code := uintptr(c.ReturnTo), c.Base, e.ssa
			exitPC, depth := int(ctx.ExitPC), int(ctx.ExitDepth)
			f.pc = uint32(exitPC)
			v, err := r.jitUnwindNative(idx, e, f.cl.fn)
			if err == nil && r.stopped == nil && back != 0 && e.ssa == code && !jit.Marking() {
				// Native code run meanwhile may have used the context.
				c.Base, c.Live, c.ReturnTo = base, 0, 0
				*(*Value)(unsafe.Pointer(&c.RetValue)) = v
				s.resumed++
				resume, pc = back, exitPC
				continue
			}
			e.ssaStats.records += ctx.Records
			r.jitApplyRecords(f, e, ctx)
			sp, ok := r.jitCallResult(f, exitPC, depth, v, err)
			if !ok || r.stopped != nil {
				return r.jitInterpret(f, sp, err)
			}
			pc, depth = int(f.pc), sp-f.base
			if e.reopt || e.inlineReopt || e.polyReopt || e.upgradeReopt {
				r.jitReoptimize(f.cl, e)
			}
			if !e.ssa.HasEntry(pc) || e.entrySlow {
				return r.jitInterpret(f, sp, nil)
			}
			continue
		}
		if s.ssaCtxs[idx+1].Live != 0 {
			// An inlined callee's exit: Go makes its frame and finishes it
			// (jitUnwindNative), then this code goes on after the call, as
			// after one Go made.
			s.hosts++
			e.ssaStats.hosts++
			// An inlined callee's exit while the collector marks leaves its
			// records, its frame's included, to Go.
			e.ssaStats.records += ctx.Records
			r.jitApplyRecords(f, e, ctx)
			exitPC, depth := int(ctx.ExitPC), int(ctx.ExitDepth)
			v, err := r.jitUnwindNative(idx, e, f.cl.fn)
			sp, ok := r.jitCallResult(f, exitPC, depth, v, err)
			if !ok || r.stopped != nil {
				return r.jitInterpret(f, sp, err)
			}
			pc, depth = int(f.pc), sp-f.base
			if e.reopt || e.inlineReopt || e.polyReopt || e.upgradeReopt {
				r.jitReoptimize(f.cl, e)
			}
			if !e.ssa.HasEntry(pc) || e.entrySlow {
				return r.jitInterpret(f, sp, nil)
			}
			continue
		}
		if ctx.ExitKind == abi.ExitReturn {
			if s := jitSource(&ctx.RetFrom); s != nil && s.ref != nil {
				return *s, nil, true
			}
			return Value{num: math.Float64frombits(ctx.Ret)}, nil, true
		}
		e.ssaStats.records += ctx.Records
		r.jitApplyRecords(f, e, ctx)
		switch ctx.ExitKind {
		case abi.ExitDeopt:
			// The frame holds the state at the guard; the interpreter runs
			// the rest of this invocation.
			s.guards++
			e.ssaStats.guards++
			if e.ssaStats.guardsAt == nil {
				e.ssaStats.guardsAt = map[int32]uint64{}
			}
			e.ssaStats.guardsAt[int32(int64(ctx.ExitSite))]++
			jitDeoptimized(e, ctx, start)
			f.pc = uint32(ctx.ExitPC)
			return r.jitInterpret(f, f.base+int(ctx.ExitDepth), nil)
		case abi.ExitHost:
			// Go runs the one instruction, then native code goes on from the
			// next entry, if there is one.
			s.hosts++
			e.ssaStats.hosts++
			// The context is every invocation's: Go may run native code of
			// its own, a call's, which overwrites it.
			exitPC := int(ctx.ExitPC)
			in := f.cl.fn.Code[exitPC]
			f.pc = uint32(exitPC)
			switch in.Op {
			case bytecode.OpCall, bytecode.OpCallMethod, bytecode.OpNew:
				r.jitCallSeen(f, e, exitPC, f.base+int(ctx.ExitDepth), in)
			case bytecode.OpGetProp, bytecode.OpGetPropThis:
				r.jitPolySeen(f, e, exitPC, f.base+int(ctx.ExitDepth), in)
			}
			sp, steps, err := r.jitHost(f, f.base+int(ctx.ExitDepth), 1)
			if err != nil || r.stopped != nil || steps == 0 {
				return r.jitInterpret(f, sp, err)
			}
			pc, depth = int(f.pc), sp-f.base
			switch in.Op {
			case bytecode.OpGetProp, bytecode.OpGetPropThis, bytecode.OpSetProp:
				jitFed(f.cl, e, uint32(exitPC))
			}
			jitPolySettled(e)
			if e.reopt || e.inlineReopt || e.polyReopt || e.upgradeReopt {
				// The code is compiled again now, not at the next call: a
				// loop in this one may run long.
				r.jitReoptimize(f.cl, e)
			}
			if !e.ssa.HasEntry(pc) || e.entrySlow {
				// No entry here; or native code does not pay (jitSSAProfit),
				// and the interpreter runs the rest of this invocation.
				return r.jitInterpret(f, sp, nil)
			}
		case abi.ExitPoll:
			// The interpreter's back-edge check, at a loop header native
			// code can be entered again at.
			s.budgets++
			e.ssaStats.polls++
			r.backEdges = backEdgeCheckInterval
			if err := r.checkInterruptNow(); err != nil {
				return Undefined, err, true
			}
			pc, depth = int(ctx.ExitPC), int(ctx.ExitDepth)
			f.pc = uint32(pc)
		}
	}
}

// jitShared is what every context native code runs in shares with the one
// it was called from (abi.Context): where the global names are, the
// VM's stack, how deep calls may go.
type jitShared struct {
	global, lexNames, stackBase unsafe.Pointer
	stackEnd, levelLimit        uint64
}

// jitShareContexts writes what the contexts share into context idx, which
// code is entered in, and every one past it, which its native calls run
// in, with each one's level, when it is not what they hold: a native call
// then writes none of it. They hold it from the last time it was written
// (jitSharedContexts), if that was from idx or before, as nothing else
// writes it.
func (r *Runtime) jitShareContexts(idx int, f *frame) {
	s := r.jit
	sh := jitShared{
		global: unsafe.Pointer(f.cl.scope()), lexNames: unsafe.Pointer(&r.lexNames),
		stackBase: unsafe.Pointer(unsafe.SliceData(r.stack)), stackEnd: uint64(len(r.stack)),
		levelLimit: uint64(min(jitContexts, idx+max(0, r.maxFrames-r.frameDepth))),
	}
	if sh == s.ssaShared && s.ssaSharedFrom <= idx {
		return
	}
	for i := idx; i < jitContexts; i++ {
		c := &s.ssaCtxs[i]
		c.Level, c.LevelLimit = uint64(i), sh.levelLimit
		c.StackBase, c.StackEnd = sh.stackBase, sh.stackEnd
		c.StackTop, c.StackHigh = &r.stackTop, &r.stackHigh
		c.BackEdges = &r.backEdges
		c.Global, c.LexNames = sh.global, sh.lexNames
	}
	s.ssaShared, s.ssaSharedFrom = sh, idx
}

// jitContexts is how many contexts native code runs in at once: one for
// each runSSA running, each native call another (abi.Context.Level).
const jitContexts = 24

// jitNativeLevel is a frame a native call made whose callee left native
// code, or an inlined callee's an exit inside it wrote (abi.Context.Live):
// what Go makes the VM's frame from, and finishes it with. An inlined
// callee's has its program's locals and its receiver's slot plus one
// (abi.Context's Inline fields), and top, the stack's top before it.
type jitNativeLevel struct {
	cl                    *closure
	base                  int
	this                  Value
	kind, pc, depth, site uint64
	inline                bool
	locals, thisSlot, top int
	// construct marks a callee a construction called, whose result, if not
	// an object, is its receiver.
	construct bool
	// ctx is the level's context's index.
	ctx int
}

// jitUnwindNative finishes the native calls the code running in context
// idx, e's, made, the innermost of which left native code, and the inlined
// callee an exit inside wrote the frame of: it makes the VM's frame of
// each, as runFD would have, from the outermost; then finishes each from
// the innermost -- that one from its exit, as runSSA does, the others from
// after their call, with what the one they called returned or threw -- and
// returns what the outermost returned or threw.
func (r *Runtime) jitUnwindNative(idx int, e *jitEntry, code *bytecode.Function) (Value, error) {
	s := r.jit
	// Where the call each level runs for is: e's code's at first, then
	// that of the level before.
	caller, callerPC := code, int(s.ssaCtxs[idx].ExitPC)
	// Its levels go past those of the unwinds it runs inside, which hold
	// theirs until they finish: they are reached by index, the slices
	// growing as the unwinds it runs need.
	start := len(s.unwinding)
	for i := idx + 1; i < jitContexts && s.ssaCtxs[i].Live != 0; i++ {
		c := &s.ssaCtxs[i]
		l := jitNativeLevel{base: int(c.Base), kind: c.ExitKind, pc: c.ExitPC, depth: c.ExitDepth, site: c.ExitSite, ctx: i}
		if c.Live == abi.LiveInline {
			// The callee its caller -- e's code or a native call's callee's
			// -- inlined at the call it left at.
			l.inline, l.locals, l.thisSlot = true, int(c.InlineLocals), int(c.InlineThis)
			// Its code is that of the nearest level below it not inlined,
			// or e's, which inlined it, and the callees it is inlined in.
			parent, parentPC := e, s.ssaCtxs[idx].ExitPC
			outer := true
			for k := len(s.unwinding) - 1; k >= start; k-- {
				if caller := &s.unwinding[k]; !caller.inline {
					parent, parentPC = s.cache[weak.Make(caller.cl.fn)], caller.pc
					break
				}
				outer = false
			}
			for _, cl := range parent.ssaInlined {
				if uintptr(unsafe.Pointer(cl)) == uintptr(c.InlineClosure) {
					l.cl = cl
				}
			}
			if l.cl == nil {
				panic("jit: an inlined callee's closure is not its caller's")
			}
			if outer {
				// Counted once, for the call the outermost is inlined at.
				r.jitInlineLeft(parent, int(parentPC))
			}
		} else {
			l.cl, l.this = (*closure)(c.Closure), *(*Value)(unsafe.Pointer(&c.This))
			l.construct = callerPC < len(caller.Code) && caller.Code[callerPC].Op == bytecode.OpNew
		}
		caller, callerPC = l.cl.fn, int(l.pc)
		s.unwinding = append(s.unwinding, l)
		c.Live, c.ReturnTo = 0, 0
	}
	n := len(s.unwinding) - start
	if n != 0 {
		// The innermost left; the others only return through Go.
		if l := &s.unwinding[len(s.unwinding)-1]; !l.inline {
			if e := s.hint(l.cl.hint()); e != nil {
				r.jitUnwound(e)
			}
		}
	}
	defer func() {
		clear(s.unwinding[start:])
		clear(s.unwindingFrames[start:])
		s.unwinding, s.unwindingFrames = s.unwinding[:start], s.unwindingFrames[:start]
	}()
	s.unwound += uint64(n)
	for k := range n {
		l := &s.unwinding[start+k]
		fn := l.cl.fn
		if l.inline {
			// The exit wrote its slots as its program has them -- locals,
			// receiver, operands -- past its caller's operands: the
			// receiver goes to the frame, the operands down to the locals.
			l.top = r.stackTop
			l.this = Undefined
			if l.thisSlot > 0 {
				l.this = r.stack[l.base+l.thisSlot-1]
			}
			from, to := l.base+l.locals, l.base+fn.LocalCount
			copy(r.stack[to:to+int(l.depth)], r.stack[from:from+int(l.depth)])
			clear(r.stack[to+int(l.depth) : from+int(l.depth)])
			r.stackTop = max(r.stackTop, l.base+fn.LocalCount+fn.MaxStack)
			r.stackHigh = max(r.stackHigh, r.stackTop)
		} else {
			l.top = l.base
		}
		f := r.pushFrame()
		if c := &s.ssaCtxs[l.ctx]; !l.inline && c.Records != 0 {
			// Records its exit left to Go, the collector marking: before
			// anything else reads its frame, an inlined callee's past it
			// included.
			if e := s.cache[weak.Make(fn)]; e != nil {
				f.cl, f.locals, f.base = l.cl, r.stack[l.base:l.base+fn.LocalCount:l.base+fn.LocalCount], l.base+fn.LocalCount
				saved := s.ssaCtx
				s.ssaCtx = c
				r.jitApplyRecords(f, e, c)
				s.ssaCtx = saved
				l.this = *(*Value)(unsafe.Pointer(&c.This))
			}
		}
		f.cl = l.cl
		f.locals = r.stack[l.base : l.base+fn.LocalCount : l.base+fn.LocalCount]
		f.base = l.base + fn.LocalCount
		f.pc = uint32(l.pc)
		f.this, f.newTarget, f.callee, f.args = l.this, Undefined, nil, nil
		f.openUpvalues, f.handlers = f.openUpvalues[:0], f.handlers[:0]
		f.thisRef, f.withScopes, f.evalVars, f.native, f.savedSP = nil, nil, nil, "", 0
		s.unwindingFrames = append(s.unwindingFrames, f)
	}
	var v Value
	var err error
	for k := n - 1; k >= 0; k-- {
		f, l := s.unwindingFrames[start+k], s.unwinding[start+k]
		e := r.jit.hint(f.cl.hint())
		if e == nil {
			e = r.jitFor(f.cl)
		}
		if k == n-1 {
			v, err = r.jitFinishExit(f, e, &l)
		} else {
			// A frame native code made goes on in native code where it can,
			// whatever Go's entries to it cost (entrySlow).
			sp, ok := r.jitCallResult(f, int(l.pc), int(l.depth), v, err)
			switch {
			case !ok || r.stopped != nil || e == nil || e.ssa == nil || !e.ssa.HasEntry(int(f.pc)):
				v, err, _ = r.jitInterpret(f, sp, err)
			default:
				var native bool
				if v, err, native = r.runSSA(f, e, int(f.pc), sp-f.base); !native {
					v, err, _ = r.jitInterpret(f, sp, nil)
				}
			}
		}
		if l.construct && err == nil && !v.IsObject() {
			v = l.this
		}
		r.popFrameOf(f, l.top)
	}
	return v, err
}

// jitUnwindProbe is how many times native code called natively leaves
// between looks at whether it leaves too often, and jitUnwindShare the
// share of its calls it may leave from, as one in so many: Go then finishes
// every native call outside it (jitUnwindNative), which costs more than Go
// making the call.
const (
	jitUnwindProbe = 64
	jitUnwindShare = 4
)

// jitUnwound counts a native call that left native code inside e's code
// (nativeOut), which native callers stop calling if it leaves too often
// (jitUnwindShare): they leave for Go at the call instead, which makes it.
func (r *Runtime) jitUnwound(e *jitEntry) {
	if e.nativeOut++; e.nativeOut%jitUnwindProbe == 0 && e.nativeOut*jitUnwindShare > e.nativeIn {
		e.notNative, e.nativeEntry, e.nativeRetry = true, 0, 0
		e.nativeBackoff = min(e.nativeBackoff+1, jitNativeBackoffs)
	}
}

// jitNativeRetry is how many calls Go makes to code native callers no
// longer call before they call it again, doubled for each time it was
// found to leave too often, up to jitNativeBackoffs times: code that left
// while it learned what to compile for, or whose callers' objects changed,
// may not now.
const (
	jitNativeRetry    = 1024
	jitNativeBackoffs = 8
)

// jitRetryNative counts a call Go makes to e's code, which native callers
// no longer call (notNative) -- one Go enters it for, or one a native
// caller left for Go to make (jitCallTarget), which a function Go does not
// enter, or calls without a frame, has only -- and has them call it again
// after jitNativeRetry, counting its leaving afresh.
func jitRetryNative(e *jitEntry) {
	if e.nativeRetry++; e.nativeRetry >= jitNativeRetry<<(e.nativeBackoff-1) {
		e.notNative, e.nativeIn, e.nativeOut = false, 0, 0
		e.nativeEntry = e.ssa.EntryAddress(0)
	}
}

// jitFinishExit finishes a frame whose native code, called natively, left
// for Go as l says, as runSSA would have had it been entered from Go: Go
// runs the instruction, or the interpreter the rest, or the interrupt
// check is made; native code goes on where it can.
func (r *Runtime) jitFinishExit(f *frame, e *jitEntry, l *jitNativeLevel) (Value, error) {
	pc, depth := int(l.pc), int(l.depth)
	f.pc = uint32(pc)
	switch l.kind {
	case abi.ExitHost:
		// What it learns here it learns as runSSA's exits do: a function
		// only native callers run is entered from Go nowhere else.
		in := f.cl.fn.Code[pc]
		if e != nil && e.ssa != nil {
			e.ssaStats.hosts++
			switch in.Op {
			case bytecode.OpCall, bytecode.OpCallMethod, bytecode.OpNew:
				r.jitCallSeen(f, e, pc, f.base+depth, in)
			case bytecode.OpGetProp, bytecode.OpGetPropThis:
				r.jitPolySeen(f, e, pc, f.base+depth, in)
			}
		}
		sp, steps, err := r.jitHost(f, f.base+depth, 1)
		if err != nil || r.stopped != nil || steps == 0 {
			v, err, _ := r.jitInterpret(f, sp, err)
			return v, err
		}
		if e != nil && e.ssa != nil {
			switch in.Op {
			case bytecode.OpGetProp, bytecode.OpGetPropThis, bytecode.OpSetProp:
				jitFed(f.cl, e, uint32(pc))
			}
			jitPolySettled(e)
			if e.inlinePending && e.inlineReopts < 2*jitInlineReoptimizations {
				e.inlinePending, e.inlineReopt = false, true
			}
			if e.reopt || e.inlineReopt || e.polyReopt || e.upgradeReopt {
				r.jitReoptimize(f.cl, e)
			}
		}
		pc, depth = int(f.pc), sp-f.base
	case abi.ExitPoll:
		r.backEdges = backEdgeCheckInterval
		if err := r.checkInterruptNow(); err != nil {
			return Undefined, err
		}
	default:
		if e != nil {
			e.ssaStats.guards++
		}
		v, err, _ := r.jitInterpret(f, f.base+depth, nil)
		return v, err
	}
	if e != nil && e.ssa != nil && e.ssa.HasEntry(pc) {
		if v, err, native := r.runSSA(f, e, pc, depth); native {
			return v, err
		}
	}
	f.pc = uint32(pc)
	v, err, _ := r.jitInterpret(f, f.base+depth, nil)
	return v, err
}

// jitCallResult has a frame whose state at the call at pc, depth operands
// deep, is in its frame go on after the call, which returned v or threw
// err: the call's operands go, its result comes, as when Go makes it
// (jitHost). It returns the operands' top, and false for a throw, which
// the interpreter takes from there.
func (r *Runtime) jitCallResult(f *frame, pc, depth int, v Value, err error) (int, bool) {
	in := f.cl.fn.Code[pc]
	sp := f.base + depth - int(in.A) - 1
	if in.Op == bytecode.OpCallMethod {
		sp--
	}
	f.pc = uint32(pc + 1)
	if err != nil {
		return sp, false
	}
	r.stack[sp] = v
	return sp + 1, true
}
