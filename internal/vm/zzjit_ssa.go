//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"math"
	"os"
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

	ObjectClass:    int32(unsafe.Offsetof(Object{}.class)),
	ObjectFlags:    int32(unsafe.Offsetof(Object{}.flags)),
	ObjectArrayLen: int32(unsafe.Offsetof(Object{}.arrayLen)),
	ObjectElems:    int32(unsafe.Offsetof(Object{}.elems)),
	ClassArray:     uint8(ClassArray),
	FlagSparse:     uint8(objHasSparseElements),
	FlagHTMLDDA:    uint8(objHTMLDDA),
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

// jitReoptimize compiles cl's function again for e, its entry, with what
// failed in its code generic, and replaces its code, which runs nowhere:
// native code leaves for Go to do anything else. If the function no longer
// compiles, the code it has is kept.
func (r *Runtime) jitReoptimize(cl *closure, e *jitEntry) {
	e.reopt = false
	e.reopts++
	fn := cl.fn
	p, err := jitcompile.LowerSSA(fn)
	if err != nil {
		return
	}
	code, shapes := r.compileSSA(fn, cl, p, r.jitAllowance(fn)+e.ssa.Size(), e)
	if code == nil {
		return
	}
	old := e.ssa
	e.ssa, e.ssaShapes, e.ssaStats = code, shapes, jitSSAStats{}
	old.Close()
	r.jit.reoptimized++
}

// compileSSA compiles a lowered function with the new pipeline, or returns
// nil. It takes functions whose slots are the frame's locals, its captured
// bindings, read through their cells, the receiver, which Go puts in the
// context, and its operands: no global slots yet.
//
// The closure's caches say where its sites' properties are (jitFeedback);
// it returns the shapes the code compares objects with, to be kept alive.
func (r *Runtime) compileSSA(fn *bytecode.Function, cl *closure, p *ir.Program, limit int, e *jitEntry) (*jit.SSACode, []*shape) {
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
	return code, fb.shapes
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
	// e is the entry being compiled again, whose failed speculations
	// (jitEntry.failed) are built generic; nil for a first compile.
	e *jitEntry
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
	if fb.cl == nil || pc >= len(fb.fn.Code) {
		return ssa.PropertySite{}, false
	}
	in := fb.fn.Code[pc]
	if in.Op != bytecode.OpGetProp && in.Op != bytecode.OpSetProp || int(in.A) >= len(fb.cl.names) {
		return ssa.PropertySite{}, false
	}
	site := ssa.PropertySite{Key: uint32(fb.cl.names[in.A])}
	if int(in.B) < len(fb.cl.ic) {
		c := &fb.cl.ic[in.B]
		if c.shape != nil && c.shape != noShape && c.p1 == nil && !c.getter && c.next == nil && c.idx >= 0 && c.idx < c.shape.n {
			fb.shapes = append(fb.shapes, remember(c.shape))
			site.Shape, site.Index = uintptr(unsafe.Pointer(c.shape)), c.idx
		}
	}
	return site, true
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
// origin.go), which word holds as native code wrote it: a slot's, or, at or
// above abi.MaxRecords, the value at a heap cell native code loaded a
// reference from; nil for a primitive's -1. The cell still holds the
// reference: native code stores no pointer, and nothing else has run since
// it read it. Reading the address back as a pointer is outside
// unsafe.Pointer's documented rules, and sound while Go's heap does not
// move (docs/jit-progress.md, D8, decided 2026-10-08).
func (r *Runtime) jitSource(f *frame, e *jitEntry, word *uint64) *Value {
	switch from := int64(*word); {
	case from < 0:
		return nil
	case from < abi.MaxRecords:
		return r.jitSlot(f, e, int(from))
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
	if n == 1 {
		rec := &ctx.Record[0]
		*r.jitSlot(f, e, int(rec.Slot&^(abi.RecordScalar|abi.RecordMaybe))) = r.jitRecordValue(f, e, rec)
		return
	}
	var buf [8]Value
	src := buf[:0]
	for i := range ctx.Record[:n] {
		src = append(src, r.jitRecordValue(f, e, &ctx.Record[i]))
	}
	for i, rec := range ctx.Record[:n] {
		*r.jitSlot(f, e, int(rec.Slot&^(abi.RecordScalar|abi.RecordMaybe))) = src[i]
	}
}

// jitRecordValue is the value a record writes, read from the frame as the
// exit left it.
func (r *Runtime) jitRecordValue(f *frame, e *jitEntry, rec *abi.Record) Value {
	switch {
	case rec.Slot&abi.RecordScalar != 0:
	case rec.Slot&abi.RecordMaybe != 0:
		if s := r.jitSource(f, e, &rec.Arg); s != nil && s.ref != nil {
			return *s
		}
	default:
		return *r.jitSlot(f, e, int(rec.Arg))
	}
	return Value{num: math.Float64frombits(rec.Word)}
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
// work a stretch that ends in an exit to Go must do, on average, for it to:
// an exit and the entry after it cost what the tree tier takes for about
// that many instructions (BenchmarkJITHostRoundTrip's go-call: a stretch
// of about ten is level with the tree tier).
const (
	jitSSAProbe   = 64
	jitSSAMinWork = 10
)

// jitSSAProfit accounts for a native stretch that started at pc, with the
// back-edge counter at edges, and ended as ctx says; every jitSSAProbe
// stretches it decides whether the function's native code pays. Code whose
// stretches mostly end leaving for Go, after little work -- a method call
// at every iteration, through Go -- costs more there than it saves, and the
// tree tier runs it from then on. The work is estimated: the distance from
// the entry to the exit, in slot IR instructions, and a loop's mean length
// for each back-edge taken natively.
func (r *Runtime) jitSSAProfit(e *jitEntry, ctx *abi.Context, start, edges int) {
	st := &e.ssaStats
	work := uint64(0)
	if back := edges - r.backEdges; back > 0 {
		work = uint64(back) * uint64(e.ssaLoop)
	}
	if ctx.ExitKind == abi.ExitHost || ctx.ExitKind == abi.ExitDeopt {
		st.ended++
		if site := int64(ctx.ExitSite); site > int64(start) {
			work += uint64(site - int64(start))
		}
	}
	st.work += work
	if st.entries%jitSSAProbe == 0 && st.ended*2 > st.entries && st.work < st.entries*jitSSAMinWork {
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
	if s.ssaCtx == nil {
		s.ssaCtx = new(abi.Context)
	}
	ctx := s.ssaCtx
	for {
		// The stack can move between entries, after Go has run something.
		ctx.Locals = unsafe.Pointer(unsafe.SliceData(f.locals))
		ctx.Stack = unsafe.Pointer(&r.stack[f.base])
		ctx.BackEdges = &r.backEdges
		ctx.Upvalues = unsafe.Pointer(unsafe.SliceData(f.cl.upvalues))
		ctx.Global, ctx.LexNames = unsafe.Pointer(f.cl.scope()), unsafe.Pointer(&r.lexNames)
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
		if err := e.ssa.Run(pc, ctx); err != nil {
			return r.jitInterpret(f, f.base+depth, nil)
		}
		r.jitSSAProfit(e, ctx, start, edges)
		if ctx.ExitKind == abi.ExitReturn {
			from := ctx.RetFrom - 1
			if s := r.jitSource(f, e, &from); s != nil && s.ref != nil {
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
			f.pc = uint32(ctx.ExitPC)
			sp, steps, err := r.jitHost(f, f.base+int(ctx.ExitDepth), 1)
			if err != nil || r.stopped != nil || steps == 0 {
				return r.jitInterpret(f, sp, err)
			}
			pc, depth = int(f.pc), sp-f.base
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
