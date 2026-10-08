//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64))

package vm

import (
	"math"
	"os"
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/jit"
	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
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

	ObjectClass:    int32(unsafe.Offsetof(Object{}.class)),
	ObjectFlags:    int32(unsafe.Offsetof(Object{}.flags)),
	ObjectArrayLen: int32(unsafe.Offsetof(Object{}.arrayLen)),
	ObjectElems:    int32(unsafe.Offsetof(Object{}.elems)),
	ClassArray:     uint8(ClassArray),
	FlagSparse:     uint8(objHasSparseElements),
}

// compileSSA compiles a lowered function with the new pipeline, or returns
// nil. It takes functions whose slots are the frame's locals, its captured
// bindings, read through their cells, the receiver, which Go puts in the
// context, and its operands: no global slots yet.
//
// The closure's caches say where its sites' properties are (jitFeedback);
// it returns the shapes the code compares objects with, to be kept alive.
func (r *Runtime) compileSSA(fn *bytecode.Function, cl *closure, p *ir.Program, limit int) (*jit.SSACode, []*shape) {
	this := 0
	if p.This {
		this = 1
	}
	if len(p.Globals) != 0 || p.Locals != fn.LocalCount+len(fn.Upvalues)+this {
		return nil, nil
	}
	fb := &jitFeedback{fn: fn, cl: cl}
	f, err := ssa.BuildWith(p, fb)
	if err != nil {
		return nil, nil
	}
	f.FrameLocals = fn.LocalCount
	if p.This {
		f.ThisSlot = fn.LocalCount + len(fn.Upvalues)
	}
	ssa.Optimize(f)
	mc, err := mir.CompileAMD64(f, jitEncoding)
	if err != nil || len(mc.Bytes) > limit {
		return nil, nil
	}
	code, err := jit.NewSSACode(mc)
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
	fn     *bytecode.Function
	cl     *closure
	shapes []*shape
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

// jitApplyRecords writes the slots an exit left to Go (abi.Record): the
// references it moved, and the primitives it put where a reference was.
// The slots records read hold their values from entry until the first
// write, so every one is read first.
func (r *Runtime) jitApplyRecords(f *frame, e *jitEntry, ctx *abi.Context) {
	n := int(ctx.Records)
	if n == 0 {
		return
	}
	r.jit.ssaRecords += uint64(n)
	var buf [8]Value
	src := buf[:0]
	for _, rec := range ctx.Record[:n] {
		v := Value{num: math.Float64frombits(rec.Word)}
		switch {
		case rec.Slot&abi.RecordScalar != 0:
		case rec.Slot&abi.RecordMaybe != 0:
			if from := int32(rec.Arg); from >= 0 {
				if s := r.jitSlot(f, e, int(from)); s.ref != nil {
					v = *s
				}
			}
		default:
			v = *r.jitSlot(f, e, int(rec.Arg))
		}
		src = append(src, v)
	}
	for i, rec := range ctx.Record[:n] {
		*r.jitSlot(f, e, int(rec.Slot&^(abi.RecordScalar|abi.RecordMaybe))) = src[i]
	}
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
		if e.this {
			this, bound := f.thisValue()
			if !bound {
				this = uninitialized
			}
			*(*Value)(unsafe.Pointer(&ctx.This)) = this
		}
		s.entries++
		s.ssaEntries++
		if err := e.ssa.Run(pc, ctx); err != nil {
			return r.jitInterpret(f, f.base+depth, nil)
		}
		if ctx.ExitKind == abi.ExitReturn {
			if ctx.RetFrom != 0 {
				if v := *r.jitSlot(f, e, int(ctx.RetFrom)-1); v.ref != nil {
					return v, nil, true
				}
			}
			return Value{num: math.Float64frombits(ctx.Ret)}, nil, true
		}
		r.jitApplyRecords(f, e, ctx)
		switch ctx.ExitKind {
		case abi.ExitDeopt:
			// The frame holds the state at the guard; the interpreter runs
			// the rest of this invocation.
			s.guards++
			f.pc = uint32(ctx.ExitPC)
			return r.jitInterpret(f, f.base+int(ctx.ExitDepth), nil)
		case abi.ExitHost:
			// Go runs the one instruction, then native code goes on from the
			// next entry, if there is one.
			s.hosts++
			f.pc = uint32(ctx.ExitPC)
			sp, steps, err := r.jitHost(f, f.base+int(ctx.ExitDepth), 1)
			if err != nil || r.stopped != nil || steps == 0 {
				return r.jitInterpret(f, sp, err)
			}
			pc, depth = int(f.pc), sp-f.base
			if !e.ssa.HasEntry(pc) {
				return r.jitInterpret(f, sp, nil)
			}
		case abi.ExitPoll:
			// The interpreter's back-edge check, at a loop header native
			// code can be entered again at.
			s.budgets++
			r.backEdges = backEdgeCheckInterval
			if err := r.checkInterruptNow(); err != nil {
				return Undefined, err, true
			}
			pc, depth = int(ctx.ExitPC), int(ctx.ExitDepth)
			f.pc = uint32(pc)
		}
	}
}
