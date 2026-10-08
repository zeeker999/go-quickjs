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

// jitEncoding is how a Value lies in memory, for generated code.
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
}

// compileSSA compiles a lowered function with the new pipeline, or returns
// nil. The walking skeleton takes functions whose slots are exactly the
// frame's locals and operands: no captured bindings, receiver snapshot or
// global slots.
func (r *Runtime) compileSSA(fn *bytecode.Function, p *ir.Program, limit int) *jit.SSACode {
	if p.This || len(p.Globals) != 0 || p.Locals != fn.LocalCount {
		return nil
	}
	f, err := ssa.Build(p)
	if err != nil {
		return nil
	}
	ssa.Optimize(f)
	mc, err := mir.CompileAMD64(f, jitEncoding)
	if err != nil || len(mc.Bytes) > limit {
		return nil
	}
	code, err := jit.NewSSACode(mc)
	if err != nil {
		return nil
	}
	return code
}

// jitSlot is a frame's slot as the JIT numbers them: its locals, then its
// operands.
func (r *Runtime) jitSlot(f *frame, i int) *Value {
	if n := f.cl.fn.LocalCount; i >= n {
		return &r.stack[f.base+i-n]
	}
	return &f.locals[i]
}

// jitApplyRecords writes the slots an exit left to Go (abi.Record): the
// references it moved, and the primitives it put where a reference was.
// The slots records read hold their values from entry until the first
// write, so every one is read first.
func (r *Runtime) jitApplyRecords(f *frame, ctx *abi.Context) {
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
				if s := r.jitSlot(f, int(from)); s.ref != nil {
					v = *s
				}
			}
		default:
			v = *r.jitSlot(f, int(rec.Arg))
		}
		src = append(src, v)
	}
	for i, rec := range ctx.Record[:n] {
		*r.jitSlot(f, int(rec.Slot&^(abi.RecordScalar|abi.RecordMaybe))) = src[i]
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
		s.entries++
		s.ssaEntries++
		if err := e.ssa.Run(pc, ctx); err != nil {
			return r.jitInterpret(f, f.base+depth, nil)
		}
		if ctx.ExitKind == abi.ExitReturn {
			if ctx.RetFrom != 0 {
				if v := *r.jitSlot(f, int(ctx.RetFrom)-1); v.ref != nil {
					return v, nil, true
				}
			}
			return Value{num: math.Float64frombits(ctx.Ret)}, nil, true
		}
		r.jitApplyRecords(f, ctx)
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
