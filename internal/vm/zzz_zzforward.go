//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"slices"
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
	jitcompile "github.com/go-quickjs/go-quickjs/internal/jit/compile"
	"github.com/go-quickjs/go-quickjs/internal/jit/ssa"
)

// Forwarding calls: f.apply(x, arguments) in a function whose arguments
// object is never made (bytecode's ForwardsArguments, apply_arguments),
// made by native code as V8's TurboFan makes JSCallForwardVarargs -- a
// call of f with the arguments the frame was given, copied from where they
// are to the callee's frame, their count read at run time
// (ssa.CallSite's Forward). A native callee's are in its context, where
// its call wrote them (jitFrameArgs); a frame Go enters the code with has
// them in the VM's frame, which Go copies to the context at each entry
// (jitForwardEntry).


// slicesHasSite reports whether sites calls in's function object already,
// or its function by its code.
func slicesHasSite(sites []ssa.CallSite, in jitInline) bool {
	for _, s := range sites {
		if s.Callee == uintptr(unsafe.Pointer(in.obj)) || s.ByCode != 0 && s.ByCode == uintptr(unsafe.Pointer(in.cl.fn)) &&
			s.Env == uintptr(unsafe.Pointer(in.cl.env)) && s.Realm == uintptr(unsafe.Pointer(in.cl.realm)) {
			return true
		}
	}
	return false
}


// jitForwardApplies reports whether the call of Go the code in ctx makes
// at pc of fn, an apply_arguments (abi.GoCallForward), is one Go makes
// from the frame's arguments: its apply the realm's, in code that hands
// them on as they were given (compile's ForwardsNatively). Any other the
// interpreter makes, which makes the arguments object.
func (r *Runtime) jitForwardApplies(ctx *abi.Context, fn *bytecode.Function, pc int) bool {
	if pc >= len(fn.Code) || fn.Code[pc].Op != bytecode.OpApplyArguments || !jitcompile.ForwardsNatively(fn) {
		return false
	}
	ap := jitGoArg(ctx, 1)
	return r.applyFn != nil && ap.IsObject() && ap.Object() == r.applyFn
}

// jitForwardMade makes f.apply(x, arguments), its operands at the top of
// the VM's stack up to sp, with args, the frame's arguments, as the
// interpreter's applyArguments makes it: through the built-in apply's
// frame (applyCall). Arguments a native call wrote to a context, which
// its next call there overwrites, are copied first (copied).
func (r *Runtime) jitForwardMade(args []Value, sp int, copied bool) (Value, error) {
	target, this := r.stack[sp-3], r.stack[sp-1]
	if !copied {
		return r.applyCall(target, this, args)
	}
	i, n := len(r.argStack), len(args)
	r.argStack = append(r.argStack, args...)
	v, err := r.applyCall(target, this, r.argStack[i:i+n:i+n])
	clear(r.argStack[i:])
	r.argStack = r.argStack[:i]
	return v, err
}

// jitForwardEntry writes args, the arguments of the frame Go enters code
// that hands them on with (ir.Program's Forwards), to its context, as a
// native call writes its callee's (abi.Context's Argc and Args): their
// count, which past the context's room for them has the code's forwarding
// calls made by Go, from the frame's; and their words, as the context
// holds them. The frame keeps them alive while the code runs.
func jitForwardEntry(ctx *abi.Context, args []Value) {
	ctx.Argc = uint64(len(args))
	if len(args) > len(ctx.Args) {
		return
	}
	for i := range args {
		ctx.Args[i] = *(*abi.GoArg)(unsafe.Pointer(&args[i]))
	}
}

// jitForwardPlain makes Array.apply(null, arguments) -- a forwarding call
// (abi.GoCallForward) of the realm's Array through its apply -- as the
// constructor called so makes it, an array of the frame's arguments,
// without a frame: what runs no script and cannot throw. Array(n) of one
// number, which makes an array of that length or throws, it leaves to
// jitGoCall, as it does more arguments than the context holds -- the
// frame's, read there -- and a runtime that meters its memory.
func (r *Runtime) jitForwardPlain(ctx *abi.Context) bool {
	if ctx.Argc > uint64(len(ctx.Args)) || r.meter != nil || r.applyFn == nil {
		return false
	}
	t, ap := jitGoArg(ctx, 0), jitGoArg(ctx, 1)
	if !t.IsObject() || t.Object() != r.proto.arrayCtor || !ap.IsObject() || ap.Object() != r.applyFn {
		return false
	}
	args := jitContextArgs(ctx)
	if len(args) == 1 && args[0].IsNumber() {
		return false
	}
	*(*Value)(unsafe.Pointer(&ctx.Keep[ctx.GoKeep])) = Obj(r.newArrayFrom(args))
	return true
}

// jitForwardLearned is the function the apply_arguments at pc of cl's
// code calls natively, as that code's own forwards learned it
// (jitForwardSeen), where it is one function object a native caller may
// call: what a call of cl inlined in a caller forwards its arguments to,
// inlined in turn (forwardInline). Or nil.
func (r *Runtime) jitForwardLearned(cl *closure, pc int) *Object {
	e := r.jit.entryFor(cl.fn)
	if e == nil || e.callSites == nil || pc >= len(e.callSites) || e.callSites[pc] != jitCallNative {
		return nil
	}
	var t *Object
	for _, x := range e.nativeCalls {
		if int(x.pc) != pc {
			continue
		}
		if t != nil || x.byCode || x.cl == nil || x.obj == nil {
			return nil
		}
		t = x.obj
	}
	if t == nil || !r.jitNativeTargetOf(t.fn(), cl, false) {
		return nil
	}
	return t
}

// jitForwardsCall reports whether fn forwards its arguments,
// f.apply(this, arguments), its arguments object made for nothing else
// (bytecode's ForwardsArguments): a call of it, not a construction, is
// inlined where jitForwardsCallInline says, as V8 inlines a callee whose
// arguments its caller knows, the forward a call with them.
func jitForwardsCall(fn *bytecode.Function) bool {
	return (fn.ForwardsArguments || fn.Leaf == bytecode.LeafForward) && !jitReadsArguments(fn)
}


// jitPlainThisCell is a cell holding the receiver a plain call of cl's
// function gives it, for its frame inlined (ssa.InlineSite's ThisCell): the
// realm's global object, a sloppy function's this coerced, or undefined, a
// strict one's, as callFunction gives it; kept by fb's compile. False for
// a function of another realm, whose global object is its own.
func (fb *jitFeedback) jitPlainThisCell(cl *closure) (uintptr, bool) {
	cell := new(Value)
	*cell = Undefined
	if cl.fn.CoerceThis {
		if cl.realm != fb.r.Realm || !fb.r.globalThis.IsObject() {
			return 0, false
		}
		*cell = fb.r.globalThis
	}
	k := fb.keep()
	k.cells = append(k.cells, cell)
	return uintptr(unsafe.Pointer(cell)), true
}

// jitForwardsCoerced reports whether a function cl's code forwards its
// arguments to (jitForwardLearned) coerces its receiver: a sloppy one.
func (r *Runtime) jitForwardsCoerced(cl *closure) bool {
	for pc, in := range cl.fn.Code {
		if in.Op != bytecode.OpApplyArguments {
			continue
		}
		if t := r.jitForwardLearned(cl, pc); t == nil || t.fn().closure.fn.CoerceThis {
			return true
		}
	}
	return false
}
