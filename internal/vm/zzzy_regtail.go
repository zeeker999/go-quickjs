package vm

import (
	"errors"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
)

// The register interpreter's tail calls (tail_call_w and
// tail_call_method_w; docs/register-interpreter.md, "Stage 2"), in a file
// that sorts after the package's others but the layout lock's last pad:
// what only a tail call runs moves none of the call path's code.

// regCallEnd is regCall's end in an error: where its frame ended in a tail
// call (errTailCall), the chain the call starts made by regTailLoop,
// unless the loop called it (tailLoopDepth). Only regCall's own frame can
// end so: a function with a tail call has no treeCall (regPlan,
// planTreeCall), so callDirect makes its tail calls itself.
//
//go:noinline
func (r *Runtime) regCallEnd(v Value, err error) (Value, error) {
	if err == errTailCall && r.tailLoopDepth != r.frameDepth {
		return r.regTailLoop()
	}
	return v, err
}

// regTailLoop is the loop over the calls a chain of frames ended in
// (errTailCall), each made in place of the last at this depth.
//
//go:noinline
func (r *Runtime) regTailLoop() (Value, error) {
	saved := r.tailLoopDepth
	r.tailLoopDepth = r.frameDepth
	defer func() { r.tailLoopDepth = saved }()
	v, err := Undefined, errTailCall
	for err == errTailCall {
		tc := r.pendingTail
		r.pendingTail = tailCall{}
		// A bound function is resolved to its target here, as run's
		// tailTarget resolves it: called through the built-in, each would
		// make its call a level deeper.
		_, this, args, o, ok, terr := r.tailTarget(tc)
		switch {
		case terr != nil:
			return Undefined, terr
		case ok:
			v, err = r.regCall(Obj(o), this, args)
		default:
			v, err = r.regCall(tc.callee, tc.this, tc.args)
		}
	}
	return v, err
}

// regMissEnd is regCallMiss's end of a call that ended in an error, which
// it returns: where the callee's frame, entered by regEnterW, was given to
// a callee in place (errRegRestart), the frame -- the one past the top, as
// regEnterW left it -- run again, and where it ended in a tail call
// (errTailCall), the call made, its value written to the stack at dst.
//
//go:noinline
func (r *Runtime) regMissEnd(err error, dst int) error {
	if err == errRegRestart {
		u := uint(r.frameDepth)
		return r.regCallWEnd(&r.frames[u/frameBlockSize][u%frameBlockSize], regFrameEnter, err, r.stackTop, dst)
	}
	return r.regCallWEnd(nil, 0, err, 0, dst)
}

// regTailRest is the call a register frame, gone, ended in (errTailCall,
// r.pendingTail), made where the frame's caller would have had its value:
// by regCall, whose loop makes every later call of the chain in place.
func (r *Runtime) regTailRest() (Value, error) {
	tc := r.pendingTail
	r.pendingTail = tailCall{}
	return r.regCall(tc.callee, tc.this, tc.args)
}

// regTailCallW is tail_call_w and tail_call_method_w: call_w and
// call_method_w in tail position, as the stack form makes tail_call. Once
// the stack is tailCallDepth deep, the frame f, an ordinary call's, is
// given to the callee in place (regTailInPlace, errRegRestart) or given up
// (regTail, errTailCall), as the tree's restarts and tailCallNode's frame
// are; otherwise, or where neither can be, the call is regCallW's.
//
//go:noinline
func (r *Runtime) regTailCallW(f *frame, c *regCode, k *regKept, wb int) error {
	if r.frameDepth >= tailCallDepth {
		pc := k.pc - 1
		w := regAt(c.code, pc)
		f.pc = regSrc(c, pc) + 1
		if r.regTailInPlace(f, c, w, wb) {
			return errRegRestart
		}
		if r.regTail(f, w, wb) {
			return errTailCall
		}
	}
	return r.regCallW(f, c, k, wb)
}

// regCallWEnd is regCallW's end of a frame nf, of kind kind, that ended
// in an error,
// which it returns, the frame closed and the stack's top saved again;
// where the frame was given to a callee in place (errRegRestart), the
// frame run again, and where it ended in a tail call (errTailCall), the
// call made, its value written to the stack at dst.
//
//go:noinline
func (r *Runtime) regCallWEnd(nf *frame, kind uint8, err error, saved, dst int) error {
	var v Value
	if err == errRegRestart {
		v, err = r.regRerun(nf, kind, saved)
	}
	if err == errTailCall {
		v, err = r.regTailRest()
	}
	if err != nil {
		return err
	}
	r.stack[dst] = v
	return nil
}

// regRerun runs again frame f, of kind kind, which its caller has closed
// and whose stack top was saved, where a tail call has given it to a callee
// in place (regTailTo): opened again at the same depth with the callee's
// registers, run by regRestartRun, and closed.
func (r *Runtime) regRerun(f *frame, kind uint8, saved int) (Value, error) {
	r.frameDepth++
	if top := f.base + int(r.regRestart.numRegs); top > r.stackTop {
		r.stackTop = top
		if top > r.stackHigh {
			r.stackHigh = top
		}
	}
	f.regKind = kind
	v, err := r.regRestartRun(f)
	f.regKind = 0
	r.frameDepth--
	r.stackTop = saved
	return v, err
}

// errRegRestart is how a call in tail position tells runReg's caller that
// its frame is now the callee's (regTailInPlace), to run from its first
// instruction with r.regRestart's code (regRestartRun): the loop leaves
// through its fail tail, which regCatch leaves to the caller.
//
// regRestartRun runs frame f again with the code a tail call gave it, as
// many times as its tail calls give it another.
//
//go:noinline
func (r *Runtime) regRestartRun(f *frame) (Value, error) {
	v, err := Undefined, errRegRestart
	for err == errRegRestart {
		rc := r.regRestart
		r.regRestart = nil
		v, err = r.runReg(f, rc)
	}
	// The captures of the code the frame was given, which its caller
	// closes only by the first code's.
	if len(f.openUpvalues) != 0 {
		regCloseUpvalues(f)
	}
	return v, err
}

var errRegRestart = errors.New("register frame restarted")

// regTailInPlace is a call in tail position, the stack deep, of a function
// of this realm whose register code is of the form's own layout and needs
// nothing of its surroundings: the frame f, regCallW's or regEnterW's, an
// ordinary call's, its arguments kept nowhere, becomes the callee's -- its
// captured bindings closed, the arguments moved to the parameters, the
// rest of the bindings undefined -- as the tree's restarts and swaps of
// frames do (selfRestart, tailSwap), with no unwinding and no frame made.
// It reports whether it did.
func (r *Runtime) regTailInPlace(f *frame, c *regCode, w bytecode.RInstr, wb int) bool {
	win := regWindow(r.stack, wb+int(w.A()))
	if !win[0].IsObject() {
		return false
	}
	o := win[0].Object()
	fd := o.fn()
	if fd == nil || fd.closure == nil || fd.bound || fd.native != nil || fd.closure.realm != r.Realm {
		return false
	}
	cl := fd.closure
	rc := cl.reg
	if rc == nil {
		rc = r.regOf(cl)
	}
	if rc == nil || isClassConstructorKind(cl.fn.Kind) {
		return false
	}
	this := Undefined
	if w.Op().MethodCall() {
		this = win[1]
	}
	if x := fd.extra; x != nil {
		if !fd.arrow || x.lexThisRef != nil || x.rare != nil {
			return false
		}
		this = x.lexThis
	} else if rc.coerceThis && !this.IsObject() {
		if !this.IsNullish() {
			return false
		}
		this = r.globalThis
	}
	return r.regTailTo(f, w, wb, o, cl, rc, this)
}

// regTailTo is regTailInPlace's frame given to o, of closure cl and
// register code rc, called with this, which its caller has found.
func (r *Runtime) regTailTo(f *frame, w bytecode.RInstr, wb int, o *Object, cl *closure, rc *regCode, this Value) bool {
	if f.regKind != regFrameCall && f.regKind != regFrameEnter || !f.newTarget.IsUndefined() ||
		f.thisRef != nil || f.args != nil || f.withScopes != nil || f.evalVars != nil ||
		!rc.wide || rc.args || rc.locals != 0 {
		return false
	}
	base := f.base
	if base+int(rc.numRegs) > len(r.stack) {
		return false
	}
	if len(f.openUpvalues) != 0 {
		regCloseUpvalues(f)
	}
	argc := min(int(w.X()), int(rc.params))
	src := wb + int(w.A()) + 2
	nw := regWindow(r.stack, base)
	copy(r.stack[base+2:base+2+argc], r.stack[src:src+argc])
	nw[0], nw[1] = Obj(o), this
	for i := min(argc, int(rc.params)); i < int(rc.params+rc.inits); i++ {
		nw[2+i] = Undefined
	}
	if top := base + int(rc.numRegs); top > r.stackTop {
		r.stackTop = top
		if top > r.stackHigh {
			r.stackHigh = top
		}
	}
	f.cl, f.callee, f.this, f.pc = cl, o, this, 0
	r.regRestart = rc
	return true
}

// regTail is call_w or call_method_w of a call in tail position, as the
// stack form's tail_call makes it once the stack is tailCallDepth deep:
// the frame f, an ordinary call's, is given up, its loop leaving with
// errTailCall and the call pending (r.pendingTail), its arguments copied
// out of the window the callee's frame may reuse. It reports false where
// f is a construct's or shares a derived constructor's this, whose call
// is made the ordinary way.
func (r *Runtime) regTail(f *frame, w bytecode.RInstr, wb int) bool {
	switch f.regKind {
	case regFrameCall:
	default:
		// regEnterW's, regCall's, or a frame the tree or run made, whose
		// runTree leaves with errTailCall (regStmt sets treeTail), as
		// tailCallNode's frame does.
		if !f.newTarget.IsUndefined() || f.thisRef != nil {
			return false
		}
	}
	win := regWindow(r.stack, wb+int(w.A()))
	this := Undefined
	if w.Op().MethodCall() {
		this = win[1]
	}
	r.pendingTail = tailCall{callee: win[0], this: this, args: append([]Value(nil), win[2:2+int(w.X())]...)}
	return true
}
