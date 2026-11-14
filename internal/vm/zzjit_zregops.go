//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
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
