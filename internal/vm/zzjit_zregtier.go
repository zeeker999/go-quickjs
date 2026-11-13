//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"slices"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
)

// Tier-up from the register interpreter (docs/register-interpreter.md,
// Stage 3, 3a-3's step B): a runtime that runs the JIT compiles a register
// function's code once it is hot, and enters it at its calls and at its
// loops' headers.
//
// Hotness is sampled, not counted: no handler of either dispatcher -- the
// Go handler table's or the native handlers' -- does any work for it. The
// two budgets they already spend are the samples. A call that spends the
// interrupt budget (rhCallW's and regEnterW's check, which the native
// call handler leaves to rhCallW) credits its callee; a back edge that
// spends the back-edge budget (regBackEdge, which every dispatcher's jump
// leaves to its Go slow function) credits its frame's function. The
// function is compiled at its jitRegSamples-th sample, from what its
// caches learned meanwhile.
//
// Its code is entered:
//   - by a call that makes its frame in Go (regEnterW, jitRegEnter): a
//     call site is sent there once it is sampled calling a function that
//     has code (regCallJIT), which no fast path enters;
//   - by on-stack replacement at the loop whose back edge took the sample
//     (jitRegLoop), in the dispatcher that runs the frame, which goes on
//     where the code leaves it: at the register pc of a deoptimization or
//     of an instruction with no entry after it, or, where the code
//     returned, at a return of the code's own with what it returned.

// regCallJITMark is regCallJIT where the JIT is built in (regCallMarks).
const regCallJITMark = regCallJIT

// jitRegSamples is how many samples make a function hot.
const jitRegSamples = 2

// jitRegDeoptLimit is how many deoptimizations a function's code is
// compiled again after: past it, the function stays in the interpreter.
const jitRegDeoptLimit = 8

// jitRegTakes reports whether cl's register code c is compiled: a plain
// function's or a method's code of the form's own layout that does not
// suspend and has a frame -- a leaf's calls are answered without one.
func jitRegTakes(cl *closure, c *regCode) bool {
	k := cl.fn.Kind
	return c.wide && !c.suspends && c.leaf == regLeafNone && (k == bytecode.KindNormal || k == bytecode.KindMethod) && !cl.fn.TopLevel &&
		c.rc == cl.fn.RegCode()
}

// jitRegHas reports whether cl's function has native code made from its
// register code.
func (r *Runtime) jitRegHas(cl *closure) bool {
	return cl.closureCode != nil && cl.closureCode.jitReg != nil
}

// jitRegSample is a sample of a run of cl's register code c (regEnterW).
func (r *Runtime) jitRegSample(cl *closure, c *regCode) {
	if r.jitEnabled && cl.closureCode != nil {
		r.jitRegHot(cl, c)
		r.jitRegStressSample()
	}
}

// jitRegStressSample makes the next call a sample too where the runtime
// compiles at a function's first call (QJS_JIT_STRESS's threshold,
// SetJITWarmUp's 1), so that every call is.
func (r *Runtime) jitRegStressSample() {
	if r.jitCallThreshold <= 1 {
		r.interruptCounter = 1
	}
}

// jitRegCallSample is a sample of a call at the site whose cache is e
// (rhCallW): its callee's hotness, and the site sent to regCallMiss once
// the callee has code, where regEnterW enters it.
func (r *Runtime) jitRegCallSample(e *regCallCache) {
	if !r.jitEnabled || e.cl == nil || e.rc == nil || e.cl.closureCode == nil {
		return
	}
	if r.jitRegHot(e.cl, e.rc) != nil {
		e.kind |= regCallJIT
	}
	r.jitRegStressSample()
}

// jitRegHot counts a sample of cl's function, its register code c, and
// returns its code: compiled at the jitRegSamples-th sample, nil before
// and where it is not compiled.
func (r *Runtime) jitRegHot(cl *closure, c *regCode) *jitRegEntry {
	cc := cl.closureCode
	if cc.jitRegRefused {
		return nil
	}
	if e := cc.jitReg; e != nil {
		if !e.invalid {
			return e
		}
		cc.jitReg = nil
	}
	if cc.jitRegHot++; cc.jitRegHot < jitRegSamples && r.jitCallThreshold > 1 {
		return nil
	}
	cc.jitRegHot = 0
	if !jitRegTakes(cl, c) {
		cc.jitRegRefused = true
		return nil
	}
	e := r.jitRegFor(cl, c)
	switch {
	case e == nil:
		// No room now.
		return nil
	case e.code == nil:
		cc.jitRegRefused = true
		return nil
	}
	cc.jitReg = e
	return e
}

// jitRegInvalidate marks e, cl's function's code, invalid after a
// deoptimization: the function is sampled anew, and compiled again from
// what the interpreter learns meanwhile, up to jitRegDeoptLimit times.
func (r *Runtime) jitRegInvalidate(cl *closure, e *jitRegEntry) {
	e.invalid = true
	cc := cl.closureCode
	if cc == nil {
		return
	}
	if cc.jitReg == e {
		cc.jitReg = nil
	}
	cc.jitRegHot = 0
	if cc.jitRegDeopts++; cc.jitRegDeopts >= jitRegDeoptLimit {
		cc.jitRegRefused = true
	}
}

// regNoteFailed notes that a guard of the code of the function whose key
// is key failed at register pc rpc (regFailed).
func (s *jitState) regNoteFailed(key jitKey, rpc int) {
	if s.regFailed == nil {
		s.regFailed = map[jitKey][]int32{}
	}
	if l := s.regFailed[key]; !slices.Contains(l, int32(rpc)) {
		s.regFailed[key] = append(l, int32(rpc))
	}
}

// jitRegEnter runs f, the frame regEnterW made for a call of cl, its
// register code c and its window at base, in cl's function's native code
// from its start, and in the interpreter where the code leaves it.
func (r *Runtime) jitRegEnter(f *frame, cl *closure, c *regCode, base int) (Value, error) {
	e := cl.closureCode.jitReg
	if e == nil || e.invalid || !r.jitEnabled {
		return r.runReg(f, c, 0)
	}
	return r.jitRegRun(f, c, e, f.base-int(c.locals), 0)
}

// jitRegLoop is the back edge to register pc target's sample, in the frame
// st runs (regBackEdge): its function's hotness, and its code entered at
// the loop's header where it has an entry there. It returns the pc the
// dispatcher goes on at: target where the code is not entered; where the
// code left the frame to it; a return of the code's own where the code
// returned, its register given the value (jitRegEntry's returns); regDone
// where the code threw what none of the frame's handlers takes, the error
// in st.
func (r *Runtime) jitRegLoop(st *regState, target int) int {
	r.backEdges = r.jit.backEdgeBudget()
	f, c := st.f, st.c
	cl := f.cl
	if !r.jitEnabled || cl == nil || cl.closureCode == nil || cl.reg != c {
		return target
	}
	e := r.jitRegHot(cl, c)
	if e == nil || !e.goesOn {
		return target
	}
	if _, ok := e.loopEntry(target); !ok {
		return target
	}
	pc, v, done, err := r.jitRegGoOn(f, c, e, st.wb, target)
	switch {
	case err != nil:
		st.err = err
		return regDone
	case !done:
		return pc
	case v.IsUndefined() && e.retUndef >= 0:
		return e.retUndef
	case e.retPC >= 0:
		st.regs[e.retReg] = v
		return e.retPC
	}
	// Unreachable: code that returns a value has a return of one.
	panic("vm: native code returned where its register code has no return")
}

// findReturns finds the returns of e's register code rc that a frame the
// code returned from goes on at in the interpreter (jitRegLoop): a
// ret_undef, and a ret whose register no closure captures, which the
// value is written to -- the frame ends there, and nothing reads it after.
// Each is -1 where the code has none. goesOn says the interpreter can end
// every frame the code returns from so: the code has such a ret where it
// has a ret at all.
func (e *jitRegEntry) findReturns(rc *bytecode.RegCode) {
	e.retPC, e.retUndef = -1, -1
	rets := false
	for pc, w := range rc.Code {
		switch w.Op() {
		case bytecode.RRetUndef:
			if e.retUndef < 0 {
				e.retUndef = pc
			}
		case bytecode.RRet:
			rets = true
			if a := int(w.A()); e.retPC < 0 && (a >= len(e.p.Captured) || !e.p.Captured[a]) {
				e.retPC, e.retReg = pc, a
			}
		}
	}
	e.goesOn = e.retPC >= 0 || !rets && e.retUndef >= 0
}
