//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"slices"
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
)

// Register callees inlined (jitRegFeedback's Inline, ssa's InlineSite's
// Window): an exit inside one -- or inside one inlined in it -- writes the
// caller's state at the call, a host exit there, and each inlined callee's
// window past the caller's slots, saying in the contexts after the
// caller's where (abi.LiveInline): its closure, its window's first slot
// in the VM's stack (Base), the register pc it left at and how. Go makes
// each callee's frame there, as regEnterW would have -- the outermost
// first -- finishes the innermost from its exit in the interpreter, as a
// frame's code goes on from one (jitRegLeft), each other from its call
// with what its callee answered (regFinishCall), and hands the outermost's
// answer to the caller's call (regPending), as a native callee's.

// jitRegInlineLevel is an inlined callee's frame an exit wrote: its
// closure, its window at wb, the register pc it left at and how.
type jitRegInlineLevel struct {
	cl   *closure
	wb   int
	pc   int
	kind uint64
	site int64
	// construct marks a construction's callee (its caller's new_w), this
	// its receiver, its window's register 1, which every state inside it
	// holds: the construction's result where the callee returns no object.
	construct bool
	this      Value
	// threw marks the innermost level whose operation Go made at its host
	// exit threw (jitRegInlineLeft).
	threw bool
}

// jitRegInlined finishes the inlined callees whose frames an exit of the
// code of owner, the entry of the function frame f runs, run in context
// idx, wrote in the contexts past it (abi.LiveInline), and returns what the
// outermost answered, or false where there are none. at is the register pc
// of the owner's call the outermost is inlined at, wb the owner's window:
// the objects and closures its code did not make, which its frame's
// registers and the callees' windows may name, are made first, once for
// both (jitRegMaterializeSpans). A callee that is a closure the owner did
// not make (ssa's calledClosures), named by its template, is the closure
// made in its place, which its caller's call's window holds.
func (r *Runtime) jitRegInlined(idx int, f *frame, owner *jitRegEntry, at, wb int) (Value, error, bool) {
	cl := f.cl
	s := r.jit
	if idx+1 >= len(s.ssaCtxs) || s.ssaCtxs[idx+1].Live != abi.LiveInline {
		return Undefined, nil, false
	}
	var buf [4]jitRegInlineLevel
	levels := buf[:0]
	for i := idx + 1; i < len(s.ssaCtxs) && s.ssaCtxs[i].Live == abi.LiveInline; i++ {
		c := s.ssaCtxs[i]
		var in *closure
		if k := owner.keep; k != nil {
			for _, x := range k.inlined {
				if uintptr(unsafe.Pointer(x)) == uintptr(c.InlineClosure) {
					in = x
				}
			}
		}
		if in != nil && in.reg == nil {
			// A template of a closure the owner makes (madeInline), which
			// no call has run yet where the owner was compiled before the
			// closure's first call, as at QJS_JIT_STRESS's threshold: its
			// register code, as a call of it gets it.
			r.regOf(in)
		}
		if in == nil || in.reg == nil {
			panic("vm: an inlined register callee's closure is not its caller's")
		}
		levels = append(levels, jitRegInlineLevel{cl: in, wb: int(c.Base), pc: int(c.ExitPC), kind: c.ExitKind, site: int64(c.ExitSite)})
		c.Live, c.ReturnTo = 0, 0
	}
	n := len(levels)
	if owner.frame != 0 || owner.keep != nil && len(owner.keep.made) != 0 {
		spans := make([][2]int, 0, n)
		for _, x := range levels {
			spans = append(spans, [2]int{x.wb, x.wb + int(x.cl.reg.numRegs)})
		}
		r.jitRegMaterializeSpans(f, owner, wb, spans)
	}
	for k := range levels {
		// The closure its caller's call called, the function in the call's
		// window: the one made in place of a template.
		caller, cwb, cpc := cl.reg, wb, at
		if k > 0 {
			caller, cwb, cpc = levels[k-1].cl.reg, levels[k-1].wb, levels[k-1].pc
		}
		if cpc >= len(caller.rc.Code) {
			continue
		}
		called := Undefined
		switch w := regAt(caller.code, cpc); w.Op() {
		case bytecode.RCallW, bytecode.RCallMW, bytecode.RNewW:
			called = r.stack[cwb+int(w.A())]
		case bytecode.RApplyArgs:
			// f of f.apply(x, arguments), inlined with the arguments
			// (ssa.InlineSite's Forward).
			called = r.stack[cwb+int(w.B())]
		}
		if loop := s.ssaCtxs[idx].Loop; k == 0 && loop != 0 {
			// A loop's callback inlined in it: the loop's callback slot
			// (jitRegLoopFinish's), which Go goes on calling.
			called = r.stack[jitRegSlotIndex(owner.p, wb, owner.p.Locals+int(loop&0xffffffff)-1)+1]
		}
		if called.IsObject() {
			if fd := called.Object().fn(); fd != nil && fd.closure != nil && fd.closure.fn == levels[k].cl.fn && fd.closure.reg != nil {
				levels[k].cl = fd.closure
			}
		}
		if kp := owner.keep; kp != nil && slices.Contains(kp.made, levels[k].cl) {
			panic("vm: an inlined closure the code did not make is not where its call is")
		}
	}
	s.unwound += uint64(n)
	saved := r.stackTop
	var fbuf [4]*frame
	frames := fbuf[:0]
	for k := range levels {
		x := &levels[k]
		rc := x.cl.reg
		if top := x.wb + int(rc.numRegs); top > r.stackTop {
			r.stackTop = top
			r.stackHigh = max(r.stackHigh, top)
		}
		pc := regSrc(rc, x.pc)
		if k < n-1 {
			// Past the call it makes, as the interpreter's frame is during
			// one, which a stack trace shows.
			pc++
		}
		// A construction's, where its caller's call is new_w: the owner's
		// at the call the outermost is inlined at, or the level's below.
		caller, at := cl.reg, at
		if k > 0 {
			caller, at = levels[k-1].cl.reg, levels[k-1].pc
		}
		if at < len(caller.rc.Code) && regAt(caller.code, at).Op() == bytecode.RNewW {
			x.construct, x.this = true, r.stack[x.wb+1]
		}
		lv := jitRegLevel{cl: x.cl, c: rc, wb: x.wb, construct: x.construct, inline: true}
		if rc.args {
			// Its call's arguments, which it keeps (ssa.InlineSite's
			// ReadsArguments and Arguments): its caller's call's window's,
			// or those its caller hands on.
			cwb, cargs := wb, f.args
			if k > 0 {
				cwb, cargs = levels[k-1].wb, frames[k-1].args
			}
			lv.args = r.jitRegInlineArgs(caller, cwb, at, cargs)
		}
		frames = append(frames, r.jitRegPushLevel(&lv, pc))
	}
	var v Value
	var err error
	for k := n - 1; k >= 0; k-- {
		x, f := &levels[k], frames[k]
		if k == n-1 {
			// Told before Go runs the instruction, which may fill the pool
			// it finds empty.
			atSite := r.jitRegLeftAtSite(x)
			var learned bool
			v, err, learned = r.jitRegInlineLeft(f, x, owner)
			l := s.regLearnedOf(r.jitRegKeyOf(cl))
			switch {
			case x.kind == abi.ExitDeopt:
				// A guard's failure: the owner's code invalid from here,
				// as at one of its own (jitRegLeft), the stack form's
				// code's at one inside a callee it inlined.
				r.jitRegInvalidate(cl, owner)
			case learned:
				// The callee's feedback learned what the owner's code was
				// not compiled for: compiled again for it (jitRegReopt).
				r.jitRegReopt(cl, owner)
			case atSite || x.threw || owner.life.reopt:
				// Left at a site of the callee's -- or by code to be
				// compiled again, which may be what it leaves for -- not
				// for being inlined, as the stack form's unwinding counts
				// none such; or left to throw, which the callee's own code,
				// called natively, leaves native code for just the same: a
				// read of null's property, at a site the callee's own code
				// made megamorphic deoptimizing for it (jitDeoptSeenOp),
				// had had constructs' Vs called natively after 16 throws.
			default:
				// Left with nothing to learn: counted against inlining the
				// callee at the call, by the stack form's own rule
				// (jitInlineLeft, jitInlineExits): past it the call is a
				// native call in the code compiled next.
				r.jitInlineLeft(l, at)
				if l.inlineReopt {
					l.inlineReopt = false
					r.jitRegReopt(cl, owner)
				}
			}
		} else {
			var next int
			next, _, err = r.regFinishCall(f, x.cl.reg, x.wb, x.pc, v, err)
			if err == nil {
				v, err = r.runReg(f, x.cl.reg, next)
			}
		}
		if err == errRegRestart {
			// Its frame given to a callee in place by a call in tail
			// position (regRestartRun).
			v, err = r.regRestartRun(f)
		}
		r.jitRegPopLevel(f, x.wb, false)
		if err == errTailCall {
			v, err = r.regTailRest()
		}
		if x.construct && err == nil && !v.IsObject() {
			// A construction's result is its receiver, unless an object.
			v = x.this
		}
	}
	r.stackTop = saved
	return v, err, true
}

// jitRegLeftAtSite reports whether the innermost inlined level x left,
// at a host exit, for a site of its own rather than for being inlined, as
// the stack form's unwinding tells (jitShapeMiss, jitPoolEmptied): a
// property's read or write whose site is not megamorphic -- a shape its
// cases lack, a receiver whose read throws -- or a construction whose
// pool ran out, which Go fills again. Neither counts against inlining the
// callee (jitInlineLeft): a callee reading a property of the null its
// caller passes now and then, a TypeError the caller catches, had been
// called natively after 16 of them.
func (r *Runtime) jitRegLeftAtSite(x *jitRegInlineLevel) bool {
	rc := x.cl.reg.rc
	if x.kind != abi.ExitHost || x.pc < 0 || x.pc >= len(rc.Code) {
		return false
	}
	w := rc.Code[x.pc]
	l := r.jit.regLearnedOf(r.jitRegKeyOf(x.cl))
	switch w.Op() {
	case bytecode.RGetProp, bytecode.RGetMethod, bytecode.RGetMethodW, bytecode.RSetProp, bytecode.RAddProp, bytecode.RAddPropI:
		if site := int(w.C()); site < len(x.cl.ic) && x.cl.ic[site].fills >= maxCacheFills {
			return false
		}
		for i := range l.poly {
			if p := &l.poly[i]; int(p.pc) == x.pc && p.mega {
				return false
			}
		}
		return true
	case bytecode.RNewW:
		for _, t := range jitTargets(l) {
			if int(t.pc) == x.pc && t.pool != nil && t.pool.Count == 0 {
				return true
			}
		}
	}
	return false
}

// jitRegInlineLeft finishes f, the innermost inlined callee's frame, from
// where it left (x), in the interpreter, learning from it as from an exit
// of its own code (jitRegLeft), what it learns its function's own: a
// guard's failure; the cases of a property site Go ran a host operation
// at; a loop's poll. learned says it learned what the code inlining it,
// owner, is to be compiled again for.
func (r *Runtime) jitRegInlineLeft(f *frame, x *jitRegInlineLevel, owner *jitRegEntry) (v Value, err error, learned bool) {
	s := r.jit
	c := x.cl.reg
	pc := x.pc
	switch x.kind {
	case abi.ExitDeopt:
		s.guards++
		if x.site < 0 {
			r.jitRegEntrySeen(f, pc)
		} else {
			r.jitRegDeoptSeen(f, c, owner, x.wb, pc)
		}
		learned = true
	case abi.ExitHost:
		s.hosts++
		f.pc = regSrc(c, pc) + 1
		s.noteRegHost(c, pc)
		l := s.regLearnedOf(r.jitRegKeyOf(x.cl))
		before, failed := l.polyReopt, len(l.failed)
		seen := r.jitRegHostSeen(f, c, x.wb, pc)
		next, ran, err := r.regWhole(f, c, x.wb, pc)
		if !ran {
			break
		}
		if seen.write && err == nil {
			r.jitRegWriteSeen(f, c, owner, pc, seen)
		}
		// The site met what the code inlining it was not compiled for --
		// its own code's to be compiled again for too (polyReopt) -- or
		// a value not a number at a read the code takes for a number's
		// (jitNumberMissed), which a guard of the callee's own code
		// deoptimizes for but one inlined leaves to Go at, the read
		// generic once compiled again: str's in callsGo, of a string, had
		// left at each. A change of its caches alone is not: the stack
		// form's code is compiled again for none (its mark only delays a
		// compile, jitLearnedMark), and counted so, each exit had the
		// code inlining it compiled again for what the interpreter's run
		// of the callee filled.
		learned = !before && l.polyReopt || len(l.failed) != failed || seen.elem
		if err != nil {
			x.threw = true
			return Undefined, err, learned
		}
		if r.stopped != nil {
			return Undefined, r.stopped, learned
		}
		pc = next
	case abi.ExitPoll:
		s.budgets++
		r.backEdges = s.backEdgeBudget()
		if err := r.checkInterruptPoll(); err != nil {
			return Undefined, err, false
		}
	}
	s.interpreted++
	v, err = r.runReg(f, c, pc)
	return v, err, learned
}
