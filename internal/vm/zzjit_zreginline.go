//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"unsafe"

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
}

// jitRegInlined finishes the inlined callees whose frames an exit of the
// code of owner, cl's function's entry, run in context idx, wrote in the
// contexts past it (abi.LiveInline), and returns what the outermost
// answered, or false where there are none. at is the register pc of the
// owner's call the outermost is inlined at.
func (r *Runtime) jitRegInlined(idx int, cl *closure, owner *jitRegEntry, at int) (Value, error, bool) {
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
		if in == nil || in.reg == nil {
			panic("vm: an inlined register callee's closure is not its caller's")
		}
		levels = append(levels, jitRegInlineLevel{cl: in, wb: int(c.Base), pc: int(c.ExitPC), kind: c.ExitKind, site: int64(c.ExitSite)})
		c.Live, c.ReturnTo = 0, 0
	}
	n := len(levels)
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
		lv := jitRegLevel{cl: x.cl, c: rc, wb: x.wb}
		frames = append(frames, r.jitRegPushLevel(&lv, pc))
	}
	var v Value
	var err error
	for k := n - 1; k >= 0; k-- {
		x, f := &levels[k], frames[k]
		if k == n-1 {
			var learned bool
			v, err, learned = r.jitRegInlineLeft(f, x, owner)
			l := s.regLearnedOf(r.jitKeyOf(cl.fn))
			if !learned {
				// Left with nothing to learn, counted against inlining
				// the callee at the call (jitInlineLeft), as the stack
				// form's code counts it: past jitInlineExits, it is
				// called natively there.
				r.jitInlineLeft(l, at)
			}
			if learned || l.inlineReopt {
				// The owner's code is invalid from here -- its runs, and
				// its native callers, leave it, as for a property site it
				// cannot hold (polyReopt) -- and is compiled again for
				// what was learned.
				l.inlineReopt = false
				r.jitRegInvalidate(cl, owner)
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
		r.jitRegPopLevel(f, x.wb)
		if err == errTailCall {
			v, err = r.regTailRest()
		}
	}
	r.stackTop = saved
	return v, err, true
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
		l := s.regLearnedOf(r.jitKeyOf(x.cl.fn))
		before, mark := l.polyReopt, jitFeedbackMark(x.cl)
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
		// its caches learned it, which the code inlining it reads, as a
		// function's own code is compiled again for its caches' changes
		// (jitFeedbackMark).
		learned = !before && l.polyReopt || jitFeedbackMark(x.cl) != mark
		if err != nil {
			return Undefined, err, learned
		}
		if r.stopped != nil {
			return Undefined, r.stopped, learned
		}
		pc = next
	case abi.ExitPoll:
		s.budgets++
		r.backEdges = s.backEdgeBudget()
		if err := r.checkInterruptNow(); err != nil {
			return Undefined, err, false
		}
	}
	s.interpreted++
	v, err = r.runReg(f, c, pc)
	return v, err, learned
}
