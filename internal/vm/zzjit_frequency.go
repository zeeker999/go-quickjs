//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"math"
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/jit/ssa"
)

// Call frequency, as V8's feedback has it: how often a call site calls for
// each invocation of its function, which V8's inliner orders its candidates
// by and cuts them at (ssa.FrequencyFeedback). V8's lower tiers count it,
// not its optimized code: each call bytecode's feedback slot counts the
// calls made there, and the function's feedback vector its invocations,
// from when the vector is allocated. Here the existing tiers count it
// where every call they make with a frame passes once the JIT is on: the
// call path's hook, tryJITFrame, which the interpreter's calls, the tree
// tier's call nodes and constructions, and the calls of built-ins and
// accessors all reach once the callee's frame is made, and the end of a
// pure body's evaluation (pureCallAt), for a call the call path answers
// without one. The callee's frame is the top one, or would be; its
// caller's, the one below, is at the call, its pc past it, as a stack
// trace reads it (jitCountCallAt, jitCountPure). Native code counts nothing:
// its calls through Go, at its exits, and of callees with no code are left
// out (jitState's goCallDepth, nativeCallDepth), and so are its
// invocations.
//
// They are counted while the function's code is to be compiled: while
// its entry warms, before it is compiled, and while its code is
// invalidated, before it is compiled again (jitCounting) -- the existing
// tiers' runs a compile is made from. A function Go does not enter the
// code of, or that has none for good, runs in them without counting.

// jitCallCounts is what the existing tiers counted of a function's calls
// while its code was to be compiled (jitCounting), from its first entry
// on, as V8's feedback vector counts them from when it is allocated: runs,
// the invocations they ran of it, and, for each of its calls and
// constructions -- pcs, their PCs in order, a call in tail position among
// them -- calls, the calls made there of a function in a frame of its own.
// code is how many instructions the function has, which a ghost's counts
// are checked against (jitGhostCounts).
//
// A call the call path answers without a frame -- a leaf's
// (bytecode.LeafKind) other than a pure body's or a forwarding
// constructor's (jitCountPure), a built-in's -- is not counted: such a
// site's frequency is not known (jitCallFrequency). One through
// Function.prototype.call or apply is counted at the site that calls them
// (jitCountCallAt). A call in tail position deep
// in the stack gives its frame up (tailCallDepth), and its callee's frame
// is counted at its caller's caller's call.
type jitCallCounts struct {
	runs  uint32
	code  int
	pcs   []int32
	calls []uint32
	// learned marks the calls native code made that the existing tiers
	// had not (jitCountLearned), by index, or is nil for none.
	learned []bool
}

// jitCountsHalved is where a count is halved, all the function's counts
// with it, so that none wraps and the frequencies stay as they were.
const jitCountsHalved = 1 << 31

// at is the index of the call at pc among c's, or -1.
func (c *jitCallCounts) at(pc int32) int {
	lo, hi := 0, len(c.pcs)
	for lo < hi {
		m := int(uint(lo+hi) >> 1)
		if c.pcs[m] < pc {
			lo = m + 1
		} else {
			hi = m
		}
	}
	if lo < len(c.pcs) && c.pcs[lo] == pc {
		return lo
	}
	return -1
}

// halve halves c's counts.
//
//go:noinline
func (c *jitCallCounts) halve() {
	c.runs >>= 1
	for i := range c.calls {
		c.calls[i] >>= 1
	}
}

// bytes is what c holds.
func (c *jitCallCounts) bytes() int {
	if c == nil {
		return 0
	}
	return int(unsafe.Sizeof(*c)) + jitCapBytes(c.pcs) + jitCapBytes(c.calls) + jitCapBytes(c.learned)
}

// jitCounting reports whether the existing tiers count the calls of e's
// function (jitCallCounts): its entry warms, or its code is invalidated --
// it is to be compiled.
func jitCounting(e *jitEntry) bool {
	return e.calls != nil && (e.warming || e.invalid)
}

// jitCountsCaller reports whether the frame at depth i, a call's caller,
// may be one whose calls are counted (jitCountCallAt): not a frame of a
// template whose calls are not (jitCodeFields' jitCounting), which is what
// almost every call meets once its caller has code, nor a built-in's, but
// Function.prototype.call's and apply's, whose calls count at the call of
// them -- asked first, and inlined where a call counts, so that it costs
// a few loads. A built-in's calls -- a callback's that forEach, map, sort
// or another makes for each element -- are no site native code compiles,
// and jitCountCallAt counts none of them: each had made its call, of every
// element, to find that out.
func (r *Runtime) jitCountsCaller(i int) bool {
	if off := i - r.frameBase; i >= 0 && uint(off) < uint(len(r.cur)) {
		g := &r.cur[off]
		if cl := g.cl; cl != nil {
			return cl.closureCode == nil || cl.jitCounting
		}
		return g.callee == r.callFn || g.callee == r.applyFn
	}
	return true
}

// jitCountPure counts a call the call path answered without the callee's
// frame tryJITFrame sees, at its caller's call site, the top frame's
// (jitCountCallAt): a pure body's (pureCallAt), and a construction with a
// constructor that forwards its arguments to a method of its object
// (leafForward), which calls the method itself. One it does not answer so
// it makes with the frame, which tryJITFrame counts. The caller asks
// first whether the top frame may count it (jitCountsCaller).
func (r *Runtime) jitCountPure() { r.jitCountCallAt(r.frameDepth + 1) }

// jitCountCallAt counts a call whose callee's frame is, or would be, the
// d-th, at its caller's call site, the frame below's (jitCallCounts),
// where the caller is a frame of a function whose calls the existing tiers
// count (jitCounting), of a template whose are (jitCountRun), not a
// built-in's -- and the call not native code's: through Go (goCallDepth),
// at an exit, or of a callee with no code (nativeCallDepth).
func (r *Runtime) jitCountCallAt(d int) {
	s := r.jit
	if s == nil || d < 2 || d == s.goCallDepth || d == s.nativeCallDepth {
		return
	}
	var g *frame
	if off := d - 2 - r.frameBase; uint(off) < uint(len(r.cur)) {
		// The caller's frame in the top's block, as almost always.
		g = &r.cur[off]
	} else {
		g = r.frameAt(d - 2)
	}
	if g.cl == nil && d >= 3 && g.callee != nil && (g.callee == r.callFn || g.callee == r.applyFn) {
		// Function.prototype.call's or apply's frame: the call is counted
		// at the call of call or apply, the frame below's, as V8's call
		// feedback counts it at the site that calls them.
		g = r.frameAt(d - 3)
	}
	if cl := g.cl; cl == nil || cl.closureCode == nil || !cl.jitCounting {
		// A built-in's frame, or one whose template's calls are not
		// counted (jitCountRun).
		return
	}
	e := s.hint(g.cl)
	if e == nil || !jitCounting(e) {
		// Its code compiled since: the template's calls are counted no
		// more, till they are again.
		g.cl.jitCounting = false
		return
	}
	c := e.calls
	if i := c.at(int32(g.pc) - 1); i >= 0 {
		if c.calls[i]++; c.calls[i] >= jitCountsHalved {
			c.halve()
		}
	}
}

// jitCountsEnough is how many of a function's invocations the existing
// tiers count before its frequencies are known (jitCallFrequency): the
// fewest in which a call made at V8's minimum frequency
// (ssa.MinInliningFrequency, 0.05) is made at least once 19 times in 20,
// 59 invocations,
// so that a call counted none of the times is cold. A function compiled
// after fewer -- by on-stack replacement at a long loop, its first
// invocations, or for a native caller -- has its calls ordered as before
// they were counted (ssa's inlineOrder). V8's optimizing compiler reads
// its feedback after its tier-up budget, rarely so few.
var jitCountsEnough = uint32(math.Ceil(math.Log(0.05) / math.Log(1-ssa.MinInliningFrequency)))

// pureCallCounted is pureCall made at a call site of the script's -- a
// tree's call node, callRest -- whose call the JIT counts for its inliner,
// as a call made with a frame is (jitCountPure): here and in callDirect,
// not in pureCallAt or pureCall, so that pureCallAt's code is the same
// with the JIT built in as without, and pureCall small enough to inline
// where callDirect calls it, which it does for a built-in's callbacks too:
// a sort's comparator, whose caller, a built-in's frame, counts nothing,
// had paid a call of pureCall and its count's tests at every comparison,
// a tenth of a typed array's sort.
func (r *Runtime) pureCallCounted(fd *funcData, this Value, args []Value) (Value, bool) {
	v, ok := r.pureCall(fd, this, args)
	if ok && r.jitPureCounted() {
		r.jitCountPureCall()
	}
	return v, ok
}

// jitPureCounted reports, for pureCall, whether the caller's frame, the
// top one, may count the pure body's call it answered: one of a function
// of the script's (jitCountsCaller asks the rest, out of line), or of
// Function.prototype.call or apply, whose calls count at the call of them;
// not another built-in's -- a sort's, a forEach's -- which counts none. It
// is one load and a test, so that pureCall stays small enough to inline
// where callDirect calls it, as it is without the JIT.
func (r *Runtime) jitPureCounted() bool {
	if off := r.frameDepth - 1 - r.frameBase; uint(off) < uint(len(r.cur)) {
		g := &r.cur[off]
		return g.cl != nil || g.callee == r.callFn || g.callee == r.applyFn
	}
	return true
}

// jitCountPureCall is pureCall's count of a pure body's call, where
// jitPureCounted said its caller may count it.
//
//go:noinline
func (r *Runtime) jitCountPureCall() {
	if r.jitOn() && r.jitCountsCaller(r.frameDepth-1) {
		r.jitCountPure()
	}
}
