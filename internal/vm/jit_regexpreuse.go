//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
)

// Native code's calls of exec at a site whose array may be made again in
// the memory of its last (regexpreuse.go) tell exec's leaf so, as the
// interpreter's do (reuseCallRest): a leaf's call (ssa.GoLeaf) of the
// realm's exec by its description (abi.GoCallState's Reuse), the site the
// register compiler marked (bytecode.RMarkReuse), of the function's own
// code -- not an inlined callee's, whose locals are not its frame's own
// (GoCallState's Inline). What native code holds of a frame's locals its
// slots cannot say, so a native frame's call reads none: it makes the
// array again only where the frame made the last itself -- the table's
// entry held in this frame's slot of the call's local, which the call's
// result is about to replace -- and the pieces' Strings only where nothing
// reads the local they are kept in past the call before it is written,
// which the code's liveness says at its compile (ExecReuse). An entry a
// native frame made is held, for any call of another frame, whatever the
// slot holds (execReuseEntry's native).
//
// The functions are generic, each instantiated with struct{} alone.

// ExecReuse is ssa.ExecReuseFeedback's: the site of the call at the
// program's instruction pc of the leaf at fn, the realm's exec, where the
// register compiler marked its array to be made again (bytecode.RMarkReuse),
// its low bit set where the pieces' local, if any, is read by nothing past
// the call before it is written, nor captured; 0 for any other.
func (f *jitRegFeedback) ExecReuse(pc int, fn uintptr) uintptr {
	fb := f.fb
	r := fb.r
	if fb.e == nil || fb.cl == nil || pc >= len(f.p.Origin) || r.regexpExecFn == nil || uintptr(unsafe.Pointer(r.regexpExecFn)) != fn {
		return 0
	}
	c := fb.cl.reg
	if c == nil || c.rc != f.rc {
		return 0
	}
	rpc := int(f.p.Origin[pc])
	if rpc >= len(c.rc.Marks) || c.rc.Marks[rpc]&bytecode.RMarkReuse == 0 || regAt(c.code, rpc).Op() != bytecode.RCallMW {
		return 0
	}
	site := regReuseAt(c, rpc)
	if site == nil {
		return 0
	}
	keep := false
	for _, x := range fb.e.reuses {
		keep = keep || x == site
	}
	if !keep {
		fb.e.reuses = append(fb.e.reuses, site)
	}
	a := uintptr(unsafe.Pointer(site))
	if site.strs && (site.s < 0 || !f.p.IsCaptured(site.s) && !f.p.LiveAt(pc+1, site.s)) {
		a |= 1
	}
	return a
}

// jitReuseForm has form, the leaf's call of exec st describes, tell the
// leaf of the call's site (abi.GoCallState's Reuse) where the call is of
// the function's own code.
func jitReuseForm[_ any](form *jitLeafForm, st *abi.GoCallState) {
	if st.Reuse == 0 || st.Inline != 0 {
		return
	}
	// The address as the description holds it, less the tag of its low
	// bit, read through the field as Leaf is (jitLeafFormOf).
	p := *(*unsafe.Pointer)(unsafe.Pointer(&st.Reuse))
	form.reuse = (*execReuseSite)(unsafe.Add(p, -int(st.Reuse&1)))
	form.reuseStrs = st.Reuse&1 != 0
}

// jitReuseTell tells exec's leaf, just before form's call of it from the
// native frame whose locals start at locals, of the call, which the leaf
// takes as it begins; untold by jitReuseUntell after.
func jitReuseTell[_ any](r *Runtime, form *jitLeafForm, this Value, locals unsafe.Pointer) {
	if r.regexpExecDead == nil || !this.IsObject() || r.meter != nil || r.debug != nil || locals == nil {
		return
	}
	site := form.reuse
	n := max(site.l, site.s) + 1
	execReuseLeaf[struct{}](r).reuse = execReuseCall{site: site, recv: this.Object(), locals: unsafe.Slice((*Value)(locals), n), native: true, strs: form.reuseStrs}
}

// jitReuseUntell is reuseCallRest's untelling, for a call the leaf did not
// take.
func jitReuseUntell[_ any](r *Runtime) {
	if r.regexpExecDead != nil {
		if l := execReuseLeaf[struct{}](r); l.reuse.site != nil {
			l.reuse = execReuseCall{}
		}
	}
}
