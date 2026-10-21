package vm

import "sync/atomic"

// A call in tail position past tailCallDepth that a tree's frame ended in
// (tailCallNode, treeTail), to a plain function of this realm the tree tier
// runs as a tree, made by tailFromTree without tailTarget's walk: what it
// asks of a callee -- a proxy, a bound function's target, an arrow's this, a
// generator, a class constructor, another realm -- is asked once, here, and
// the call made on the callee's tree. A chain of mutual tail calls between
// functions of the tier, each hop a call in tail position, then settles each
// callee once. Anything else is left in pendingTail as it was, for
// tailFromTree's own loop to make.
//
// The code is a closure assigned in this file's init, whose closures are
// laid out after every other closure of the package's (see
// ztree_zkself.go): this file's init comes first. Grown in zcall_tail.go
// and zcall_tree.go -- tailArgsOf settling the callee as it copied the
// arguments, tailFromTree making the call -- the change moved every tree
// closure by 1 KB: tail_self, which runs none of it, was 13% slower, and
// AWFY 0.7% to 1.6%.

// tailFast makes the call in pendingTail, and the calls in tail position
// it ends in, where each is to a plain function with a tree, and reports
// whether it made the last: its result and error, then. Where it reports
// false, pendingTail holds a call it did not make, which tailFromTree
// makes the ordinary way.
var tailFast func(r *Runtime) (Value, error, bool)

func init() {
	tailFast = func(r *Runtime) (Value, error, bool) {
		for {
			tc := r.pendingTail
			if !tc.callee.IsObject() {
				return Undefined, nil, false
			}
			o := tc.callee.object()
			fd := o.fn()
			if fd == nil || fd.closure == nil || fd.bound || fd.native != nil || fd.arrow || fd.extra != nil ||
				fd.closure.realm != r.Realm {
				return Undefined, nil, false
			}
			fn := fd.closure.fn
			if isGeneratorTemplate(fn) || isClassConstructorKind(fn.Kind) || fn.CoerceThis && !tc.this.IsObject() {
				return Undefined, nil, false
			}
			var t *tree
			switch {
			case fd.treeCall != nil:
				t = fd.treeCall
			case fd.treeRest:
				t = (*tree)(atomic.LoadPointer(&fn.VMCode))
			}
			if t == nil {
				return Undefined, nil, false
			}
			r.pendingTail = tailCall{}
			if err := r.tick(); err != nil {
				return Undefined, err, true
			}
			v, err := r.callTreeWith(o, fd, t, tc.this, tc.args, Undefined, true)
			if err != errTailCall {
				return v, err, true
			}
		}
	}
}
