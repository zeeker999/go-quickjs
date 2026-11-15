package vm

import "github.com/go-quickjs/go-quickjs/internal/bytecode"

// yield* and for await in the tree tier.
//
// Both are loops the compiler writes out as instructions around a cursor in
// a stack slot, as for...of is (see ztree_zof.go), each turn of which
// suspends the frame at least once.
//
// yield* starts a cursor over its operand, and then, from a stack of the
// cursor, the value sent in and the kind of resumption, iter_resume calls
// the delegate's next, throw or return with the value, an async generator
// awaits what that returned, iter_unpack_delegate reads the result, and
// yield_star hands it out and suspends: a suspension that ends a block as
// yield's does (see suspendEnd), whose resumption pushes the kind of
// resumption above the value sent in rather than acting on it
// (runDelegating). The jump back to iter_resume follows it. Each of the
// three is a node that does what the interpreter's case does, calling the
// helper it calls, and the two that can finish the delegation end a block,
// going on to the code after it, which decides what the delegate's last
// value means, or on round the loop.
//
// for await starts a cursor as for...of does, and each turn async_iter_next
// asks for a promise of the next result, await suspends until it settles,
// and iter_result_or_jump ends a block, going on with the value or to
// where the loop ends. A loop left abruptly closes its cursor as a for...of
// loop's is closed.
//
// Its functions are variables, each assigned a closure in init, rather
// than functions or methods: Go lays out all of a package's functions
// before any of its closures, so code added here as functions would sit
// between the interpreter's hot functions and the tree nodes' hot
// closures, and move one against the other, where a closure is laid out
// after every closure already there (see AGENTS.md).
var (
	// treeBuildsDelegate reports whether the tier builds op, one of
	// yield*'s or for await's instructions other than the suspensions, or
	// one treeBuildsUsing names.
	treeBuildsDelegate func(op bytecode.Op) bool

	// asyncIterClose is async_iter_close, for both tiers: with mode 0, the
	// call of the return method of the iterator of v, a for await's cursor,
	// which reports what it returned, or false where an asynchronous
	// iterator has none; with mode 1, the check of the result awaited, v;
	// with mode 2, v marked done, unclosed.
	asyncIterClose func(r *Runtime, v Value, mode uint32) (Value, bool, error)
)

func init() {
	treeBuildsDelegate = func(op bytecode.Op) bool {
		switch op {
		case bytecode.OpForAwaitOfStart, bytecode.OpAsyncIterNext, bytecode.OpAsyncIterClose,
			bytecode.OpIterResume, bytecode.OpIterUnpackDelegate, bytecode.OpIterResultOrJump:
			return true
		}
		return treeBuildsUsing(op)
	}

	asyncIterClose = func(r *Runtime, v Value, mode uint32) (Value, bool, error) {
		switch mode {
		case 1:
			if !v.IsObject() {
				return Undefined, false, r.throwTypeError("the iterator's return method must return an object")
			}
			return Undefined, false, nil
		case 2:
			if st := cursorOf(v); st != nil {
				st.done = true
			}
			return Undefined, false, nil
		}
		st := cursorOf(v)
		if st == nil || st.done {
			return Undefined, false, nil
		}
		st.done = true
		if st.asyncIter {
			ret, err := r.getValueProp(st.iter, atomReturn)
			if err != nil {
				return Undefined, false, err
			}
			if ret.IsNullish() {
				return Undefined, false, nil
			}
			if !isCallable(ret) {
				return Undefined, false, r.throwTypeError("the iterator's return method is not callable")
			}
			res, err := r.call(ret, st.iter, nil)
			if err != nil {
				return Undefined, false, err
			}
			return res, true, nil
		}
		// A synchronous iterator is driven through the spec's wrapper of it,
		// whose return method always answers: with what the iterator's own
		// return gives, its value awaited, or, where it has none, a result
		// saying it is done.
		if st.arr == nil {
			ret, err := r.getValueProp(st.iter, atomReturn)
			if err != nil {
				return Undefined, false, err
			}
			if !ret.IsNullish() {
				if !isCallable(ret) {
					return Undefined, false, r.throwTypeError("the iterator's return method is not callable")
				}
				res, err := r.call(ret, st.iter, nil)
				if err != nil {
					return Undefined, false, err
				}
				return r.asyncResult(st, res), true, nil
			}
		}
		return Obj(r.iterResult(Undefined, true)), true, nil
	}
}
