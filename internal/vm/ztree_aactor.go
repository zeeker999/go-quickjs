package vm

import (
	"math"
)

// Constructions on the tree tier's path. A construction ran its
// constructor on the tree planTreeCall gave it, callTreeWith's, only where
// a call could run it too, so that a class's constructor, which a call
// refuses (DirectCall), was run by runFD at every new -- its frame set up
// for anything, a with's scopes and an eval's bindings asked after -- and
// a pure body's, whose plan waits for pureCall to give up on calls that a
// construction never makes, was never planned at all. A derived class's
// was worse off: runFD gave its frame a this binding of the heap, its
// super() reached the parent through callObject and runFD again, and
// copied its arguments for it, three allocations and two frames set up for
// anything for each new of a subclass -- each of Are We Fast Yet's
// DeltaBlue constraints and Json values, and a class hierarchy's every
// object in real code.
//
// planTreeNew decides once what a construction may run on, as planTreeCall
// decides for calls, and constructTree makes the constructions it allows:
// a base class's on callTreeWith; a derived class's on a frame of its own
// making, whose this binding is the runtime's cell for the frame's depth
// (thisCells) where the body makes no closure and no direct eval, the two
// things that can keep a binding past its frame. super() calls the parent
// the same way. What each construction then does is what runFD would have
// done for it, in the same order.
//
// The code is closures assigned in this file's init, the package's first
// init with closures, which are laid out after every other closure (see
// ztree_aaeq.go): the tree tier's own closures do not move.

var (
	// constructRun is constructWithTarget's construction of a compiled
	// constructor, with this the object made for it: runFD's.
	constructRun func(r *Runtime, this Value, args []Value, newTarget Value, o *Object, fd *funcData) (Value, error)

	// superConstruct is super()'s construction of its parent in frame f,
	// with f's object as this and new.target, after the parent's instance
	// initializer: callObject's.
	superConstruct func(r *Runtime, parent *Object, f *frame, args []Value, newTarget Value) (Value, error)

	// constructNativeAt is constructWithTarget's construction by a
	// built-in: a native leaf whose construction is its call
	// (elemNewFirst...) applied without a frame where new.target is the
	// built-in itself, and constructNative otherwise, where it had called
	// constructNative.
	constructNativeAt func(r *Runtime, o *Object, args []Value, newTarget Value, derived bool) (Value, error)

	// superInit is super()'s instance initializer of a base parent, in
	// frame f, run on f's object before the parent's body as runFD's
	// construction of it would -- where super() has not bound this yet: a
	// second super()'s parent makes an object of its own, which
	// superConstruct's construction initializes. superCall calls it where
	// it had called initInstanceElements.
	superInit func(r *Runtime, parent *Object, f *frame) error

	// setArrayCtor records the realm's Array constructor, made by
	// initArrayBuiltins, and declares its native leaf (elemArrayNew), in
	// place of the store initArrayBuiltins had made.
	setArrayCtor func(r *Runtime, ctor *Object)
)

func init() {
	// Array(...) and new Array(...) make an array and run nothing: of its
	// arguments, or of a length, which only throws where it is not one.
	leafChecks[elemArrayNew] = func(_ *Runtime, _ Value, args []Value) bool {
		if len(args) == 1 && args[0].IsNumber() {
			n := args[0].num
			return n >= 0 && n <= math.MaxUint32 && n == math.Trunc(n)
		}
		return true
	}

	superInit = func(r *Runtime, parent *Object, f *frame) error {
		if f.thisRef.init {
			return nil
		}
		// initInstanceElements, written out: a class without fields or
		// private methods has no initializer, the common case.
		fd := parent.fn()
		if fd == nil || fd.fieldInit() == nil {
			return nil
		}
		_, err := r.callObject(fd.fieldInit(), f.this, nil, Undefined)
		return err
	}

	setArrayCtor = func(r *Runtime, ctor *Object) {
		r.proto.arrayCtor = ctor
		setElemOp(r, ctor, elemArrayNew)
	}

	constructNativeAt = func(r *Runtime, o *Object, args []Value, newTarget Value, derived bool) (Value, error) {
		fd := o.fn()
		if op := fd.elemOp; op >= elemNewFirst && !derived && r.meter == nil &&
			(fd.realm == nil || fd.realm == r.Realm) && leafChecks[op](r, Undefined, args) {
			// A leaf whose construction is its call, by new.target itself:
			// applied without a frame, as its call is.
			if v, err := fd.native(r, Undefined, args); err == nil {
				return v, nil
			}
		}
		return r.constructNative(o, args, newTarget, derived)
	}

	constructRun = func(r *Runtime, this Value, args []Value, newTarget Value, o *Object, fd *funcData) (Value, error) {
		return r.runFD(fd.closure, this, args, newTarget, o, fd)
	}

	superConstruct = func(r *Runtime, parent *Object, f *frame, args []Value, newTarget Value) (Value, error) {
		if f.thisRef.init {
			// A second super(): the parent's [[Construct]] with new.target
			// makes an object of its own and runs the parent on it, its
			// fields included (superInit left them to it), and only then is
			// binding this an error (BindThisValue); the object the first
			// super() bound is left as it is.
			if _, err := r.constructWithTarget(Obj(parent), args, newTarget); err != nil {
				return Undefined, err
			}
			return Undefined, r.throwError(errReference, "super() has already been called")
		}
		return r.callObject(parent, f.this, args, newTarget)
	}
}
