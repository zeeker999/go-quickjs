package vm

import (
	"unsafe"
)

// A tree's loop stepping a generator, with neither the layers of a step
// nor a recover of the resumption's own.
//
// A for...of loop in a tree steps its cursor at every value
// (iterNextOrJump). Over a generator that had been iterNext, stepIter,
// resumeTree, runGeneratorTree and runTreeProtected, each of them
// checking what the one before had, the last setting a recover so that
// an exception the body throws finishes the generator before it goes on
// out. Where the body has no handler and no loop's cursor of its own -- a
// yield*'s it may have, which no exception closes -- nothing in it
// catches an exception or has an iterator to close, and the exception
// goes from the body straight to the loop's: the step runs
// the body's blocks with no recover of its own, as runTreeNested runs a
// callee's, and lets the exception go on to the recover of the tree whose
// loop it is (runTree, runTreeProtected), which unwinds the frames the
// step left -- next's and the generator's, whose captures it closes, as
// finishing the generator does (unwindTreeFrames). What else finishing it
// does is left for the next resumption to find: the cursor is marked done
// before the body runs, so that the loop's unwinding does not close its
// iterator, which threw; and the generator, executing in a tree's step
// (generator.treeRun), is executing while its frame is on the stack and
// finished once it is not (stillRunning). A yield*'s step of its delegate
// in a tree (delegateNext) resumes the delegate so too (resumeTreeBare):
// its exception goes on out of the delegating generator's body, which
// does not close a delegation's cursor.
//
// Its functions are variables, each assigned a closure in init, as the
// tree tier's are (see ztree_zyield.go). The file sorts before every other
// file of the tier with an init, so its closures, made by the init that
// runs first of theirs, are laid out after all of theirs, and move none.

var (

	// stillRunning is the TypeError a resumption of g, executing, throws,
	// or nil where g is a generator a tree's step resumed whose body threw
	// past it (see above): its frame is not on the stack. It finishes such
	// a generator, as finishResume would have, but for its captures, which
	// were closed with its frame.
	stillRunning func(r *Runtime, g *generator) error

	// genAlloc makes a generator's object and its generator with room for
	// n arguments and locals, in the one allocation where there are few,
	// as kind says: an async function's with its promise, and, for
	// genLoopKind, with room for the cursor of the first loop over it
	// (genCursor) -- a generator of a function whose generators loops
	// step is made with its loop's cursor, in one allocation where there
	// had been two. The generator's room is set where it has it.
	genAlloc func(n int, kind int) (*generatorObject, []Value)

	// genCursor is the cursor of a loop over o, a generator whose next,
	// the realm's, is next: the one in its allocation, where it has room
	// for one no loop has taken, and otherwise one of its own -- which
	// teaches its function to make its generators with the room, unless
	// it is an async function's, which no loop steps.
	genCursor func(r *Runtime, o *Object, next *Object) Value
)

// What tgen.plain says.
const (
	genPlainYes = 1 + iota
	genPlainNo
)

func init() {
	genAlloc = func(n int, kind int) (*generatorObject, []Value) {
		switch kind {
		case genLoopKind:
			var gl *generatorLoop
			var vals []Value
			switch {
			case n <= 2:
				o := &generatorLoopVals[[2]Value]{}
				gl, vals = &o.generatorLoop, o.vals[:n]
			case n <= 4:
				o := &generatorLoopVals[[4]Value]{}
				gl, vals = &o.generatorLoop, o.vals[:n]
			case n <= 8:
				o := &generatorLoopVals[[8]Value]{}
				gl, vals = &o.generatorLoop, o.vals[:n]
			default:
				gl, vals = &generatorLoop{}, make([]Value, n)
			}
			gl.g.room = true
			return &gl.generatorObject, vals
		case genAsyncKind:
			switch {
			case n <= 2:
				o := &asyncVals[[2]Value]{}
				return &o.generatorObject, o.vals[:n]
			case n <= 4:
				o := &asyncVals[[4]Value]{}
				return &o.generatorObject, o.vals[:n]
			case n <= 8:
				o := &asyncVals[[8]Value]{}
				return &o.generatorObject, o.vals[:n]
			}
			return &(&asyncObject{}).generatorObject, make([]Value, n)
		}
		switch {
		case n <= 2:
			o := &generatorVals[[2]Value]{}
			return &o.generatorObject, o.vals[:n]
		case n <= 4:
			o := &generatorVals[[4]Value]{}
			return &o.generatorObject, o.vals[:n]
		case n <= 8:
			o := &generatorVals[[8]Value]{}
			return &o.generatorObject, o.vals[:n]
		}
		return &generatorObject{}, make([]Value, n)
	}

	genCursor = func(r *Runtime, o *Object, next *Object) Value {
		g := genOf(o)
		if g.room {
			g.room = false
			io := &(*generatorLoop)(unsafe.Pointer(o)).cur
			*io = iterObject{Object: Object{class: ClassCursor}, st: iterState{iter: Obj(o), next: Obj(next)}}
			return Obj(&io.Object)
		}
		if !g.async {
			g.cl.genLoops = true
		}
		return r.newIterObject(iterState{iter: Obj(o), next: Obj(next)})
	}

	stillRunning = func(r *Runtime, g *generator) error {
		if g.treeRun {
			locals := unsafe.SliceData(g.vals[g.nargs:])
			live := false
			for d := r.frameDepth - 1; d >= 0 && !live; d-- {
				f := r.frameAt(d)
				live = f.cl == g.cl && unsafe.SliceData(f.locals) == locals
			}
			if !live {
				g.state, g.treeRun, g.saved = genCompleted, false, nil
				return nil
			}
		}
		return r.throwTypeError("the generator is already running")
	}
}
