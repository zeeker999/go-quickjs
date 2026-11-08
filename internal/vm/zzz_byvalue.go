package vm

import "github.com/go-quickjs/go-quickjs/internal/bytecode"

// A closure may take by value what its maker never assigns once the
// closure is made (bytecode.UpvalueDesc's ByValue): a const initialized
// by then, a strict function's parameter nothing stores to. It copies the
// value into a cell of its own, closed from the start, which no frame
// keeps open and closes as it returns: Havlak's and DeltaBlue's callbacks,
// made at every call of their maker to read its parameters and consts,
// had each been a cell besides, opened on the frame and closed with it.
// It is in the package's last files, out of the way of the closures made
// otherwise (newClosureIn), which are made as they were.

// The variants whose captures are all taken by value carry their cells in
// the same allocation too.
type scriptFuncObject1v struct {
	scriptFuncObject
	captured [1]*upvalue
	own      [1]upvalue
}

type scriptFuncObject2v struct {
	scriptFuncObject
	captured [2]*upvalue
	own      [2]upvalue
}

type arrowFuncObject1v struct {
	arrowFuncObject
	captured [1]*upvalue
	own      [1]upvalue
}

type arrowFuncObject2v struct {
	arrowFuncObject
	captured [2]*upvalue
	own      [2]upvalue
}

type arrowFuncObject3v struct {
	arrowFuncObject
	captured [3]*upvalue
	own      [3]upvalue
}

// newOwnFuncObject is newScriptFuncObject for a function n of whose
// captures, all of them, are taken by value, with their cells in the same
// allocation; or nil where there is no such variant.
func newOwnFuncObject[_ any](proto *Object, class Class, n int, arrow bool) (*Object, *funcData, *closure, []*upvalue, *funcExtra, []upvalue) {
	var fo *scriptFuncObject
	var captured []*upvalue
	var extra *funcExtra
	var cells []upvalue
	switch {
	case arrow && n == 1:
		x := &arrowFuncObject1v{}
		fo, extra, captured, cells = &x.scriptFuncObject, &x.extra, x.captured[:], x.own[:]
	case arrow && n == 2:
		x := &arrowFuncObject2v{}
		fo, extra, captured, cells = &x.scriptFuncObject, &x.extra, x.captured[:], x.own[:]
	case arrow && n == 3:
		x := &arrowFuncObject3v{}
		fo, extra, captured, cells = &x.scriptFuncObject, &x.extra, x.captured[:], x.own[:]
	case !arrow && n == 1:
		x := &scriptFuncObject1v{}
		fo, captured, cells = &x.scriptFuncObject, x.captured[:], x.own[:]
	case !arrow && n == 2:
		x := &scriptFuncObject2v{}
		fo, captured, cells = &x.scriptFuncObject, x.captured[:], x.own[:]
	default:
		return nil, nil, nil, nil, nil, nil
	}
	fo.Object = Object{shape: rootFor(class, proto), class: class, flags: objExtensible}
	fo.embedExt(&fo.side)
	fo.side.data = &fo.fn
	return &fo.Object, &fo.fn, &fo.cl, captured, extra, cells
}

// newClosureByValue is newClosureIn's object, function data, closure and
// funcExtra for the template tmpl, own of whose captures are by value, made
// in f, whose captured locals are on its slots. Where they are all by value
// and few, their cells are in the object's allocation; else they are in the
// room the bindings it is the first to share are given, with them.
func newClosureByValue[_ any](r *Runtime, f *frame, tmpl *closure, arrow bool, own int) (*Object, *funcData, *closure, *funcExtra) {
	descs := tmpl.fn.Upvalues
	proto := r.funcProtoFor(tmpl.fn)
	var o *Object
	var fd *funcData
	var child *closure
	var upvalues []*upvalue
	var extra *funcExtra
	var cells []upvalue
	if own == len(descs) {
		o, fd, child, upvalues, extra, cells = newOwnFuncObject[struct{}](proto, ClassFunction, own, arrow)
	}
	if o == nil {
		o, fd, child, upvalues, extra = newScriptFuncObject(proto, ClassFunction, len(descs), arrow)
	}
	*child = closure{
		fn: tmpl.fn, closureCode: tmpl.closureCode,
		realm: r.Realm, env: f.cl.env, upvalues: upvalues,
	}
	var room []upvalue
	for i, desc := range descs {
		if !desc.FromParent {
			upvalues[i] = f.cl.upvalues[desc.Index]
			continue
		}
		var u *upvalue
		switch {
		case desc.ByValue && len(cells) != 0:
			u, cells = &cells[0], cells[1:]
		case desc.ByValue:
			if len(room) == 0 {
				room = make([]upvalue, cellsWanted[struct{}](f, descs[i:]))
			}
			u, room = &room[0], room[1:]
		default:
			if u = openUpvalue(f, int(desc.Index)); u == nil {
				if len(room) == 0 {
					room = make([]upvalue, cellsWanted[struct{}](f, descs[i:]))
				}
				u, room = &room[0], room[1:]
				u.slot = &f.locals[desc.Index]
				f.openUpvalues = append(f.openUpvalues, u)
			}
			upvalues[i] = u
			continue
		}
		u.closed = f.locals[desc.Index]
		u.slot = &u.closed
		upvalues[i] = u
	}
	return o, fd, child, extra
}

// cellsWanted is uncaptured for a closure that takes some of its captures
// by value: the bindings no upvalue is open on yet, and those it copies,
// each of which has a cell of its own.
func cellsWanted[_ any](f *frame, descs []bytecode.UpvalueDesc) int {
	n := 0
	for _, desc := range descs {
		if desc.FromParent && (desc.ByValue || openUpvalue(f, int(desc.Index)) == nil) {
			n++
		}
	}
	return n
}
