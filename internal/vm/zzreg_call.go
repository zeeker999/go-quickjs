package vm

import (
	"sync/atomic"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
)

// The register interpreter's calls and slow paths: what its loop does out
// of line, so that the loop stays small enough for Go to inline into it what
// it inlines (see zzreg_run.go).

// regCall calls callee with this and args from the register interpreter. A
// function of this realm that has register code, and that a call runs with
// nothing to establish but its frame, is run here directly: its frame made
// as callTree makes one, its registers run by runReg, with no tree, no
// recover and no other Go call between the two loops. Anything else is
// called as the stack interpreter calls it.
func (r *Runtime) regCall(callee, this Value, args []Value) (Value, error) {
	if !callee.IsObject() {
		return r.callDirect(callee, this, args)
	}
	o := callee.Object()
	fd := o.fn()
	if fd == nil || fd.closure == nil || fd.bound {
		return r.callDirect(callee, this, args)
	}
	cl := fd.closure
	c := cl.reg
	if c == nil {
		c = r.regOf(cl)
	}
	if c == nil || cl.realm != r.Realm {
		return r.callDirect(callee, this, args)
	}
	if fd.extra != nil {
		// An arrow whose surroundings are its this alone; anything else a
		// function carries of its surroundings is runFD's to establish.
		x := fd.extra
		if !fd.arrow || x.lexThisRef != nil || x.rare != nil {
			return r.callDirect(callee, this, args)
		}
		this = x.lexThis
	}
	if c.coerceThis && !this.IsObject() {
		if !this.IsNullish() {
			return r.callDirect(callee, this, args)
		}
		this = r.globalThis
	}
	if err := r.tick(); err != nil {
		return Undefined, err
	}
	if r.frameDepth >= r.maxFrames {
		return Undefined, r.throwRangeError("maximum call stack size exceeded")
	}
	base := r.stackTop
	top := base + int(c.numRegs)
	if top > len(r.stack) {
		return Undefined, r.throwRangeError("maximum call stack size exceeded")
	}
	r.stackTop = top
	if top > r.stackHigh {
		r.stackHigh = top
	}
	lc := int(c.locals)
	locals := r.stack[base : base+lc : base+lc]
	m := min(int(c.params), len(locals), len(args))
	//lint:ignore S1001 a loop, not copy: see callTree
	for i, a := range args[:m] {
		locals[i] = a
	}
	for i := m; i < len(locals); i++ {
		locals[i] = Undefined
	}
	off := r.frameDepth - r.frameBase
	if uint(off) >= uint(len(r.cur)) {
		r.seekFrameBlock()
		off = r.frameDepth - r.frameBase
	}
	r.frameDepth++
	if r.frameDepth > r.frameHigh {
		r.frameHigh = r.frameDepth
	}
	f := &r.cur[off]
	f.cl = cl
	f.locals = locals
	f.base = base + lc
	f.pc = 0
	f.this = this
	f.newTarget = Undefined
	f.callee = o
	f.args = args
	if len(f.openUpvalues) != 0 {
		f.openUpvalues = f.openUpvalues[:0]
	}
	if f.thisRef != nil {
		f.thisRef = nil
	}
	if f.withScopes != nil {
		f.withScopes = nil
	}
	if f.evalVars != nil {
		f.evalVars = nil
	}
	if len(f.handlers) != 0 {
		f.handlers = f.handlers[:0]
	}
	if f.native != "" {
		f.native = ""
	}
	if f.savedSP != 0 {
		f.savedSP = 0
	}
	v, err := r.runReg(f, c)
	r.popFrameOf(f, base)
	return v, err
}

// regOf is the register code of cl's function, built if the function has
// not been run yet, and kept in its template for regCall's next call of it;
// or nil.
func (r *Runtime) regOf(cl *closure) *regCode {
	if !regTier.Load() {
		return nil
	}
	t := (*tree)(atomic.LoadPointer(&cl.fn.VMCode))
	if t == nil {
		t = firstTree(cl.fn)
	}
	cl.reg = t.reg
	return t.reg
}

// regTctx is f's tree context, made enough of for the tree tier's slow
// paths the register interpreter shares.
func regTctx(r *Runtime, f *frame, cl *closure) *tctx {
	c := &f.tc
	c.r, c.f, c.cl = r, f, cl
	c.closureCode = cl.closureCode
	return c
}

// regSlow is the slow path of the instruction at pc, whatever the loop's
// fast path for it did not answer: it sets the frame's pc to the stack
// form's instruction, does what the stack interpreter does, and stores the
// result in its register. A comparison's jump is left to the loop, which is
// given the comparison's result.
//
// It reads the instruction from the code rather than being given it, so that
// the loop keeps nothing of the instruction across the call (see runReg).
func (r *Runtime) regSlow(f *frame, c *regCode, pc int, regs *[1 << 16]Value) (Value, error) {
	w := regAt(c.code, pc)
	f.pc = c.src[pc] + 1
	cl := f.cl
	wb := f.base - int(c.locals)
	var v Value
	var err error
	switch w.Op() {
	case bytecode.RGetGlobal:
		v, err = r.regGetGlobal(f, cl, w)
	case bytecode.RSetGlobal:
		return Undefined, r.regSetGlobal(f, cl, w, regs[w.A()])
	case bytecode.RGetProp:
		v, err = r.regGetProp(cl, c, w, regs[w.B()])
	case bytecode.RGetMethod:
		v, err = r.regGetMethod(cl, c, w, regs[w.B()])
	case bytecode.RSetProp:
		return Undefined, r.regSetProp(cl, c, w, regs[w.A()], regs[w.B()])
	case bytecode.RGetIndex:
		v, err = r.getIndexed(regs[w.B()], regs[w.C()])
	case bytecode.RSetIndex:
		return Undefined, r.setIndexed(regs[w.A()], regs[w.B()], regs[w.C()], cl.fn.Strict)
	case bytecode.RAdd:
		v, err = r.add(regs[w.B()], regs[w.C()])
	case bytecode.RSub:
		v, err = r.arith(bytecode.OpSub, regs[w.B()], regs[w.C()])
	case bytecode.RMul:
		v, err = r.arith(bytecode.OpMul, regs[w.B()], regs[w.C()])
	case bytecode.RBin:
		v, err = r.regBinary(bytecode.Op(w.X()), regs[w.B()], regs[w.C()])
	case bytecode.RBinI:
		v, err = r.regBinImm(bytecode.Op(w.X()), regs[w.B()], w.SC())
	case bytecode.RInc:
		v, err = r.regStep(regs[w.B()], 1)
	case bytecode.RDec:
		v, err = r.regStep(regs[w.B()], -1)
	case bytecode.RUnary:
		v, err = r.regUnary(bytecode.Op(w.X()), regs[w.B()])
	case bytecode.RDefineField:
		return Undefined, r.regDefineField(cl, w, regs[w.A()], regs[w.B()])
	case bytecode.RNewObject:
		v = Obj(r.regNewObject(cl, w))
	case bytecode.RNewArray:
		b := wb + int(w.B())
		v = Obj(r.newArrayFrom(r.stack[b : b+int(w.C())]))
	case bytecode.RClosure:
		v = Obj(r.makeClosure(f, cl.consts[w.BC()]))
	case bytecode.RNew:
		b := wb + int(w.A())
		if v, err = r.construct(regs[w.A()], r.stack[b+1:b+1+int(w.X())]); err == nil {
			regs[w.B()] = v
		}
		return v, err
	case bytecode.RCloseUpvals:
		if len(f.openUpvalues) != 0 {
			r.closeUpvaluesFrom(f, int(w.A()))
		}
		return Undefined, nil
	case bytecode.RJmpCmp, bytecode.RJmpCmpI:
		b := Int32(w.SB())
		if w.Op() == bytecode.RJmpCmp {
			b = regs[w.B()]
		}
		res, err := r.regCompare(bytecode.Op(w.X()&^bytecode.RCmpTrue), regs[w.A()], b)
		return Bool(res), err
	default:
		panic("register code: no slow path for " + w.Op().String())
	}
	if err == nil {
		regs[w.A()] = v
	}
	return v, err
}

// regThrowDeadZone is the ReferenceError of a read of a binding in its dead
// zone, at the check at pc.
func (r *Runtime) regThrowDeadZone(f *frame, c *regCode, pc int) error {
	f.pc = c.src[pc] + 1
	return r.throwReferenceError("cannot access %q before initialization", f.cl.fn.Locals[regAt(c.code, pc).B()].Name)
}

// regGetGlobal is get_global's slow path: what the stack interpreter does,
// then the layout of the global object the site found the slot in noted,
// for the loop's check.
func (r *Runtime) regGetGlobal(f *frame, cl *closure, w bytecode.RInstr) (Value, error) {
	in := bytecode.Instr{Op: bytecode.OpGetGlobal, A: uint32(w.B()), B: uint32(w.C())}
	v, err := r.getGlobalAt(regTctx(r, f, cl), in, int(f.pc-1))
	if err == nil {
		r.regNoteGlobal(&cl.ic[w.C()], false)
	}
	return v, err
}

// regSetGlobal is set_global's, or set_global_strict's, slow path.
func (r *Runtime) regSetGlobal(f *frame, cl *closure, w bytecode.RInstr, v Value) error {
	in := bytecode.Instr{Op: bytecode.OpSetGlobal, A: uint32(w.B()), B: uint32(w.C())}
	var err error
	if w.X() != 0 {
		in.Op = bytecode.OpSetGlobalStrict
		err = r.setGlobalStrict(f, cl, in, v)
	} else {
		err = r.setGlobalIn(f, cl, in, v)
	}
	if err == nil {
		r.regNoteGlobal(&cl.ic[w.C()], true)
	}
	return err
}

// regNoteGlobal remembers the layout of the global object a site found a
// plain data property in -- a writable one for a write -- which the loop
// compares the object's with.
func (r *Runtime) regNoteGlobal(s *propCache, write bool) {
	env := s.p1
	if env == nil || s.idx < 0 || int(s.idx) >= int(env.valn) || env.shape == nil {
		return
	}
	mask := propAccessor | propPrivate | propDeleted | propUninit
	want := propFlags(0)
	if write {
		mask |= propWritable
		want = propWritable
	}
	if p := env.shape.keys[s.idx]; p.flags&mask == want {
		noteGlobalShape(s, env.shape)
	}
}

// regGetProp is get_prop's slow path, as the stack interpreter's case.
func (r *Runtime) regGetProp(cl *closure, c *regCode, w bytecode.RInstr, obj Value) (Value, error) {
	site := w.C()
	name := cl.names[c.siteName[site]]
	if obj.IsObject() {
		o := obj.Object()
		if v, ok := plainOwn(o, name); ok {
			return v, nil
		}
		if v, ok, err := r.cachedGet(&cl.ic[site], o, name); ok {
			return v, err
		}
	}
	return r.getValueProp(obj, name)
}

// regGetMethod is get_prop_this's slow path.
func (r *Runtime) regGetMethod(cl *closure, c *regCode, w bytecode.RInstr, obj Value) (Value, error) {
	site := w.C()
	name := cl.names[c.siteName[site]]
	if obj.IsObject() {
		if v, ok := r.cachedProp(&cl.ic[site], obj.Object(), name); ok {
			return v, nil
		}
	}
	return r.getValueProp(obj, name)
}

// regSetProp is set_prop's slow path.
func (r *Runtime) regSetProp(cl *closure, c *regCode, w bytecode.RInstr, obj, v Value) error {
	site := w.C()
	name := cl.names[c.siteName[site]]
	if obj.IsObject() {
		return r.setPropCached(&cl.ic[site], obj.Object(), name, v, cl.fn.Strict)
	}
	return r.setValueProp(obj, name, v, cl.fn.Strict)
}

// regDefineField is define_field.
func (r *Runtime) regDefineField(cl *closure, w bytecode.RInstr, obj, v Value) error {
	if !obj.IsObject() {
		return nil
	}
	name := cl.names[w.C()]
	if o := obj.Object(); o.class == ClassObject && o.flags&objExtensible != 0 {
		if next := o.transition(name, propDefault); next != nil {
			o.appendTransition(v, next)
			return nil
		}
	}
	return r.defineOwnProp(obj.Object(), name, v, propDefault)
}

// regNewObject is new_object.
func (r *Runtime) regNewObject(cl *closure, w bytecode.RInstr) *Object {
	root := r.shapes.siteRoot(&cl.ic[w.C()], r.proto.object)
	return newShapedLiteral(root.orRoot(ClassObject, r.proto.object), ClassObject, int(w.B()))
}
