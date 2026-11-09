package vm

import (
	"math"
	"os"
	"sync/atomic"
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/compiler/regform"
	"github.com/go-quickjs/go-quickjs/internal/jsnum"
)

// The register interpreter (docs/register-interpreter.md), stage 1: a
// function the register form covers runs as its register code, every other
// as before. QJS_REG in the environment turns it on.
//
// A frame's registers are a window of the runtime's value stack laid out as
// the stack form's frame is -- the locals, then the temporaries -- so that a
// closure captures a local, a stack trace reads a pc and the frame is pushed
// and popped as ever. The loop indexes the window with 16-bit register
// numbers through a pointer to an array of 65,536 values, which needs no
// bounds check: the register form's verifier (regform.VerifyRegisterCode)
// proves every register an instruction names is below the frame's count,
// and a frame is given that many slots before it runs. The code is read the
// same way, every jump proven to land in it.
//
// The files sort after every other of the package's, so that the functions
// they add come after every function there was: the interpreter's hot
// functions keep their addresses, and only the closures move.

// regTier turns the register interpreter on for functions not yet run.
var regTier atomic.Bool

func init() { regTier.Store(os.Getenv("QJS_REG") != "") }

// SetRegTier turns the register interpreter on or off for functions not yet
// run, for tests that compare it with the other tiers.
func SetRegTier(on bool) { regTier.Store(on) }

// regsBuilt counts the functions given register code.
var regsBuilt atomic.Int64

// RegsBuilt is how many functions have been given register code, for tests
// that check it ran.
func RegsBuilt() int64 { return regsBuilt.Load() }

// regCode is a function's register code as the VM runs it.
type regCode struct {
	// code is the first instruction, read with no bounds check.
	code unsafe.Pointer
	src  []uint32
	// siteName is each property site's name, by its cache's number.
	siteName []uint32
	// numRegs is how many registers a frame uses; locals and params are
	// the function's counts.
	numRegs, locals, params int32
	coerceThis              bool
	// wide marks code of the register form's own frame layout: register 0
	// the function, 1 its this, the parameters from 2, and inits bindings
	// after them that a call makes undefined. Only call_w enters it.
	wide  bool
	inits int32
	rc    *bytecode.RegCode
	fn    *bytecode.Function
}

// regWindow is the frame's registers from base. The register form's
// verifier proves that every register an instruction names is below the
// frame's count, which the frame has, so the array's length is never
// reached past them.
//
//go:nocheckptr
func regWindow(stack []Value, base int) *[1 << 16]Value {
	return (*[1 << 16]Value)(unsafe.Add(unsafe.Pointer(unsafe.SliceData(stack)), uintptr(base)*unsafe.Sizeof(Value{})))
}

// regAt is the instruction at pc, which the verifier proves is in the code.
func regAt(code unsafe.Pointer, pc int) bytecode.RInstr {
	return *(*bytecode.RInstr)(unsafe.Add(code, uintptr(pc)*8))
}

// firstForm is what firstTree builds for a function: its register code,
// run by a tree of one node, where the register interpreter is on and the
// form covers it, and its tree otherwise.
func firstForm(fn *bytecode.Function) *tree {
	if regTier.Load() {
		if t := regTreeOf(fn); t != nil {
			return t
		}
	}
	return buildTree(fn)
}

// regTreeOf translates fn into register code and returns the tree that runs
// it, or nil.
func regTreeOf(fn *bytecode.Function) *tree {
	if regHandOn {
		if t := regHandTree(fn); t != nil {
			return t
		}
	}
	rc, why := regform.RegisterCode(fn)
	if rc == nil {
		if regTrace {
			println("reg: not", fn.Name, why)
		}
		return nil
	}
	regsBuilt.Add(1)
	if regDump {
		println("reg:", fn.Name)
		println(rc.String())
	}
	c := &regCode{
		code: unsafe.Pointer(unsafe.SliceData(rc.Code)), src: rc.SrcPC, siteName: rc.SiteName,
		numRegs: int32(rc.NumRegs), locals: int32(fn.LocalCount), params: int32(fn.ParamCount),
		coerceThis: fn.CoerceThis, rc: rc, fn: fn,
	}
	t := &tree{reg: c}
	t.blocks = []tblock{{body: []tstmt{regStmt(c)}, next: regNext}}
	return t
}

// regTrace prints why a function has no register code.
var regTrace = os.Getenv("QJS_REGTRACE") != ""

// regDump prints each function's register code as it is made.
var regDump = os.Getenv("QJS_REGDUMP") != ""

// regStmt is the one node of a register function's tree: it runs the frame
// a call through runFD or callTree has made as register code. A frame of
// the few calls the code cannot run in -- an arrow sharing a derived
// constructor's this, a function made inside a with or beside an eval -- is
// run by the stack interpreter instead.
func regStmt(c *regCode) tstmt {
	return func(t *tctx) {
		r, f := t.r, t.f
		if c.wide || f.thisRef != nil || f.withScopes != nil || f.evalVars != nil {
			v, err := r.execute(f)
			if err != nil {
				t.throw(err)
			}
			t.ret = v
			return
		}
		// The frame was made for the stack form's slots; the register form
		// has a few more of its own.
		base := f.base - int(c.locals)
		if top := base + int(c.numRegs); top > r.stackTop {
			if top > len(r.stack) {
				t.throw(r.throwRangeError("maximum call stack size exceeded"))
			}
			r.stackTop = top
			if top > r.stackHigh {
				r.stackHigh = top
			}
		}
		v, err := r.runReg(f, c)
		if err != nil {
			t.throw(err)
		}
		t.ret = v
	}
}

func regNext(*tctx) int { return -1 }

// regTruthy is ToBoolean, a boolean's without the call.
func regTruthy(v Value) bool {
	return math.Float64bits(v.num) == trueBits || math.Float64bits(v.num) != falseBits && v.Truthy()
}

// runReg runs frame f as register code c and returns what it returns.
//
// A case that finishes its instruction continues the loop; one that can
// fail sets the frame's pc to the stack form's instruction first, which is
// what a stack trace or an error's position reads, and returns the error.
// A frame of the register form has no handler: a function with a try
// statement is not translated.
func (r *Runtime) runReg(f *frame, c *regCode) (Value, error) {
	if r.stopped != nil {
		return Undefined, r.stopped
	}
	cl := f.cl
	wb := f.base - int(c.locals)
	regs := regWindow(r.stack, wb)
	code := c.code
	// The loop's pc stays in a register only between calls: a case that
	// calls out keeps it in kept across the call and reads it back, as it
	// reads its instruction back from the code, so that neither is live
	// across the call and Go does not store them at every dispatch.
	var kept regKept
	pc := 0
	for {
		w := regAt(code, pc)
		pc++
		switch w.Op() & 127 {
		case bytecode.RNop:
		case bytecode.RMove:
			regs[w.A()] = regs[w.B()]
		case bytecode.RLoadInt:
			regs[w.A()] = Int32(int32(w.BC()))
		case bytecode.RLoadConst:
			regs[w.A()] = cl.consts[w.BC()]
		case bytecode.RLoadUndef:
			regs[w.A()] = Undefined
		case bytecode.RLoadNull:
			regs[w.A()] = Null
		case bytecode.RLoadTrue:
			regs[w.A()] = True
		case bytecode.RLoadFalse:
			regs[w.A()] = False
		case bytecode.RLoadEmpty:
			regs[w.A()] = Str(emptyString)
		case bytecode.RLoadUninit:
			regs[w.A()] = uninitialized
		case bytecode.RLoadThis:
			regs[w.A()] = f.this
		case bytecode.RCheckRef:
			if regs[w.A()].IsUninitialized() {
				return Undefined, r.regThrowDeadZone(f, c, pc-1)
			}
		case bytecode.RGetUpval:
			regs[w.A()] = cl.upvalues[w.BC()].get()
		case bytecode.RSetUpval:
			cl.upvalues[w.BC()].set(regs[w.A()])

		case bytecode.RGetGlobal:
			// The global object's slot the site remembers, in the layout it
			// read it in: nothing can shadow it where no script has a
			// lexical binding and no eval a variable, which a frame of the
			// register form never has.
			s := &cl.ic[w.C()]
			if env := s.p1; env != nil && (r.globalLex.valn == 0 || !r.lexShadows(cl.names[w.B()])) {
				if i := uint(s.idx); i < uint(env.valn) {
					// The layout the site read the slot in, or, for one that
					// changes in place, the slot's own attributes.
					if sh := env.shape; sh == s.shape ||
						sh.keys[i].flags&(propAccessor|propPrivate|propDeleted|propUninit) == 0 {
						regs[w.A()] = env.valAt(int(i))
						continue
					}
				}
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc
		case bytecode.RSetGlobal:
			s := &cl.ic[w.C()]
			if env := s.p1; env != nil && (r.globalLex.valn == 0 || !r.lexShadows(cl.names[w.B()])) {
				if i := uint(s.idx); i < uint(env.valn) {
					if sh := env.shape; sh == s.shape ||
						sh.keys[i].flags&(propAccessor|propPrivate|propDeleted|propUninit|propWritable) == propWritable {
						env.vals()[i] = regs[w.A()]
						continue
					}
				}
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc

		case bytecode.RGetProp, bytecode.RGetMethod:
			if o := regs[w.B()]; o.IsObject() {
				obj := o.Object()
				if s := &cl.ic[w.C()]; obj.shape == s.ownShape {
					regs[w.A()] = obj.valAt(int(s.idx))
					continue
				}
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc
		case bytecode.RSetProp:
			if o := regs[w.A()]; o.IsObject() {
				obj := o.Object()
				if s := &cl.ic[w.C()]; obj.shape == s.ownShape {
					obj.vals()[s.idx] = regs[w.B()]
					continue
				}
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc
		case bytecode.RGetIndex:
			obj, key := regs[w.B()], regs[w.C()]
			if obj.IsObject() && key.IsNumber() {
				o := obj.Object()
				if i := uint32(key.Number()); float64(i) == key.Number() &&
					uint(i) < uint(len(o.elemsOf())) && o.flags&objMappedArguments == 0 {
					if v := o.elemsOf()[i]; !isHole(v) {
						regs[w.A()] = v
						continue
					}
				}
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc
		case bytecode.RSetIndex:
			obj, key := regs[w.A()], regs[w.B()]
			if obj.IsObject() && key.IsNumber() {
				o := obj.Object()
				if i := uint32(key.Number()); float64(i) == key.Number() &&
					uint(i) < uint(len(o.elemsOf())) && o.flags&(objMappedArguments|objSharedElems) == 0 && !isHole(o.elemsOf()[i]) {
					o.extOf().elems[i] = regs[w.C()]
					continue
				}
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc

		case bytecode.RAdd:
			a, b := regs[w.B()], regs[w.C()]
			if a.IsNumber() && b.IsNumber() {
				regs[w.A()] = regNum(a.Number() + b.Number())
				continue
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc
		case bytecode.RSub:
			a, b := regs[w.B()], regs[w.C()]
			if a.IsNumber() && b.IsNumber() {
				regs[w.A()] = regNum(a.Number() - b.Number())
				continue
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc
		case bytecode.RMul:
			a, b := regs[w.B()], regs[w.C()]
			if a.IsNumber() && b.IsNumber() {
				regs[w.A()] = regNum(a.Number() * b.Number())
				continue
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc
		case bytecode.RBinI:
			if a := regs[w.B()]; a.IsNumber() {
				x, y := a.Number(), w.SC()
				switch bytecode.Op(w.X()) {
				case bytecode.OpAdd:
					regs[w.A()] = regNum(x + float64(y))
					continue
				case bytecode.OpSub:
					regs[w.A()] = regNum(x - float64(y))
					continue
				case bytecode.OpBitOr:
					if xi := int32(x); float64(xi) == x {
						regs[w.A()] = Int32(xi | y)
						continue
					}
				case bytecode.OpBitAnd:
					if xi := int32(x); float64(xi) == x {
						regs[w.A()] = Int32(xi & y)
						continue
					}
				}
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc
		case bytecode.RInc:
			if a := regs[w.B()]; a.IsNumber() {
				regs[w.A()] = regNum(a.Number() + 1)
				continue
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc
		case bytecode.RDec:
			if a := regs[w.B()]; a.IsNumber() {
				regs[w.A()] = regNum(a.Number() - 1)
				continue
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc
		case bytecode.RBin, bytecode.RUnary, bytecode.RDefineField, bytecode.RNewObject,
			bytecode.RNewArray, bytecode.RClosure, bytecode.RNew, bytecode.RCloseUpvals:
			// What the loop has no fast path of its own for.
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc

		case bytecode.RJmp:
			off := int32(w.BC())
			if off < 0 {
				if r.backEdges--; r.backEdges <= 0 {
					kept.pc = pc
					if err := r.regBackEdge(f, c, pc-1); err != nil {
						return Undefined, err
					}
					pc = kept.pc
					off = int32(regAt(code, pc-1).BC())
				}
			}
			pc += int(off)
		case bytecode.RJmpF, bytecode.RJmpT:
			if regTruthy(regs[w.A()]) == (w.Op() == bytecode.RJmpT) {
				off := int32(w.BC())
				if off < 0 {
					if r.backEdges--; r.backEdges <= 0 {
						kept.pc = pc
						if err := r.regBackEdge(f, c, pc-1); err != nil {
							return Undefined, err
						}
						pc = kept.pc
						off = int32(regAt(code, pc-1).BC())
					}
				}
				pc += int(off)
			}
		case bytecode.RJmpCmp, bytecode.RJmpCmpI:
			a := regs[w.A()]
			var b Value
			if w.Op() == bytecode.RJmpCmp {
				b = regs[w.B()]
			} else {
				b = Int32(w.SB())
			}
			var res bool
			if a.IsNumber() && b.IsNumber() {
				switch u, v := a.Number(), b.Number(); bytecode.Op(w.X() &^ bytecode.RCmpTrue) {
				case bytecode.OpLt:
					res = u < v
				case bytecode.OpLe:
					res = u <= v
				case bytecode.OpGt:
					res = u > v
				case bytecode.OpGe:
					res = u >= v
				case bytecode.OpStrictEq, bytecode.OpEq:
					res = u == v
				default:
					res = u != v
				}
			} else {
				kept.pc = pc
				v, err := r.regSlow(f, c, pc-1, regs)
				if err != nil {
					return Undefined, err
				}
				pc = kept.pc
				w = regAt(code, pc-1)
				res = v.sameBits(True)
			}
			if res == (w.X()&bytecode.RCmpTrue != 0) {
				off := w.SC()
				if off < 0 {
					if r.backEdges--; r.backEdges <= 0 {
						kept.pc = pc
						if err := r.regBackEdge(f, c, pc-1); err != nil {
							return Undefined, err
						}
						pc = kept.pc
						off = regAt(code, pc-1).SC()
					}
				}
				pc += int(off)
			}

		case bytecode.RRet:
			return regs[w.A()], nil
		case bytecode.RRetUndef:
			return Undefined, nil
		case bytecode.RCall, bytecode.RCallMethod:
			f.pc = c.src[pc-1] + 1
			kept.pc = pc
			b := wb + int(w.A())
			var v Value
			var err error
			if w.Op() == bytecode.RCall {
				v, err = r.regCall(regs[w.A()], Undefined, r.stack[b+1:b+1+int(w.X())])
			} else {
				v, err = r.regCall(regs[w.A()+1], regs[w.A()], r.stack[b+2:b+2+int(w.X())])
			}
			if err != nil {
				return Undefined, err
			}
			pc = kept.pc
			regs[regAt(code, pc-1).B()] = v
		case bytecode.RAndI:
			if x, ok := regInt32(regs[w.B()]); ok {
				regs[w.A()] = Int32(x & w.SC())
				continue
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc
		case bytecode.ROrI:
			if x, ok := regInt32(regs[w.B()]); ok {
				regs[w.A()] = Int32(x | w.SC())
				continue
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc
		case bytecode.RXorI:
			if x, ok := regInt32(regs[w.B()]); ok {
				regs[w.A()] = Int32(x ^ w.SC())
				continue
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc
		case bytecode.RShlI:
			if x, ok := regInt32(regs[w.B()]); ok {
				regs[w.A()] = Int32(x << (uint32(w.SC()) & 31))
				continue
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc
		case bytecode.RShrI:
			if x, ok := regInt32(regs[w.B()]); ok {
				regs[w.A()] = Int32(x >> (uint32(w.SC()) & 31))
				continue
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc
		case bytecode.RUShrI:
			if x, ok := regInt32(regs[w.B()]); ok {
				regs[w.A()] = Uint32(uint32(x) >> (uint32(w.SC()) & 31))
				continue
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc
		case bytecode.RBitAnd:
			if x, ok := regInt32(regs[w.B()]); ok {
				if y, ok := regInt32(regs[w.C()]); ok {
					regs[w.A()] = Int32(x & y)
					continue
				}
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc
		case bytecode.RBitOr:
			if x, ok := regInt32(regs[w.B()]); ok {
				if y, ok := regInt32(regs[w.C()]); ok {
					regs[w.A()] = Int32(x | y)
					continue
				}
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc
		case bytecode.RBitXor:
			if x, ok := regInt32(regs[w.B()]); ok {
				if y, ok := regInt32(regs[w.C()]); ok {
					regs[w.A()] = Int32(x ^ y)
					continue
				}
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc
		case bytecode.RShl:
			if x, ok := regInt32(regs[w.B()]); ok {
				if y, ok := regInt32(regs[w.C()]); ok {
					regs[w.A()] = Int32(x << (uint32(y) & 31))
					continue
				}
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc
		case bytecode.RShr:
			if x, ok := regInt32(regs[w.B()]); ok {
				if y, ok := regInt32(regs[w.C()]); ok {
					regs[w.A()] = Int32(x >> (uint32(y) & 31))
					continue
				}
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc
		case bytecode.RUShr:
			if x, ok := regInt32(regs[w.B()]); ok {
				if y, ok := regInt32(regs[w.C()]); ok {
					regs[w.A()] = Uint32(uint32(x) >> (uint32(y) & 31))
					continue
				}
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc
		case bytecode.RCallW:
			f.pc = c.src[pc-1] + 1
			kept.pc = pc
			v, err := r.regCallW(f, c, pc-1, wb)
			if err != nil {
				return Undefined, err
			}
			pc = kept.pc
			regs[regAt(code, pc-1).B()] = v
		case bytecode.RAddI:
			if a := regs[w.B()]; a.IsNumber() {
				regs[w.A()] = regNum(a.Number() + float64(w.SC()))
				continue
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc
		case bytecode.RSubI:
			if a := regs[w.B()]; a.IsNumber() {
				regs[w.A()] = regNum(a.Number() - float64(w.SC()))
				continue
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, pc-1, regs); err != nil {
				return Undefined, err
			}
			pc = kept.pc
		case bytecode.RJLt:
			if a, b := regs[w.A()], regs[w.B()]; a.IsNumber() && b.IsNumber() {
				if u, v := a.Number(), b.Number(); u < v {
					if off := w.SC(); off >= 0 {
						pc += int(off)
					} else if r.backEdges--; r.backEdges > 0 {
						pc += int(off)
					} else {
						kept.pc = pc
						if err := r.regBackEdge(f, c, pc-1); err != nil {
							return Undefined, err
						}
						pc = kept.pc + int(regAt(code, kept.pc-1).SC())
					}
				}
				continue
			}
			kept.pc = pc
			t, err := r.regSlow(f, c, pc-1, regs)
			if err != nil {
				return Undefined, err
			}
			pc = kept.pc
			if t.sameBits(True) {
				w = regAt(code, pc-1)
				if off := w.SC(); off >= 0 {
					pc += int(off)
				} else if r.backEdges--; r.backEdges > 0 {
					pc += int(off)
				} else {
					kept.pc = pc
					if err := r.regBackEdge(f, c, pc-1); err != nil {
						return Undefined, err
					}
					pc = kept.pc + int(regAt(code, kept.pc-1).SC())
				}
			}
		case bytecode.RJLtI:
			if a := regs[w.A()]; a.IsNumber() {
				if u, v := a.Number(), float64(w.SB()); u < v {
					if off := w.SC(); off >= 0 {
						pc += int(off)
					} else if r.backEdges--; r.backEdges > 0 {
						pc += int(off)
					} else {
						kept.pc = pc
						if err := r.regBackEdge(f, c, pc-1); err != nil {
							return Undefined, err
						}
						pc = kept.pc + int(regAt(code, kept.pc-1).SC())
					}
				}
				continue
			}
			kept.pc = pc
			t, err := r.regSlow(f, c, pc-1, regs)
			if err != nil {
				return Undefined, err
			}
			pc = kept.pc
			if t.sameBits(True) {
				w = regAt(code, pc-1)
				if off := w.SC(); off >= 0 {
					pc += int(off)
				} else if r.backEdges--; r.backEdges > 0 {
					pc += int(off)
				} else {
					kept.pc = pc
					if err := r.regBackEdge(f, c, pc-1); err != nil {
						return Undefined, err
					}
					pc = kept.pc + int(regAt(code, kept.pc-1).SC())
				}
			}
		case bytecode.RJNLt:
			if a, b := regs[w.A()], regs[w.B()]; a.IsNumber() && b.IsNumber() {
				if u, v := a.Number(), b.Number(); !(u < v) {
					if off := w.SC(); off >= 0 {
						pc += int(off)
					} else if r.backEdges--; r.backEdges > 0 {
						pc += int(off)
					} else {
						kept.pc = pc
						if err := r.regBackEdge(f, c, pc-1); err != nil {
							return Undefined, err
						}
						pc = kept.pc + int(regAt(code, kept.pc-1).SC())
					}
				}
				continue
			}
			kept.pc = pc
			t, err := r.regSlow(f, c, pc-1, regs)
			if err != nil {
				return Undefined, err
			}
			pc = kept.pc
			if t.sameBits(True) {
				w = regAt(code, pc-1)
				if off := w.SC(); off >= 0 {
					pc += int(off)
				} else if r.backEdges--; r.backEdges > 0 {
					pc += int(off)
				} else {
					kept.pc = pc
					if err := r.regBackEdge(f, c, pc-1); err != nil {
						return Undefined, err
					}
					pc = kept.pc + int(regAt(code, kept.pc-1).SC())
				}
			}
		case bytecode.RJNLtI:
			if a := regs[w.A()]; a.IsNumber() {
				if u, v := a.Number(), float64(w.SB()); !(u < v) {
					if off := w.SC(); off >= 0 {
						pc += int(off)
					} else if r.backEdges--; r.backEdges > 0 {
						pc += int(off)
					} else {
						kept.pc = pc
						if err := r.regBackEdge(f, c, pc-1); err != nil {
							return Undefined, err
						}
						pc = kept.pc + int(regAt(code, kept.pc-1).SC())
					}
				}
				continue
			}
			kept.pc = pc
			t, err := r.regSlow(f, c, pc-1, regs)
			if err != nil {
				return Undefined, err
			}
			pc = kept.pc
			if t.sameBits(True) {
				w = regAt(code, pc-1)
				if off := w.SC(); off >= 0 {
					pc += int(off)
				} else if r.backEdges--; r.backEdges > 0 {
					pc += int(off)
				} else {
					kept.pc = pc
					if err := r.regBackEdge(f, c, pc-1); err != nil {
						return Undefined, err
					}
					pc = kept.pc + int(regAt(code, kept.pc-1).SC())
				}
			}
		case bytecode.RJLe:
			if a, b := regs[w.A()], regs[w.B()]; a.IsNumber() && b.IsNumber() {
				if u, v := a.Number(), b.Number(); u <= v {
					if off := w.SC(); off >= 0 {
						pc += int(off)
					} else if r.backEdges--; r.backEdges > 0 {
						pc += int(off)
					} else {
						kept.pc = pc
						if err := r.regBackEdge(f, c, pc-1); err != nil {
							return Undefined, err
						}
						pc = kept.pc + int(regAt(code, kept.pc-1).SC())
					}
				}
				continue
			}
			kept.pc = pc
			t, err := r.regSlow(f, c, pc-1, regs)
			if err != nil {
				return Undefined, err
			}
			pc = kept.pc
			if t.sameBits(True) {
				w = regAt(code, pc-1)
				if off := w.SC(); off >= 0 {
					pc += int(off)
				} else if r.backEdges--; r.backEdges > 0 {
					pc += int(off)
				} else {
					kept.pc = pc
					if err := r.regBackEdge(f, c, pc-1); err != nil {
						return Undefined, err
					}
					pc = kept.pc + int(regAt(code, kept.pc-1).SC())
				}
			}
		case bytecode.RJLeI:
			if a := regs[w.A()]; a.IsNumber() {
				if u, v := a.Number(), float64(w.SB()); u <= v {
					if off := w.SC(); off >= 0 {
						pc += int(off)
					} else if r.backEdges--; r.backEdges > 0 {
						pc += int(off)
					} else {
						kept.pc = pc
						if err := r.regBackEdge(f, c, pc-1); err != nil {
							return Undefined, err
						}
						pc = kept.pc + int(regAt(code, kept.pc-1).SC())
					}
				}
				continue
			}
			kept.pc = pc
			t, err := r.regSlow(f, c, pc-1, regs)
			if err != nil {
				return Undefined, err
			}
			pc = kept.pc
			if t.sameBits(True) {
				w = regAt(code, pc-1)
				if off := w.SC(); off >= 0 {
					pc += int(off)
				} else if r.backEdges--; r.backEdges > 0 {
					pc += int(off)
				} else {
					kept.pc = pc
					if err := r.regBackEdge(f, c, pc-1); err != nil {
						return Undefined, err
					}
					pc = kept.pc + int(regAt(code, kept.pc-1).SC())
				}
			}
		case bytecode.RJNLe:
			if a, b := regs[w.A()], regs[w.B()]; a.IsNumber() && b.IsNumber() {
				if u, v := a.Number(), b.Number(); !(u <= v) {
					if off := w.SC(); off >= 0 {
						pc += int(off)
					} else if r.backEdges--; r.backEdges > 0 {
						pc += int(off)
					} else {
						kept.pc = pc
						if err := r.regBackEdge(f, c, pc-1); err != nil {
							return Undefined, err
						}
						pc = kept.pc + int(regAt(code, kept.pc-1).SC())
					}
				}
				continue
			}
			kept.pc = pc
			t, err := r.regSlow(f, c, pc-1, regs)
			if err != nil {
				return Undefined, err
			}
			pc = kept.pc
			if t.sameBits(True) {
				w = regAt(code, pc-1)
				if off := w.SC(); off >= 0 {
					pc += int(off)
				} else if r.backEdges--; r.backEdges > 0 {
					pc += int(off)
				} else {
					kept.pc = pc
					if err := r.regBackEdge(f, c, pc-1); err != nil {
						return Undefined, err
					}
					pc = kept.pc + int(regAt(code, kept.pc-1).SC())
				}
			}
		case bytecode.RJNLeI:
			if a := regs[w.A()]; a.IsNumber() {
				if u, v := a.Number(), float64(w.SB()); !(u <= v) {
					if off := w.SC(); off >= 0 {
						pc += int(off)
					} else if r.backEdges--; r.backEdges > 0 {
						pc += int(off)
					} else {
						kept.pc = pc
						if err := r.regBackEdge(f, c, pc-1); err != nil {
							return Undefined, err
						}
						pc = kept.pc + int(regAt(code, kept.pc-1).SC())
					}
				}
				continue
			}
			kept.pc = pc
			t, err := r.regSlow(f, c, pc-1, regs)
			if err != nil {
				return Undefined, err
			}
			pc = kept.pc
			if t.sameBits(True) {
				w = regAt(code, pc-1)
				if off := w.SC(); off >= 0 {
					pc += int(off)
				} else if r.backEdges--; r.backEdges > 0 {
					pc += int(off)
				} else {
					kept.pc = pc
					if err := r.regBackEdge(f, c, pc-1); err != nil {
						return Undefined, err
					}
					pc = kept.pc + int(regAt(code, kept.pc-1).SC())
				}
			}
		case bytecode.RJGt:
			if a, b := regs[w.A()], regs[w.B()]; a.IsNumber() && b.IsNumber() {
				if u, v := a.Number(), b.Number(); u > v {
					if off := w.SC(); off >= 0 {
						pc += int(off)
					} else if r.backEdges--; r.backEdges > 0 {
						pc += int(off)
					} else {
						kept.pc = pc
						if err := r.regBackEdge(f, c, pc-1); err != nil {
							return Undefined, err
						}
						pc = kept.pc + int(regAt(code, kept.pc-1).SC())
					}
				}
				continue
			}
			kept.pc = pc
			t, err := r.regSlow(f, c, pc-1, regs)
			if err != nil {
				return Undefined, err
			}
			pc = kept.pc
			if t.sameBits(True) {
				w = regAt(code, pc-1)
				if off := w.SC(); off >= 0 {
					pc += int(off)
				} else if r.backEdges--; r.backEdges > 0 {
					pc += int(off)
				} else {
					kept.pc = pc
					if err := r.regBackEdge(f, c, pc-1); err != nil {
						return Undefined, err
					}
					pc = kept.pc + int(regAt(code, kept.pc-1).SC())
				}
			}
		case bytecode.RJGtI:
			if a := regs[w.A()]; a.IsNumber() {
				if u, v := a.Number(), float64(w.SB()); u > v {
					if off := w.SC(); off >= 0 {
						pc += int(off)
					} else if r.backEdges--; r.backEdges > 0 {
						pc += int(off)
					} else {
						kept.pc = pc
						if err := r.regBackEdge(f, c, pc-1); err != nil {
							return Undefined, err
						}
						pc = kept.pc + int(regAt(code, kept.pc-1).SC())
					}
				}
				continue
			}
			kept.pc = pc
			t, err := r.regSlow(f, c, pc-1, regs)
			if err != nil {
				return Undefined, err
			}
			pc = kept.pc
			if t.sameBits(True) {
				w = regAt(code, pc-1)
				if off := w.SC(); off >= 0 {
					pc += int(off)
				} else if r.backEdges--; r.backEdges > 0 {
					pc += int(off)
				} else {
					kept.pc = pc
					if err := r.regBackEdge(f, c, pc-1); err != nil {
						return Undefined, err
					}
					pc = kept.pc + int(regAt(code, kept.pc-1).SC())
				}
			}
		case bytecode.RJNGt:
			if a, b := regs[w.A()], regs[w.B()]; a.IsNumber() && b.IsNumber() {
				if u, v := a.Number(), b.Number(); !(u > v) {
					if off := w.SC(); off >= 0 {
						pc += int(off)
					} else if r.backEdges--; r.backEdges > 0 {
						pc += int(off)
					} else {
						kept.pc = pc
						if err := r.regBackEdge(f, c, pc-1); err != nil {
							return Undefined, err
						}
						pc = kept.pc + int(regAt(code, kept.pc-1).SC())
					}
				}
				continue
			}
			kept.pc = pc
			t, err := r.regSlow(f, c, pc-1, regs)
			if err != nil {
				return Undefined, err
			}
			pc = kept.pc
			if t.sameBits(True) {
				w = regAt(code, pc-1)
				if off := w.SC(); off >= 0 {
					pc += int(off)
				} else if r.backEdges--; r.backEdges > 0 {
					pc += int(off)
				} else {
					kept.pc = pc
					if err := r.regBackEdge(f, c, pc-1); err != nil {
						return Undefined, err
					}
					pc = kept.pc + int(regAt(code, kept.pc-1).SC())
				}
			}
		case bytecode.RJNGtI:
			if a := regs[w.A()]; a.IsNumber() {
				if u, v := a.Number(), float64(w.SB()); !(u > v) {
					if off := w.SC(); off >= 0 {
						pc += int(off)
					} else if r.backEdges--; r.backEdges > 0 {
						pc += int(off)
					} else {
						kept.pc = pc
						if err := r.regBackEdge(f, c, pc-1); err != nil {
							return Undefined, err
						}
						pc = kept.pc + int(regAt(code, kept.pc-1).SC())
					}
				}
				continue
			}
			kept.pc = pc
			t, err := r.regSlow(f, c, pc-1, regs)
			if err != nil {
				return Undefined, err
			}
			pc = kept.pc
			if t.sameBits(True) {
				w = regAt(code, pc-1)
				if off := w.SC(); off >= 0 {
					pc += int(off)
				} else if r.backEdges--; r.backEdges > 0 {
					pc += int(off)
				} else {
					kept.pc = pc
					if err := r.regBackEdge(f, c, pc-1); err != nil {
						return Undefined, err
					}
					pc = kept.pc + int(regAt(code, kept.pc-1).SC())
				}
			}
		case bytecode.RJGe:
			if a, b := regs[w.A()], regs[w.B()]; a.IsNumber() && b.IsNumber() {
				if u, v := a.Number(), b.Number(); u >= v {
					if off := w.SC(); off >= 0 {
						pc += int(off)
					} else if r.backEdges--; r.backEdges > 0 {
						pc += int(off)
					} else {
						kept.pc = pc
						if err := r.regBackEdge(f, c, pc-1); err != nil {
							return Undefined, err
						}
						pc = kept.pc + int(regAt(code, kept.pc-1).SC())
					}
				}
				continue
			}
			kept.pc = pc
			t, err := r.regSlow(f, c, pc-1, regs)
			if err != nil {
				return Undefined, err
			}
			pc = kept.pc
			if t.sameBits(True) {
				w = regAt(code, pc-1)
				if off := w.SC(); off >= 0 {
					pc += int(off)
				} else if r.backEdges--; r.backEdges > 0 {
					pc += int(off)
				} else {
					kept.pc = pc
					if err := r.regBackEdge(f, c, pc-1); err != nil {
						return Undefined, err
					}
					pc = kept.pc + int(regAt(code, kept.pc-1).SC())
				}
			}
		case bytecode.RJGeI:
			if a := regs[w.A()]; a.IsNumber() {
				if u, v := a.Number(), float64(w.SB()); u >= v {
					if off := w.SC(); off >= 0 {
						pc += int(off)
					} else if r.backEdges--; r.backEdges > 0 {
						pc += int(off)
					} else {
						kept.pc = pc
						if err := r.regBackEdge(f, c, pc-1); err != nil {
							return Undefined, err
						}
						pc = kept.pc + int(regAt(code, kept.pc-1).SC())
					}
				}
				continue
			}
			kept.pc = pc
			t, err := r.regSlow(f, c, pc-1, regs)
			if err != nil {
				return Undefined, err
			}
			pc = kept.pc
			if t.sameBits(True) {
				w = regAt(code, pc-1)
				if off := w.SC(); off >= 0 {
					pc += int(off)
				} else if r.backEdges--; r.backEdges > 0 {
					pc += int(off)
				} else {
					kept.pc = pc
					if err := r.regBackEdge(f, c, pc-1); err != nil {
						return Undefined, err
					}
					pc = kept.pc + int(regAt(code, kept.pc-1).SC())
				}
			}
		case bytecode.RJNGe:
			if a, b := regs[w.A()], regs[w.B()]; a.IsNumber() && b.IsNumber() {
				if u, v := a.Number(), b.Number(); !(u >= v) {
					if off := w.SC(); off >= 0 {
						pc += int(off)
					} else if r.backEdges--; r.backEdges > 0 {
						pc += int(off)
					} else {
						kept.pc = pc
						if err := r.regBackEdge(f, c, pc-1); err != nil {
							return Undefined, err
						}
						pc = kept.pc + int(regAt(code, kept.pc-1).SC())
					}
				}
				continue
			}
			kept.pc = pc
			t, err := r.regSlow(f, c, pc-1, regs)
			if err != nil {
				return Undefined, err
			}
			pc = kept.pc
			if t.sameBits(True) {
				w = regAt(code, pc-1)
				if off := w.SC(); off >= 0 {
					pc += int(off)
				} else if r.backEdges--; r.backEdges > 0 {
					pc += int(off)
				} else {
					kept.pc = pc
					if err := r.regBackEdge(f, c, pc-1); err != nil {
						return Undefined, err
					}
					pc = kept.pc + int(regAt(code, kept.pc-1).SC())
				}
			}
		case bytecode.RJNGeI:
			if a := regs[w.A()]; a.IsNumber() {
				if u, v := a.Number(), float64(w.SB()); !(u >= v) {
					if off := w.SC(); off >= 0 {
						pc += int(off)
					} else if r.backEdges--; r.backEdges > 0 {
						pc += int(off)
					} else {
						kept.pc = pc
						if err := r.regBackEdge(f, c, pc-1); err != nil {
							return Undefined, err
						}
						pc = kept.pc + int(regAt(code, kept.pc-1).SC())
					}
				}
				continue
			}
			kept.pc = pc
			t, err := r.regSlow(f, c, pc-1, regs)
			if err != nil {
				return Undefined, err
			}
			pc = kept.pc
			if t.sameBits(True) {
				w = regAt(code, pc-1)
				if off := w.SC(); off >= 0 {
					pc += int(off)
				} else if r.backEdges--; r.backEdges > 0 {
					pc += int(off)
				} else {
					kept.pc = pc
					if err := r.regBackEdge(f, c, pc-1); err != nil {
						return Undefined, err
					}
					pc = kept.pc + int(regAt(code, kept.pc-1).SC())
				}
			}
		case bytecode.RJSEq:
			if regs[w.A()].StrictEquals(regs[w.B()]) {
				if off := w.SC(); off >= 0 {
					pc += int(off)
				} else if r.backEdges--; r.backEdges > 0 {
					pc += int(off)
				} else {
					kept.pc = pc
					if err := r.regBackEdge(f, c, pc-1); err != nil {
						return Undefined, err
					}
					pc = kept.pc + int(regAt(code, kept.pc-1).SC())
				}
			}
		case bytecode.RJSNe:
			if !regs[w.A()].StrictEquals(regs[w.B()]) {
				if off := w.SC(); off >= 0 {
					pc += int(off)
				} else if r.backEdges--; r.backEdges > 0 {
					pc += int(off)
				} else {
					kept.pc = pc
					if err := r.regBackEdge(f, c, pc-1); err != nil {
						return Undefined, err
					}
					pc = kept.pc + int(regAt(code, kept.pc-1).SC())
				}
			}
		case bytecode.RJSEqI:
			if regs[w.A()].StrictEquals(Int32(w.SB())) {
				if off := w.SC(); off >= 0 {
					pc += int(off)
				} else if r.backEdges--; r.backEdges > 0 {
					pc += int(off)
				} else {
					kept.pc = pc
					if err := r.regBackEdge(f, c, pc-1); err != nil {
						return Undefined, err
					}
					pc = kept.pc + int(regAt(code, kept.pc-1).SC())
				}
			}
		case bytecode.RJSNeI:
			if !regs[w.A()].StrictEquals(Int32(w.SB())) {
				if off := w.SC(); off >= 0 {
					pc += int(off)
				} else if r.backEdges--; r.backEdges > 0 {
					pc += int(off)
				} else {
					kept.pc = pc
					if err := r.regBackEdge(f, c, pc-1); err != nil {
						return Undefined, err
					}
					pc = kept.pc + int(regAt(code, kept.pc-1).SC())
				}
			}
		// Every value of the switch is a case, so that it is a jump table
		// that needs no range check: Go makes one only of a switch whose
		// cases are dense, and the mask tells it the value is below 128. A
		// new operation takes its value out of this list, or the switch does
		// not compile.
		case 79:
			goto invalid
		case 80:
			goto invalid
		case 81:
			goto invalid
		case 82:
			goto invalid
		case 83:
			goto invalid
		case 84:
			goto invalid
		case 85:
			goto invalid
		case 86:
			goto invalid
		case 87:
			goto invalid
		case 88:
			goto invalid
		case 89:
			goto invalid
		case 90:
			goto invalid
		case 91:
			goto invalid
		case 92:
			goto invalid
		case 93:
			goto invalid
		case 94:
			goto invalid
		case 95:
			goto invalid
		case 96:
			goto invalid
		case 97:
			goto invalid
		case 98:
			goto invalid
		case 99:
			goto invalid
		case 100:
			goto invalid
		case 101:
			goto invalid
		case 102:
			goto invalid
		case 103:
			goto invalid
		case 104:
			goto invalid
		case 105:
			goto invalid
		case 106:
			goto invalid
		case 107:
			goto invalid
		case 108:
			goto invalid
		case 109:
			goto invalid
		case 110:
			goto invalid
		case 111:
			goto invalid
		case 112:
			goto invalid
		case 113:
			goto invalid
		case 114:
			goto invalid
		case 115:
			goto invalid
		case 116:
			goto invalid
		case 117:
			goto invalid
		case 118:
			goto invalid
		case 119:
			goto invalid
		case 120:
			goto invalid
		case 121:
			goto invalid
		case 122:
			goto invalid
		case 123:
			goto invalid
		case 124:
			goto invalid
		case 125:
			goto invalid
		case 126:
			goto invalid
		case 127:
			goto invalid
		}
		continue
	invalid:
		panic("register code: invalid instruction")
	}
}

// regNaN is the canonical NaN as a Value, in a variable rather than a
// constant: Go's loop-invariant pass takes a constant out of the loop into a
// register of its own, one for each use, which the loop then saves and
// restores around every case that calls anything.
var regNaN = Value{num: math.Float64frombits(canonicalNaN)}

// regNum is Float, its NaN read from regNaN.
func regNum(f float64) Value {
	if f != f {
		return regNaN
	}
	return Value{num: f}
}

// regInt32 is v as an int32, where it is a number that is one exactly:
// what a bitwise operator's fast path takes without converting.
func regInt32(v Value) (int32, bool) {
	if v.IsNumber() {
		f := v.Number()
		if x := int32(f); float64(x) == f {
			return x, true
		}
	}
	return 0, false
}

// regKept is what the loop keeps in memory across a call: a struct Go does
// not keep in registers (it has an array of more than one element), so that
// what is stored in it is stored where the loop says, not at every dispatch.
type regKept struct {
	pc  int
	pad [2]int
}

// regBinImm is bin_imm's slow path.
func (r *Runtime) regBinImm(op bytecode.Op, a Value, imm int32) (Value, error) {
	if a.IsNumber() {
		x, y := a.Number(), imm
		switch op {
		case bytecode.OpAdd:
			return Float(x + float64(y)), nil
		case bytecode.OpSub:
			return Float(x - float64(y)), nil
		case bytecode.OpMul:
			return Float(x * float64(y)), nil
		}
		xi := int32(x)
		if float64(xi) != x {
			xi = jsnum.ToInt32(x)
		}
		switch op {
		case bytecode.OpBitAnd:
			return Int32(xi & y), nil
		case bytecode.OpBitOr:
			return Int32(xi | y), nil
		case bytecode.OpBitXor:
			return Int32(xi ^ y), nil
		case bytecode.OpShl:
			return Int32(xi << (uint32(y) & 31)), nil
		case bytecode.OpShr:
			return Int32(xi >> (uint32(y) & 31)), nil
		default:
			return Uint32(uint32(xi) >> (uint32(y) & 31)), nil
		}
	}
	return r.binImm(op, a, Int32(imm))
}

// regBinary is a binary operator the loop has no case of its own for.
func (r *Runtime) regBinary(op bytecode.Op, a, b Value) (Value, error) {
	switch op {
	case bytecode.OpDiv, bytecode.OpMod, bytecode.OpPow:
		if a.IsNumber() && b.IsNumber() {
			if op == bytecode.OpDiv {
				return Float(a.Number() / b.Number()), nil
			}
			return Float(numericOp(op, a.Number(), b.Number())), nil
		}
		return r.arith(op, a, b)
	case bytecode.OpBitAnd, bytecode.OpBitOr, bytecode.OpBitXor, bytecode.OpShl, bytecode.OpShr, bytecode.OpUShr:
		if a.IsNumber() && b.IsNumber() {
			fa, fb := a.Number(), b.Number()
			x, y := int32(fa), int32(fb)
			if float64(x) != fa {
				x = jsnum.ToInt32(fa)
			}
			if float64(y) != fb {
				y = jsnum.ToInt32(fb)
			}
			switch op {
			case bytecode.OpBitAnd:
				return Int32(x & y), nil
			case bytecode.OpBitOr:
				return Int32(x | y), nil
			case bytecode.OpBitXor:
				return Int32(x ^ y), nil
			case bytecode.OpShl:
				return Int32(x << (uint32(y) & 31)), nil
			case bytecode.OpShr:
				return Int32(x >> (uint32(y) & 31)), nil
			}
			return Uint32(uint32(x) >> (uint32(y) & 31)), nil
		}
		return r.bitwise(op, a, b)
	}
	res, err := r.regCompare(op, a, b)
	return Bool(res), err
}

// regCompare is a comparison: an equality or a relational operator.
func (r *Runtime) regCompare(op bytecode.Op, a, b Value) (bool, error) {
	switch op {
	case bytecode.OpStrictEq:
		return a.StrictEquals(b), nil
	case bytecode.OpStrictNe:
		return !a.StrictEquals(b), nil
	case bytecode.OpEq, bytecode.OpNe:
		eq, err := r.looseEquals(a, b)
		return eq == (op == bytecode.OpEq), err
	}
	if a.IsNumber() && b.IsNumber() {
		return compareFloats(op, a.Number(), b.Number()), nil
	}
	c, err := r.compare(a, b)
	if err != nil {
		return false, err
	}
	return relationalResult(op, c), nil
}

// regStep is ++ or -- of a value that is not a number.
func (r *Runtime) regStep(a Value, delta float64) (Value, error) {
	n, err := r.toNumeric(a)
	if err != nil {
		return Undefined, err
	}
	if n.IsBigInt() {
		return r.arith(bytecode.OpAdd, n, shortBig(int64(delta)))
	}
	return Float(n.Number() + delta), nil
}

// regUnary is a unary operator.
func (r *Runtime) regUnary(op bytecode.Op, a Value) (Value, error) {
	switch op {
	case bytecode.OpNeg:
		if a.IsNumber() {
			return Float(-a.Number()), nil
		}
		return r.negate(a)
	case bytecode.OpPos:
		n, err := r.toNumber(a)
		return Float(n), err
	case bytecode.OpNot:
		return Bool(!regTruthy(a)), nil
	case bytecode.OpTypeOf:
		return Str(r.typeofString(a)), nil
	}
	// bit_not.
	if !a.IsNumber() {
		n, err := r.toNumeric(a)
		if err != nil {
			return Undefined, err
		}
		if n.IsBigInt() {
			return bigNot(n), nil
		}
		a = n
	}
	x, err := r.toInt32(a)
	return Int32(^x), err
}

// regBackEdge is a backward jump's interrupt check, when the budget is
// spent.
func (r *Runtime) regBackEdge(f *frame, c *regCode, pc int) error {
	r.backEdges = backEdgeCheckInterval
	f.pc = c.src[pc] + 1
	if err := r.checkInterruptNow(); err != nil {
		return err
	}
	r.sweepStaleSlots(f.base - int(c.locals) + int(c.numRegs))
	return nil
}
