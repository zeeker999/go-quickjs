package vm

import (
	"math"
	"sync/atomic"
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
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
// bounds check: the register backend's verifier (regcomp.Verify)
// proves every register an instruction names is below the frame's count,
// and a frame is given that many slots before it runs. The code is read the
// same way, every jump proven to land in it.
//
// The files sort after every other of the package's, so that the functions
// they add come after every function there was: the interpreter's hot
// functions keep their addresses, and only the closures move.

// regTier turns the register interpreter on for functions not yet run.
var regTier atomic.Bool

// regsBuilt counts the functions given register code.
var regsBuilt atomic.Int64

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
	wide bool
	// keep is the first parameter whose argument f.arguments reads as
	// the call passed it, which the call keeps if the frame may overwrite
	// it (regKeepWindow): params, but in a sloppy function whose
	// parameter list is not plain (0) or one of whose parameters a
	// closure captures (the first such). It fills the padding after wide.
	keep  uint16
	inits int32
	// ncap is how many of the function's stack slots a closure captures,
	// the cells a closure it makes is given (regClosure).
	ncap int32
	// newTarget says the function reads new.target, which its frame keeps.
	newTarget bool
	// args says a frame keeps its arguments as passed, for RArguments.
	args bool
	// suspends marks a generator's or an async function's code, which
	// only a generator's resumption runs (zzreg_zgen.go).
	suspends bool
	// leaf is what a call may answer without a frame: a pure body that
	// calls nothing, or one that only reads or writes its this's
	// properties; read here rather than in the function, a line away.
	leaf uint8
	// leafArg is the parameter a regLeafArg body returns, and leafVal the
	// value a regLeafConst body does.
	leafArg int32
	leafVal Value
	// leafOps is a regLeafCode body's code as regLeafRun runs it.
	leafOps []regLeafOp
	// switches is each switch instruction's table, made from the code's.
	switches []*regSwitchTable
	rc       *bytecode.RegCode
	fn       *bytecode.Function
	// reuse is the sites of the code's calls of exec marked RMarkReuse
	// whose arrays may be made again, by call site (regReuseSites).
	reuse []*execReuseSite
}

// runReg runs frame f as register code c, from instruction start, and
// returns what it returns. Only a generator's resumption starts past the
// first instruction (regGenRun).
//
// A case that finishes its instruction continues the loop; one that can
// fail sets the frame's pc to the stack form's instruction first, which is
// what a stack trace or an error's position reads, keeps the pc, and goes
// to the fail tail with the error, where a handler of the frame's may
// catch it (regCatch).
func (r *Runtime) runReg(f *frame, c *regCode, start int) (Value, error) {
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
	var failed error
	pc := start
	for {
		var off int32
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
			regs[w.A()] = Value{num: mkTag(KindString, 0), ref: unsafe.Pointer(emptyString)}
		case bytecode.RLoadUninit:
			regs[w.A()] = uninitialized
		case bytecode.RLoadThis:
			regs[w.A()] = f.this
		case bytecode.RCheckRef:
			if regTagged(math.Float64bits(regs[w.A()].num), KindUninitialized) {
				kept.pc = pc
				failed = r.regThrowDeadZone(f, c, pc)
				goto fail
			}
		case bytecode.RGetUpval:
			regs[w.A()] = cl.upvalues[w.BC()].get()
		case bytecode.RSetUpval:
			cl.upvalues[w.BC()].set(regs[w.A()])

		case bytecode.RGetGlobal:
			// The global object's slot the site remembers, while the global
			// epoch is the one it was filled at (regNoteGlobal): no binding
			// of the global object has moved, gone or changed its attributes
			// since, and no script-level lexical binding shadows it, nor an
			// eval's variable, which a frame of the register form never has.
			if ep := cl.regEpochs; uint(w.C()) < uint(len(ep)) && ep[w.C()] == r.shapes.globalEpoch+1 {
				s := &cl.ic[w.C()]
				if env, i := s.p1, uint(s.idx); i < uint(env.valn) {
					regs[w.A()] = env.valAt(int(i))
					continue
				}
			}
			goto slow
		case bytecode.RSetGlobal:
			if ep := cl.regEpochs; uint(w.C()) < uint(len(ep)) && ep[w.C()] == r.shapes.globalEpoch+1 {
				s := &cl.ic[w.C()]
				if env, i := s.p1, uint(s.idx); i < uint(env.valn) {
					env.vals()[i] = regs[w.A()]
					continue
				}
			}
			kept.pc = pc
			if _, err := r.regSlow(f, c, &kept, regs); err != nil {
				failed = err
				goto fail
			}
			pc = kept.pc

		case bytecode.RGetProp, bytecode.RGetMethod:
			if o := regs[w.B()]; o.IsObject() {
				obj := o.Object()
				s := &cl.ic[w.C()]
				if obj.shape == s.ownShape {
					regs[w.A()] = obj.valAt(int(s.idx))
					continue
				}
				// A prototype's data property the site remembers: a method,
				// in the holder propCache.holder finds, written out.
				if obj.shape == s.shape {
					h := obj
					if s.cell != nil {
						h = nil
						if s.cell.valid {
							h = s.p2
						}
					} else if p1 := s.p1; p1 != nil {
						h = nil
						if p1.shape == s.s1 {
							h = p1
							if q := s.p2; q != nil {
								h = nil
								if q.shape == s.s2 {
									h = q
								}
							}
						}
					}
					if h != nil && !s.getter {
						regs[w.A()] = h.valAt(int(s.idx))
						continue
					}
				}
			}
			goto slow
		case bytecode.RGetMethodW:
			// The receiver put in the call's window beside its method.
			if o := regs[w.B()]; o.IsObject() {
				obj := o.Object()
				s := &cl.ic[w.C()]
				p := (*[2]Value)(unsafe.Pointer(&regs[w.A()]))
				if obj.shape == s.ownShape {
					p[0], p[1] = obj.valAt(int(s.idx)), o
					continue
				}
				if obj.shape == s.shape {
					h := obj
					if s.cell != nil {
						h = nil
						if s.cell.valid {
							h = s.p2
						}
					} else if p1 := s.p1; p1 != nil {
						h = nil
						if p1.shape == s.s1 {
							h = p1
							if q := s.p2; q != nil {
								h = nil
								if q.shape == s.s2 {
									h = q
								}
							}
						}
					}
					if h != nil && !s.getter {
						p[0], p[1] = h.valAt(int(s.idx)), o
						continue
					}
				}
			}
			goto slow
		case bytecode.RSetProp:
			if o := regs[w.A()]; o.IsObject() {
				obj := o.Object()
				if s := &cl.ic[w.C()]; obj.shape == s.ownShape {
					obj.vals()[s.idx] = regs[w.B()]
					continue
				}
			}
			goto slow
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
			goto typed
		case bytecode.RGetIndexPre:
			// x[++i]: the key stepped, then the element read; neither is
			// written until both are known, so that the slow path does both.
			obj, key := regs[w.B()], regs[w.C()]
			if obj.IsObject() && key.IsNumber() {
				k := key.Number() + float64(int8(w.X()))
				o := obj.Object()
				if i := uint32(k); float64(i) == k &&
					uint(i) < uint(len(o.elemsOf())) && o.flags&objMappedArguments == 0 {
					if v := o.elemsOf()[i]; !isHole(v) {
						regs[w.C()] = Value{num: k}
						regs[w.A()] = v
						continue
					}
				}
			}
			goto slow
		case bytecode.RGetIndexPost:
			// x[i++]: the element read at the key, then the key stepped;
			// neither is written until both are known.
			obj, key := regs[w.B()], regs[w.C()]
			if obj.IsObject() && key.IsNumber() {
				o := obj.Object()
				if i := uint32(key.Number()); float64(i) == key.Number() &&
					uint(i) < uint(len(o.elemsOf())) && o.flags&objMappedArguments == 0 {
					if v := o.elemsOf()[i]; !isHole(v) {
						regs[w.C()] = Value{num: key.Number() + float64(int8(w.X()))}
						regs[w.A()] = v
						continue
					}
				}
			}
			goto slow
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
			if r.regSetMiss(c, &kept, regs) {
				pc = kept.pc
				continue
			}
			if _, err := r.regSlow(f, c, &kept, regs); err != nil {
				failed = err
				goto fail
			}
			pc = kept.pc

		case bytecode.RAdd:
			a, b := regs[w.B()], regs[w.C()]
			if a.IsNumber() && b.IsNumber() {
				regs[w.A()] = regNum(a.Number() + b.Number())
				continue
			}
			goto slow
		case bytecode.RSub:
			a, b := regs[w.B()], regs[w.C()]
			if a.IsNumber() && b.IsNumber() {
				regs[w.A()] = regNum(a.Number() - b.Number())
				continue
			}
			goto slow
		case bytecode.RMul:
			a, b := regs[w.B()], regs[w.C()]
			if a.IsNumber() && b.IsNumber() {
				regs[w.A()] = regNum(a.Number() * b.Number())
				continue
			}
			goto slow
		case bytecode.RInc:
			if a := regs[w.B()]; a.IsNumber() {
				regs[w.A()] = regNum(a.Number() + 1)
				continue
			}
			goto slow
		case bytecode.RDec:
			if a := regs[w.B()]; a.IsNumber() {
				regs[w.A()] = regNum(a.Number() - 1)
				continue
			}
			goto slow
		case bytecode.RBin, bytecode.RUnary, bytecode.RDefineField, bytecode.RNewObject,
			bytecode.RNewArray, bytecode.RCold, bytecode.RGetPrivate, bytecode.RSetPrivate,
			bytecode.RAddIndex, bytecode.RAddIndexI:
			// What the loop has no fast path of its own for, cold's
			// operations, which may throw (regColdSub), and what runs
			// per turn of no loop the suites have: a[i] += v and a[i] += 1
			// of a key in a register (regAddIndex).
			kept.pc = pc
			if _, err := r.regSlow(f, c, &kept, regs); err != nil {
				failed = err
				goto fail
			}
			pc = kept.pc

		case bytecode.RJmp:
			off = int32(w.BC())
			goto jump
		case bytecode.RJmpF:
			// What needs a call is the compare's slow path's, so that
			// nothing of the instruction is live across one. ToBoolean of
			// what needs no call -- a boolean, an object, a number --
			// written out in each.
			v := regs[w.A()]
			var t bool
			switch b := math.Float64bits(v.num); {
			case b == trueBits:
				t = true
			case b == falseBits:
			case b == objectBits:
				t = (*Object)(v.ref).flags&objHTMLDDA == 0
			case v.IsNumber():
				t = v.num != 0 && v.num == v.num
			default:
				goto cmpSlow
			}
			if !t {
				off = int32(w.BC())
				goto jump
			}
		case bytecode.RJmpT:
			v := regs[w.A()]
			var t bool
			switch b := math.Float64bits(v.num); {
			case b == trueBits:
				t = true
			case b == falseBits:
			case b == objectBits:
				t = (*Object)(v.ref).flags&objHTMLDDA == 0
			case v.IsNumber():
				t = v.num != 0 && v.num == v.num
			default:
				goto cmpSlow
			}
			if t {
				off = int32(w.BC())
				goto jump
			}
		case bytecode.RRet:
			return regs[w.A()], nil
		case bytecode.RRetUndef:
			return Undefined, nil
		case bytecode.RAndI:
			if a := regs[w.B()]; a.IsNumber() && regIsInt32(a.num) {
				x := int32(a.num)
				regs[w.A()] = Int32(x & w.SC())
				continue
			}
			goto slow
		case bytecode.ROrI:
			if a := regs[w.B()]; a.IsNumber() && regIsInt32(a.num) {
				x := int32(a.num)
				regs[w.A()] = Int32(x | w.SC())
				continue
			}
			goto slow
		case bytecode.RXorI:
			if a := regs[w.B()]; a.IsNumber() && regIsInt32(a.num) {
				x := int32(a.num)
				regs[w.A()] = Int32(x ^ w.SC())
				continue
			}
			goto slow
		case bytecode.RShlI:
			if a := regs[w.B()]; a.IsNumber() && regIsInt32(a.num) {
				x := int32(a.num)
				regs[w.A()] = Int32(x << (uint32(w.SC()) & 31))
				continue
			}
			goto slow
		case bytecode.RShrI:
			if a := regs[w.B()]; a.IsNumber() && regIsInt32(a.num) {
				x := int32(a.num)
				regs[w.A()] = Int32(x >> (uint32(w.SC()) & 31))
				continue
			}
			goto slow
		case bytecode.RUShrI:
			if a := regs[w.B()]; a.IsNumber() && regIsInt32(a.num) {
				x := int32(a.num)
				regs[w.A()] = Uint32(uint32(x) >> (uint32(w.SC()) & 31))
				continue
			}
			goto slow
		case bytecode.RBitAnd:
			if a := regs[w.B()]; a.IsNumber() && regIsInt32(a.num) {
				if b := regs[w.C()]; b.IsNumber() && regIsInt32(b.num) {
					x, y := int32(a.num), int32(b.num)
					regs[w.A()] = Int32(x & y)
					continue
				}
			}
			goto slow
		case bytecode.RBitOr:
			if a := regs[w.B()]; a.IsNumber() && regIsInt32(a.num) {
				if b := regs[w.C()]; b.IsNumber() && regIsInt32(b.num) {
					x, y := int32(a.num), int32(b.num)
					regs[w.A()] = Int32(x | y)
					continue
				}
			}
			goto slow
		case bytecode.RBitXor:
			if a := regs[w.B()]; a.IsNumber() && regIsInt32(a.num) {
				if b := regs[w.C()]; b.IsNumber() && regIsInt32(b.num) {
					x, y := int32(a.num), int32(b.num)
					regs[w.A()] = Int32(x ^ y)
					continue
				}
			}
			goto slow
		case bytecode.RShl:
			if a := regs[w.B()]; a.IsNumber() && regIsInt32(a.num) {
				if b := regs[w.C()]; b.IsNumber() && regIsInt32(b.num) {
					x, y := int32(a.num), int32(b.num)
					regs[w.A()] = Int32(x << (uint32(y) & 31))
					continue
				}
			}
			goto slow
		case bytecode.RShr:
			if a := regs[w.B()]; a.IsNumber() && regIsInt32(a.num) {
				if b := regs[w.C()]; b.IsNumber() && regIsInt32(b.num) {
					x, y := int32(a.num), int32(b.num)
					regs[w.A()] = Int32(x >> (uint32(y) & 31))
					continue
				}
			}
			goto slow
		case bytecode.RUShr:
			if a := regs[w.B()]; a.IsNumber() && regIsInt32(a.num) {
				if b := regs[w.C()]; b.IsNumber() && regIsInt32(b.num) {
					x, y := int32(a.num), int32(b.num)
					regs[w.A()] = Uint32(uint32(x) >> (uint32(y) & 31))
					continue
				}
			}
			goto slow
		case bytecode.RCallW:
			kept.pc = pc
			if err := r.regCallW(f, c, &kept, wb); err != nil {
				failed = err
				goto fail
			}
			pc = kept.pc
		case bytecode.RAddI:
			if a := regs[w.B()]; a.IsNumber() {
				regs[w.A()] = regNum(a.Number() + float64(w.SC()))
				continue
			}
			goto slow
		case bytecode.RSubI:
			if a := regs[w.B()]; a.IsNumber() {
				regs[w.A()] = regNum(a.Number() - float64(w.SC()))
				continue
			}
			goto slow
		case bytecode.RJLt:
			if a, b := regs[w.A()], regs[w.B()]; a.IsNumber() && b.IsNumber() {
				if a.Number() < b.Number() {
					goto taken
				}
				continue
			}
			goto cmpSlow
		case bytecode.RJLtU:
			if a, b := regs[w.A()], cl.upvalues[w.B()].get(); a.IsNumber() && b.IsNumber() {
				if a.Number() < b.Number() {
					goto taken
				}
				continue
			}
			goto cmpSlow
		case bytecode.RJNLtU:
			if a, b := regs[w.A()], cl.upvalues[w.B()].get(); a.IsNumber() && b.IsNumber() {
				if !(a.Number() < b.Number()) {
					goto taken
				}
				continue
			}
			goto cmpSlow
		case bytecode.RJLeU:
			if a, b := regs[w.A()], cl.upvalues[w.B()].get(); a.IsNumber() && b.IsNumber() {
				if a.Number() <= b.Number() {
					goto taken
				}
				continue
			}
			goto cmpSlow
		case bytecode.RJNLeU:
			if a, b := regs[w.A()], cl.upvalues[w.B()].get(); a.IsNumber() && b.IsNumber() {
				if !(a.Number() <= b.Number()) {
					goto taken
				}
				continue
			}
			goto cmpSlow
		case bytecode.RJLtI:
			if a := regs[w.A()]; a.IsNumber() {
				if a.Number() < float64(w.SB()) {
					goto taken
				}
				continue
			}
			goto cmpSlow
		case bytecode.RJNLt:
			if a, b := regs[w.A()], regs[w.B()]; a.IsNumber() && b.IsNumber() {
				if !(a.Number() < b.Number()) {
					goto taken
				}
				continue
			}
			goto cmpSlow
		case bytecode.RJNLtI:
			if a := regs[w.A()]; a.IsNumber() {
				if !(a.Number() < float64(w.SB())) {
					goto taken
				}
				continue
			}
			goto cmpSlow
		case bytecode.RJLe:
			if a, b := regs[w.A()], regs[w.B()]; a.IsNumber() && b.IsNumber() {
				if a.Number() <= b.Number() {
					goto taken
				}
				continue
			}
			goto cmpSlow
		case bytecode.RJLeI:
			if a := regs[w.A()]; a.IsNumber() {
				if a.Number() <= float64(w.SB()) {
					goto taken
				}
				continue
			}
			goto cmpSlow
		case bytecode.RJNLe:
			if a, b := regs[w.A()], regs[w.B()]; a.IsNumber() && b.IsNumber() {
				if !(a.Number() <= b.Number()) {
					goto taken
				}
				continue
			}
			goto cmpSlow
		case bytecode.RJNLeI:
			if a := regs[w.A()]; a.IsNumber() {
				if !(a.Number() <= float64(w.SB())) {
					goto taken
				}
				continue
			}
			goto cmpSlow
		case bytecode.RJGt:
			if a, b := regs[w.A()], regs[w.B()]; a.IsNumber() && b.IsNumber() {
				if a.Number() > b.Number() {
					goto taken
				}
				continue
			}
			goto cmpSlow
		case bytecode.RJGtI:
			if a := regs[w.A()]; a.IsNumber() {
				if a.Number() > float64(w.SB()) {
					goto taken
				}
				continue
			}
			goto cmpSlow
		case bytecode.RJNGt:
			if a, b := regs[w.A()], regs[w.B()]; a.IsNumber() && b.IsNumber() {
				if !(a.Number() > b.Number()) {
					goto taken
				}
				continue
			}
			goto cmpSlow
		case bytecode.RJNGtI:
			if a := regs[w.A()]; a.IsNumber() {
				if !(a.Number() > float64(w.SB())) {
					goto taken
				}
				continue
			}
			goto cmpSlow
		case bytecode.RJGe:
			if a, b := regs[w.A()], regs[w.B()]; a.IsNumber() && b.IsNumber() {
				if a.Number() >= b.Number() {
					goto taken
				}
				continue
			}
			goto cmpSlow
		case bytecode.RJGeI:
			if a := regs[w.A()]; a.IsNumber() {
				if a.Number() >= float64(w.SB()) {
					goto taken
				}
				continue
			}
			goto cmpSlow
		case bytecode.RJNGe:
			if a, b := regs[w.A()], regs[w.B()]; a.IsNumber() && b.IsNumber() {
				if !(a.Number() >= b.Number()) {
					goto taken
				}
				continue
			}
			goto cmpSlow
		case bytecode.RJNGeI:
			if a := regs[w.A()]; a.IsNumber() {
				if !(a.Number() >= float64(w.SB())) {
					goto taken
				}
				continue
			}
			goto cmpSlow
		case bytecode.RJEq, bytecode.RJEqI:
			// B is the immediate form's operand, no register: read only
			// where it names one, which the verifier proves in the frame.
			if a := regs[w.A()]; w.Op() == bytecode.RJEqI {
				if a.IsNumber() {
					if a.Number() == float64(w.SB()) {
						goto taken
					}
					continue
				}
			} else if b := regs[w.B()]; a.IsNumber() && b.IsNumber() {
				if a.Number() == b.Number() {
					goto taken
				}
				continue
			}
			goto cmpSlow
		case bytecode.RJNe, bytecode.RJNeI:
			// B is the immediate form's operand, no register: read only
			// where it names one, which the verifier proves in the frame.
			if a := regs[w.A()]; w.Op() == bytecode.RJNeI {
				if a.IsNumber() {
					if a.Number() != float64(w.SB()) {
						goto taken
					}
					continue
				}
			} else if b := regs[w.B()]; a.IsNumber() && b.IsNumber() {
				if a.Number() != b.Number() {
					goto taken
				}
				continue
			}
			goto cmpSlow
		case bytecode.RJNull, bytecode.RJNotNull:
			// An object other than document.all's stand-in is not == null.
			if a := regs[w.A()]; !a.IsObject() || a.object().flags&objHTMLDDA == 0 {
				// Value.IsNullish, written out.
				b := math.Float64bits(a.num)
				k := Kind(b & 0xFF)
				if (b&tagMask == tagBase && (k == KindNull || k == KindUndefined)) == (w.Op() == bytecode.RJNull) {
					goto taken
				}
				continue
			}
			goto cmpSlow
		case bytecode.RJSEq, bytecode.RJSNe:
			// Two numbers, or two values one of which is undefined, null, a
			// boolean, a symbol or an object, equal only if their words
			// are; two strings or BigInts are the compare's slow path's.
			a, b := regs[w.A()], regs[w.B()]
			var eq bool
			if a.IsNumber() && b.IsNumber() {
				eq = a.num == b.num
			} else if x, y := math.Float64bits(a.num), math.Float64bits(b.num); regWordKind(x) || regWordKind(y) {
				eq = x == y && a.ref == b.ref
			} else {
				goto cmpSlow
			}
			if eq == (w.Op() == bytecode.RJSEq) {
				goto taken
			}
		case bytecode.RJSEqI:
			// Only a number is === a number.
			if a := regs[w.A()]; a.IsNumber() && a.Number() == float64(w.SB()) {
				goto taken
			}
		case bytecode.RJSNeI:
			if a := regs[w.A()]; !a.IsNumber() || a.Number() != float64(w.SB()) {
				goto taken
			}
		case bytecode.RGetIndexI:
			if obj := regs[w.B()]; obj.IsObject() {
				o := obj.Object()
				if i := uint(w.C()); i < uint(len(o.elemsOf())) && o.flags&objMappedArguments == 0 {
					if v := o.elemsOf()[i]; !isHole(v) {
						regs[w.A()] = v
						continue
					}
				}
			}
			goto typed
		case bytecode.RSetIndexI:
			if obj := regs[w.A()]; obj.IsObject() {
				o := obj.Object()
				if i := uint(w.B()); i < uint(len(o.elemsOf())) && o.flags&(objMappedArguments|objSharedElems) == 0 && !isHole(o.elemsOf()[i]) {
					o.extOf().elems[i] = regs[w.C()]
					continue
				}
			}
			goto typed
		case bytecode.RCallMW:
			kept.pc = pc
			if err := r.regCallW(f, c, &kept, wb); err != nil {
				failed = err
				goto fail
			}
			pc = kept.pc
		case bytecode.RPostInc:
			if a := regs[w.B()]; a.IsNumber() {
				regs[w.A()] = a
				regs[w.B()] = regNum(a.Number() + 1)
				continue
			}
			goto slow
		case bytecode.RPostDec:
			if a := regs[w.B()]; a.IsNumber() {
				regs[w.A()] = a
				regs[w.B()] = regNum(a.Number() - 1)
				continue
			}
			goto slow
		case bytecode.RGetLength:
			if o := regs[w.B()]; o.IsObject() && o.object().class == ClassArray {
				// Object.arrayLength, written out.
				obj := o.object()
				n := uint32(len(obj.elemsOf()))
				if obj.flags&objHasSparseElements != 0 && obj.arrayExt().arrayLen > n {
					n = obj.arrayExt().arrayLen
				}
				regs[w.A()] = Uint32(n)
				continue
			} else if regTagged(math.Float64bits(o.num), KindString) {
				regs[w.A()] = Int((*String)(o.ref).Len())
				continue
			}
			goto slow
		case bytecode.RAddProp, bytecode.RAddPropI:
			if o := regs[w.A()]; o.IsObject() {
				obj := o.Object()
				if s := &cl.ic[w.C()]; obj.shape == s.ownShape {
					p := &obj.vals()[s.idx]
					var b Value
					if w.Op() == bytecode.RAddProp {
						b = regs[w.B()]
					} else {
						b = Int32(w.SB())
					}
					if p.IsNumber() && b.IsNumber() {
						p.num = regNum(p.Number() + b.Number()).num
						continue
					}
				}
			}
			goto slow
		case bytecode.RAddIndexK:
			if o, b := regs[w.A()], regs[w.B()]; o.IsObject() && b.IsNumber() {
				obj := o.Object()
				if i, es := uint(w.C()), obj.elemsOf(); i < uint(len(es)) && obj.flags&(objMappedArguments|objSharedElems) == 0 {
					if p := &es[i]; p.IsNumber() {
						p.num = regNum(p.Number() + b.Number()).num
						continue
					}
				}
			}
			goto slow
		case bytecode.RNewArrayE, bytecode.RSwitch, bytecode.RSwitchTypeof, bytecode.RTypeof, bytecode.RNewObjectLit:
			// Operations not run per turn of a hot loop: one case, which
			// moves none of the others' code, and a second dispatch.
			kept.pc = pc
			r.regCold(f, c, &kept, regs)
			pc = kept.pc
		case bytecode.RJTypeof:
			// An object's type asked inline where it is a function's or
			// an ordinary object's; a proxy and document.all's kind are
			// the slow path's.
			v := regs[w.A()]
			// typeof's answer of v's kind (Value.Kind, regTypeOf), written
			// out.
			var k Kind
			switch bits := math.Float64bits(v.num); {
			case bits>>51 != tagBase>>51:
				k = KindNumber
			case bits >= heapBigBits:
				k = KindBigInt
			default:
				k = Kind(bits & 0xFF)
			}
			t := uint8(typeUndefined)
			switch k {
			case KindNumber:
				t = typeNumber
			case KindNull, KindObject:
				t = typeObject
			case KindBool:
				t = typeBoolean
			case KindString:
				t = typeString
			case KindSymbol:
				t = typeSymbol
			case KindBigInt:
				t = typeBigInt
			}
			if v.IsObject() {
				o := v.object()
				d := o.dataOf()
				if _, isProxy := d.(*proxyData); isProxy || o.flags&objHTMLDDA != 0 {
					goto cmpSlow
				}
				if _, isFn := d.(*funcData); isFn {
					t = typeFunction
				}
			}
			if (t == w.X()&^bytecode.RCmpTrue) == (w.X()&bytecode.RCmpTrue != 0) {
				goto taken
			}
		case bytecode.RThrow:
			kept.pc = pc
			failed = r.regThrow(f, c, pc, regs)
			goto fail
		case bytecode.RNewW, bytecode.RClosureW, bytecode.RInstanceOf, bytecode.RIn, bytecode.RDelete, bytecode.RArguments, bytecode.RNewArrayK,
			bytecode.RApplyArgs, bytecode.RArgsLen, bytecode.RArgsIndex, bytecode.RConcat:
			// What the loop has no fast path of its own for.
			goto slow
		case bytecode.RKeyOfBase:
			// A key whose conversion runs nothing is left as it is, and a
			// null or undefined object is the read's to report.
			if v := regs[w.C()]; regTagged(math.Float64bits(v.num), KindString) || v.IsNumber() || regTagged(math.Float64bits(v.num), KindSymbol) {
				regs[w.A()] = v
				continue
			}
			goto slow
		case bytecode.RStrPart:
			// A string or a number is joined as it is.
			if v := regs[w.A()]; regTagged(math.Float64bits(v.num), KindString) || v.IsNumber() {
				continue
			}
			goto slow
		case bytecode.RAppend:
			kept.pc = pc
			if err := r.regAppend(f, c, &kept, regs); err != nil {
				failed = err
				goto fail
			}
			pc = kept.pc
		case bytecode.RStrFreeze:
			// A read of a binding an append grows, which what read it may
			// keep: only a string an append owns is written.
			if v := regs[w.A()]; regTagged(math.Float64bits(v.num), KindString) {
				if s := (*String)(v.ref); s.building {
					s.building = false
				}
			}
		case bytecode.RCharCodeAt:
			kept.pc = pc
			if err := r.regCharCodeAt(f, c, &kept, regs); err != nil {
				failed = err
				goto fail
			}
			pc = kept.pc
		case bytecode.RIterNext:
			kept.pc = pc
			if err := r.regIterNext(f, c, &kept, regs); err != nil {
				failed = err
				goto fail
			}
			pc = kept.pc
		case bytecode.RLit:
			kept.pc = pc
			if err := r.regLit(f, c, &kept, regs); err != nil {
				failed = err
				goto fail
			}
			pc = kept.pc
		case bytecode.RYield:
			// A generator's step, suspended with what it yields: the
			// frame's suspension, which no handler of its own takes
			// (regCatch), as no other error leaves without the fail tail.
			kept.pc = pc
			failed = r.regYield(f, regs[w.B()], pc)
			goto fail
		// Every value of the switch is a case, so that it is a jump table
		// that needs no range check: Go makes one only of a switch whose
		// cases are dense, and the mask tells it the value is below 128. A
		// new operation takes its value out of this list, or the switch does
		// not compile: the values the translation reference's operations
		// had, and 127, invalid's.
		case bytecode.RCallWT, bytecode.RCallMWT:
			goto tail
		case 42:
			goto invalid
		case 127:
			goto invalid
		}
		continue
	taken:
		// A compare-and-branch taken: its offset is SC.
		off = w.SC()
	jump:
		// A jump taken, by off: one back counts the interrupt budget.
		if off < 0 {
			if r.backEdges--; r.backEdges <= 0 {
				kept.pc = pc
				if err := r.regBackEdge(f, c, &kept); err != nil {
					failed = err
					goto fail
				}
				pc = kept.pc
				// regJumpOff, written out.
				switch w := regAt(code, pc-1); w.Op() {
				case bytecode.RJmp, bytecode.RJmpF, bytecode.RJmpT:
					off = int32(w.BC())
				default:
					off = w.SC()
				}
			}
		}
		pc += int(off)
		continue
	cmpSlow:
		// A comparison of anything but two numbers, and its jump.
		kept.pc = pc
		if t, err := r.regSlow(f, c, &kept, regs); err != nil {
			failed = err
			goto fail
		} else if pc = kept.pc; math.Float64bits(t.num) == math.Float64bits(True.num) && t.ref == True.ref {
			w = regAt(code, pc-1)
			switch w.Op() {
			case bytecode.RJmp, bytecode.RJmpF, bytecode.RJmpT:
				off = int32(w.BC())
			default:
				off = w.SC()
			}
			goto jump
		}
		continue
	typed:
		// An element get_index and the immediate forms have not found
		// dense: a typed array's, read here (regTypedAt, regTypedLoad,
		// written out),
		// the operation and the key decoded again as a miss path may,
		// or out of line (regElemMiss), which set_index_imm's store
		// takes. Here rather than in get_index's case, whose code the
		// hot loops run: there it moved the loop's blocks, and empty
		// loops were 13% slower at every placement.
		if w.Op() != bytecode.RSetIndexI {
			if o := regs[w.B()]; o.IsObject() {
				k := float64(w.C())
				if w.Op() == bytecode.RGetIndex {
					k = regs[w.C()].num
				}
				if obj := o.Object(); obj.class == ClassTypedArray {
					if t, ok := obj.dataOf().(*typedArrayData); ok && t.epoch == r.typedEpoch && t.kind <= elemFloat64 {
						if i := uint32(k); float64(i) == k && uint(i) < uint(t.length) {
							p := unsafe.Add(t.data, uintptr(i)<<elemShifts[t.kind&15])
							var n float64
							ok := true
							switch t.kind {
							case elemInt32:
								n = float64(*(*int32)(p))
							case elemUint8, elemUint8Clamped:
								n = float64(*(*uint8)(p))
							case elemUint16:
								n = float64(*(*uint16)(p))
							case elemInt16:
								n = float64(*(*int16)(p))
							case elemInt8:
								n = float64(*(*int8)(p))
							case elemUint32:
								n = float64(*(*uint32)(p))
							case elemFloat64:
								n = *(*float64)(p)
							case elemFloat32:
								n = float64(*(*float32)(p))
							default:
								ok = false
							}
							if ok {
								regs[w.A()] = regNum(n)
								continue
							}
						}
					}
				}
			}
		}
		kept.pc = pc
		if regElemMiss(c, &kept, regs) {
			pc = kept.pc
			continue
		}
		pc = kept.pc
	slow:
		// What a case's fast path did not answer.
		kept.pc = pc
		if _, err := r.regSlow(f, c, &kept, regs); err != nil {
			failed = err
			goto fail
		}
		pc = kept.pc
		continue
	invalid:
		panic("register code: invalid instruction")
	fail:
		// An exception: the instruction at kept.pc-1 threw failed, which
		// a handler of the frame's may catch (regCatch).
		if pc = r.regCatch(f, c, &kept, regs, failed); pc < 0 {
			return Undefined, failed
		}
		continue
	tail:
		// tail_call_w and tail_call_method_w, after every other case and
		// tail: here they move none of the loop's code.
		kept.pc = pc
		if err := r.regTailCallW(f, c, &kept, wb); err != nil {
			failed = err
			goto fail
		}
		pc = kept.pc
	}
}

// regCallW is call_w and call_method_w: a call whose callee's frame
// starts at the caller's window, R[A], its parameters the arguments where
// they lie. It writes the call's value to R[B] itself, so that the loop
// keeps nothing of the instruction across the call.
//
// A callee the site called last, of the form's own layout and with nothing
// of its surroundings to establish but a this its cache holds, is entered
// here: its frame is the window, the frame record the few fields a stack
// trace and a closure read, and runReg is the only call between the two
// loops. What the frame needs is read from the callee's register code,
// whose facts are the function's, precomputed. Anything else is
// regCallMiss's.
func (r *Runtime) regCallW(f *frame, c *regCode, k *regKept, wb int) error {
	pc := k.pc - 1
	w := regAt(c.code, pc)
	base := wb + int(w.A())
	win := regWindow(r.stack, base)
	if calls := f.cl.regCalls; uint(w.C()) < uint(len(calls)) {
		e := &calls[w.C()]
		// The callee's tag and pointer read as words: read as a Value, the
		// tag goes through a float register to be compared.
		if regIsObject(&win[0]) && (*Object)(win[0].ref) == e.callee {
			// A leaf is answered first, before the frame's pc is written:
			// its body reads nothing of the frame and throws nothing.
			if e.kind&regCallLeaf != 0 {
				if leaf := e.rc.leaf; leaf >= regLeafArg {
					v := e.rc.leafVal
					if p := int(e.rc.leafArg); leaf == regLeafArg {
						v = Undefined
						if p < int(w.X()) {
							v = win[2+p]
						}
					}
					regWindow(r.stack, wb)[w.B()] = v
					return nil
				}
				if e.rc.leaf == regLeafCode {
					if v, ok := r.regLeafRun(e.cl, e.rc, base, int(w.X())); ok {
						regWindow(r.stack, wb)[w.B()] = v
						return nil
					}
					if e.cl.pureMiss >= pureMissLimit {
						// A body that keeps meeting what it cannot run
						// without a frame is entered with one from now on.
						e.kind &^= regCallOut | regCallLeaf
					}
					f.pc = regSrc(c, pc) + 1
					return r.regCallMiss(f, c, w, wb, pc)
				}
				// The this is given to the body, not written to the
				// window: a miss writes it (regCallSlow).
				this, ok := win[1], true
				switch e.kind &^ (regCallOut | regCallLeaf) {
				case regCallThis:
					this = e.this
				case regCallObjThis:
					ok = this.IsObject()
				}
				if ok {
					var v Value
					argc := int(w.X())
					if e.rc.leaf == regLeafPure {
						// pureCallAt checks the closure's misses.
						r.pureRefused = false
						v, ok = r.pureCallAt(e.callee.fn(), this, win[2:2+argc], 0, true)
					} else if ok = this.IsObject(); ok {
						v, ok = r.leafCall(e.cl, this.Object(), win[2:2+argc])
					}
					if ok {
						regWindow(r.stack, wb)[w.B()] = v
						return nil
					}
				}
				f.pc = regSrc(c, pc) + 1
				return r.regCallMiss(f, c, w, wb, pc)
			}
			f.pc = regSrc(c, pc) + 1
			if e.kind == regCallNative {
				return r.regNative(e.callee, w, wb)
			}
			if e.kind == regCallMarked {
				v, err := r.regMarkedCall(f, c, base, int(w.X()), pc, c.rc.Marks[pc], regAuxAt(c, pc))
				if err == nil {
					regWindow(r.stack, wb)[w.B()] = v
				}
				return err
			}
			ok := e.kind == regCallEnter
			if e.kind == regCallThis {
				win[1] = e.this
				ok = true
			} else if e.kind == regCallObjThis {
				ok = win[1].IsObject()
			}
			if ok {
				o, cl, rc := e.callee, e.cl, e.rc
				argc := int(w.X())
				top := base + int(rc.numRegs)
				if top > len(r.stack) || r.frameDepth >= r.maxFrames {
					return r.throwRangeError("maximum call stack size exceeded")
				}
				if r.interruptCounter--; r.interruptCounter <= 0 {
					if err := r.checkInterruptNow(); err != nil {
						return err
					}
				}
				saved := r.stackTop
				if top > saved {
					r.stackTop = top
					if top > r.stackHigh {
						r.stackHigh = top
					}
				}
				if argc > int(rc.keep) {
					// The arguments past the parameters, which the
					// bindings overwrite: f.arguments reads them.
					r.regKeepWindow(base, int(rc.keep), argc)
				}
				// Parameters no argument was given for, and the bindings
				// past them.
				for i := min(argc, int(rc.params)); i < int(rc.params+rc.inits); i++ {
					win[2+i] = Undefined
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
				nf := &r.cur[off]
				nf.cl = cl
				nf.callee = o
				nf.base = base
				nf.pc = 0
				if len(nf.openUpvalues) != 0 {
					nf.openUpvalues = nf.openUpvalues[:0]
				}
				if nf.thisRef != nil || nf.withScopes != nil || nf.evalVars != nil || nf.native != "" || nf.args != nil {
					regClearFrame(nf)
				}
				if rc.newTarget {
					nf.newTarget = Undefined
				}
				nf.regKind = regFrameCall
				v, err := r.runReg(nf, rc, 0)
				nf.regKind = 0
				if rc.ncap != 0 && len(nf.openUpvalues) != 0 {
					regCloseUpvalues(nf)
				}
				r.frameDepth--
				r.stackTop = saved
				if err != nil {
					return r.regCallWEnd(nf, regFrameCall, err, saved, wb+int(w.B()))
				}
				regWindow(r.stack, wb)[w.B()] = v
				return nil
			}
		}
	}
	f.pc = regSrc(c, pc) + 1
	return r.regCallMiss(f, c, w, wb, pc)
}

// regEnterW runs a register function of the form's own layout in the
// window at base, which holds the function, its this and its argc
// arguments.
//
// The frame record is written only where a reader needs it: the function,
// the window's base and the pc, which stack traces read, and the fields a
// frame's last user may have left set, cleared where they are. Its this is
// the window's register 1, which regClosure gives the frame when it makes
// an arrow that captures it.
func (r *Runtime) regEnterW(o *Object, cl *closure, rc *regCode, base, argc int, args []Value, newTarget Value) (Value, error) {
	top := base + int(rc.numRegs)
	if top > len(r.stack) || r.frameDepth >= r.maxFrames {
		return Undefined, r.throwRangeError("maximum call stack size exceeded")
	}
	if r.interruptCounter--; r.interruptCounter <= 0 {
		if err := r.checkInterruptNow(); err != nil {
			return Undefined, err
		}
	}
	saved := r.stackTop
	if top > saved {
		r.stackTop = top
		if top > r.stackHigh {
			r.stackHigh = top
		}
	}
	var kept []Value
	if rc.args {
		ka := len(r.argStack)
		// Before the bindings take the window past the parameters, which
		// args may be; kept on the runtime's argument stack until the call
		// returns.
		r.argStack = append(r.argStack, args...)
		kept = r.argStack[ka:len(r.argStack):len(r.argStack)]
	}
	// Parameters no argument was given for, and the bindings past them.
	for i := base + 2 + min(argc, int(rc.params)); i < base+2+int(rc.params)+int(rc.inits); i++ {
		r.stack[i] = Undefined
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
	f.callee = o
	f.base = base
	f.pc = 0
	if len(f.openUpvalues) != 0 {
		f.openUpvalues = f.openUpvalues[:0]
	}
	if f.thisRef != nil || f.withScopes != nil || f.evalVars != nil || f.native != "" || f.args != nil {
		regClearFrame(f)
	}
	// What an arrow made here captures of new.target, and what a stack
	// trace reads.
	f.newTarget = newTarget
	f.regKind = regFrameEnter
	if rc.args {
		f.args = kept
		p := int(rc.params)
		f.locals = r.stack[base+2 : base+2+p : base+2+p]
	}
	v, err := r.runReg(f, rc, 0)
	f.regKind = 0
	if len(f.openUpvalues) != 0 {
		regCloseUpvalues(f)
	}
	r.frameDepth--
	r.stackTop = saved
	if a := f.args; a != nil {
		// The arguments kept for it, the last on the argument stack.
		r.argStack = r.argStack[:len(r.argStack)-len(a)]
	}
	return v, err
}

// regDirect is callDirect's call of a function whose tree runs register
// code of the form's own layout: its window is made at the top of the stack
// and run there, without callTree's frame, runTree's recover and the copy
// regEnterFrame makes. callDirect has counted the call.
func (r *Runtime) regDirect(o *Object, fd *funcData, c *regCode, this Value, args []Value) (Value, error) {
	if regReportOn {
		regNoteCross(regCrossDirect, c.fn, "")
	}
	// A body answered without a frame is answered here, as callDirect
	// answers one without register code and regCallW one a register call
	// makes: a built-in's callback is most often such a body.
	switch c.leaf {
	case regLeafArg:
		if p := int(c.leafArg); p < len(args) {
			return args[p], nil
		}
		return Undefined, nil
	case regLeafConst:
		return c.leafVal, nil
	case regLeafCode:
		// The parameters' arguments in a window at the top of the stack,
		// the rest of them unread; the body reads neither its callee's
		// register nor its this's (regLeafRunnable).
		m := min(int(c.params), len(args))
		if base := r.stackTop; base+2+m <= len(r.stack) {
			w := r.stack[base+2 : base+2+m : base+2+m]
			for i, a := range args[:m] {
				w[i] = a
			}
			if v, ok := r.regLeafRun(fd.closure, c, base, m); ok {
				return v, nil
			}
		}
	case regLeafPure:
		if fd.closure.pureMiss < pureMissLimit {
			if v, ok := r.pureCall(fd, this, args); ok {
				return v, nil
			}
		}
	case regLeafThis:
		if this.IsObject() {
			if v, ok := r.leafCall(fd.closure, this.Object(), args); ok {
				return v, nil
			}
		}
	}
	if fd.arrow {
		this = fd.extra.lexThis
	} else if c.coerceThis && !this.IsObject() {
		if !this.IsNullish() {
			// The this made an object as runFD makes it.
			v, err, _ := r.regRunFD(fd.closure, this, args, Undefined, o, fd)
			return v, err
		}
		this = r.globalThis
	}
	base := r.stackTop
	if base+int(c.numRegs) > len(r.stack) {
		return Undefined, r.throwRangeError("maximum call stack size exceeded")
	}
	m := min(int(c.params), len(args))
	w := r.stack[base : base+2+m : base+2+m]
	w[0] = Obj(o)
	w[1] = this
	for i, a := range args[:m] {
		w[2+i] = a
	}
	if r.nodeQuirks {
		// f.arguments reads the arguments as the built-in passed them.
		return r.regDirectLegacy(o, fd.closure, c, base, m, args)
	}
	return r.regEnterW(o, fd.closure, c, base, m, args, Undefined)
}

// regNative is call_w's call of the built-in o a site has cached, made as
// callDirect makes one: Math's functions, push and pop and the native
// leaves answered without a frame where their checks pass, anything else
// with the built-in's frame. Its value goes to R[B].
func (r *Runtime) regNative(o *Object, w bytecode.RInstr, wb int) error {
	base := wb + int(w.A())
	args := r.stack[base+2 : base+2+int(w.X())]
	this := Undefined
	if w.Op().MethodCall() {
		this = r.stack[base+1]
	}
	fd := o.fn()
	v, ok := Undefined, false
	if fd.mathOp != 0 {
		v, ok = mathCall(fd.mathOp, args)
	} else if op := fd.elemOp; op >= elemNativeFirst {
		if r.meter == nil && leafChecks[op](r, this, args) {
			var err error
			v, err = fd.native(r, this, args)
			ok = err == nil
		}
	} else if op != 0 {
		v, ok = r.pureElemOp(op, this, args, true)
	}
	if !ok {
		if err := r.tick(); err != nil {
			return err
		}
		if err := r.nativeFrame(o, fd, this, args, Undefined); err != nil {
			return err
		}
		var err error
		v, err = fd.native(r, this, args)
		r.frameDepth--
		if r.meter != nil && err == nil {
			err = r.chargeResult(v)
		}
		if err != nil {
			return err
		}
	}
	r.stack[wb+int(w.B())] = v
	return nil
}

// regLeafRun runs cl's register code c, a regLeafCode body, in the window
// at base, where the callee, its this and argc arguments are, without a
// frame: an arithmetic operation of what is no number gives up, before any
// store, and the call is then made with one. It is the frameless evaluator
// of a pure body (pureCallAt) for the register form, whose instructions
// are fewer, dispatched by a switch of their own (regLeafOp).
func (r *Runtime) regLeafRun(cl *closure, c *regCode, base, argc int) (Value, bool) {
	if cl.pureMiss >= pureMissLimit {
		return Undefined, false
	}
	top := base + int(c.numRegs)
	if top > len(r.stack) {
		return Undefined, false
	}
	if top > r.stackHigh {
		r.stackHigh = top
	}
	regs := regWindow(r.stack, base)
	for i := argc; i < int(c.params+c.inits); i++ {
		regs[2+i] = Undefined
	}
	// The stores to upvalues, made when the body returns.
	var held [regLeafStores]regHeld
	n := 0
	for i := range c.leafOps {
		o := &c.leafOps[i]
		switch o.op {
		case rlMove:
			regs[o.a] = regs[o.b]
		case rlLoad:
			regs[o.a] = o.k
		case rlGetUp:
			regs[o.a] = cl.upvalues[o.b].get()
		case rlSetUp:
			held[n] = regHeld{cl.upvalues[o.b].slot, regs[o.a]}
			n++
		case rlAdd:
			a, b := regs[o.b], regs[o.c]
			if !a.IsNumber() || !b.IsNumber() {
				cl.pureMiss++
				return Undefined, false
			}
			regs[o.a] = regNum(a.Number() + b.Number())
		case rlSub:
			a, b := regs[o.b], regs[o.c]
			if !a.IsNumber() || !b.IsNumber() {
				cl.pureMiss++
				return Undefined, false
			}
			regs[o.a] = regNum(a.Number() - b.Number())
		case rlMul:
			a, b := regs[o.b], regs[o.c]
			if !a.IsNumber() || !b.IsNumber() {
				cl.pureMiss++
				return Undefined, false
			}
			regs[o.a] = regNum(a.Number() * b.Number())
		case rlAddK:
			a := regs[o.b]
			if !a.IsNumber() {
				cl.pureMiss++
				return Undefined, false
			}
			regs[o.a] = regNum(a.Number() + o.k.num)
		case rlRet:
			regLeafStore(held[:n])
			return regs[o.a], true
		case rlRetUndef:
			regLeafStore(held[:n])
			return Undefined, true
		}
	}
	return Undefined, true
}

// regCloseUpvalues closes the upvalues a frame's closures left open on its
// registers.
//
//go:noinline
func regCloseUpvalues(f *frame) {
	for _, u := range f.openUpvalues {
		u.close()
	}
}

// regSlow is the slow path of the instruction at pc, whatever the loop's
// fast path for it did not answer: it sets the frame's pc to the stack
// form's instruction, does what the stack interpreter does, and stores the
// result in its register. A comparison's jump is left to the loop, which is
// given the comparison's result.
//
// It reads the instruction from the code rather than being given it, so that
// the loop keeps nothing of the instruction across the call (see runReg).
func (r *Runtime) regSlow(f *frame, c *regCode, k *regKept, regs *[1 << 16]Value) (Value, error) {
	pc := k.pc - 1
	w := regAt(c.code, pc)
	f.pc = c.src[pc] + 1
	cl := f.cl
	wb := f.base - int(c.locals)
	var v Value
	var err error
	switch w.Op() {
	case bytecode.RCold, bytecode.RGetPrivate, bytecode.RSetPrivate:
		return Undefined, r.regColdSub(f, c, k, regs)
	case bytecode.RGetGlobal:
		v, err = r.regGetGlobal(f, cl, w)
	case bytecode.RSetGlobal:
		return Undefined, r.regSetGlobal(f, cl, w, regs[w.A()])
	case bytecode.RGetProp:
		v, err = r.regGetProp(cl, c, w, regs[w.B()])
	case bytecode.RGetMethod:
		v, err = r.regGetMethod(cl, c, w, regs[w.B()])
	case bytecode.RGetMethodW:
		o := regs[w.B()]
		v, err = r.regGetMethod(cl, c, w, o)
		(*[2]Value)(unsafe.Pointer(&regs[w.A()]))[1] = o
	case bytecode.RSetProp:
		return Undefined, r.regSetProp(cl, c, w, regs[w.A()], regs[w.B()])
	case bytecode.RGetIndex:
		// A key string the runtime handed out -- a for-in's -- read as
		// the tree's element read reads it (stringKeyRead): a[k] in for
		// (k in a) an element by the key's index.
		if o, key := regs[w.B()], regs[w.C()]; o.IsObject() && key.IsString() {
			if x, ok := r.stringKeyRead(o.Object(), key.String()); ok {
				v = x
				break
			}
		}
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
	case bytecode.RAndI, bytecode.ROrI, bytecode.RXorI, bytecode.RShlI, bytecode.RShrI, bytecode.RUShrI:
		v, err = r.regBinImm(regStackOp[w.Op()], regs[w.B()], w.SC())
	case bytecode.RBitAnd, bytecode.RBitOr, bytecode.RBitXor, bytecode.RShl, bytecode.RShr, bytecode.RUShr:
		v, err = r.regBinary(regStackOp[w.Op()], regs[w.B()], regs[w.C()])
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
	case bytecode.RAddIndex:
		return Undefined, r.regAddIndex(f, c.rc.Aux[pc], regs[w.A()], regs[w.B()], regs[w.C()], false)
	case bytecode.RAddIndexK:
		return Undefined, r.regAddIndex(f, c.rc.Aux[pc], regs[w.A()], Int32(int32(w.C())), regs[w.B()], w.X() != 0)
	case bytecode.RAddIndexI:
		return Undefined, r.regAddIndex(f, c.rc.Aux[pc], regs[w.A()], regs[w.B()], Int32(w.SC()), w.X() != 0)
	case bytecode.RNewArray:
		b := wb + int(w.B())
		v = Obj(r.newArrayFrom(r.stack[b : b+int(w.C())]))
	case bytecode.RGetIndexPre:
		// x[++i]: the step, then the read, each as the stack form makes it.
		// Assigned, not declared: the switch's err is what the call returns.
		var n Value
		if n, err = r.regStep(regs[w.C()], float64(int8(w.X()))); err != nil {
			return Undefined, err
		}
		regs[w.C()] = n
		v, err = r.getIndexed(regs[w.B()], n)
	case bytecode.RGetIndexPost:
		// x[i++]: the step, its old value the key, then the read, as the
		// stack form makes them.
		var old, n Value
		if old, err = r.toNumeric(regs[w.C()]); err != nil {
			return Undefined, err
		}
		if n, err = r.regStep(old, float64(int8(w.X()))); err != nil {
			return Undefined, err
		}
		o := regs[w.B()]
		regs[w.C()] = n
		v, err = r.getIndexed(o, old)
	case bytecode.RGetIndexI:
		v, err = r.getIndexed(regs[w.B()], Int32(int32(w.C())))
	case bytecode.RSetIndexI:
		return Undefined, r.setIndexed(regs[w.A()], Int32(int32(w.B())), regs[w.C()], cl.fn.Strict)
	case bytecode.RNewW:
		if regReportOn {
			regCrossOf(regCrossNew, regs[w.A()])
		}
		b := wb + int(w.A())
		if v, err = r.regConstruct(regs[w.A()], b, int(w.X())); err == nil {
			regs[w.B()] = v
		}
		return v, err
	case bytecode.RClosureW:
		v = Obj(r.regClosure(f, c, w, wb))
	case bytecode.RPostInc, bytecode.RPostDec:
		old, err := r.toNumeric(regs[w.B()])
		if err != nil {
			return Undefined, err
		}
		delta := 1.0
		if w.Op() == bytecode.RPostDec {
			delta = -1
		}
		next, err := r.regStep(old, delta)
		if err != nil {
			return Undefined, err
		}
		regs[w.A()] = old
		regs[w.B()] = next
		return old, nil
	case bytecode.RGetLength:
		if o := regs[w.B()]; o.IsString() {
			v = Int(o.String().Len())
		} else {
			v, err = r.getValueProp(o, atomLength)
		}
	case bytecode.RInstanceOf:
		var ok bool
		ok, err = r.instanceOfAt(&cl.ic[c.rc.Aux[pc]], regs[w.B()], regs[w.C()])
		v = Bool(ok)
	case bytecode.RIn:
		v, err = r.regIn(regs[w.B()], regs[w.C()])
	case bytecode.RDelete:
		v, err = r.regDelete(cl, regs[w.B()], regs[w.C()])
	case bytecode.RArguments:
		v = Obj(r.newArgumentsObject(f))
	case bytecode.RArgsLen:
		if a := regs[w.B()]; a.IsObject() {
			v, err = r.getValueProp(a, atomLength)
		} else {
			v = Int(len(f.args))
		}
	case bytecode.RConcat:
		// The registers are the parts' own, which concatParts may convert
		// in place: nothing reads them after.
		b := int(w.B())
		v, err = concatParts(r, regs[b:b+int(w.C())])
	case bytecode.RKeyOfBase:
		if base := regs[w.B()]; base.IsNullish() {
			return Undefined, r.throwTypeError("cannot read property of %s", r.describe(base))
		}
		var k Atom
		if k, err = r.toPropertyKey(regs[w.C()]); err == nil {
			// An object's Symbol.toPrimitive may give a symbol: the key
			// keeps whichever kind it turned out to be.
			v = r.keyToValue(k)
		}
	case bytecode.RStrPart:
		v, err = r.add(Str(emptyString), regs[w.A()])
	case bytecode.RArgsIndex:
		key := regs[w.C()]
		if !regs[w.B()].IsObject() {
			if i := uint32(key.num); key.IsNumber() && float64(i) == key.num && uint(i) < uint(len(f.args)) &&
				!(f.cl.fn.MappedArguments && uint(i) < uint(f.cl.fn.ParamCount)) {
				v = f.args[i]
				break
			}
			regs[w.B()] = Obj(r.newArgumentsObject(f))
		}
		v, err = r.getIndexed(regs[w.B()], key)
	case bytecode.RApplyArgs:
		b := int(w.B())
		v, err = r.regApplyArgs(f, regs[b], regs[b+1], regs[b+2], &regs[w.C()])
	case bytecode.RAddProp, bytecode.RAddPropI:
		b := Int32(w.SB())
		if w.Op() == bytecode.RAddProp {
			b = regs[w.B()]
		}
		return Undefined, r.regAddProp(cl, c, w, regs[w.A()], b)
	case bytecode.RNewArrayK:
		// The tree's shared literal (ztree_a0shared.go): the site keeps the
		// array of the constants, and each array shares its elements until
		// one is written.
		s := &cl.ic[w.C()]
		t := s.p1
		if t == nil {
			lit := c.rc.Literals[c.rc.Aux[pc]]
			vals := make([]Value, len(lit))
			for i, e := range lit {
				switch e.Kind {
				case bytecode.LitConst:
					vals[i] = cl.consts[e.K]
				case bytecode.LitTrue:
					vals[i] = True
				case bytecode.LitFalse:
					vals[i] = False
				case bytecode.LitNull:
					vals[i] = Null
				default:
					vals[i] = Undefined
				}
			}
			t = sharedMade(r, vals)
			s.p1 = t
		}
		v = Obj(sharedShare(r, s, t))
	case bytecode.RAddI:
		v, err = r.regBinImm(bytecode.OpAdd, regs[w.B()], w.SC())
	case bytecode.RSubI:
		v, err = r.regBinImm(bytecode.OpSub, regs[w.B()], w.SC())
	case bytecode.RJLt, bytecode.RJNLt, bytecode.RJLe, bytecode.RJNLe, bytecode.RJGt, bytecode.RJNGt, bytecode.RJGe, bytecode.RJNGe,
		bytecode.RJLtI, bytecode.RJNLtI, bytecode.RJLeI, bytecode.RJNLeI, bytecode.RJGtI, bytecode.RJNGtI, bytecode.RJGeI, bytecode.RJNGeI,
		bytecode.RJEq, bytecode.RJNe, bytecode.RJEqI, bytecode.RJNeI, bytecode.RJNull, bytecode.RJNotNull,
		bytecode.RJLtU, bytecode.RJNLtU, bytecode.RJLeU, bytecode.RJNLeU:
		k := regJumpCmp[w.Op()]
		b := Int32(w.SB())
		if k.reg {
			b = regs[w.B()]
		} else if w.Op() == bytecode.RJNull || w.Op() == bytecode.RJNotNull {
			b = Null
		} else if k.upval {
			b = cl.upvalues[w.B()].get()
		}
		res, err := r.regCompare(k.op, regs[w.A()], b)
		return Bool(res != k.not), err
	case bytecode.RJmpF, bytecode.RJmpT:
		// A truthiness the loop's regTruth does not answer.
		return Bool(regs[w.A()].Truthy() == (w.Op() == bytecode.RJmpT)), nil
	case bytecode.RJTypeof:
		t := typeofKind[struct{}](regs[w.A()])
		return Bool((t == int(w.X()&^bytecode.RCmpTrue)) == (w.X()&bytecode.RCmpTrue != 0)), nil
	case bytecode.RJSEq, bytecode.RJSNe:
		// A === the loop's regStrictEq does not answer: strings, BigInts.
		return Bool(regs[w.A()].StrictEquals(regs[w.B()]) == (w.Op() == bytecode.RJSEq)), nil
	default:
		panic("register code: no slow path for " + w.Op().String())
	}
	if err == nil {
		regs[w.A()] = v
	}
	return v, err
}
