package vm

import (
	"encoding/binary"
	"math"
	"os"
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/jsnum"
)

// The register interpreter's hot block after regSlow (zzreg_run.go): its
// operators, the miss paths and what an iteration runs out of line. The
// layout lock's pad P, between the two files, puts this half at a fixed
// address modulo 4 KB, so that a case regSlow takes moves none of it
// (docs/register-interpreter.md, "The loop").

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
	if res, ok := pureBigCompare[struct{}](op, a, b); ok {
		// Two BigInts an int64 holds, as the tree's compare nodes order
		// them.
		return res, nil
	}
	c, err := r.compare(a, b)
	if err != nil {
		return false, err
	}
	return relationalResult(op, c), nil
}

// regStep is ++ or -- of a value that is not a number.
func (r *Runtime) regStep(a Value, delta float64) (Value, error) {
	if a.IsBigInt() {
		if v, ok := shortArith(bytecode.OpAdd, a, shortBig(int64(delta))); ok {
			return v, nil
		}
		return r.bigOp(bytecode.OpAdd, a, shortBig(int64(delta)))
	}
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
	case bytecode.OpToNumeric:
		return r.toNumeric(a)
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

// regElemMiss is the element get_index, set_index and their immediate
// forms read or write where their fast path has found no dense element, as
// the tree's getIndexMiss and setIndexMiss do: a typed array's, from or to
// its buffer as elemAt and setElem read and write it, typedRead and
// typedStore the rest they can; an element appended at an array's length,
// or a hole filled (appendElem, fillHole). False sends what is left -- a
// hole read, a value a typed array converts with code, a getter, a proxy
// -- to the slow path. None of it can throw or run code.
//
// It is the one call of a label get_index and the immediate forms share,
// so that the loop grows by a jump each, and set_index's before its slow
// path, which keeps the code after that case where it was: with the case
// shorter, crypto_am3 was 5.5% slower at every placement. It switches on
// the operation again, as regSlow does, out of the loop.
//
//go:noinline
func regElemMiss(c *regCode, k *regKept, regs *[1 << 16]Value) bool {
	w := regAt(c.code, k.pc-1)
	var obj, key Value
	switch w.Op() {
	case bytecode.RGetIndex, bytecode.RGetIndexI:
		obj, key = regs[w.B()], Value{num: float64(w.C())}
		if w.Op() == bytecode.RGetIndex {
			key = regs[w.C()]
		}
		if obj.IsObject() && obj.object().class == ClassTypedArray {
			// The element read here, as elemAt reads it, rather than
			// through a second call.
			t, ok := obj.object().dataOf().(*typedArrayData)
			i := uint32(key.num)
			if ok && float64(i) == key.num && uint(i) < uint(t.fixedLength) {
				if st := t.st; st != nil && t.end <= len(st.bytes) {
					b, j := st.bytes[t.byteOffset:t.end], int(i)
					var v Value
					switch t.kind {
					case elemInt8:
						v = Int(int(int8(b[j])))
					case elemUint8, elemUint8Clamped:
						v = Int(int(b[j]))
					case elemInt16:
						v = Int(int(int16(binary.LittleEndian.Uint16(b[j<<1:]))))
					case elemUint16:
						v = Int(int(binary.LittleEndian.Uint16(b[j<<1:])))
					case elemInt32:
						v = Int32(int32(binary.LittleEndian.Uint32(b[j<<2:])))
					case elemUint32:
						v = Uint32(binary.LittleEndian.Uint32(b[j<<2:]))
					case elemFloat32:
						v = Float(float64(math.Float32frombits(binary.LittleEndian.Uint32(b[j<<2:]))))
					case elemFloat64:
						v = Float(math.Float64frombits(binary.LittleEndian.Uint64(b[j<<3:])))
					default:
						goto long
					}
					regs[w.A()] = v
					return true
				}
			}
		}
	long:
		if v, ok := elemAt(obj, key); ok {
			regs[w.A()] = v
			return true
		}
		if v, ok := typedRead(obj, key.num); ok {
			regs[w.A()] = v
			return true
		}
		return false
	case bytecode.RSetIndex, bytecode.RSetIndexI:
		obj, key = regs[w.A()], Value{num: float64(w.B())}
		if w.Op() == bytecode.RSetIndex {
			key = regs[w.B()]
		}
		v := regs[w.C()]
		if obj.IsObject() && obj.object().class == ClassTypedArray && v.IsNumber() {
			// A number stored here, as setElem stores it.
			t, ok := obj.object().dataOf().(*typedArrayData)
			i := uint32(key.num)
			if ok && float64(i) == key.num && uint(i) < uint(t.fixedLength) {
				if st := t.st; st != nil && t.end <= len(st.bytes) && !st.immutable {
					b, n, j := st.bytes[t.byteOffset:t.end], v.num, int(i)
					switch t.kind {
					case elemInt8, elemUint8:
						b[j] = byte(toInt32Wrap(n))
					case elemUint8Clamped:
						b[j] = clampUint8(n)
					case elemInt16, elemUint16:
						binary.LittleEndian.PutUint16(b[j<<1:], uint16(toInt32Wrap(n)))
					case elemInt32, elemUint32:
						binary.LittleEndian.PutUint32(b[j<<2:], uint32(toInt32Wrap(n)))
					case elemFloat32:
						binary.LittleEndian.PutUint32(b[j<<2:], math.Float32bits(float32(n)))
					case elemFloat64:
						binary.LittleEndian.PutUint64(b[j<<3:], math.Float64bits(n))
					default:
						return typedStore(obj, key.num, v)
					}
					return true
				}
			}
		}
		if setElem(obj, key, v) || typedStore(obj, key.num, v) {
			return true
		}
		// An element appended at an array's length, or a hole filled,
		// as the tree's setIndexMiss stores them.
		return obj.IsObject() && (appendElem(obj.object(), key.num, v) || fillHole(obj.object(), key.num, v))
	}
	return false
}

// regSetMiss is set_index's miss: on an array, an element appended at its
// length or a hole filled, as the tree's setIndexMiss stores them, where
// setElem would store nothing the loop's fast path has not; on anything
// else, regElemMiss's typed array's element. False sends what is left --
// a shared literal's element, a setter, a proxy -- to the slow path. None
// of it can throw or run code.
//
//go:noinline
func regSetMiss(c *regCode, k *regKept, regs *[1 << 16]Value) bool {
	w := regAt(c.code, k.pc-1)
	if obj := regs[w.A()]; obj.IsObject() && obj.object().class == ClassArray {
		key, v := regs[w.B()], regs[w.C()]
		o := obj.object()
		return appendElem(o, key.num, v) || fillHole(o, key.num, v)
	}
	return regElemMiss(c, k, regs)
}

// regBackEdge is a backward jump's interrupt check, when the budget is
// spent.
func (r *Runtime) regBackEdge(f *frame, c *regCode, k *regKept) error {
	r.backEdges = backEdgeCheckInterval
	f.pc = c.src[k.pc-1] + 1
	if err := r.checkInterruptNow(); err != nil {
		return err
	}
	// The temporaries past what the loop's head has live are stale: a
	// value a call's window was given stays there until the window is
	// used again.
	r.sweepStaleSlots(f.base - int(c.locals) + c.rc.LiveAt(k.pc-1))
	return nil
}

// regAppend is append, a case of runReg's: an append to a local, its
// string grown in place where it alone holds it, as the tree's
// strAppendNode grows it. The case keeps nothing of the instruction across
// the call; this reads it back.
//
//go:noinline
func (r *Runtime) regAppend(f *frame, c *regCode, k *regKept, regs *[1 << 16]Value) error {
	pc := k.pc - 1
	w := regAt(c.code, pc)
	f.pc = c.src[pc] + 1
	v, err := strAppendOp[struct{}](r, regs[w.A()], regs[w.B()])
	if err == nil {
		regs[w.A()] = v
	}
	return err
}

// regCharCodeAt is char_code_at, a case of runReg's: s.charCodeAt(i) of a
// string, its method the built-in read from the cache primProp fills and
// its argument a number, read here as the tree's charCodeAtCall reads it,
// without the method's read on the slow path, the call's site and the
// built-in's conversions; anything else is regCharCodeAtSlow's. The case
// keeps nothing of the instruction across the call; this reads it back.
//
//go:noinline
func (r *Runtime) regCharCodeAt(f *frame, c *regCode, k *regKept, regs *[1 << 16]Value) error {
	pc := k.pc - 1
	w := regAt(c.code, pc)
	if o := regs[w.B()]; o.IsString() {
		key := f.cl.names[c.siteName[w.C()]]
		p := r.proto.str
		if e := &r.primProps[key&63]; e.key == key && e.proto == p && e.shape == p.shape {
			if m := p.vals()[e.idx]; m.IsObject() {
				if fd := m.Object().fn(); fd != nil && fd.elemOp == elemStrCharCodeAt && !fd.bound {
					if a := regs[w.X()]; a.IsNumber() {
						x := a.Number()
						if x != x {
							x = 0
						}
						// ToIntegerOrInfinity truncates toward zero, so any
						// position above -1 and below the length is a code
						// unit's.
						v := Float(math.NaN())
						if s := o.String(); x > -1 && x < float64(s.Len()) {
							v = Int(s.CharCodeAt(int(x)))
						}
						regs[w.A()] = v
						return nil
					}
				}
			}
		}
	}
	return r.regCharCodeAtSlow(f, c, pc, regs)
}

// regIterNext is iter_next: R[A] = the next value of cursor R[B], or a
// jump by SC where it has none, the step at the head of each turn of a
// for-of or a for-in loop. An array's element in its dense storage is
// read here, as the tree's iterNextTree reads it; a generator's cursor is
// stepped as the tree steps it (regIterNextTree); anything else as
// iterNext steps it.
//
//go:noinline
func (r *Runtime) regIterNext(f *frame, c *regCode, k *regKept, regs *[1 << 16]Value) error {
	pc := k.pc - 1
	w := regAt(c.code, pc)
	cur := regs[w.B()]
	var v Value
	var more bool
	var err error
	if st := cursorOf(cur); st != nil && !st.done {
		if a := st.arr; a != nil {
			if i := st.idx; i < int(a.arrayLength()) && i < len(a.elemsOf()) {
				if x := a.elemsOf()[i]; !isHole(x) {
					st.idx++
					regs[w.A()] = x
					return nil
				}
			}
		} else if !st.forIn && r.genNextFn != nil && st.next.ref == unsafe.Pointer(r.genNextFn) {
			f.pc = c.src[pc] + 1
			v, more, err = r.regIterNextTree(f, cur)
			goto stepped
		} else if !st.forIn {
			f.pc = c.src[pc] + 1
			v, more, err = r.regStepNext(st)
			goto stepped
		}
	}
	f.pc = c.src[pc] + 1
	v, more, err = r.iterNext(cur)
stepped:
	if err != nil {
		return err
	}
	if !more {
		k.pc += int(w.SC())
		return nil
	}
	regs[w.A()] = v
	return nil
}

// regLit is regexp_lit: the literal's RegExp made or given back in the
// window's receiver and its method in the callee's register -- the
// built-in, unread, where reading it would find it running nothing to
// find it (regexpLitMethod), as the tree's literal call does, and read
// otherwise, at the instruction's position or, for one that calls, at the
// stack form's instruction of the literal (Aux) -- and, with RLitCall,
// the method called with the window's one argument as a call_method_w
// marked RMarkLitRecv calls it.
//
//go:noinline
func (r *Runtime) regLit(f *frame, c *regCode, k *regKept, regs *[1 << 16]Value) error {
	pc := k.pc - 1
	w := regAt(c.code, pc)
	f.pc = c.src[pc] + 1
	a, x, kc := w.A(), w.X(), uint32(w.C())
	lit, err := r.newRegExpLiteral(regConst(f.cl.fn, c, kc))
	if err != nil {
		return err
	}
	regs[a+1] = lit
	test := x&bytecode.RLitTest != 0
	if m := r.regexpLitMethod(test); m != nil {
		if x&(bytecode.RLitCall|bytecode.RLitDead) == bytecode.RLitCall && !regs[a+2].IsObject() {
			// The built-in on the literal given a primitive, called
			// at once, and the RegExp given back (RMarkLitRecv).
			v, err := r.callRest(m, m.fn(), lit, regs[a+2:a+3])
			if err != nil {
				return err
			}
			if !test || r.regexpLitMethod(false) != nil {
				// test calls exec, which must be the built-in too.
				r.regexpLiteralDone(regConst(f.cl.fn, c, kc), lit.Object())
			}
			regs[w.B()] = v
			return nil
		}
		regs[a] = Obj(m)
	} else {
		name := atomExec
		if test {
			name = r.atoms.intern("test")
		}
		if x&bytecode.RLitCall != 0 {
			f.pc = c.rc.Aux[pc] + 1
		}
		v, err := r.getValueProp(lit, name)
		if err != nil {
			return err
		}
		regs[a] = v
		f.pc = c.src[pc] + 1
	}
	if x&bytecode.RLitCall == 0 {
		return nil
	}
	m := uint8(bytecode.RMarkLitRecv)
	if x&bytecode.RLitDead != 0 {
		m |= bytecode.RMarkDead
	}
	v, err := r.regMarkedCall(f, c, f.base-int(c.locals)+int(a), 1, pc, m, kc)
	if err != nil {
		return err
	}
	regs[w.B()] = v
	return nil
}

// regCallSlow is the call of instruction w at pc, its window at base,
// that its site's cache does not answer.
// regMarkedCall is the call of call_method_w pc, its window at base, of
// argc arguments, which its Aux, aux, marks, and what the tree's nodes of
// such calls do around it, x its operand (Aux). RMarkLitArg makes the
// call's first argument,
// the literal; a literal's RegExp is given back where the call keeps
// nothing of it and runs nothing of the script's (regexpLiteralDone):
// exec or test, the realm's own, given a primitive, of the literal's
// RegExp (RMarkLitRecv); replace, split or match searching with it
// directly (regexpArgCall). A call of the realm's exec or replace whose
// value nothing reads but its truth (RMarkDead) tells the function so, as
// deadExec's node does (regDeadCall); one of exec whose value goes to a
// local that keeps it only as execReuseOf says (RMarkReuse) has its array
// made in the memory of the last (reuseCallRest).
func (r *Runtime) regMarkedCall(f *frame, c *regCode, base, argc, pc int, m uint8, x uint32) (Value, error) {
	k := x
	args := r.stack[base+2 : base+2+argc]
	if m&bytecode.RMarkLitArg != 0 {
		lit, err := r.newRegExpLiteral(regConst(f.cl.fn, c, k))
		if err != nil {
			return Undefined, err
		}
		args[0] = lit
	}
	fv, recv := r.stack[base], r.stack[base+1]
	if !fv.IsObject() {
		return r.regCall(fv, recv, args)
	}
	o := fv.Object()
	var rx *Object
	if m&bytecode.RMarkLitArg != 0 {
		if r.regexpArgCall(o, recv, args) {
			rx = args[0].Object()
		}
	} else if m&bytecode.RMarkLitRecv != 0 && recv.IsObject() && (argc == 0 || !args[0].IsObject()) {
		if exec := r.regexpLitMethod(false); exec != nil && (o == exec || o == r.regexpLitMethod(true)) {
			rx = recv.Object()
		}
	}
	var v Value
	var err error
	switch {
	case m&bytecode.RMarkReuse != 0:
		if site := regReuseAt(c, pc); site != nil && o == r.regexpExecFn {
			wb := f.base - int(c.locals)
			v, err = reuseCallRest[struct{}](r, site, r.stack[wb:wb+int(c.numRegs)], o, o.fn(), recv, args)
		} else {
			v, err = r.regMarkedRest(o, recv, args)
		}
	case m&bytecode.RMarkDead != 0 && (o == r.regexpExecFn || o == r.strReplaceFn):
		v, err = r.regDeadCall(o, recv, args)
	default:
		v, err = r.regMarkedRest(o, recv, args)
	}
	if err != nil {
		return Undefined, err
	}
	if rx != nil {
		r.regexpLiteralDone(regConst(f.cl.fn, c, k), rx)
	}
	return v, nil
}

// regApplyArgs is apply_args: target.apply(this, the frame's arguments),
// which the built-in apply makes with the arguments as passed, no object
// made; any other apply is given one, made once and kept in the arguments
// binding's register, slot, so that every call is given the same one.
func (r *Runtime) regApplyArgs(f *frame, target, apply, this Value, slot *Value) (Value, error) {
	if !apply.IsObject() || apply.Object() != r.applyFn {
		o := *slot
		if !o.IsObject() {
			o = Obj(r.newArgumentsObject(f))
			*slot = o
		}
		return r.call(apply, target, []Value{this, o})
	}
	args := f.args
	if f.cl.fn.MappedArguments {
		n := min(f.cl.fn.ParamCount, len(args), len(f.locals))
		for i := 0; i < n; i++ {
			if !f.locals[i].sameBits(args[i]) {
				// A parameter has been assigned, which a mapped object
				// would show.
				args = append([]Value(nil), args...)
				copy(args, f.locals[:n])
				break
			}
		}
	}
	return r.applyCall(target, this, args)
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

// The kinds of regCode's leaf.
const (
	regLeafNone = iota
	regLeafPure
	regLeafThis
	// regLeafCode is run as its register code by regLeafRun, with no
	// frame, at the call site.
	regLeafCode
	// regLeafArg returns its parameter leafArg, and regLeafConst the
	// constant leafVal, undefined for nothing: regCallW answers them from
	// the window and the code.
	regLeafArg
	regLeafConst
)

// regTrace prints why a function has no register code.
var regTrace = os.Getenv("QJS_REGTRACE") != ""

// regDump prints each function's register code as it is made.
var regDump = os.Getenv("QJS_REGDUMP") != ""

// regTruth is ToBoolean of what needs no call -- a boolean, an object, a
// number -- and ok false for anything else.
func regTruth(v Value) (t, ok bool) {
	switch b := math.Float64bits(v.num); {
	case b == trueBits:
		return true, true
	case b == falseBits:
		return false, true
	case b == objectBits:
		return (*Object)(v.ref).flags&objHTMLDDA == 0, true
	case v.IsNumber():
		return v.num != 0 && v.num == v.num, true
	}
	return false, false
}

// regWordKind reports whether a value of tag bits is one that is === only
// to a value of the same words: undefined, null, a boolean, a symbol or an
// object.
func regWordKind(bits uint64) bool {
	return bits-tagBase < heapBigBits-tagBase &&
		1<<(bits&0xFF)&(1<<KindUndefined|1<<KindNull|1<<KindBool|1<<KindSymbol|1<<KindObject) != 0
}

// regTruthy is ToBoolean, a boolean's without the call.
func regTruthy(v Value) bool {
	return math.Float64bits(v.num) == trueBits || math.Float64bits(v.num) != falseBits && v.Truthy()
}

// regJumpOff is a jump's offset: BC for a jump of a register's
// truthiness or none, SC for a compare-and-branch.
func regJumpOff(w bytecode.RInstr) int32 {
	switch w.Op() {
	case bytecode.RJmp, bytecode.RJmpF, bytecode.RJmpT:
		return int32(w.BC())
	}
	return w.SC()
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
// what a bitwise operator's fast path takes without converting. The tag
// test is not redundant in time: without it (a NaN fails the round trip's
// compare as well), loop_int32 was 8.1% slower at every placement.
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
// not keep in registers (it has an array of more than one element), whose
// address every call the loop makes is given, so that Go cannot take a load
// of it after the call for the value stored before: the loop's pc is then
// not live across the call, and is not saved at every dispatch.
type regKept struct {
	pc  int
	pad [2]int
}
