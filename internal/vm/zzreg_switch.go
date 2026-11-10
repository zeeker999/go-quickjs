package vm

import (
	"math"
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
)

// regTypeOf is the index in typeofNames of typeof a value of kind k but
// an object, whose is its own: undefined for the rest. A switch, not a
// table: package data moves the runtime's data lines.
func regTypeOf(k Kind) uint8 {
	switch k {
	case KindNumber:
		return typeNumber
	case KindNull, KindObject:
		return typeObject
	case KindBool:
		return typeBoolean
	case KindString:
		return typeString
	case KindSymbol:
		return typeSymbol
	case KindBigInt:
		return typeBigInt
	}
	return typeUndefined
}

// regSwitchTable is a switch instruction's table, as the tree tier's
// switchLookup makes one of the same cases: a number by an array's index
// where the cases are integers of a small range, by a short list or a map
// otherwise; a string by its code units, past the longest case's length
// not even flattened; true, false and null by their kind. Each selector's
// first case is the one it goes to, as the chain of === would.
type regSwitchTable struct {
	dense  []int32
	lo, hi float64
	few    []float64
	fewTo  []int32
	nums   map[float64]int32
	// byLen is where flat's strings of each length start, as the tree's
	// switchLookup groups them.
	byLen []int32
	flat  []bytecode.SwitchCase
	other [3]int32
	// typeTo is the body of each type's name in typeofNames, for
	// switch_typeof.
	typeTo [8]int32
}

// newRegSwitchTable is the table of cases.
func newRegSwitchTable(cases []bytecode.SwitchCase) *regSwitchTable {
	t := &regSwitchTable{other: [3]int32{-1, -1, -1}, lo: math.Inf(1), hi: math.Inf(-1)}
	for i := range t.typeTo {
		t.typeTo[i] = -1
	}
	nums := map[float64]int32{}
	var order []float64
	seen := map[string]bool{}
	var strs []bytecode.SwitchCase
	ints := true
	for _, c := range cases {
		switch c.Kind {
		case bytecode.SwitchNum:
			f := c.Num
			if f != f {
				continue // NaN equals nothing.
			}
			if _, ok := nums[f]; !ok {
				nums[f] = int32(c.To)
				order = append(order, f)
				t.lo, t.hi = min(t.lo, f), max(t.hi, f)
				ints = ints && f == math.Trunc(f) && !math.IsInf(f, 0)
			}
		case bytecode.SwitchStr:
			if !seen[c.Str] {
				seen[c.Str] = true
				strs = append(strs, c)
				for i, name := range typeofNames {
					if name == c.Str {
						t.typeTo[i] = int32(c.To)
					}
				}
			}
		default:
			if i := c.Kind - bytecode.SwitchTrue; t.other[i] < 0 {
				t.other[i] = int32(c.To)
			}
		}
	}
	if len(strs) > 0 {
		most := 0
		lens := make([]int, len(strs))
		for i, s := range strs {
			lens[i] = NewString(s.Str).length
			most = max(most, lens[i])
		}
		t.byLen = make([]int32, most+2)
		for n := 0; n <= most; n++ {
			t.byLen[n] = int32(len(t.flat))
			for i, s := range strs {
				if lens[i] == n {
					t.flat = append(t.flat, s)
				}
			}
		}
		t.byLen[most+1] = int32(len(t.flat))
	}
	switch {
	case len(nums) == 0:
	case ints && t.hi-t.lo < switchDense:
		t.dense = make([]int32, int(t.hi-t.lo)+1)
		for i := range t.dense {
			t.dense[i] = -1
		}
		for f, to := range nums {
			t.dense[int(f-t.lo)] = to
		}
	case len(nums) <= 8:
		for _, f := range order {
			t.few, t.fewTo = append(t.few, f), append(t.fewTo, nums[f])
		}
	default:
		t.nums = nums
	}
	return t
}

// lookup is the pc of the body v goes to, or -1: a string the constant a
// case was made from first by its pointer, as the tree's lookup asks.
func (t *regSwitchTable) lookup(v Value, consts []Value) int32 {
	switch {
	case v.IsNumber():
		f := v.num
		if t.dense != nil {
			if f >= t.lo && f <= t.hi {
				if i := int(f - t.lo); float64(i)+t.lo == f {
					return t.dense[i]
				}
			}
			return -1
		}
		for i, g := range t.few {
			if f == g {
				return t.fewTo[i]
			}
		}
		if t.nums != nil {
			if to, ok := t.nums[f]; ok {
				return to
			}
		}
	case v.IsString():
		s := (*String)(v.ref)
		if s.length >= len(t.byLen)-1 {
			return -1
		}
		for _, e := range t.flat[t.byLen[s.length]:t.byLen[s.length+1]] {
			if int(e.K) < len(consts) && consts[e.K].ref == unsafe.Pointer(s) {
				return int32(e.To)
			}
			if s.left != nil {
				s.Go()
			}
			if s.s == e.Str {
				return int32(e.To)
			}
		}
	case v.IsBool():
		if v.Truthy() {
			return t.other[0]
		}
		return t.other[1]
	case v.IsNull():
		return t.other[2]
	}
	return -1
}

// regCold is the operations runReg leaves to a case of one call: each
// case added to the loop moves the code of every other, which no placement
// varies, so only what runs per turn of a hot loop is a case of its own.
// It switches on the operation again.
func (r *Runtime) regCold(f *frame, c *regCode, k *regKept, regs *[1 << 16]Value) {
	w := regAt(c.code, k.pc-1)
	switch w.Op() {
	case bytecode.RNewArrayE:
		r.regNewArrayE(f, c, k, regs)
	case bytecode.RSwitch:
		regSwitch(c, k, regs[w.A()], f.cl.consts)
	case bytecode.RSwitchTypeof:
		regSwitchTypeof(c, k, regs[w.A()])
	case bytecode.RTypeof:
		r.regTypeof(c, k, regs)
	case bytecode.RNewObjectLit:
		r.regNewObjectLit(f, c, k, regs)
	}
}

// regNewObjectLit is new_object_lit: the literal made with the room its
// site's objects came to need, as new_object makes it, and each property
// defined as define_field defines it, the values read from their
// registers. A fresh ordinary object takes each new key by a transition,
// and defining one can neither throw nor run code.
func (r *Runtime) regNewObjectLit(f *frame, c *regCode, k *regKept, regs *[1 << 16]Value) {
	pc := k.pc - 1
	w := regAt(c.code, pc)
	cl := f.cl
	n := int(w.X())
	ic := &cl.ic[w.C()]
	root := r.shapes.siteRoot(ic, r.proto.object)
	m := n
	if root != nil && int(root.slack) > m {
		m = int(root.slack)
	}
	o := newShapedLiteral(root.orRoot(ClassObject, r.proto.object), ClassObject, m)
	vals := regs[w.B() : int(w.B())+n]
	if final := ic.next; final != nil && root != nil && ic.s1 == root && n <= int(o.valc) {
		// The shape the keys led to from this root before, which the
		// site keeps as the tree's literal node does: transitions are
		// never undone, so the values go in place.
		dst := o.vals()[:n]
		copy(dst, vals)
		o.setVals(dst)
		o.shape = final
		if verifyShapes {
			checkShape(o)
		}
		regs[w.A()] = Obj(o)
		return
	}
	along := root != nil
	keys := c.rc.ObjKeys[c.rc.Aux[pc]]
	for i, key := range keys {
		name := cl.names[key]
		if next := o.transition(name, propDefault); next != nil {
			o.appendTransition(vals[i], next)
		} else {
			// No transition where the shape tree is full: the long way,
			// which for a fresh extensible object of distinct keys
			// cannot fail.
			along = false
			_ = r.defineOwnProp(o, name, vals[i], propDefault)
		}
	}
	if s := o.shape; along && s != nil && !s.unique && !s.loose && int(s.n) == n && int(o.valn) == n &&
		(s.index != nil || n <= linearScanLimit) {
		ic.next, ic.s1 = s, root
	}
	regs[w.A()] = Obj(o)
}

// regSwitch is switch: k's pc, past the instruction, made the body's
// where v equals a case.
func regSwitch(c *regCode, k *regKept, v Value, consts []Value) {
	w := regAt(c.code, k.pc-1)
	if to := c.switches[w.BC()].lookup(v, consts); to >= 0 {
		k.pc = int(to)
	}
}

// regSwitchTypeof is switch_typeof: k's pc made the body of v's type's
// name, where a case has it.
func regSwitchTypeof(c *regCode, k *regKept, v Value) {
	w := regAt(c.code, k.pc-1)
	if to := c.switches[w.BC()].typeTo[typeofKind[struct{}](v)]; to >= 0 {
		k.pc = int(to)
	}
}

// regTypeof is typeof: the type's name the runtime keeps, as
// typeofString answers it.
func (r *Runtime) regTypeof(c *regCode, k *regKept, regs *[1 << 16]Value) {
	w := regAt(c.code, k.pc-1)
	regs[w.A()] = Str(r.typeofString(regs[w.B()]))
}

// regNewArrayE is new_array_empty: the tree's node for the literal, made
// here, its site in the closure's arrays, named to native code by the
// stack form's pc. Making an array throws nothing, so the frame's pc is
// left as it is.
func (r *Runtime) regNewArrayE(f *frame, c *regCode, k *regKept, regs *[1 << 16]Value) {
	pc := k.pc - 1
	w := regAt(c.code, pc)
	n := arrayEmptyNode[struct{}]{site: int(w.B()), pc: int32(regSrc(c, pc))}
	regs[w.A()] = n.make(regTctx(r, f, f.cl))
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
