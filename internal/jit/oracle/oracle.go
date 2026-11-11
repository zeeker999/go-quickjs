// Package oracle runs register code's programs (compile.LowerReg) through
// the slot IR's evaluator with a heap of its own, which makes what the
// evaluator leaves to Go: globals, objects, arrays and strings, property
// and element reads and writes, closures and calls of the program's own
// functions, each called by evaluating its program in turn. It is the
// differential tests' model of the VM at a host exit, not a tier: what it
// does not model it declines, and the evaluation leaves there, at the
// register instruction it names (Machine.Stop).
package oracle

import (
	"encoding/json"
	"math"
	"slices"
	"strconv"
	"strings"
	"unicode/utf16"
	"unicode/utf8"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/jit/compile"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

// The kinds of an object.
const (
	kindObject = iota
	kindArray
	kindFunction
	// kindGlobal is the global object, whose properties are the machine's
	// globals.
	kindGlobal
	// kindCharCodeAt is String.prototype.charCodeAt's permission, which
	// ir's StringMethod reads.
	kindCharCodeAt
	// kindBuiltin is a built-in function the machine models, by its name
	// (builtin); kindIterator a for-of loop's cursor over an array's
	// elements or a string's code points.
	kindBuiltin
	kindIterator
)

type object struct {
	kind  int
	elems []ir.Value
	keys  []string
	props map[string]ir.Value
	// hidden are the keys of properties that are not enumerable.
	hidden map[string]bool
	proto  *object
	// noProto marks an object whose prototype is null, not
	// Object.prototype.
	noProto bool
	// errorKind is an error's constructor's name, Error's or another's
	// the machine makes; opaque marks one the machine threw, whose message
	// it does not model.
	errorKind string
	opaque    bool
	// builtins marks a built-in namespace, Math's or JSON's, of which the
	// machine models some functions: a read of another is declined, not
	// undefined.
	builtins bool
	// fn, cells and this are a function's: its template, its captured
	// bindings, and an arrow's this.
	fn    *bytecode.Function
	cells []*box
	this  ir.Value
	self  int
	// name is a built-in's; iterated and at an iterator's.
	name     string
	iterated ir.Value
	at       int
}

// A box is a captured binding: open, the frame's register it points at;
// closed, its own value (close_regs).
type box struct {
	p *ir.Value
	v ir.Value
}

func (o *object) get(k string) (ir.Value, bool) {
	for p := o; p != nil; p = p.proto {
		if v, ok := p.props[k]; ok {
			return v, true
		}
	}
	return ir.Value{}, false
}

func (o *object) set(k string, v ir.Value) {
	if o.props == nil {
		o.props = map[string]ir.Value{}
	}
	if _, ok := o.props[k]; !ok {
		o.keys = append(o.keys, k)
	}
	o.props[k] = v
}

type frame struct {
	p     *ir.Program
	fn    *bytecode.Function
	rc    *bytecode.RegCode
	cells []*box
	args  []ir.Value
	// open are the boxes of the frame's registers closures captured, by
	// register, until a scope's end closes them.
	open map[int]*box
	// pc is the instruction the machine runs in the frame, or the call it
	// waits at; slots are its registers.
	pc    int
	slots []ir.Value
}

// An unwind is an exception on its way to the handler of frame, which
// takes value.
type unwind struct {
	frame *frame
	value ir.Value
}

// A Machine is a heap and the frames evaluating over it.
type Machine struct {
	// Global, if set, is asked for a global the code reads that the machine
	// has none of yet; its answer is kept.
	Global func(name string) (ir.Value, bool)
	// Budget is how many instructions the machine's calls may run in all.
	Budget uint64
	// Stop is the register instruction, or the IR instruction, the last
	// run declined, the innermost: where the evaluation left.
	Stop string
	// Thrown, where the last call left by an exception the machine is sure
	// of and no handler of its frames would catch, is a JavaScript
	// condition on the exception, e, that holds of it: what kind of error
	// it is, and its message where the code gave one, or the primitive it
	// is.
	Thrown string

	objs []*object
	strs []string
	// unwinding is the exception the frames leave for a handler of one of
	// theirs, nil for none.
	unwinding *unwind
	// bytes is about how much the heap holds, which the machine bounds
	// (maxBytes): what would grow it past declines.
	bytes   int
	globals map[string]ir.Value
	progs   map[*bytecode.Function]*ir.Program
	frames  []*frame
	steps   uint64
	global  ir.Value
}

// New is a machine with nothing on its heap but the global object.
func New() *Machine {
	m := &Machine{globals: map[string]ir.Value{}, progs: map[*bytecode.Function]*ir.Program{}, Budget: 1 << 20}
	m.global = m.alloc(&object{kind: kindGlobal})
	return m
}

func (m *Machine) alloc(o *object) ir.Value {
	m.bytes += 64 + 16*len(o.elems)
	o.self = len(m.objs)
	m.objs = append(m.objs, o)
	return ir.Value{Kind: ir.Opaque, Bits: uint64(o.self)}
}

func (m *Machine) obj(v ir.Value) *object {
	if v.Kind != ir.Opaque || v.Bits >= uint64(len(m.objs)) {
		return nil
	}
	return m.objs[v.Bits]
}

// maxBytes and maxString bound the machine's heap and a string on it: a
// program building a string of a turn of each of a million iterations is
// the interpreter's to run, not a model's.
const (
	maxBytes  = 16 << 20
	maxString = 1 << 16
)

// String is s on the heap.
func (m *Machine) String(s string) ir.Value {
	m.bytes += len(s) + 16
	m.strs = append(m.strs, s)
	return ir.Value{Kind: ir.String, Bits: uint64(len(m.strs) - 1)}
}

func (m *Machine) str(v ir.Value) (string, bool) {
	if v.Kind != ir.String || v.Bits >= uint64(len(m.strs)) {
		return "", false
	}
	return m.strs[v.Bits], true
}

// Array is an array of vs.
func (m *Machine) Array(vs []ir.Value) ir.Value {
	return m.alloc(&object{kind: kindArray, elems: slices.Clone(vs)})
}

// Object is a plain object of keys and their values, in order.
func (m *Machine) Object(keys []string, vals []ir.Value) ir.Value {
	m.bytes += 64 * len(keys)
	o := &object{}
	for i, k := range keys {
		o.set(k, vals[i])
	}
	return m.alloc(o)
}

// Function is a closure of fn with no captured bindings.
func (m *Machine) Function(fn *bytecode.Function) ir.Value {
	return m.alloc(&object{kind: kindFunction, fn: fn, cells: make([]*box, len(fn.Upvalues))})
}

// SetGlobal makes name's global v.
func (m *Machine) SetGlobal(name string, v ir.Value) { m.globals[name] = v }

// Text is v as a JavaScript literal, for a primitive: a number, a string,
// a boolean, null or undefined.
func (m *Machine) Text(v ir.Value) (string, bool) {
	switch v.Kind {
	case ir.Number:
		x := math.Float64frombits(v.Bits)
		if x == 0 && math.Signbit(x) {
			return "-0", true
		}
		return NumberString(x), true
	case ir.String:
		s, ok := m.str(v)
		if !ok {
			return "", false
		}
		q, err := json.Marshal(s)
		return string(q), err == nil && utf8.ValidString(s)
	case ir.Boolean:
		return strconv.FormatBool(v.Bits != 0), true
	case ir.Null:
		return "null", true
	case ir.Undefined:
		return "undefined", true
	}
	return "", false
}

// Call calls fnv with this and args, as a call of the language does, and
// returns its value: ok false where the evaluation left before the end.
func (m *Machine) Call(fnv, this ir.Value, args []ir.Value) (ir.Value, bool, error) {
	m.Stop, m.Thrown, m.unwinding = "", "", nil
	return m.call(fnv, this, args, false)
}

func (m *Machine) program(fn *bytecode.Function) (*ir.Program, string) {
	if p, ok := m.progs[fn]; ok {
		if p == nil {
			return nil, "refused"
		}
		return p, ""
	}
	if fn.Reg == nil {
		m.progs[fn] = nil
		return nil, "no register code"
	}
	p, err := compile.LowerRegInline(fn.Reg, fn)
	if err == nil {
		err = p.Validate()
	}
	if err != nil {
		m.progs[fn] = nil
		return nil, "refused"
	}
	m.progs[fn] = p
	return p, ""
}

func (m *Machine) call(fnv, this ir.Value, args []ir.Value, construct bool) (ir.Value, bool, error) {
	f := m.obj(fnv)
	if f != nil && f.kind == kindBuiltin && !construct {
		v, ok := m.builtin(f, this, args)
		if !ok {
			m.decline("call of " + f.name)
		}
		return v, ok, nil
	}
	if f == nil && fnv.Kind != ir.Opaque || f != nil && (f.kind == kindObject || f.kind == kindArray) {
		m.typeError()
		return ir.Value{}, false, nil
	}
	if f == nil || f.kind != kindFunction || f.fn == nil || len(m.frames) >= 200 {
		m.decline("call of no function")
		return ir.Value{}, false, nil
	}
	fn := f.fn
	switch fn.Kind {
	case bytecode.KindNormal, bytecode.KindArrow, bytecode.KindMethod:
	case bytecode.KindConstructor:
		if !construct {
			m.decline("call of a class without new")
			return ir.Value{}, false, nil
		}
	default:
		m.decline("call of a special function")
		return ir.Value{}, false, nil
	}
	if fn.Async || fn.Generator || construct && fn.Kind != bytecode.KindNormal && fn.Kind != bytecode.KindConstructor {
		m.decline("call of a special function")
		return ir.Value{}, false, nil
	}
	p, why := m.program(fn)
	if p == nil {
		m.decline("call of a function " + why)
		return ir.Value{}, false, nil
	}
	slots := make([]ir.Value, p.Locals)
	slots[0] = fnv
	switch {
	case fn.Kind == bytecode.KindArrow:
		this = f.this
	case !fn.Strict && (this.Kind == ir.Undefined || this.Kind == ir.Null):
		this = m.global
	}
	slots[1] = this
	for i := range p.Params {
		if i < len(args) {
			slots[p.ParamBase+i] = args[i]
		}
	}
	if m.bytes > maxBytes {
		m.decline("heap budget")
		return ir.Value{}, false, nil
	}
	m.frames = append(m.frames, &frame{p: p, fn: fn, rc: fn.Reg, cells: f.cells, args: args, slots: slots})
	defer func() { m.frames = m.frames[:len(m.frames)-1] }()
	if m.steps >= m.Budget {
		m.decline("budget")
		return ir.Value{}, false, nil
	}
	exit, err := p.EvaluateHost(slots, nil, m, 0, m.Budget-m.steps)
	m.steps += exit.Steps
	if err != nil {
		return ir.Value{}, false, err
	}
	if exit.Kind != ir.Returned {
		if exit.Kind == ir.BudgetExit {
			m.decline("budget")
		} else if m.Stop == "" && m.unwinding == nil {
			m.Stop = fn.Reg.Format(int(exit.State.PC))
		}
		return ir.Value{}, false, nil
	}
	return exit.Value, true, nil
}

// kindOf is what the receiver of a property's read or write in is, for
// Stop: an array, a string, a function, an object, a primitive.
func (m *Machine) kindOf(slots []ir.Value, in *ir.Instruction) string {
	if in.Op == ir.BindingRead || in.Op == ir.BindingWrite || in.Left.Slot < 0 {
		return ""
	}
	v := slots[in.Left.Slot]
	if o := m.obj(v); o != nil {
		return [...]string{" of an object", " of an array", " of a function", " of the global", " of a method", " of a built-in", " of a cursor"}[o.kind]
	}
	t, _ := m.typeOf(v)
	return " of a " + t
}

// raise throws v from what the machine runs: to the innermost frame with
// a handler where it is, which takes it (Run); or, none, out of every
// frame, which check -- a JavaScript condition on the exception, e, that
// holds of it -- says what it is (Thrown); with no check, the machine
// declines it.
func (m *Machine) raise(v ir.Value, check string) {
	if m.Stop != "" || m.Thrown != "" || m.unwinding != nil {
		return
	}
	for i := len(m.frames) - 1; i >= 0; i-- {
		if f := m.frames[i]; f.p.HandlerAt != nil && f.p.HandlerAt[f.pc] != 0 {
			m.unwinding = &unwind{frame: f, value: v}
			return
		}
	}
	if check == "" {
		m.decline("throw of an object")
		return
	}
	m.Thrown = check
	m.Stop = "throw"
}

// typeError throws a TypeError, whose message the machine does not model.
func (m *Machine) typeError() {
	m.raise(m.alloc(&object{errorKind: "TypeError", opaque: true}), "e instanceof TypeError")
}

// decline notes why the machine does not run something, the innermost.
func (m *Machine) decline(why string) {
	if m.Stop == "" {
		m.Stop = why
	}
}

// Run is ir.Runner's: it runs in, instruction pc of p, the top frame's,
// as Go does at a host exit, where it models it.
func (m *Machine) Run(p *ir.Program, pc int, in *ir.Instruction, slots []ir.Value) (int, bool, error) {
	f := m.frames[len(m.frames)-1]
	if f.p != p {
		return 0, false, nil
	}
	f.pc = pc
	if m.bytes > maxBytes {
		m.decline("heap budget")
		return 0, false, nil
	}
	next, ok := m.run(f, pc, in, slots)
	if !ok && m.unwinding != nil {
		if u := m.unwinding; u.frame == f {
			// The handler of this frame's takes the exception: a catch
			// clause, or a finally clause's copy, which throws it again.
			k := p.HandlerAt[pc]
			slots[f.rc.Handlers[len(f.rc.Handlers)-int(k)].Reg] = u.value
			m.unwinding = nil
			return int(p.Handlers[k-1].PC), true, nil
		}
		return 0, false, nil
	}
	if !ok && m.Stop == "" {
		w := f.rc.Code[p.Origin[pc]]
		m.Stop = w.Op().String()
		if w.Op() == bytecode.RCold {
			m.Stop = bytecode.RColdOp(w.X()).String()
		}
		// What it read, where a name says: a global's, a property's.
		switch in.Op {
		case ir.BindingRead, ir.BindingWrite, ir.PropertyRead, ir.ReferenceRead, ir.PropertyWrite:
			m.Stop += " " + f.fn.Names[in.Key] + m.kindOf(slots, in)
		}
	}
	return next, ok, nil
}

func (m *Machine) run(f *frame, pc int, in *ir.Instruction, slots []ir.Value) (int, bool) {
	read := func(o ir.Operand) ir.Value {
		if o.Slot < 0 {
			return o.Literal
		}
		return slots[o.Slot]
	}
	set := func(d int, v ir.Value) (int, bool) {
		slots[d] = v
		return pc + 1, true
	}
	w := f.rc.Code[f.p.Origin[pc]]
	switch in.Op {
	case ir.Host:
		return m.host(f, pc, w, slots)
	case ir.Copy:
		// A dead zone's check: the interpreter's to throw.
		return 0, false
	case ir.BindingRead:
		name := f.fn.Names[in.Key]
		v, ok := m.globals[name]
		if !ok {
			if v, ok = m.builtinGlobal(name); ok {
				m.globals[name] = v
			}
		}
		if !ok && m.Global != nil {
			v, ok = m.Global(name)
			if ok {
				m.globals[name] = v
			}
		}
		if !ok {
			return 0, false
		}
		return set(in.Dest, v)
	case ir.BindingWrite:
		name := f.fn.Names[in.Key]
		if _, ok := m.globals[name]; !ok && m.Global != nil {
			if _, ok := m.Global(name); !ok && f.fn.Strict {
				return 0, false
			}
		}
		m.globals[name] = read(in.Left)
		return pc + 1, true
	case ir.StringConst:
		s := ""
		if w.Op() != bytecode.RLoadEmpty {
			s = f.fn.Constants[in.Key].Str
		}
		return set(in.Dest, m.String(s))
	case ir.UpvalueRead, ir.UpvalueWrite:
		if int(in.Key) >= len(f.cells) || f.cells[in.Key] == nil {
			return 0, false
		}
		c := f.cells[in.Key].p
		if in.Op == ir.UpvalueRead {
			if in.Check && c.Kind == ir.Uninitialized {
				return 0, false
			}
			return set(in.Dest, *c)
		}
		if in.Check && c.Kind == ir.Uninitialized {
			return 0, false
		}
		*c = read(in.Left)
		return pc + 1, true
	case ir.ObjectLiteral:
		if w.Op() == bytecode.RCold {
			// A regular expression literal's.
			return 0, false
		}
		return set(in.Dest, m.alloc(&object{}))
	case ir.ArrayLiteral:
		return set(in.Dest, m.Array(slots[in.Dest:in.Dest+in.Extra]))
	case ir.FieldDefine, ir.PropertyWrite:
		if k := read(in.Left).Kind; in.Op == ir.PropertyWrite && (k == ir.Undefined || k == ir.Null) {
			m.typeError()
			return 0, false
		}
		o, key := m.obj(read(in.Left)), f.fn.Names[in.Key]
		if v := read(in.Left); in.Op == ir.PropertyWrite && v.Kind != ir.Opaque && v.Kind != ir.Uninitialized {
			// A primitive's property: nothing in sloppy code, a TypeError
			// in strict.
			if f.fn.Strict {
				m.typeError()
				return 0, false
			}
			return pc + 1, true
		}
		if o == nil || o.kind != kindObject && o.kind != kindFunction || in.Private {
			return 0, false
		}
		m.bytes += 64
		o.set(key, read(in.Right))
		return pc + 1, true
	case ir.PropertyRead, ir.ReferenceRead:
		if in.Private || in.When {
			return 0, false
		}
		if k := read(in.Left).Kind; k == ir.Undefined || k == ir.Null {
			m.typeError()
			return 0, false
		}
		v, ok := m.property(read(in.Left), f.fn.Names[in.Key])
		if !ok {
			return 0, false
		}
		return set(in.Dest, v)
	case ir.ArrayRead:
		v, ok := m.element(read(in.Left), read(in.Right))
		if !ok {
			return 0, false
		}
		return set(in.Dest, v)
	case ir.ArrayWrite:
		o, k := m.obj(read(in.Left)), read(in.Right)
		if o != nil && o.kind == kindObject {
			ks, ok := m.key(k)
			if !ok || objectProto[ks] {
				return 0, false
			}
			o.set(ks, read(in.Third))
			return pc + 1, true
		}
		i, ok := index(k)
		if o == nil || o.kind != kindArray || !ok || i > len(o.elems) {
			return 0, false
		}
		if i == len(o.elems) {
			m.bytes += 16
			o.elems = append(o.elems, read(in.Third))
		} else {
			o.elems[i] = read(in.Third)
		}
		return pc + 1, true
	case ir.ArrayLength:
		v, ok := m.property(read(in.Left), "length")
		if !ok {
			return 0, false
		}
		return set(in.Dest, v)
	case ir.ArrayKey:
		if k := read(in.Right); k.Kind == ir.Number || k.Kind == ir.String {
			if b := read(in.Left); b.Kind != ir.Null && b.Kind != ir.Undefined {
				return pc + 1, true
			}
		}
		return 0, false
	case ir.Binary, ir.StringAdd:
		v, ok := m.binary(in.Operator, read(in.Left), read(in.Right), in.Strict, in.Op == ir.StringAdd)
		if !ok {
			return 0, false
		}
		return set(in.Dest, v)
	case ir.Update:
		x, ok := number(read(in.Left))
		if !ok {
			return 0, false
		}
		d := 1.0
		if in.Operator == ir.Sub {
			d = -1
		}
		slots[in.Dest] = ir.Float(x + d)
		if in.Extra >= 0 {
			if in.Postfix {
				slots[in.Extra] = ir.Float(x)
			} else {
				slots[in.Extra] = ir.Float(x + d)
			}
		}
		return pc + 1, true
	case ir.IterStep:
		it := m.obj(read(in.Left))
		if it == nil || it.kind != kindIterator {
			return 0, false
		}
		if in.When {
			// A destructuring's: undefined once done; its rest, an array of
			// what is left.
			if in.Rest {
				var rest []ir.Value
				for {
					v, done := m.step(it)
					if done {
						break
					}
					rest = append(rest, v)
				}
				return set(in.Dest, m.Array(rest))
			}
			v, _ := m.step(it)
			return set(in.Dest, v)
		}
		v, done := m.step(it)
		if done {
			return in.Target, true
		}
		return set(in.Dest, v)
	case ir.IterClose:
		if it := m.obj(read(in.Left)); it != nil && it.kind == kindIterator {
			return pc + 1, true
		}
		return 0, false
	case ir.Unary:
		if in.Operator != ir.Not {
			x, ok := number(read(in.Left))
			if !ok {
				return 0, false
			}
			switch in.Operator {
			case ir.Neg:
				x = -x
			case ir.BitNot:
				x = float64(^int32(ir.ToUint32(x)))
			case ir.Int32:
				x = float64(int32(ir.ToUint32(x)))
			}
			return set(in.Dest, ir.Float(x))
		}
		t, ok := m.truth(read(in.Left))
		if !ok {
			return 0, false
		}
		return set(in.Dest, ir.Bool(!t))
	case ir.Branch:
		var c bool
		switch in.Operator {
		case ir.Truth:
			t, ok := m.truth(read(in.Left))
			if !ok {
				return 0, false
			}
			c = t
		case ir.Nullish:
			k := read(in.Left).Kind
			c = k == ir.Null || k == ir.Undefined
		default:
			v, ok := m.binary(in.Operator, read(in.Left), read(in.Right), in.Strict, false)
			if !ok {
				return 0, false
			}
			c = v.Bits != 0
		}
		if c == in.When {
			return in.Target, true
		}
		return pc + 1, true
	case ir.TypeTest:
		t, ok := m.typeOf(read(in.Left))
		if !ok {
			return 0, false
		}
		eq := typeIndex[t] == int(in.Key)
		return set(in.Dest, ir.Bool(eq == in.When))
	case ir.StringConcat:
		var sb strings.Builder
		for i := range in.Extra {
			s, ok := m.toString(slots[in.Dest+i])
			if !ok {
				return 0, false
			}
			sb.WriteString(s)
			if sb.Len() > maxString {
				return 0, false
			}
		}
		return set(in.Dest, m.String(sb.String()))
	case ir.StringPart:
		v := read(in.Left)
		if v.Kind == ir.Opaque || v.Kind == ir.Uninitialized {
			return 0, false
		}
		return set(in.Dest, v)
	case ir.StringMethod:
		if _, ok := m.str(read(in.Left)); !ok {
			return 0, false
		}
		return set(in.Dest, m.alloc(&object{kind: kindCharCodeAt}))
	case ir.StringCode:
		c := m.obj(read(in.Left))
		s, ok := m.str(read(in.Right))
		i, iok := index(read(in.Third))
		if c == nil || c.kind != kindCharCodeAt || !ok {
			return 0, false
		}
		u := utf16.Encode([]rune(s))
		if !iok || i >= len(u) {
			return set(in.Dest, ir.Float(math.NaN()))
		}
		return set(in.Dest, ir.Float(float64(u[i])))
	case ir.Closure:
		tmpl := f.fn.Constants[in.Key].Fn
		caps := f.rc.Caps[in.Key]
		o := &object{kind: kindFunction, fn: tmpl, cells: make([]*box, len(tmpl.Upvalues)), this: slots[1]}
		for i, u := range tmpl.Upvalues {
			switch {
			case u.FromParent && i < len(caps) && int(caps[i]) < f.p.Regs:
				if u.ByValue {
					b := &box{v: slots[caps[i]]}
					b.p = &b.v
					o.cells[i] = b
				} else {
					r := int(caps[i])
					b := f.open[r]
					if b == nil {
						b = &box{p: &slots[r]}
						if f.open == nil {
							f.open = map[int]*box{}
						}
						f.open[r] = b
					}
					o.cells[i] = b
				}
			case !u.FromParent && int(u.Index) < len(f.cells):
				o.cells[i] = f.cells[u.Index]
			}
		}
		return set(in.Dest, m.alloc(o))
	case ir.ArgumentRead:
		if in.When {
			return set(in.Dest, ir.Float(float64(len(f.args))))
		}
		i, ok := index(read(in.Left))
		if !ok {
			return 0, false
		}
		if i >= len(f.args) {
			return set(in.Dest, ir.Value{})
		}
		return set(in.Dest, f.args[i])
	case ir.HasProperty:
		o, k := m.obj(read(in.Right)), read(in.Left)
		ks, ok := m.key(k)
		if o == nil || o.kind != kindObject || !ok || objectProto[ks] {
			return 0, false
		}
		_, has := o.get(ks)
		return set(in.Dest, ir.Bool(has))
	case ir.StringFreeze:
		return pc + 1, true
	}
	return 0, false
}

// host runs the register instruction w, which the program leaves to Go
// whole (an ir.Host).
func (m *Machine) host(f *frame, pc int, w bytecode.RInstr, slots []ir.Value) (int, bool) {
	a, b, c := int(w.A()), int(w.B()), int(w.C())
	switch w.Op() {
	case bytecode.RCallW, bytecode.RCallMW, bytecode.RNewW:
		args := slices.Clone(slots[a+2 : a+2+int(w.X())])
		var v ir.Value
		var ok bool
		var err error
		if w.Op() == bytecode.RNewW {
			v, ok, err = m.construct(slots[a], args)
		} else {
			this := ir.Value{}
			if w.Op() == bytecode.RCallMW {
				this = slots[a+1]
			}
			v, ok, err = m.call(slots[a], this, args, false)
		}
		if err != nil || !ok {
			return 0, false
		}
		slots[b] = v
		return pc + 1, true
	case bytecode.RNewArray:
		slots[a] = m.Array(slots[b : b+c])
		return pc + 1, true
	case bytecode.RNewArrayK:
		var vs []ir.Value
		for _, e := range f.rc.Literals[f.rc.Aux[f.p.Origin[pc]]] {
			switch e.Kind {
			case bytecode.LitTrue:
				vs = append(vs, ir.Bool(true))
			case bytecode.LitFalse:
				vs = append(vs, ir.Bool(false))
			case bytecode.LitNull:
				vs = append(vs, ir.Value{Kind: ir.Null})
			case bytecode.LitUndefined:
				vs = append(vs, ir.Value{})
			default:
				k := f.fn.Constants[e.K]
				switch k.Kind {
				case bytecode.ConstNumber:
					vs = append(vs, ir.Float(k.Num))
				case bytecode.ConstString:
					vs = append(vs, m.String(k.Str))
				default:
					return 0, false
				}
			}
		}
		slots[a] = m.Array(vs)
		return pc + 1, true
	case bytecode.RCold:
		switch bytecode.RColdOp(w.X()) {
		case bytecode.RColdForOfStart:
			v := slots[b]
			if o := m.obj(v); (o == nil || o.kind != kindArray) && v.Kind != ir.String {
				return 0, false
			}
			slots[a] = m.alloc(&object{kind: kindIterator, iterated: v})
			return pc + 1, true
		case bytecode.RColdForInStart:
			// A plain object's enumerable keys, its own and then its
			// prototypes', none of them an index, which come first.
			o := m.obj(slots[b])
			if o == nil || o.kind != kindObject || o.errorKind != "" {
				return 0, false
			}
			var keys []ir.Value
			seen := map[string]bool{}
			for q := o; q != nil; q = q.proto {
				for _, k := range q.keys {
					if isIndex(k) {
						return 0, false
					}
					if !seen[k] && !q.hidden[k] {
						keys = append(keys, m.String(k))
					}
					seen[k] = true
				}
			}
			slots[a] = m.alloc(&object{kind: kindIterator, iterated: m.Array(keys)})
			return pc + 1, true
		case bytecode.RColdCloseRegs:
			// A scope's end: what closures captured of its registers is
			// theirs from here, the registers free for the next entry.
			for r := a; r < a+b; r++ {
				if bx := f.open[r]; bx != nil {
					bx.v = *bx.p
					bx.p = &bx.v
					delete(f.open, r)
				}
			}
			return pc + 1, true
		}
		return 0, false
	case bytecode.RThrow:
		v := slots[a]
		check := ""
		if t, ok := m.Text(v); ok {
			check = "Object.is(e, " + t + ")"
		} else if e := m.obj(v); e != nil && e.errorKind != "" && !e.opaque {
			check = "e.constructor === " + e.errorKind
			if msg, ok := e.props["message"]; ok {
				if t, ok := m.Text(msg); ok {
					check += " && e.message === " + t
				} else {
					check = ""
				}
			} else {
				check += ` && !Object.prototype.hasOwnProperty.call(e, "message")`
			}
		} else if e != nil && e.opaque {
			check = "e instanceof " + e.errorKind
		}
		m.raise(v, check)
		return 0, false
	case bytecode.RInstanceOf:
		ctor := m.obj(slots[c])
		if ctor == nil || ctor.kind != kindFunction || ctor.fn.Kind != bytecode.KindNormal && ctor.fn.Kind != bytecode.KindConstructor {
			return 0, false
		}
		proto := m.obj(m.prototype(ctor))
		r := false
		if o := m.obj(slots[b]); o != nil {
			if o.kind != kindObject {
				return 0, false
			}
			for q := o.proto; q != nil; q = q.proto {
				r = r || q == proto
			}
		}
		slots[a] = ir.Bool(r)
		return pc + 1, true
	case bytecode.RTypeof, bytecode.RUnary:
		if w.Op() == bytecode.RUnary && bytecode.Op(w.X()) != bytecode.OpTypeOf {
			return 0, false
		}
		t, ok := m.typeOf(slots[b])
		if !ok {
			return 0, false
		}
		slots[a] = m.String(t)
		return pc + 1, true
	}
	return 0, false
}

func (m *Machine) construct(fnv ir.Value, args []ir.Value) (ir.Value, bool, error) {
	f := m.obj(fnv)
	if f != nil && f.kind == kindBuiltin && errorNames[f.name] {
		// An error: its kind, and its message, a string or none.
		e := &object{errorKind: f.name}
		if len(args) > 0 && args[0].Kind != ir.Undefined {
			msg, ok := m.str(args[0])
			if !ok {
				m.decline("an error's message of no string")
				return ir.Value{}, false, nil
			}
			e.set("message", m.String(msg))
			e.hidden = map[string]bool{"message": true}
		}
		return m.alloc(e), true, nil
	}
	if f == nil || f.kind != kindFunction || f.fn.Kind != bytecode.KindNormal && f.fn.Kind != bytecode.KindConstructor {
		m.decline("construction of no plain function")
		return ir.Value{}, false, nil
	}
	this := m.alloc(&object{proto: m.obj(m.prototype(f))})
	v, ok, err := m.call(fnv, this, args, true)
	if err != nil || !ok {
		return ir.Value{}, ok, err
	}
	if v.Kind == ir.Opaque {
		return v, true, nil
	}
	return this, true, nil
}

// prototype is plain function f's prototype property, made the first
// time it is read.
func (m *Machine) prototype(f *object) ir.Value {
	pv, ok := f.props["prototype"]
	if !ok {
		pv = m.alloc(&object{})
		m.obj(pv).set("constructor", ir.Value{Kind: ir.Opaque, Bits: uint64(f.self)})
		f.set("prototype", pv)
	}
	return pv
}

// objectProto are Object.prototype's properties, which an object the
// machine makes does not have: a read of one is declined.
var objectProto = map[string]bool{"toString": true, "valueOf": true, "hasOwnProperty": true, "isPrototypeOf": true,
	"propertyIsEnumerable": true, "toLocaleString": true, "constructor": true, "__proto__": true,
	"__defineGetter__": true, "__defineSetter__": true, "__lookupGetter__": true, "__lookupSetter__": true}

// property is v[key] where the machine knows it.
func (m *Machine) property(v ir.Value, key string) (ir.Value, bool) {
	if s, ok := m.str(v); ok {
		if key == "length" {
			return ir.Float(float64(len(utf16.Encode([]rune(s))))), true
		}
		if stringMethods[key] {
			return m.builtinOf("String.prototype." + key), true
		}
		if isIndex(key) || objectProto[key] || stringProto[key] {
			return ir.Value{}, false
		}
		return ir.Value{}, true
	}
	switch v.Kind {
	case ir.Number, ir.Boolean:
		if key == "toString" && v.Kind == ir.Number {
			return m.builtinOf("Number.prototype.toString"), true
		}
		// Number.prototype's and Boolean.prototype's, which the machine
		// does not model; anything else a primitive has not.
		if objectProto[key] || numberProto[key] {
			return ir.Value{}, false
		}
		return ir.Value{}, true
	}
	o := m.obj(v)
	if o == nil {
		return ir.Value{}, false
	}
	switch o.kind {
	case kindArray:
		if key == "length" {
			return ir.Float(float64(len(o.elems))), true
		}
		if arrayMethods[key] {
			return m.builtinOf("Array.prototype." + key), true
		}
		return ir.Value{}, false
	case kindBuiltin:
		r, ok := o.get(key)
		return r, ok
	case kindGlobal:
		g, ok := m.globals[key]
		if !ok && m.Global != nil {
			g, ok = m.Global(key)
		}
		return g, ok
	case kindObject, kindFunction:
		if o.errorKind != "" {
			switch key {
			case "message":
				if o.opaque {
					return ir.Value{}, false
				}
				if r, ok := o.props["message"]; ok {
					return r, true
				}
				return m.String(""), true
			case "name":
				return m.String(o.errorKind), true
			}
			return ir.Value{}, false
		}
		if o.kind == kindFunction && (key == "call" || key == "apply") {
			if _, own := o.get(key); !own {
				return m.builtinOf("Function.prototype." + key), true
			}
		}
		if o.kind == kindFunction && key == "prototype" && (o.fn.Kind == bytecode.KindNormal || o.fn.Kind == bytecode.KindConstructor) {
			return m.prototype(o), true
		}
		if r, ok := o.get(key); ok {
			return r, true
		}
		if o.kind == kindFunction || o.builtins || objectProto[key] && !o.chainEndsNull() {
			return ir.Value{}, false
		}
		return ir.Value{}, true
	}
	return ir.Value{}, false
}

// element is v[k] where the machine knows it.
func (m *Machine) element(v, k ir.Value) (ir.Value, bool) {
	if s, ok := m.str(v); ok {
		i, ok := index(k)
		if !ok {
			return ir.Value{}, false
		}
		u := utf16.Encode([]rune(s))
		if i >= len(u) {
			return ir.Value{}, true
		}
		return m.String(string(utf16.Decode(u[i : i+1]))), utf16.IsSurrogate(rune(u[i])) == false
	}
	o := m.obj(v)
	if o == nil {
		return ir.Value{}, false
	}
	if o.kind == kindArray {
		i, ok := index(k)
		if !ok {
			return ir.Value{}, false
		}
		if i >= len(o.elems) {
			return ir.Value{}, true
		}
		return o.elems[i], true
	}
	ks, ok := m.key(k)
	if !ok {
		return ir.Value{}, false
	}
	return m.property(v, ks)
}

func (m *Machine) key(k ir.Value) (string, bool) {
	if k.Kind == ir.String {
		return m.str(k)
	}
	if k.Kind == ir.Number {
		return NumberString(math.Float64frombits(k.Bits)), true
	}
	return "", false
}

// index is k as an array index, where it is one.
func index(k ir.Value) (int, bool) {
	if k.Kind != ir.Number {
		return 0, false
	}
	x := math.Float64frombits(k.Bits)
	if x < 0 || x >= 1<<31 || x != math.Trunc(x) {
		return 0, false
	}
	return int(x), true
}

func (m *Machine) truth(v ir.Value) (bool, bool) {
	switch v.Kind {
	case ir.Undefined, ir.Null:
		return false, true
	case ir.Boolean:
		return v.Bits != 0, true
	case ir.Number:
		x := math.Float64frombits(v.Bits)
		return x != 0 && !math.IsNaN(x), true
	case ir.String:
		s, ok := m.str(v)
		return s != "", ok
	case ir.Opaque:
		return m.obj(v) != nil, m.obj(v) != nil
	}
	return false, false
}

// typeIndex is each of typeof's answers as ir's TypeTest names it.
var typeIndex = map[string]int{"number": ir.TypeNumber, "string": ir.TypeString, "boolean": ir.TypeBoolean,
	"undefined": ir.TypeUndefined, "function": ir.TypeFunction, "object": ir.TypeObject}

func (m *Machine) typeOf(v ir.Value) (string, bool) {
	switch v.Kind {
	case ir.Number:
		return "number", true
	case ir.String:
		return "string", true
	case ir.Boolean:
		return "boolean", true
	case ir.Undefined:
		return "undefined", true
	case ir.Null:
		return "object", true
	case ir.Opaque:
		if o := m.obj(v); o != nil {
			if o.kind == kindFunction || o.kind == kindCharCodeAt || o.kind == kindBuiltin {
				return "function", true
			}
			return "object", true
		}
	}
	return "", false
}

// toString is ToString of a primitive.
func (m *Machine) toString(v ir.Value) (string, bool) {
	switch v.Kind {
	case ir.String:
		return m.str(v)
	case ir.Number:
		return NumberString(math.Float64frombits(v.Bits)), true
	case ir.Boolean:
		return strconv.FormatBool(v.Bits != 0), true
	case ir.Null:
		return "null", true
	case ir.Undefined:
		return "undefined", true
	}
	return "", false
}

// number is ToNumber of a primitive that is no string.
func number(v ir.Value) (float64, bool) {
	switch v.Kind {
	case ir.Number:
		return math.Float64frombits(v.Bits), true
	case ir.Boolean:
		return float64(v.Bits), true
	case ir.Null:
		return 0, true
	case ir.Undefined:
		return math.NaN(), true
	}
	return 0, false
}

// binary is a op b where the machine knows it: a string's addition, a
// comparison of strings, an equality of any two values, arithmetic of
// primitives that are no strings.
func (m *Machine) binary(op ir.Operator, a, b ir.Value, strict, add bool) (ir.Value, bool) {
	sa, aStr := m.str(a)
	sb, bStr := m.str(b)
	switch {
	case op == ir.Add && (aStr || bStr):
		x, ok1 := m.toString(a)
		y, ok2 := m.toString(b)
		if !ok1 || !ok2 || len(x)+len(y) > maxString {
			return ir.Value{}, false
		}
		return m.String(x + y), true
	case op == ir.Eq || op == ir.Ne:
		eq, ok := m.equal(a, b, strict)
		if !ok {
			return ir.Value{}, false
		}
		return ir.Bool(eq == (op == ir.Eq)), true
	case aStr && bStr && op >= ir.Lt && op <= ir.Ge:
		c := slices.Compare(utf16.Encode([]rune(sa)), utf16.Encode([]rune(sb)))
		var r bool
		switch op {
		case ir.Lt:
			r = c < 0
		case ir.Le:
			r = c <= 0
		case ir.Gt:
			r = c > 0
		case ir.Ge:
			r = c >= 0
		}
		return ir.Bool(r), true
	}
	if a.Kind == ir.Opaque || b.Kind == ir.Opaque || aStr || bStr {
		return ir.Value{}, false
	}
	x, ok1 := number(a)
	y, ok2 := number(b)
	if !ok1 || !ok2 || op == ir.Pow {
		return ir.Value{}, false
	}
	return arith(op, x, y)
}

func arith(op ir.Operator, x, y float64) (ir.Value, bool) {
	if op >= ir.BitAnd && op <= ir.UShr {
		l, r := ir.ToUint32(x), ir.ToUint32(y)
		switch op {
		case ir.BitAnd:
			l &= r
		case ir.BitOr:
			l |= r
		case ir.BitXor:
			l ^= r
		case ir.Shl:
			l <<= r & 31
		case ir.Shr:
			l = uint32(int32(l) >> (r & 31))
		case ir.UShr:
			return ir.Float(float64(l >> (r & 31))), true
		}
		return ir.Float(float64(int32(l))), true
	}
	switch op {
	case ir.Add:
		return ir.Float(x + y), true
	case ir.Sub:
		return ir.Float(x - y), true
	case ir.Mul:
		return ir.Float(x * y), true
	case ir.Div:
		return ir.Float(x / y), true
	case ir.Mod:
		return ir.Float(math.Mod(x, y)), true
	case ir.Lt:
		return ir.Bool(x < y), true
	case ir.Le:
		return ir.Bool(x <= y), true
	case ir.Gt:
		return ir.Bool(x > y), true
	case ir.Ge:
		return ir.Bool(x >= y), true
	}
	return ir.Value{}, false
}

// equal is a == b, or a === b, where the machine knows it.
func (m *Machine) equal(a, b ir.Value, strict bool) (bool, bool) {
	nullish := func(v ir.Value) bool { return v.Kind == ir.Null || v.Kind == ir.Undefined }
	switch {
	case a.Kind == ir.Number && b.Kind == ir.Number:
		return math.Float64frombits(a.Bits) == math.Float64frombits(b.Bits), true
	case a.Kind == ir.String && b.Kind == ir.String:
		x, ok1 := m.str(a)
		y, ok2 := m.str(b)
		return x == y, ok1 && ok2
	case a.Kind == ir.Opaque && b.Kind == ir.Opaque:
		return a.Bits == b.Bits, true
	case nullish(a) && nullish(b):
		return !strict || a.Kind == b.Kind, true
	case a.Kind == ir.Boolean && b.Kind == ir.Boolean:
		return a.Bits == b.Bits, true
	case strict:
		return false, true
	case nullish(a) || nullish(b):
		// An object may be document.all's stand-in, which the machine
		// makes none of.
		return false, true
	case a.Kind != ir.String && b.Kind != ir.String && a.Kind != ir.Opaque && b.Kind != ir.Opaque:
		x, _ := number(a)
		y, _ := number(b)
		return x == y, true
	}
	return false, false
}

// NumberString is Number::toString(x), radix 10.
func NumberString(x float64) string {
	switch {
	case math.IsNaN(x):
		return "NaN"
	case math.IsInf(x, 1):
		return "Infinity"
	case math.IsInf(x, -1):
		return "-Infinity"
	case x == 0:
		return "0"
	}
	sign := ""
	if x < 0 {
		sign, x = "-", -x
	}
	e := strconv.FormatFloat(x, 'e', -1, 64)
	mant, exp, _ := strings.Cut(e, "e")
	digits := strings.Replace(mant, ".", "", 1)
	ex, _ := strconv.Atoi(exp)
	k, n := len(digits), ex+1
	switch {
	case k <= n && n <= 21:
		return sign + digits + strings.Repeat("0", n-k)
	case 0 < n && n <= 21:
		return sign + digits[:n] + "." + digits[n:]
	case -6 < n && n <= 0:
		return sign + "0." + strings.Repeat("0", -n) + digits
	}
	s := digits[:1]
	if k > 1 {
		s += "." + digits[1:]
	}
	if n-1 >= 0 {
		return sign + s + "e+" + strconv.Itoa(n-1)
	}
	return sign + s + "e-" + strconv.Itoa(1-n)
}

// step is the next value of iterator it, or done.
func (m *Machine) step(it *object) (ir.Value, bool) {
	if s, ok := m.str(it.iterated); ok {
		u := utf16.Encode([]rune(s))
		if it.at >= len(u) {
			return ir.Value{}, true
		}
		n := 1
		if utf16.IsSurrogate(rune(u[it.at])) && it.at+1 < len(u) {
			n = 2
		}
		v := m.String(string(utf16.Decode(u[it.at : it.at+n])))
		it.at += n
		return v, false
	}
	a := m.obj(it.iterated)
	if it.at >= len(a.elems) {
		return ir.Value{}, true
	}
	it.at++
	return a.elems[it.at-1], false
}

// The prototypes' properties a primitive's read does not model.
var (
	numberProto = map[string]bool{"toFixed": true, "toExponential": true, "toPrecision": true}
	stringProto = map[string]bool{"at": true, "charAt": true, "codePointAt": true, "concat": true, "endsWith": true,
		"includes": true, "lastIndexOf": true, "localeCompare": true, "match": true, "matchAll": true, "normalize": true,
		"padEnd": true, "padStart": true, "repeat": true, "replace": true, "replaceAll": true, "search": true, "split": true,
		"startsWith": true, "substr": true, "toLocaleLowerCase": true, "toLocaleUpperCase": true, "trim": true,
		"trimEnd": true, "trimStart": true, "isWellFormed": true, "toWellFormed": true, "anchor": true, "big": true,
		"blink": true, "bold": true, "fixed": true, "fontcolor": true, "fontsize": true, "italics": true, "link": true,
		"small": true, "strike": true, "sub": true, "sup": true, "trimLeft": true, "trimRight": true}
)

// The built-ins the machine models: arrays' and strings' methods it has,
// and Math's and String's functions.
var (
	arrayMethods  = map[string]bool{"push": true, "pop": true, "map": true, "forEach": true, "indexOf": true, "join": true, "slice": true}
	stringMethods = map[string]bool{"charCodeAt": true, "indexOf": true, "slice": true, "substring": true, "toLowerCase": true, "toUpperCase": true}
	mathFunctions = []string{"floor", "ceil", "abs", "sqrt", "min", "max", "round", "imul"}
)

// builtinOf is the built-in of name, one object for each machine.
func (m *Machine) builtinOf(name string) ir.Value {
	key := "\x00builtin " + name
	if v, ok := m.globals[key]; ok {
		return v
	}
	v := m.alloc(&object{kind: kindBuiltin, name: name})
	m.globals[key] = v
	return v
}

// builtinGlobal is the global name where the machine models it: Math and
// String.
func (m *Machine) builtinGlobal(name string) (ir.Value, bool) {
	switch name {
	case "Math":
		o := &object{builtins: true}
		for _, f := range mathFunctions {
			o.set(f, m.builtinOf("Math."+f))
		}
		o.set("PI", ir.Float(math.Pi))
		o.set("E", ir.Float(math.E))
		return m.alloc(o), true
	case "String":
		v := m.builtinOf("String")
		m.obj(v).set("fromCharCode", m.builtinOf("String.fromCharCode"))
		return v, true
	case "Error", "TypeError", "RangeError":
		return m.builtinOf(name), true
	case "JSON", "Object", "Array":
		v := m.builtinOf(name)
		if name == "JSON" {
			// An object, not a function.
			v = m.alloc(&object{builtins: true})
		}
		for _, f := range builtinStatics[name] {
			m.obj(v).set(f, m.builtinOf(name+"."+f))
		}
		return v, true
	}
	return ir.Value{}, false
}

// errorNames are the errors the machine makes.
var errorNames = map[string]bool{"Error": true, "TypeError": true, "RangeError": true}

// builtinStatics are the functions of JSON, Object and Array the machine
// models.
var builtinStatics = map[string][]string{"JSON": {"stringify", "parse"}, "Object": {"keys", "create", "getPrototypeOf", "is"},
	"Array": {"isArray"}}

// builtin is a call of built-in f with this and args, where the machine
// models it.
func (m *Machine) builtin(f *object, this ir.Value, args []ir.Value) (ir.Value, bool) {
	arg := func(i int) ir.Value {
		if i < len(args) {
			return args[i]
		}
		return ir.Value{}
	}
	nums := func() ([]float64, bool) {
		xs := make([]float64, len(args))
		for i, a := range args {
			if a.Kind != ir.Number {
				return nil, false
			}
			xs[i] = math.Float64frombits(a.Bits)
		}
		return xs, true
	}
	if strings.HasPrefix(f.name, "Math.") {
		xs, ok := nums()
		if !ok {
			return ir.Value{}, false
		}
		x := math.NaN()
		if len(xs) > 0 {
			x = xs[0]
		}
		switch f.name {
		case "Math.floor":
			return ir.Float(math.Floor(x)), true
		case "Math.ceil":
			return ir.Float(math.Ceil(x)), true
		case "Math.abs":
			return ir.Float(math.Abs(x)), true
		case "Math.sqrt":
			return ir.Float(math.Sqrt(x)), true
		case "Math.round":
			r := math.Floor(x)
			if x-r >= 0.5 {
				r++
			}
			if r == 0 && math.Signbit(x) || r == 0 && x < 0 {
				r = math.Copysign(0, -1)
			}
			return ir.Float(r), true
		case "Math.imul":
			y := math.NaN()
			if len(xs) > 1 {
				y = xs[1]
			}
			return ir.Float(float64(int32(ir.ToUint32(x) * ir.ToUint32(y)))), true
		case "Math.min", "Math.max":
			r := math.Inf(1)
			if f.name == "Math.max" {
				r = math.Inf(-1)
			}
			for _, y := range xs {
				switch {
				case math.IsNaN(y) || math.IsNaN(r):
					r = math.NaN()
				case f.name == "Math.min" && (y < r || y == 0 && r == 0 && math.Signbit(y)):
					r = y
				case f.name == "Math.max" && (y > r || y == 0 && r == 0 && !math.Signbit(y)):
					r = y
				}
			}
			return ir.Float(r), true
		}
		return ir.Value{}, false
	}
	switch f.name {
	case "Function.prototype.call":
		var rest []ir.Value
		if len(args) > 1 {
			rest = args[1:]
		}
		v, ok, err := m.call(this, arg(0), rest, false)
		return v, ok && err == nil
	case "Function.prototype.apply":
		var rest []ir.Value
		if a := arg(1); a.Kind != ir.Undefined && a.Kind != ir.Null {
			o := m.obj(a)
			if o == nil || o.kind != kindArray {
				return ir.Value{}, false
			}
			rest = o.elems
		}
		v, ok, err := m.call(this, arg(0), slices.Clone(rest), false)
		return v, ok && err == nil
	case "Number.prototype.toString":
		if this.Kind != ir.Number || len(args) > 0 && arg(0).Kind != ir.Undefined && !(arg(0).Kind == ir.Number && math.Float64frombits(arg(0).Bits) == 10) {
			return ir.Value{}, false
		}
		return m.String(NumberString(math.Float64frombits(this.Bits))), true
	case "JSON.stringify":
		if len(args) != 1 {
			return ir.Value{}, false
		}
		if t, ok := m.JSON(arg(0)); ok {
			return m.String(t), true
		}
		return ir.Value{}, false
	case "JSON.parse":
		t, ok := m.str(arg(0))
		if !ok || len(args) != 1 {
			return ir.Value{}, false
		}
		return m.ParseJSON(t)
	case "Object.keys":
		o := m.obj(arg(0))
		if o == nil || o.kind != kindObject {
			return ir.Value{}, false
		}
		var ks []ir.Value
		for _, k := range o.keys {
			if isIndex(k) {
				return ir.Value{}, false
			}
			if !o.hidden[k] {
				ks = append(ks, m.String(k))
			}
		}
		return m.Array(ks), true
	case "Object.create":
		if len(args) != 1 {
			return ir.Value{}, false
		}
		if arg(0).Kind == ir.Null {
			return m.alloc(&object{noProto: true}), true
		}
		p := m.obj(arg(0))
		if p == nil || p.kind != kindObject {
			return ir.Value{}, false
		}
		return m.alloc(&object{proto: p}), true
	case "Object.getPrototypeOf":
		o := m.obj(arg(0))
		if o == nil || o.kind != kindObject || o.proto == nil {
			return ir.Value{}, false
		}
		return ir.Value{Kind: ir.Opaque, Bits: uint64(o.proto.self)}, true
	case "Object.is":
		a, b := arg(0), arg(1)
		if a.Kind == ir.Number && b.Kind == ir.Number {
			return ir.Bool(a.Bits == b.Bits || math.IsNaN(math.Float64frombits(a.Bits)) && math.IsNaN(math.Float64frombits(b.Bits))), true
		}
		eq, ok := m.equal(a, b, true)
		return ir.Bool(eq), ok
	case "Array.isArray":
		o := m.obj(arg(0))
		return ir.Bool(o != nil && o.kind == kindArray), arg(0).Kind != ir.Opaque || o != nil
	case "String.fromCharCode":
		xs, ok := nums()
		if !ok {
			return ir.Value{}, false
		}
		u := make([]uint16, len(xs))
		for i, x := range xs {
			u[i] = uint16(ir.ToUint32(x))
			if utf16.IsSurrogate(rune(u[i])) {
				return ir.Value{}, false
			}
		}
		return m.String(string(utf16.Decode(u))), true
	case "String":
		s, ok := m.toString(arg(0))
		if len(args) == 0 {
			s, ok = "", true
		}
		if !ok {
			return ir.Value{}, false
		}
		return m.String(s), true
	}
	if s, ok := m.str(this); ok && strings.HasPrefix(f.name, "String.prototype.") {
		return m.stringMethod(strings.TrimPrefix(f.name, "String.prototype."), s, args)
	}
	a := m.obj(this)
	if a == nil || a.kind != kindArray || !strings.HasPrefix(f.name, "Array.prototype.") {
		return ir.Value{}, false
	}
	switch strings.TrimPrefix(f.name, "Array.prototype.") {
	case "push":
		m.bytes += 16 * len(args)
		a.elems = append(a.elems, args...)
		return ir.Float(float64(len(a.elems))), true
	case "pop":
		if len(a.elems) == 0 {
			return ir.Value{}, true
		}
		v := a.elems[len(a.elems)-1]
		a.elems = a.elems[:len(a.elems)-1]
		return v, true
	case "indexOf":
		if len(args) > 1 {
			return ir.Value{}, false
		}
		for i, e := range a.elems {
			if eq, ok := m.equal(e, arg(0), true); !ok {
				return ir.Value{}, false
			} else if eq && !(e.Kind == ir.Number && math.IsNaN(math.Float64frombits(e.Bits))) {
				return ir.Float(float64(i)), true
			}
		}
		return ir.Float(-1), true
	case "join":
		sep := ","
		if len(args) > 0 && arg(0).Kind != ir.Undefined {
			s, ok := m.str(arg(0))
			if !ok {
				return ir.Value{}, false
			}
			sep = s
		}
		var sb strings.Builder
		for i, e := range a.elems {
			if i > 0 {
				sb.WriteString(sep)
			}
			if e.Kind == ir.Undefined || e.Kind == ir.Null {
				continue
			}
			s, ok := m.toString(e)
			if !ok {
				return ir.Value{}, false
			}
			sb.WriteString(s)
			if sb.Len() > maxString {
				return ir.Value{}, false
			}
		}
		return m.String(sb.String()), true
	case "slice":
		lo, hi, ok := m.span(len(a.elems), args)
		if !ok {
			return ir.Value{}, false
		}
		return m.Array(a.elems[lo:hi]), true
	case "map", "forEach":
		cb := arg(0)
		var out []ir.Value
		for i := 0; i < len(a.elems); i++ {
			v, ok, err := m.call(cb, arg(1), []ir.Value{a.elems[i], ir.Float(float64(i)), this}, false)
			if err != nil || !ok {
				return ir.Value{}, false
			}
			out = append(out, v)
		}
		if f.name == "Array.prototype.forEach" {
			return ir.Value{}, true
		}
		return m.Array(out), true
	}
	return ir.Value{}, false
}

// span is a slice's or a substring's start and end of n, as slice takes
// its arguments, which are numbers or undefined.
func (m *Machine) span(n int, args []ir.Value) (int, int, bool) {
	at := func(i int, def int) (int, bool) {
		if i >= len(args) || args[i].Kind == ir.Undefined {
			return def, true
		}
		if args[i].Kind != ir.Number {
			return 0, false
		}
		x := math.Trunc(math.Float64frombits(args[i].Bits))
		if math.IsNaN(x) {
			x = 0
		}
		if x < 0 {
			x = math.Max(float64(n)+x, 0)
		}
		return int(math.Min(x, float64(n))), true
	}
	lo, ok1 := at(0, 0)
	hi, ok2 := at(1, n)
	if !ok1 || !ok2 {
		return 0, 0, false
	}
	return lo, max(lo, hi), true
}

// stringMethod is s.name(args...), where the machine models it.
func (m *Machine) stringMethod(name, s string, args []ir.Value) (ir.Value, bool) {
	u := utf16.Encode([]rune(s))
	ascii := true
	for i := 0; i < len(s); i++ {
		ascii = ascii && s[i] < 0x80
	}
	switch name {
	case "charCodeAt":
		i := 0
		if len(args) > 0 {
			x, ok := number(args[0])
			if !ok || x != math.Trunc(x) {
				return ir.Value{}, false
			}
			if x < 0 || x >= float64(len(u)) {
				return ir.Float(math.NaN()), true
			}
			i = int(x)
		}
		if len(u) == 0 {
			return ir.Float(math.NaN()), true
		}
		return ir.Float(float64(u[i])), true
	case "indexOf":
		t, ok := m.str(argAt(args, 0))
		if !ok || len(args) > 1 || !ascii {
			return ir.Value{}, false
		}
		return ir.Float(float64(strings.Index(s, t))), true
	case "slice":
		lo, hi, ok := m.span(len(u), args)
		if !ok || !ascii {
			return ir.Value{}, false
		}
		return m.String(s[lo:hi]), true
	case "substring":
		if !ascii {
			return ir.Value{}, false
		}
		clamp := func(i int, def int) (int, bool) {
			if i >= len(args) || args[i].Kind == ir.Undefined {
				return def, true
			}
			x, ok := number(args[i])
			if !ok {
				return 0, false
			}
			if math.IsNaN(x) || x < 0 {
				x = 0
			}
			return int(math.Min(math.Trunc(x), float64(len(s)))), true
		}
		lo, ok1 := clamp(0, 0)
		hi, ok2 := clamp(1, len(s))
		if !ok1 || !ok2 {
			return ir.Value{}, false
		}
		if lo > hi {
			lo, hi = hi, lo
		}
		return m.String(s[lo:hi]), true
	case "toLowerCase", "toUpperCase":
		if !ascii {
			return ir.Value{}, false
		}
		if name == "toLowerCase" {
			return m.String(strings.ToLower(s)), true
		}
		return m.String(strings.ToUpper(s)), true
	}
	return ir.Value{}, false
}

func argAt(args []ir.Value, i int) ir.Value {
	if i < len(args) {
		return args[i]
	}
	return ir.Value{}
}

// JSON is JSON.stringify(v), where v is an array or an object of the
// machine's of primitives and such, and nothing JSON.stringify would ask
// anything of: no function, no cycle.
func (m *Machine) JSON(v ir.Value) (string, bool) {
	var sb strings.Builder
	var write func(v ir.Value, depth int) bool
	write = func(v ir.Value, depth int) bool {
		if depth > 20 || sb.Len() > maxString {
			return false
		}
		switch v.Kind {
		case ir.Number:
			x := math.Float64frombits(v.Bits)
			if math.IsNaN(x) || math.IsInf(x, 0) {
				sb.WriteString("null")
			} else {
				sb.WriteString(NumberString(x))
			}
			return true
		case ir.Boolean, ir.Null:
			t, _ := m.Text(v)
			sb.WriteString(t)
			return true
		case ir.String:
			s, _ := m.str(v)
			sb.WriteString(jsonQuote(s))
			return true
		}
		o := m.obj(v)
		switch {
		case o == nil:
			return false
		case o.kind == kindArray:
			sb.WriteByte('[')
			for i, e := range o.elems {
				if i > 0 {
					sb.WriteByte(',')
				}
				if f := m.obj(e); e.Kind == ir.Undefined || f != nil && (f.kind == kindFunction || f.kind == kindBuiltin) {
					sb.WriteString("null")
				} else if !write(e, depth+1) {
					return false
				}
			}
			sb.WriteByte(']')
			return true
		case o.kind == kindObject && o.proto == nil && !o.noProto:
			sb.WriteByte('{')
			n := 0
			for _, k := range o.keys {
				e := o.props[k]
				if e.Kind == ir.Undefined || o.hidden[k] {
					continue
				}
				if f := m.obj(e); f != nil && (f.kind == kindFunction || f.kind == kindBuiltin) {
					continue
				}
				if isIndex(k) {
					return false
				}
				if n > 0 {
					sb.WriteByte(',')
				}
				n++
				sb.WriteString(jsonQuote(k))
				sb.WriteByte(':')
				if !write(e, depth+1) {
					return false
				}
			}
			sb.WriteByte('}')
			return true
		}
		return false
	}
	if !write(v, 0) {
		return "", false
	}
	return sb.String(), true
}

// jsonQuote is s as JSON.stringify quotes it.
func jsonQuote(s string) string {
	var sb strings.Builder
	sb.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			sb.WriteString(`\"`)
		case '\\':
			sb.WriteString(`\\`)
		case '\b':
			sb.WriteString(`\b`)
		case '\f':
			sb.WriteString(`\f`)
		case '\n':
			sb.WriteString(`\n`)
		case '\r':
			sb.WriteString(`\r`)
		case '\t':
			sb.WriteString(`\t`)
		default:
			if r < 0x20 {
				sb.WriteString(`\u00`)
				sb.WriteByte("0123456789abcdef"[r>>4])
				sb.WriteByte("0123456789abcdef"[r&15])
			} else {
				sb.WriteRune(r)
			}
		}
	}
	sb.WriteByte('"')
	return sb.String()
}

// isIndex reports whether key is an array index's name.
func isIndex(key string) bool {
	n, err := strconv.ParseUint(key, 10, 32)
	return err == nil && n < 1<<32-1 && strconv.FormatUint(n, 10) == key
}

// chainEndsNull reports whether o's prototypes end in null, not in
// Object.prototype.
func (o *object) chainEndsNull() bool {
	for p := o; p != nil; p = p.proto {
		if p.proto == nil {
			return p.noProto
		}
	}
	return false
}

// ParseJSON is JSON.parse(t) on the heap, its objects' keys in order; ok
// false for text the machine does not parse (not JSON, or an object with
// keys JSON.parse would order otherwise).
func (m *Machine) ParseJSON(t string) (ir.Value, bool) {
	d := json.NewDecoder(strings.NewReader(t))
	d.UseNumber()
	var value func() (ir.Value, bool)
	value = func() (ir.Value, bool) {
		tok, err := d.Token()
		if err != nil {
			return ir.Value{}, false
		}
		switch x := tok.(type) {
		case nil:
			return ir.Value{Kind: ir.Null}, true
		case bool:
			return ir.Bool(x), true
		case string:
			return m.String(x), true
		case json.Number:
			f, err := strconv.ParseFloat(string(x), 64)
			return ir.Float(f), err == nil || errorsIsRange(err)
		case json.Delim:
			var keys []string
			var vs []ir.Value
			for d.More() {
				if x == '{' {
					k, err := d.Token()
					if err != nil {
						return ir.Value{}, false
					}
					if isIndex(k.(string)) || slices.Contains(keys, k.(string)) {
						return ir.Value{}, false
					}
					keys = append(keys, k.(string))
				}
				v, ok := value()
				if !ok {
					return ir.Value{}, false
				}
				vs = append(vs, v)
			}
			if _, err := d.Token(); err != nil {
				return ir.Value{}, false
			}
			if x == '[' {
				return m.Array(vs), true
			}
			return m.Object(keys, vs), true
		}
		return ir.Value{}, false
	}
	v, ok := value()
	if _, err := d.Token(); err == nil {
		// More than one value.
		return ir.Value{}, false
	}
	return v, ok
}

// errorsIsRange reports whether err is strconv's out of range, whose value
// is the infinity JSON.parse gives too.
func errorsIsRange(err error) bool {
	ne, ok := err.(*strconv.NumError)
	return ok && ne.Err == strconv.ErrRange
}

// NewObject, NewArray, NewFunction and NewForeign make a heap's object of
// each kind, empty, for a test that copies a graph of the runtime's into
// the machine: a foreign function is one the machine has no template of,
// which a call declines.
func (m *Machine) NewObject() ir.Value { return m.alloc(&object{}) }

// NewArray is an empty array (NewObject).
func (m *Machine) NewArray() ir.Value { return m.alloc(&object{kind: kindArray}) }

// NewFunction is a closure of fn with no captured bindings (NewObject).
func (m *Machine) NewFunction(fn *bytecode.Function) ir.Value { return m.Function(fn) }

// NewForeign is a function of no template's (NewObject).
func (m *Machine) NewForeign() ir.Value {
	return m.alloc(&object{kind: kindBuiltin, name: "a foreign function"})
}

// Push appends e to array a's elements.
func (m *Machine) Push(a, e ir.Value) {
	if o := m.obj(a); o != nil && o.kind == kindArray {
		o.elems = append(o.elems, e)
	}
}

// Define gives o the own data property key, enumerable or not.
func (m *Machine) Define(o ir.Value, key string, v ir.Value, enumerable bool) {
	if x := m.obj(o); x != nil {
		x.set(key, v)
		if !enumerable {
			if x.hidden == nil {
				x.hidden = map[string]bool{}
			}
			x.hidden[key] = true
		}
	}
}

// SetProto gives o, a plain object, the prototype p, an object, or, null,
// none.
func (m *Machine) SetProto(o, p ir.Value, null bool) {
	x := m.obj(o)
	if x == nil {
		return
	}
	x.noProto = null
	x.proto = m.obj(p)
}
