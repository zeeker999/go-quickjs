package vm

import "github.com/go-quickjs/go-quickjs/internal/bytecode"

// Destructuring and parameter lists in the register form
// (docs/register-interpreter.md, "Stage 2: zero refusals"): the cold
// operations of a pattern and of a rest parameter, each what the stack
// form's instruction of the same name does.

// regColdPattern runs the pattern's cold operation w in frame f.
func (r *Runtime) regColdPattern(f *frame, c *regCode, pc int, w bytecode.RInstr, regs *[1 << 16]Value) error {
	var v Value
	var err error
	switch bytecode.RColdOp(w.X()) {
	case bytecode.RColdIterStep:
		v, err = r.iterStep(regs[w.B()])
	case bytecode.RColdIterRest:
		v, err = r.iterRest(regs[w.B()])
	case bytecode.RColdCheckCoercible:
		if x := regs[w.A()]; x.IsNullish() {
			return r.throwTypeError("cannot destructure %s", r.describe(x))
		}
		return nil
	case bytecode.RColdToPropertyKey:
		// A string or a symbol is a key already.
		x := regs[w.A()]
		if x.IsString() || x.IsSymbol() {
			return nil
		}
		k, err := r.toPropertyKey(x)
		if err != nil {
			return err
		}
		v = r.keyToValue(k)
	case bytecode.RColdObjectRest:
		b := int(w.C())
		v, err = r.objectRest(regs[w.B()], regs[b:b+int(c.rc.Aux[pc])])
	case bytecode.RColdRestParam:
		var rest []Value
		if i := int(w.B()); i < len(f.args) {
			rest = f.args[i:]
		}
		v = Obj(r.newArrayFrom(rest))
	}
	if err != nil {
		return err
	}
	regs[w.A()] = v
	return nil
}
