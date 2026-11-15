package vm

import "github.com/go-quickjs/go-quickjs/internal/bytecode"

// Conversions, destructuring, spreads, computed members and private class
// members in the tree tier.
//
// Each instruction here is a node that does what the interpreter's does,
// calling the helper it calls, or, where its work is written out in the
// interpreter's loop, doing that work with the same errors. Without them a
// function with a template literal, an object destructuring pattern, a call
// of o[k](), a rest parameter, a spread or a private member ran in the
// interpreter.
//
// getGlobalOpt and the helpers after it copy work executeAt does in its own
// cases. executeAt is kept as it is: it is the hot path, whose code and
// size decide how much of it the processor's op cache holds, and moving
// those cases out of it slowed loops that run none of them. TestTreeOps
// runs each on both tiers.
//
// Its functions are variables, each assigned a closure in init, rather
// than functions or methods: Go lays out all of a package's functions
// before any of its closures, so code added here as functions would sit
// between the interpreter's hot functions and the tree nodes' hot
// closures, and move one against the other, where a closure is laid out
// after every closure already there (see AGENTS.md).

var (
	// treeBuildsOps reports whether the tier builds op, one of those opsOp
	// builds, or one treeBuildsSuspend names.
	treeBuildsOps func(op bytecode.Op) bool

	// getGlobalOpt is get_global_opt, typeof's read of a global: a name
	// declared nowhere is undefined rather than a ReferenceError. A lexical
	// binding in its dead zone is declared, though, so that one still throws.
	getGlobalOpt func(r *Runtime, f *frame, cl *closure, name Atom) (Value, error)

	// inOperator is key in obj.
	inOperator func(r *Runtime, key, obj Value) (Value, error)

	// deleteIndexed is delete obj[key], and delete obj.name.
	deleteIndexed func(r *Runtime, obj, key Value, strict bool) (Value, error)

	// toObjectValue is ToObject as a value, the base a delete of a
	// primitive's property deletes from (executeAt's delete_prop,
	// deleteIndexed): a closure here, not a function, so that it moves no
	// code of the interpreter's.
	toObjectValue func(r *Runtime, v Value) (Value, error)

	// deleteVar is delete name. Deleting a binding only succeeds for a
	// configurable global property, which is why a var declaration cannot be
	// deleted.
	deleteVar func(r *Runtime, f *frame, cl *closure, name Atom) (Value, error)

	// superIndexGet is super[key], read from base, which super_base found.
	superIndexGet func(r *Runtime, f *frame, base, rawKey Value) (Value, error)

	// superIndexSet is super[key] = val, on base, which super_base found.
	superIndexSet func(r *Runtime, f *frame, base, rawKey, val Value, strict bool) error

	// concatParts is concat, as executeAt's case does it: a template
	// literal's parts joined. Every part is converted before any of them is
	// joined: a toString may run user code, and when it runs is fixed. A
	// number is left as it is, having no conversion anything can observe and
	// no need of a string of its own. The parts are the operand stack's,
	// which are converted in place.
	concatParts func(r *Runtime, parts []Value) (Value, error)
)

func init() {

	getGlobalOpt = func(r *Runtime, f *frame, cl *closure, name Atom) (Value, error) {
		env := cl.scope()
		if f.evalVars != nil {
			if p := evalVarProp(f.evalVars, name); p.ok() {
				return p.value(), nil
			}
		}
		if p := r.globalLexProp(env, name); p.ok() {
			if p.value().IsUninitialized() {
				return Undefined, r.throwReferenceError("cannot access %q before it is initialized", r.atoms.name(name))
			}
			return p.value(), nil
		}
		if !r.isGlobalScope(env) {
			if p := r.moduleLexProp(env, name); p.ok() {
				if p.flags()&propUninit != 0 {
					return Undefined, r.throwReferenceError("cannot access %q before it is initialized", r.atoms.name(name))
				}
				if p.flags()&(propAccessor|propPrivate|propDeleted) == 0 {
					return p.value(), nil
				}
			}
		}
		if chainHasProxy(env) {
			// Asked whether it has the name first, as for a read; what is not
			// there is undefined.
			has, err := r.hasPropErr(env, name)
			if err != nil {
				return Undefined, err
			}
			if !has {
				return Undefined, nil
			}
		}
		return r.getProp(env, name, r.globalReceiver(env))
	}

	inOperator = func(r *Runtime, key, obj Value) (Value, error) {
		if !obj.IsObject() {
			return Undefined, r.throwTypeError("the right operand of \"in\" must be an object")
		}
		n, err := r.toPropertyName(key)
		if err != nil {
			return Undefined, err
		}
		k, known := r.keyFor(obj.Object(), n)
		if !known {
			return False, nil
		}
		has, err := r.hasPropErr(obj.Object(), k)
		if err != nil {
			return Undefined, err
		}
		return Bool(has), nil
	}

	toObjectValue = func(r *Runtime, v Value) (Value, error) {
		o, err := r.toObject(v)
		if err != nil {
			return Undefined, err
		}
		return Obj(o), nil
	}

	deleteIndexed = func(r *Runtime, obj, key Value, strict bool) (Value, error) {
		if obj.IsNullish() {
			// There is nothing to delete from, which is a TypeError before the
			// key is even converted.
			return Undefined, r.throwTypeError("cannot delete a property of %s", r.describe(obj))
		}
		n, err := r.toPropertyName(key)
		if err != nil {
			return Undefined, err
		}
		if !obj.IsObject() {
			// A primitive's properties are its wrapper's, which ToObject
			// makes, as executeAt's case does.
			if obj, err = toObjectValue(r, obj); err != nil {
				return Undefined, err
			}
		}
		k, has := r.keyFor(obj.Object(), n)
		if !has {
			// Deleting what is not there succeeds.
			return True, nil
		}
		ok, err := r.deleteProp(obj.Object(), k, strict)
		if err != nil {
			return Undefined, err
		}
		return Bool(ok), nil
	}

	deleteVar = func(r *Runtime, f *frame, cl *closure, name Atom) (Value, error) {
		// What a direct eval declared can be deleted again, unlike a var the
		// source named: the evaluated code could have declared it anywhere, so
		// nothing may rely on it being there.
		if deleteEvalVar(f.evalVars, name) {
			return True, nil
		}
		if r.globalLexProp(cl.scope(), name).ok() {
			// A lexical binding is not a property and cannot be removed.
			return False, nil
		}
		ok, err := r.deleteProp(cl.scope(), name, false)
		if err != nil {
			return Undefined, err
		}
		return Bool(ok), nil
	}

	superIndexGet = func(r *Runtime, f *frame, base, rawKey Value) (Value, error) {
		if !base.IsObject() {
			return Undefined, r.throwTypeError("\"super\" has no prototype to read from")
		}
		key, err := r.toPropertyKey(rawKey)
		if err != nil {
			return Undefined, err
		}
		this, _ := f.thisValue()
		return r.getProp(base.Object(), key, this)
	}

	superIndexSet = func(r *Runtime, f *frame, base, rawKey, val Value, strict bool) error {
		if !base.IsObject() {
			return r.throwTypeError("\"super\" has no prototype to assign to")
		}
		key, err := r.toPropertyKey(rawKey)
		if err != nil {
			return err
		}
		return r.superSetFrom(f, base, key, val, strict)
	}

	concatParts = func(r *Runtime, parts []Value) (Value, error) {
		for i := range parts {
			if parts[i].IsString() || parts[i].IsNumber() {
				continue
			}
			s, err := r.toString(parts[i])
			if err != nil {
				return Undefined, err
			}
			parts[i] = Str(s)
		}
		// The strings alone may be too long, which is known before anything is
		// written out.
		total := 0
		for _, p := range parts {
			if p.IsString() && total <= maxStringLength {
				total += p.String().length
			}
		}
		if total > maxStringLength {
			return Undefined, r.throwStringLength()
		}
		out, err := r.joinParts(parts)
		if err != nil {
			return Undefined, err
		}
		if out.length > maxStringLength {
			return Undefined, r.throwStringLength()
		}
		return Str(out), nil
	}

}
