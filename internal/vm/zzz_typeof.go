package vm

// typeof's tests in the tree tier -- typeof f === 'function', typeof x !==
// 'undefined', typeof v == 'object' -- as one node that asks the value's
// kind and makes no string, as V8's TestTypeOf does. A typeof had been a
// node making its name, typeofString's switch over the value's kind and
// then over the name, and the equality a node comparing that string with
// the constant: three calls, two switches and a string comparison for a
// test of the value's tag.
//
// The test is the one Value.TypeOf answers: an object with
// [[IsHTMLDDA]] (Annex B's document.all) is "undefined", any other
// callable one "function", and null and every other object "object". Its
// name is the constant's: a constant that is no type's name -- typeof x
// === 'strnig' -- is a test that never holds, the value still evaluated.
// ==, != and their strict forms answer alike, the two sides strings.
//
// A typeof that no such test takes is still marked (tentryRare's typeOf,
// its operand), so that a switch on it can test the kind as well (see
// zzz_switch.go).
//
// Its functions are generic, each instantiated with struct{} alone: Go
// lays a generic function's instantiations out after every function and
// closure of the package (see zzz_direct.go), where code written anywhere
// else moves the closures, or the functions after it, against the hot
// ones (AGENTS.md). The tree builder reaches it through opHook, called
// where buildBlock's call of op was, and eqJumpAtHook, which eqJumpAt is,
// the function eqJump asks first: each asks this file first and then
// makes the call it stands for. buildBlock's code grows within the
// padding that rounds it to 32 bytes, op, eqJump and the closure eqJumpAt
// had been are as they were, and every function and closure of the
// package stays at its address (go tool nm).

// The types' indices in typeofNames.
const (
	typeUndefined = iota
	typeObject
	typeBoolean
	typeNumber
	typeString
	typeSymbol
	typeBigInt
	typeFunction
)

// typeofKind is the index in typeofNames of typeof v, as Value.TypeOf
// answers it.
func typeofKind[_ any](v Value) int {
	switch v.Kind() {
	case KindNumber:
		return typeNumber
	case KindNull:
		return typeObject
	case KindBool:
		return typeBoolean
	case KindString:
		return typeString
	case KindSymbol:
		return typeSymbol
	case KindBigInt:
		return typeBigInt
	case KindObject:
		o := v.object()
		if o.flags&objHTMLDDA != 0 {
			return typeUndefined
		}
		if o.IsCallable() {
			return typeFunction
		}
		return typeObject
	}
	return typeUndefined
}

// typeofOfStr is r's string of the type of index i in typeofNames, as
// typeofString makes it.
func typeofOfStr[_ any](r *Runtime, i int) *String {
	if s := r.typeofStrs[i]; s != nil {
		return s
	}
	s := NewString(typeofNames[i])
	r.typeofStrs[i] = s
	return s
}
