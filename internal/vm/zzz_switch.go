package vm

// A switch's cases whose selectors are literals -- strings, numbers,
// booleans, null and undefined, a template without substitutions among
// the strings -- are one lookup of the discriminant, the register
// compiler's switch instruction (regSwitchTable): it goes to the first
// case whose selector the discriminant strictly equals, in the cases'
// order, or else past the run. Evaluating a literal has no effect, so
// skipping those not compared is not seen; what strict equality says is
// kept: +0 equals -0, NaN nothing, a string a string of its code units,
// however it was made, a number no string.
//
// Strings are looked up by length, then by the constant itself -- a
// string a constant was stored from is it -- and then by their bytes,
// which a flat string's WTF-8 spells one way: a rope is flattened only if
// a case has its length. Integers in a range of no more than
// switchDense are an array's index, other numbers a map's key; a
// switch (typeof x) of type names a table of the eight types, x's kind
// asked (see zzz_typeof.go) and its name stored where the discriminant
// goes.

// switchCase is a case of a run switchStart looks up: its selector, a
// literal, and the instruction its body begins at.
type switchCase struct {
	lit Value
	// str is the selector of a string's case, k its constant, or -1 for
	// the empty string push_empty_string pushes.
	str string
	k   int64
	to  int
}

// switchStr is a string case of a lookup: the selector's WTF-8, its
// constant, -1 for none, and the block its body begins.
type switchStr struct {
	str string
	k   int64
	to  int
}

// switchDense is the most integers' range an array of their blocks is
// made for.
const switchDense = 256
