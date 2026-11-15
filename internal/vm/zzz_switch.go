package vm

// A switch's cases whose selectors are literals -- strings, numbers,
// booleans, null and undefined, a template without substitutions among
// the strings -- as one lookup of the discriminant in the tree tier. Each
// case is compiled to dup, the selector's push, strict_eq and jump_if_true
// to its body, and each jump ended a block: a match at the nth case had
// been n blocks, each a statement copying the discriminant, an equality's
// node and the node making its boolean, then the jump's test of it, and a
// trip round runTree's loop -- a lexer's keyword table of forty cases is
// forty of each where its last keyword matches, and every miss as many.
//
// A run of such cases, from a dup the block being built has got to, is
// instead looked up as one: the block's end evaluates the discriminant
// into its slot, as the dup would have left it, and looks it up in a
// table made of the selectors, which goes to the first case whose
// selector it strictly equals, in the cases' order, or else to the
// instruction after the run -- the switch's jump to its default clause or
// past it, or the next case's test, one whose selector is not a literal,
// where the chain goes on as it was compiled. Evaluating a literal has no
// effect, so skipping those not compared is not seen; what strict
// equality says is kept: +0 equals -0, NaN nothing, a string a string of
// its code units, however it was made, a number no string. The run's
// later cases are built as they were compiled, the way on of the first
// case's jump, and never run.
//
// Strings are looked up by length, then by the constant itself -- a
// string a constant was stored from is it -- and then by their bytes,
// which a flat string's WTF-8 spells one way: a rope is flattened only if
// a case has its length. Integers in a range of no more than
// switchDense are an array's index, other numbers a map's key; a
// switch (typeof x) of type names a table of the eight types, x's kind
// asked (see zzz_typeof.go) and its name stored where the discriminant
// goes.
//
// opHook gives the run's dup to switchStart, which leaves the lookup for
// the first case's jump_if_true; truthJump, which buildBlock asks for that
// jump's end, is truthJumpHook, which makes it the lookup. Its functions
// are generic, instantiated with struct{} alone, as zzz_typeof.go's are,
// so that every function and closure there was stays at its address.

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
