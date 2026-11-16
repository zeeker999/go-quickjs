package vm

// An object literal's fields, in the tree tier, as one node that makes the
// object with its final shape, as V8 makes a literal of its boilerplate's
// map (CreateShallowObjectLiteral) and never walks its transitions again.
// A literal had been a new_object's node, then a statement for each
// define_field, which found the transition from the shape the object had
// (Object.transition) and appended the value along it
// (appendTransition): for { a: j, b: j + 1, c: "key", ... } of eight
// fields, eight closures, eight transitions and eight shape stores per
// object, the define_fields alone 40% of the loop making it.
//
// The new_object's entry (tentryRare's lit) takes each define_field that
// follows it while it is still a tree: the field's value, its own tree or
// what it reads in place -- a local, a constant -- is evaluated by the
// literal's node, in the fields' order, where the define_field's statement
// would have evaluated it. Anything that evaluates the entry before --
// a spill, for a statement or a call, a spread or an accessor's
// definition -- takes it with the fields it has, and the rest are
// define_fields as before, on the object in its slot. So the values are
// evaluated in the order they were, and nothing else runs between them
// and the object's making that had not before: the object is nothing
// anything can see until the node returns it.
//
// The node makes the object as the new_object's did, from the site's root
// (siteRoot) with its room. Its first evaluations define the fields as the
// define_fields did, along the transitions from the object's shape, or by
// defineOwnProp; where every field was added along a transition from the
// root, the shape it ends with is the literal's, which the site keeps
// (propCache's next, with its root in s1: a new_object's site holds
// nothing else), and an object made from that root again is given its
// values in place and that shape: transitions are never undone, so the
// root's chain for the same keys leads to it every time. Keys that repeat,
// which the second defines again, are not taken.
//
// Its functions are generic, instantiated with struct{} alone, which Go
// lays out after every function and closure of the package (see
// zzz_direct.go); opHook (zzz_typeof.go) asks this file first, and
// tentryRare has a word more, for lit.

// The ways a field's value is had.
const (
	litTree = iota
	litConst
	litLocal
)
