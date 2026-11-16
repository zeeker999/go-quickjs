package vm

import "unsafe"

// The in operator answered from what plain objects and arrays hold: the
// register interpreter's handler (rhIn) asks hasIndexQuick and hasCached,
// and native code's call of Go (jitHas) hasQuick, before the slow path,
// which answers the rest -- a proxy's trap, a key to convert, a typed
// array's index, a module namespace, an exotic object's own keys.
//
// The answers hasQuick finds for a key string that names no index are
// remembered in the realm's hasCache, by the string, the object's layout
// and its prototype, as V8's megamorphic stub cache remembers a load's: the
// interpreter's handler looks there first, and so does native code, before
// it calls Go (mir's keyedHas).
//
// Its functions are variables, each assigned a closure in init, laid
// out after every other closure of the package, so that no hot code
// moves for them (see AGENTS.md).

// hasCache is a realm's cache of the in operator's answers for key
// strings that name no index, made when one is first remembered. An entry
// answers key in o -- has -- where key's string is the entry's, o's layout
// is shape and its prototype proto, and cell, the validity cell of the
// chain from proto up (chainCell), is valid: the layout tells o's own keys,
// and its class, which an array's length is of; the cell, that no object up
// the chain has changed its keys or its prototype since. Only an ordinary
// object's or an array's answer, of a chain of them (hasQuick), is
// remembered, whose layout is one a cache may remember, so that it is
// replaced at any change of the object's keys. Elements do not matter to a
// name. An entry is one of the two of its set (hasSet), found by what the
// string and the layout hold, never by their addresses: which answers
// share a set, and so displace each other, is the same in every run of a
// program, however its objects fall in memory -- by addresses, three of
// prop_in_keyed's four keys had shared one in a run, and every visit of
// them called Go, and not in another. The first of a set is the newer,
// and a new entry takes its place, the older going to the second's, so
// that two answers asked in turn -- "a" in o and "b" in o -- keep their
// places. Native code finds them so too, their fields at the offsets the
// JIT's abi names (abi.HasEntrySize and the others).
type hasCache [hasCacheSize]hasEntry

// hasEntry is an entry of hasCache, of 64 bytes.
type hasEntry struct {
	key   unsafe.Pointer
	shape *shape
	proto *Object
	cell  *protoCell
	has   bool
	_     [3]uint64
}

const (
	// hasCacheBits is the bits of an entry's index: 256 entries, in sets
	// of two, the set the top hasCacheBits-1 bits of the hash.
	hasCacheBits = 8
	hasCacheSize = 1 << hasCacheBits
	// hasCacheMul is the multiplier of the hash, Fibonacci's, and
	// hasCacheKeyMul the key's part's (hasSet).
	hasCacheMul    = 0x9E3779B97F4A7C15
	hasCacheKeyMul = 0xFF51AFD7ED558CCD
)

var (
	// hasQuick is key in o where the answer is in o's and its prototypes'
	// own storage and asking it runs nothing: each of them an ordinary
	// object or an array, whose own properties are its elements and its
	// table, and an array's length -- as hasPropErr and hasOwnProp find
	// them -- by a key that is an index, a number's or a string's, or a
	// string whose atom the runtime remembers with it (keyAtoms), as
	// toPropertyName finds it. An index absent from an object's elements
	// is looked for in its table only where the table may have one
	// (noIndexKeys). It reports false for anything else, which inOperator
	// answers: a key of another kind, a string whose atom is not
	// remembered, a chain with any other object on it.
	hasQuick func(r *Runtime, key Value, o *Object) (has, ok bool)

	// hasIndexQuick is hasQuick of the index i, below atomIndexTag.
	hasIndexQuick func(o *Object, i uint32) (has, ok bool)

	// hasCached is key in o where the realm's hasCache remembers it.
	hasCached func(r *Runtime, key Value, o *Object) (has, ok bool)

	// hasNote remembers in the realm's hasCache that key, a string that
	// names no index, is in o where has says, as hasQuick found it, where
	// o's layout and its chain's cell can be kept.
	hasNote func(r *Runtime, key Value, o *Object, has bool)

	// hasSet is the set of hasCache that answers key in an object of
	// layout sh: of what the string holds -- its length in bytes and,
	// of eight or more, its first eight and its last eight, of fewer, its
	// first two and its last two, those of a flat string (a rope's are
	// none) -- multiplied, with the layout's count of keys and its last
	// key's atom, then all of it multiplied, as native code makes it
	// (abi.HasCacheMul). Names that differ only in their middle --
	// "item1" to "item99", "k12" and "k21", "prop_1_x" to "prop_99_x" --
	// differ in length or within eight bytes of an end. A remembered layout's keys never change (remember), so its
	// count and its last key are the same at every look.
	hasSet func(key *String, sh *shape) uint64

	// hasLE is the first n bytes of s, a little-endian number, as native
	// code loads them.
	hasLE func(s string, n int) uint64
)

func init() {
	hasCached = func(r *Runtime, key Value, o *Object) (bool, bool) {
		c, sh := r.has, o.shape
		if c == nil || !key.IsString() || sh == nil {
			return false, false
		}
		set := hasSet(key.String(), sh)
		for i := set * 2; i < set*2+2; i++ {
			// The layout says the prototype, but for an object that keeps
			// it in its table (objPolyProto): only that one is compared.
			if e := &c[i]; e.key == key.ref && e.shape == sh && e.cell.valid && (o.flags&objPolyProto == 0 && e.proto == sh.proto || o.flags&objPolyProto != 0 && e.proto == (*Object)(o.valp.ref)) {
				if o.flags&objPolyProto != 0 {
					// Asked again of a singleton made on a prototype of its
					// own: its layout its own, holding the prototype, as a
					// second read past it makes it (polyReread), for the
					// next answer to be remembered by that layout.
					depoly[struct{}](o)
				}
				return e.has, true
			}
		}
		return false, false
	}
	hasNote = func(r *Runtime, key Value, o *Object, has bool) {
		// An object with no properties is given its class's first layout,
		// as a read's cache gives it (fillPropCache).
		sh := r.ensureShape(o)
		if sh == nil || r.shapes.building || o.flags&objPolyProto != 0 && o.proto() == nil {
			// Or o keeps a prototype of none in its table, which native code
			// would read as its layout's, none, for an object of the layout
			// with another (mir's keyedHas).
			return
		}
		cell := r.chainCell(o.proto())
		if cell == nil {
			return
		}
		if r.has == nil {
			r.has = new(hasCache)
		}
		set := hasSet(key.String(), sh)
		ways := r.has[set*2 : set*2+2]
		if e := &ways[0]; e.key != key.ref || e.shape != sh || e.proto != o.proto() {
			// The newer is now the older, unless it is this one's, which
			// its new cell replaces.
			ways[1] = ways[0]
		}
		ways[0] = hasEntry{key: key.ref, shape: remember(sh), proto: o.proto(), cell: cell, has: has}
	}
	hasLE = func(s string, n int) uint64 {
		var v uint64
		for i := n - 1; i >= 0; i-- {
			v = v<<8 | uint64(s[i])
		}
		return v
	}
	hasSet = func(key *String, sh *shape) uint64 {
		var x uint64
		switch b, n := key.s, len(key.s); {
		case key.left != nil || n == 0:
		case n >= 8:
			x = (uint64(n)^hasLE(b[n-8:], 8))*hasCacheKeyMul ^ hasLE(b, 8)
		case n >= 2:
			x = uint64(n) ^ hasLE(b, 2)<<32 ^ hasLE(b[n-2:], 2)<<48
		default:
			x = 1 ^ uint64(b[0])<<32 ^ uint64(b[0])<<48
		}
		h := x*hasCacheKeyMul ^ uint64(uint32(sh.key)) ^ uint64(uint32(sh.n))<<32
		return h * hasCacheMul >> (64 - hasCacheBits + 1)
	}
	hasIndexQuick = func(o *Object, i uint32) (bool, bool) {
		for p := o; p != nil; p = p.proto() {
			if p.class > ClassArray {
				return false, false
			}
			if e := p.elemsOf(); uint(i) < uint(len(e)) && !isHole(e[i]) {
				return true, true
			}
			if !p.noIndexKeys() {
				if j := p.findOwn(Atom(atomIndexTag | i)); j >= 0 && p.shape.keys[j].flags&propPrivate == 0 {
					return true, true
				}
			}
		}
		return false, true
	}
	hasQuick = func(r *Runtime, key Value, o *Object) (bool, bool) {
		var a Atom
		switch {
		case key.IsNumber():
			// An index, as toPropertyName makes it: -0 is 0.
			f := key.num
			if i := uint32(f); float64(i) == f && i < atomIndexTag {
				return hasIndexQuick(o, i)
			}
			return false, false
		case key.IsString():
			s := key.String()
			e := &r.keyAtoms[keySlot(s)]
			if e.s != s {
				// Looked up, and its atom, or that it names none,
				// remembered there (toPropertyName): two keys whose
				// strings share a slot had each found the other's there
				// at every evaluation, and left every one to inOperator,
				// never remembered in hasCache.
				if _, err := r.toPropertyName(key); err != nil || e.s != s {
					return false, false
				}
			}
			if e.none {
				// A key no property has, while it names no atom, on a
				// chain of objects that holds its keys.
				if e.atom != r.atoms.next() {
					return false, false
				}
				for p := o; p != nil; p = p.proto() {
					if p.class > ClassArray {
						return false, false
					}
				}
				hasNote(r, key, o, false)
				return false, true
			}
			a = e.atom
		default:
			return false, false
		}
		if a.IsIndex() {
			return hasIndexQuick(o, a.Index())
		}
		for p := o; p != nil; p = p.proto() {
			if p.class > ClassArray {
				return false, false
			}
			if a == atomLength && p.class == ClassArray {
				hasNote(r, key, o, true)
				return true, true
			}
			if j := p.findOwn(a); j >= 0 && p.shape.keys[j].flags&propPrivate == 0 {
				hasNote(r, key, o, true)
				return true, true
			}
		}
		hasNote(r, key, o, false)
		return false, true
	}
}
