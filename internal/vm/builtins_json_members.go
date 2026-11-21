package vm

import (
	"math"
	"strconv"
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/wtf8"
)

// JSON.stringify's arrays and objects: their elements, their properties
// and the layouts that write an ordinary object's properties at once.

// encodeElements writes an array, or a proxy of one.
func (e *jsonEncoder) encodeElements(buf []byte, v Value, o *Object) ([]byte, error) {
	a := arrayLike{o: o}
	if o.class == ClassArray {
		a.n = int64(o.arrayLength())
	} else if err := e.rt.readArrayLike(&a, v); err != nil {
		return buf, err
	}
	if a.n == 0 {
		return append(buf, "[]"...), nil
	}
	var inner []byte
	if e.indent != "" {
		inner = e.margin(e.depth)
	}
	buf = append(buf, '[')
	buf = append(buf, inner...)
	for i := int64(0); i < a.n; i++ {
		// A dense array is read without the script running, so the walk
		// checks for an interrupt itself (arrayLike.get); and what it writes
		// may grow past what a string can be.
		if len(buf) >= e.limit {
			if err := e.charged(len(buf)); err != nil {
				return buf, err
			}
		}
		if i > 0 {
			buf = append(buf, ',')
			buf = append(buf, inner...)
		}
		el, err := a.get(e.rt, i)
		if err != nil {
			return buf, err
		}
		if e.needsApply(el) {
			// The key reaches the replacer as a string, the way a property
			// name does: an array index is not a number here.
			var key Atom
			if i <= math.MaxUint32 {
				key = e.rt.atoms.indexAtom(uint32(i))
			} else {
				// A proxy over an array can report a length past uint32.
				key = e.rt.atoms.intern(strconv.FormatInt(i, 10))
			}
			el, err = e.apply(v, key, el)
			if err != nil {
				return buf, err
			}
		}
		var ok bool
		buf, ok, err = e.encode(buf, el)
		if err != nil {
			return buf, err
		}
		if !ok {
			// An element with no JSON form becomes null, unlike a
			// property, which is omitted.
			buf = append(buf, "null"...)
		}
	}
	if inner != nil {
		buf = append(buf, e.margin(e.depth-1)...)
	}
	return append(buf, ']'), nil
}

// encodeProps writes an object that is no array.
func (e *jsonEncoder) encodeProps(buf []byte, v Value, o *Object) ([]byte, error) {
	if e.hasAllowed {
		return e.writeKeys(buf, v, o, e.allowed)
	}
	// Small ordinary objects can snapshot their keys on the stack. Keep
	// the snapshot even for getters: one may change the keys still to come.
	var localKeys [8]Atom
	var keys []Atom
	plain := o.class == ClassObject && o.flags&objHasExt == 0 && int(o.valn) <= len(localKeys)
	if plain {
		keys = localKeys[:0]
		for _, p := range o.keys() {
			if p.flags&propDeleted != 0 || p.flags&propEnumerable == 0 || e.rt.atoms.symbol(p.key) != nil {
				continue
			}
			if _, index := e.rt.atoms.arrayIndex(p.key); index {
				plain = false
				break
			}
			keys = append(keys, p.key)
		}
	}
	if !plain {
		keys = nil
		var own []Atom
		if p := proxyOf(o); p != nil {
			// A proxy's keys come from its ownKeys trap, which the plain
			// property table would not reflect.
			var err error
			if own, err = e.rt.proxyOwnKeys(p); err != nil {
				return buf, err
			}
		} else {
			var err error
			if own, err = e.rt.ownKeysOf(o, false); err != nil {
				return buf, err
			}
		}
		keys = own[:0]
		for _, k := range own {
			if e.rt.atoms.symbol(k) != nil {
				continue
			}
			enumerable, err := e.rt.isEnumerable(o, k)
			if err != nil {
				return buf, err
			}
			if !enumerable {
				continue
			}
			keys = append(keys, k)
		}
	}
	return e.writeKeys(buf, v, o, keys)
}

// writeKeys writes the properties of o that keys name, in their order: an
// explicit key list fixes both which properties appear and their order,
// and otherwise they are its own enumerable string keys.
func (e *jsonEncoder) writeKeys(buf []byte, v Value, o *Object, keys []Atom) ([]byte, error) {
	var inner []byte
	colon := ":"
	if e.indent != "" {
		inner = e.margin(e.depth)
		colon = ": "
	}
	start := len(buf)
	buf = append(buf, '{')
	buf = append(buf, inner...)
	written := 0
	for _, k := range keys {
		if err := e.rt.tick(); err != nil {
			return buf, err
		}
		if len(buf) >= e.limit {
			if err := e.charged(len(buf)); err != nil {
				return buf, err
			}
		}
		val, err := e.rt.getProp(o, k, v)
		if err != nil {
			return buf, err
		}
		if e.needsApply(val) {
			if val, err = e.apply(v, k, val); err != nil {
				return buf, err
			}
		}
		// A property whose value has no JSON form is omitted, key and all, so
		// the separator and the key are written first and taken back when the
		// value turns out not to be there.
		mark := len(buf)
		if written > 0 {
			buf = append(buf, ',')
			buf = append(buf, inner...)
		}
		name := e.rt.atoms.name(k)
		ascii := wtf8.IsASCII(name)
		if !ascii {
			e.rescan = true
		}
		buf, _ = appendJSONString(buf, name, ascii)
		buf = append(buf, colon...)
		var ok bool
		buf, ok, err = e.encode(buf, val)
		if err != nil {
			return buf, err
		}
		if !ok {
			buf = buf[:mark]
			continue
		}
		written++
	}
	if written == 0 {
		return append(buf[:start], "{}"...), nil
	}
	if inner != nil {
		buf = append(buf, e.margin(e.depth-1)...)
	}
	return append(buf, '}'), nil
}

// jsonLayout is what JSON.stringify keeps of a layout objects share, on its
// side: its keys, their text, and whether a toJSON can be there. The first
// object of a layout makes it for the rest, as V8's stringifier reads a map's
// descriptors rather than ask each object.
//
// The keys are the object's as it is reached. Like any snapshot of them they
// are not looked at again, so that a getter or a toJSON that changes the
// object changes the values the rest of the walk reads, not the walk: a
// value is read straight from its slot only while the object still has the
// layout, and past that by Get, which is what the specification does for each.
type jsonLayout struct {
	// ordinary says keys is what the object's properties are, or else that
	// it holds none: an array's, or one too long to keep (jsonLayoutKeys).
	ordinary bool
	// ascii says text is all ASCII.
	ascii bool
	// asks says toJSON has to be looked up on such an object, because it may
	// be there -- on the object, on its prototypes, or behind a proxy --
	// while cell stands.
	asks bool
	// cell is the validity cell of the chain above the layout: while it is
	// valid, what was found there is still so. nil means none can be kept.
	cell *protoCell
	keys []jsonLayoutKey
	// text is the keys' text, each a comma, the key quoted, and a colon.
	text string
}

// jsonLayoutKey is a key of a jsonLayout: the end of its text in the
// layout's, its atom, and, for a property that holds a value, its slot.
type jsonLayoutKey struct {
	atom Atom
	end  uint32
	slot int32
	data bool
}

// jsonLayoutOf is the jsonLayout of o, an ordinary object or an array, or nil
// for a layout only o has, which may change under it.
func (r *Runtime) jsonLayoutOf(o *Object) *jsonLayout {
	s := o.shape
	if s == nil || s.unique || s.loose || o.flags&objPolyProto != 0 {
		return nil
	}
	if sd := s.side; sd != nil && sd.json != nil {
		return sd.json
	}
	if !r.jsonCaches().metBefore(s) {
		return nil
	}
	return r.newJSONLayout(o)
}

// metBefore reports whether a layout with no jsonLayout has been met before,
// and notes that it has: most are, and those made for one object -- with keys
// of its own, or of a document written once -- are spared what a jsonLayout
// costs to make and to keep. Layouts are told apart by a hash of their address
// and each one's bit stands till the bits are cleared, so that one may be
// taken for another, which only makes its record sooner.
func (st *jsonState) metBefore(s *shape) bool {
	h := uint64(uintptr(unsafe.Pointer(s))) * 0x9E3779B97F4A7C15 >> (64 - jsonMetBits)
	word, bit := &st.met[h/64], uint64(1)<<(h%64)
	if *word&bit != 0 {
		return true
	}
	*word |= bit
	if st.metCount++; st.metCount >= 1<<(jsonMetBits-1) {
		clear(st.met[:])
		st.metCount = 0
	}
	return false
}

// jsonLayoutKeys and jsonLayoutText are the most keys, and bytes of their
// text, a jsonLayout holds: a layout past them keeps what it says of toJSON
// alone, and its objects are written by their keys. Each layout of a chain
// has its keys' text whole, so that an object given a key at a time, and
// written at each, had made the square of what it holds -- 500 MB for a
// thousand keys of a kilobyte, kept as long as the layouts are, and charged
// to the script's memory limit. Records are smaller than this.
const (
	jsonLayoutKeys = 128
	jsonLayoutText = 4096
)

// newJSONLayout makes o's layout's jsonLayout.
func (r *Runtime) newJSONLayout(o *Object) *jsonLayout {
	l := &jsonLayout{ordinary: o.class == ClassObject, ascii: true}
	if l.ordinary {
		var text []byte
		for _, k := range o.ownKeys(false, r.atoms) {
			i := o.findOwn(k)
			if i < 0 {
				continue
			}
			flags := o.shape.keys[i].flags
			if flags&(propEnumerable|propPrivate|propDeleted) != propEnumerable || r.atoms.symbol(k) != nil {
				continue
			}
			text = append(text, ',')
			name := r.atoms.name(k)
			ascii := wtf8.IsASCII(name)
			if !ascii {
				l.ascii = false
			}
			text, _ = appendJSONString(text, name, ascii)
			text = append(text, ':')
			l.keys = append(l.keys, jsonLayoutKey{atom: k, end: uint32(len(text)), slot: i, data: flags&propAccessor == 0})
			if len(l.keys) > jsonLayoutKeys || len(text) > jsonLayoutText {
				l.ordinary, l.keys, text = false, nil, nil
				break
			}
		}
		l.text = string(text)
	}
	r.lookForToJSON(l, o.shape)
	sideOf[struct{}](o.shape).json = l
	return l
}

// lookForToJSON says in l what objects of layout s make of toJSON, and how
// long that holds.
func (r *Runtime) lookForToJSON(l *jsonLayout, s *shape) {
	l.cell = r.chainCell(s.proto)
	l.asks = l.cell == nil
	for _, k := range s.keys[:s.n] {
		if k.key == atomToJSON {
			l.asks = true
		}
	}
	for p := s.proto; p != nil && !l.asks; p = p.proto() {
		// A proxy or a namespace answers for itself; the rest hold it in
		// their tables.
		l.asks = proxyOf(p) != nil || p.class == ClassModuleNamespace || p.getOwn(atomToJSON).ok()
	}
}

// lacksToJSON reports whether o is known to have no toJSON, without looking
// one up, which walks its prototypes.
func (r *Runtime) lacksToJSON(o *Object) bool {
	if o.class != ClassObject && o.class != ClassArray {
		return false
	}
	l := r.jsonLayoutOf(o)
	if l == nil {
		return false
	}
	if c := l.cell; c == nil || !c.valid {
		r.lookForToJSON(l, o.shape)
	}
	return !l.asks
}

// writeLayout is writeKeys for an object of a layout with a jsonLayout.
func (e *jsonEncoder) writeLayout(buf []byte, v Value, o *Object, l *jsonLayout) ([]byte, error) {
	if !l.ascii {
		e.rescan = true
	}
	var inner []byte
	if e.indent != "" {
		inner = e.margin(e.depth)
	}
	s := o.shape
	start := len(buf)
	buf = append(buf, '{')
	buf = append(buf, inner...)
	written := false
	from := 0
	for i := range l.keys {
		k := &l.keys[i]
		text := l.text[from:k.end]
		from = int(k.end)
		if err := e.rt.tick(); err != nil {
			return buf, err
		}
		if len(buf) >= e.limit {
			if err := e.charged(len(buf)); err != nil {
				return buf, err
			}
		}
		var val Value
		var err error
		if k.data && o.shape == s {
			val = o.valAt(int(k.slot))
		} else if val, err = e.rt.getProp(o, k.atom, v); err != nil {
			return buf, err
		}
		if e.needsApply(val) {
			if val, err = e.apply(v, k.atom, val); err != nil {
				return buf, err
			}
		}
		// As in writeKeys, a value with no JSON form takes its key back.
		mark := len(buf)
		if inner != nil {
			if written {
				buf = append(buf, ',')
				buf = append(buf, inner...)
			}
			buf = append(buf, text[1:len(text)-1]...)
			buf = append(buf, ": "...)
		} else if written {
			buf = append(buf, text...)
		} else {
			buf = append(buf, text[1:]...)
		}
		var ok bool
		buf, ok, err = e.encode(buf, val)
		if err != nil {
			return buf, err
		}
		if !ok {
			buf = buf[:mark]
			continue
		}
		written = true
	}
	if !written {
		return append(buf[:start], "{}"...), nil
	}
	if inner != nil {
		buf = append(buf, e.margin(e.depth-1)...)
	}
	return append(buf, '}'), nil
}
