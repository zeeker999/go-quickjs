package vm

import "unsafe"

// A closure's appends to a string its binding alone holds: the compiler's
// ownedAppend says which, and what makes it safe.
//
// The string in the binding is an ordinary one at every moment, a rope that
// anything may read, so a tier that knows nothing of this is right: only the
// additions the compiler marked, in the tree tier and native code, come here.
// building says the string is the binding's alone, and a read of the binding
// clears it (str_freeze, strFreeze). It is set only on what an append has
// just made.
//
// An append makes the rope any addition would, and marks it: x += y and
// return x cost what they did. The strAppendRopes-th append in a row finds its
// left side marked, so no one else can have it, and gives it a right side with
// room, a strTail, which that append and the ones after it grow where it is,
// in bytes, with no node and nothing for the collector to walk. A read clears
// the mark, and the append after it starts again with a rope over what was
// read: appends with a read every few cost a rope each, as they did. A tail
// costs what about three ropes cost, so a run has paid that much in ropes
// before it buys one: bought at the second append, x += a; x += b; read x
// took half as long again, for a tail made and thrown away each time.
//
// A string past the longest, a meter's count, a surrogate's halves meeting
// across the join and an empty operand are concat's, as ever: all of them
// may give back something not new, which is never marked.

// strAppendRopes is how many appends in a row make a rope before the next
// gives the string a tail.
const strAppendRopes = 3

// strTail is the right side of an append's string: a flat string over buf,
// which is grown in place.
type strTail struct {
	String
	buf []byte
}

// strAppend is a + b at an addition the compiler marked, a and b strings.
func strAppend[_ any](r *Runtime, a, b *String) (*String, error) {
	if a.length == 0 || b.length == 0 || a.endsHigh && b.startsLow || r.meter != nil ||
		a.length+b.length > maxStringLength {
		strFreeze[struct{}](Str(a))
		return r.concat(a, b)
	}
	if !a.tailed && (!a.building || a.left == nil || a.ropes < strAppendRopes-1) {
		s := a.Concat(b)
		// Neither operand, both not being empty: a new rope, or a short
		// string copied, which has nothing to grow.
		if s.left != nil {
			s.building = true
			if a.building {
				s.ropes = a.ropes + 1
			}
		}
		return s, nil
	}
	if !a.building || a.left == nil {
		// Tailed once, and read, or flattened, since.
		s := a.Concat(b)
		s.building = s.left != nil
		return s, nil
	}
	if !a.tailed {
		strTailed(a)
	}
	t := (*strTail)(a.aux)
	t.buf = append(t.buf, b.Go()...)
	t.s = unsafe.String(unsafe.SliceData(t.buf), len(t.buf))
	t.length += b.length
	// Only a surrogate's join reads a right side's flags, and its pair is no
	// ASCII: ascii here is for what reads a piece by itself.
	t.ascii = t.ascii && b.ascii
	t.endsHigh = b.endsHigh
	a.length += b.length
	a.ascii = a.ascii && b.ascii
	a.endsHigh = b.endsHigh
	return a, nil
}

// strTailed gives a, a rope, a right side with room, holding what it had.
//
//go:noinline
func strTailed(a *String) {
	right := a.rightSide()
	s := right.Go()
	t := &strTail{buf: make([]byte, len(s), max(64, 4*len(s)))}
	copy(t.buf, s)
	t.s = unsafe.String(unsafe.SliceData(t.buf), len(t.buf))
	t.length, t.ascii, t.endsHigh, t.startsLow = right.length, right.ascii, right.endsHigh, right.startsLow
	a.aux, a.tailed = unsafe.Pointer(&t.String), true
}

// strFreeze is str_freeze: v, if it is a string an append owns, is not any
// more. It writes only such a string: another may be a constant other
// runtimes read.
//
//go:noinline
func strFreeze[_ any](v Value) {
	if v.IsString() {
		if s := v.String(); s.building {
			s.building = false
		}
	}
}
