package vm

// Literals with the same keys, in any code of a realm, share a layout.
//
// V8 keeps, for each native context, a map for each count of an object
// literal's properties (Factory::ObjectLiteralMapFromCache), and a literal's
// keys take the transitions out of it, whatever script it is in. Here a
// literal's site has a root of its own (siteRoot), made in a few instructions
// of the interpreter's loop, which holds no key: what it adds, as its first
// object is given its first key, is the realm's root for the room that object
// was made with, the first transition it takes -- as a constructor's root
// takes its kept root's (ctorChild) -- and its objects are of the layouts any
// literal with the same keys and room has. So code new for each request makes
// no shape but its sites' roots, and code that is kept reads the objects of the
// next request's literals with the caches that read the last one's.
//
// A root and what grows from it is the realm's: it holds them for as long as
// it lives. So they are counted against a budget of their own, and past a
// quarter of the tree, or with a root's transitions filled, the realm starts
// again with new roots: those of code still running stay for as long as that
// code does, as a site's own did, and what no code has goes.

const (
	// sharedRootClass is the rootClass of a realm's literal root and of the
	// shapes grown from it: no object's class. Those shapes' room is noted
	// on the layout (noteRoom), a root being many literals'.
	sharedRootClass = ctorRootClass + 1
	// literalRooms is how many rooms have a root: a literal of more stays on
	// its site's.
	literalRooms = 64
	// sharedFanShift is how many times over a shape of a realm's roots may
	// have maxShapeFanout's transitions, as a shift (shape.used): 2048, near
	// V8's kMaxNumberOfTransitions, each literal having one somewhere, where
	// what maxShapeFanout bounds is keys made up.
	sharedFanShift = 5
	// literalShapes is the most shapes a realm's roots may have grown.
	literalShapes = maxShapes / 4
)

// literalTable is a realm's literals' roots.
type literalTable struct {
	byRoom [literalRooms]*shape
	// budget is the slot of a budget of its own, which all the roots hold
	// (chargeChild).
	budget *budgetSlot
	// full says a root or a shape grown from one has no room for another
	// transition: the realm starts again.
	full bool
}

// chainClass is the rootClass of a shape grown from s: s's, if it is of a
// realm's roots.
func (s *shape) chainClass() Class {
	if s.rootClass == sharedRootClass {
		return sharedRootClass
	}
	return 0
}

// chainFan is the used of a shape grown from s: s's, if it is of a realm's
// roots, whose shapes it lets have more transitions than others (child).
func (s *shape) chainFan() uint8 {
	if s.rootClass == sharedRootClass {
		return s.used
	}
	return 0
}

// literalChild is the shape an object of s, a root that has no transition yet,
// takes for e, if s is a literal's site's: the child for e of the realm's root
// for the room the object has (shapeTree.room), which s takes as its first
// transition. It is nil for any other root, a keyed store's key, which is the
// program's and not the literal's, and where the realm's roots can hold no
// more: s then grows shapes of its own.
//
//go:noinline
func literalChild[_ any](s *shape, e shapeEdge) *shape {
	t, p, room := s.tree, s.proto, int(s.tree.room)
	if s.generic || s.rootClass != ClassObject || p == nil || room >= literalRooms || t.keyed != nil {
		return nil
	}
	side := sideOf[struct{}](p.ownLayout())
	ps := side.programs
	if ps == nil {
		ps = &programTable{}
		side.programs = ps
	}
	l := ps.literals
	if l == nil || l.full || l.budget.b.n.Load() >= literalShapes {
		p.flags |= objProtoMet
		t.chunk = nil
		l = &literalTable{budget: budgetSlotOf[struct{}](t)}
		t.chunk = nil // The chunk is the realm's roots' alone.
		ps.literals = l
	}
	root := l.byRoom[room]
	if root == nil {
		root = &shape{tree: t, proto: p, budget: l.budget, rootClass: sharedRootClass, used: sharedFanShift}
		l.byRoom[room] = root
	}
	c := root.child(e)
	if c == nil {
		l.full = true
		return nil
	}
	s.setNext(c, e)
	return c
}

// noteRoom records that an object of layout s came to need room for n
// properties: on its root, or, on a realm's root, which is many
// literals', on the layouts from it to s, each at least its children's, for the
// sites whose objects end there to ask (siteRoom).
//
//go:noinline
func noteRoom[_ any](s *shape, n int) {
	v := uint8(min(n, maxRoom))
	if s.rootClass != sharedRootClass {
		if root := s.root(); v > root.slack {
			root.slack = v
		}
		return
	}
	for ; s.parent != nil && s.slack < v; s = s.parent {
		s.slack = v
	}
}

// siteRoom is the room the objects of a literal's site came to need, of root,
// its root, and ic, its cache, if there is one: what its root learned, or,
// where its objects' layouts are a realm's, what its final layout did, or,
// that not known yet, its first.
func siteRoom[_ any](root *shape, ic *propCache) int {
	n := int(root.slack)
	if c := root.next; c != nil {
		if ic != nil && ic.s1 == root && ic.next != nil {
			c = ic.next
		}
		if int(c.slack) > n {
			n = int(c.slack)
		}
	}
	return n
}

// shapeAlong is the shape s has a transition by e to, or nil.
func shapeAlong[_ any](s *shape, e shapeEdge) *shape {
	if s.next == nil {
		return nil
	}
	if s.nextEdge() == e {
		return s.next
	}
	return s.more[e]
}
