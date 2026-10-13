//go:build quickjs_young

package vm

import "unsafe"

func objectWord(o *Object) uintptr { return uintptr(unsafe.Pointer(o)) }
func valueWord(v Value) uintptr    { return uintptr(v.ref) }

const youngBuilt = true

type youngFields struct {
	youngCapacity int
	young         *youngHeap
}

func (r *Runtime) initYoung(capacity int) {
	// Memory accounting and debugger callbacks need their own root protocol.
	if r.meter == nil && r.debug == nil {
		r.youngCapacity = capacity
	}
}

func (r *Runtime) promoteYoung() {
	if r.young != nil {
		r.young.promoteAll()
	}
}

func (r *Runtime) releaseYoung() {
	r.promoteYoung()
	r.young, r.youngCapacity = nil, 0
}

// YoungGenerationStats reports nursery activity in this runtime.
func (r *Runtime) YoungGenerationStats() YoungGenerationStats {
	if r.young == nil {
		return YoungGenerationStats{}
	}
	return r.young.stats
}

// youngHeap owns independent Go allocations until they die or are tenured.
// Its roots must be supplied at a boundary that accounts for every live
// reference. It deliberately does not use the memory meter's approximate walk.
type youngHeap struct {
	slots  []youngSlot
	free   []int
	owners map[uintptr]int
	work   []int
	stats  YoungGenerationStats
}

type youngSlot struct {
	object       *literalObject2
	used, marked bool
	age          uint8
}

func newYoungHeap(capacity int) *youngHeap {
	h := &youngHeap{slots: make([]youngSlot, capacity), free: make([]int, capacity), owners: make(map[uintptr]int, capacity*3)}
	for i := range h.free {
		h.free[i] = capacity - 1 - i
	}
	return h
}

func (h *youngHeap) allocate(proto *Object, root *shape) *Object {
	if len(h.free) == 0 {
		return nil
	}
	i := h.free[len(h.free)-1]
	h.free = h.free[:len(h.free)-1]
	s := &h.slots[i]
	if s.object == nil {
		s.object = new(literalObject2)
		h.stats.Allocated++
	} else {
		h.stats.Reused++
	}
	o := &s.object.Object
	o.proto, o.shape = proto, root
	o.class, o.flags = ClassObject, objExtensible|objInlineProps
	o.props = s.object.inline[:0]
	s.used, s.marked, s.age = true, false, 0
	h.owners[uintptr(unsafe.Pointer(o))] = i
	// Native references can be shadows of property cells or addresses of
	// their inline storage. Keep the owner alive for both kinds of pointer.
	for j := range s.object.inline {
		h.owners[uintptr(unsafe.Pointer(&s.object.inline[j]))] = i
		h.owners[uintptr(unsafe.Pointer(&s.object.inline[j].value))] = i
	}
	h.stats.Young++
	return o
}

func (h *youngHeap) markWord(word uintptr) {
	if i, ok := h.owners[word]; ok && !h.slots[i].marked {
		h.slots[i].marked = true
		h.work = append(h.work, i)
	}
}

func (h *youngHeap) edges(o *Object) {
	h.markWord(uintptr(unsafe.Pointer(o.proto)))
	for _, p := range o.props {
		h.markWord(uintptr(p.value.ref))
	}
	for _, v := range o.elems {
		h.markWord(uintptr(v.ref))
	}
}

func (h *youngHeap) trace() {
	for len(h.work) != 0 {
		i := h.work[len(h.work)-1]
		h.work = h.work[:len(h.work)-1]
		h.edges(&h.slots[i].object.Object)
	}
}

func (h *youngHeap) forget(i int, promote bool) {
	s := &h.slots[i]
	delete(h.owners, uintptr(unsafe.Pointer(&s.object.Object)))
	for j := range s.object.inline {
		delete(h.owners, uintptr(unsafe.Pointer(&s.object.inline[j])))
		delete(h.owners, uintptr(unsafe.Pointer(&s.object.inline[j].value)))
	}
	if promote {
		// Release ownership without changing the address or contents. Go
		// reclaims this allocation once its ordinary references disappear.
		s.object = nil
		h.stats.Promoted++
	} else {
		// Clear the whole allocation, including inline fields past props' len.
		*s.object = literalObject2{}
		h.stats.Reclaimed++
	}
	s.used, s.marked, s.age = false, false, 0
	h.free = append(h.free, i)
	h.stats.Young--
}

// collect follows young edges, sweeps dead slots, and tenures the reachable
// graph of aged survivors. Promotion must not leave an old-to-young edge.
func (h *youngHeap) collect() {
	h.trace()
	h.stats.Collections++
	for i := range h.slots {
		s := &h.slots[i]
		if !s.used {
			continue
		}
		if !s.marked {
			h.forget(i, false)
		} else {
			s.age++
		}
	}
	for i := range h.slots {
		h.slots[i].marked = false
	}
	for i := range h.slots {
		if s := &h.slots[i]; s.used && s.age >= 2 {
			h.markWord(uintptr(unsafe.Pointer(&s.object.Object)))
		}
	}
	h.trace()
	for i := range h.slots {
		if s := &h.slots[i]; s.used && s.marked {
			h.forget(i, true)
		}
	}
}

// promoteAll is the conservative escape barrier used before untracked Go
// code or a native exit can expose a value to the interpreter or host.
func (h *youngHeap) promoteAll() {
	for i := range h.slots {
		if h.slots[i].used {
			h.forget(i, true)
		}
	}
	h.work = h.work[:0]
}
