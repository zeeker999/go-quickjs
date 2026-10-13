//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"runtime"
	"testing"
	"unsafe"
	"weak"
)

func TestYoungGenerationReuse(t *testing.T) {
	h := newYoungHeap(2)
	a, b := h.allocate(nil, nil), h.allocate(nil, nil)
	child := newObject(nil, ClassObject)
	ref := weak.Make(child)
	a.props = append(a.props, Property{value: Obj(child)})
	h.markWord(uintptr(unsafe.Pointer(b)))
	h.collect()
	c := h.allocate(nil, nil)
	if c != a || len(c.props) != 0 || c.elems != nil || c.data != nil || c.weakMapRefs != nil {
		t.Fatal("dead object was not cleared and reused")
	}
	runtime.GC()
	if ref.Value() != nil {
		t.Fatal("recycled object retains a dead child")
	}
	if h.stats.Reused != 1 || h.stats.Reclaimed != 1 {
		t.Fatalf("stats: %+v", h.stats)
	}
	runtime.KeepAlive(h)
}

func TestYoungGenerationCyclesAndTenure(t *testing.T) {
	h := newYoungHeap(4)
	a, b := h.allocate(nil, nil), h.allocate(nil, nil)
	a.props = append(a.props, Property{value: Obj(b)})
	b.props = append(b.props, Property{value: Obj(a)})
	h.allocate(nil, nil)
	h.markWord(uintptr(unsafe.Pointer(&a.props[0].value)))
	h.collect()
	if h.stats.Young != 2 || h.stats.Reclaimed != 1 {
		t.Fatalf("cycle lost: %+v", h.stats)
	}
	c := h.allocate(nil, nil)
	b.props = append(b.props, Property{value: Obj(c)})
	h.markWord(uintptr(unsafe.Pointer(a)))
	h.collect()
	if h.stats.Young != 0 || h.stats.Promoted != 3 || a.props[0].value.Object() != b || b.props[1].value.Object() != c {
		t.Fatalf("tenure moved objects or left a young descendant: %+v", h.stats)
	}
	h.allocate(nil, nil)
	if a.props[0].value.Object() != b {
		t.Fatal("tenured allocation reused")
	}
}

func TestYoungGenerationDeadCycle(t *testing.T) {
	h := newYoungHeap(2)
	a, b := h.allocate(nil, nil), h.allocate(nil, nil)
	a.props = append(a.props, Property{value: Obj(b)})
	b.props = append(b.props, Property{value: Obj(a)})
	h.collect()
	if h.stats.Young != 0 || h.stats.Reclaimed != 2 {
		t.Fatalf("dead cycle retained: %+v", h.stats)
	}
}

func TestYoungGenerationPromotionReleasesOwnership(t *testing.T) {
	h := newYoungHeap(2)
	a, b := h.allocate(nil, nil), h.allocate(nil, nil)
	ref := weak.Make(b)
	h.promoteAll()
	runtime.GC()
	if ref.Value() != nil {
		t.Fatal("nursery retains promoted object")
	}
	if h.stats.Young != 0 || h.stats.Promoted != 2 || len(h.owners) != 0 {
		t.Fatalf("stats: %+v", h.stats)
	}
	runtime.KeepAlive(a)
	runtime.KeepAlive(h)
}
