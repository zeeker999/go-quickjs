package ssa

import (
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

func TestPropertySlotFallback(t *testing.T) {
	v := &Value{Op: OpPropCell, Index: 1, Key: 7, IC: &InlineCache{Cases: []PropertyCase{{Shape: 2, Index: 1}}}}
	v.Const.Bits = 1
	if !v.SlotFallback() || checksShape(v) {
		t.Fatal("a read of a common slot must accept the slot without proving a shape")
	}
	for _, change := range []func(*Value){
		func(v *Value) { v.Op = OpPropWrite },
		func(v *Value) { v.IC.Mega = true },
		func(v *Value) { v.IC.Cases = nil },
		func(v *Value) { v.IC.Cases = []PropertyCase{{Shape: 2, Index: 0}} },
		func(v *Value) {
			v.IC.Cases = []PropertyCase{{Shape: 2, Index: 1, Holders: [2]Holder{{Object: 3, Shape: 4}}}}
		},
		func(v *Value) { v.IC.Holders = &HolderChain{Holders: [2]Holder{{Object: 3, Shape: 4}}} },
	} {
		x, ic := *v, *v.IC
		x.IC = &ic
		change(&x)
		if x.SlotFallback() {
			t.Fatalf("read without a common own slot has fallback: %+v, %+v", x, ic)
		}
	}
	h := &Heap{}
	for _, o := range []Object{
		{Shape: 3, Ordinary: true, Keys: []uint32{6, 7}, Data: []bool{true, true}},
		{Shape: 3, Ordinary: true, Keys: []uint32{6, 8}, Data: []bool{true, true}},
		{Shape: 3, Ordinary: true, Keys: []uint32{6, 7}, Data: []bool{true, false}},
		{Shape: 3, Ordinary: true, Keys: []uint32{6, 7}, Data: []bool{true, true}, Uninit: []bool{false, true}},
		{Shape: 3, Keys: []uint32{6, 7}, Data: []bool{true, true}},
		{Shape: 3, Ordinary: true},
	} {
		o.Props = make([]ir.Value, len(o.Keys))
		got, i := h.holder(&o, v)
		want := o.Ordinary && len(o.Keys) > 1 && o.Keys[1] == 7 && o.Data[1] && len(o.Uninit) == 0
		if (got == &o && i == 1) != want {
			t.Fatalf("object %+v: holder %p, index %d, want hit %v", o, got, i, want)
		}
	}
}
