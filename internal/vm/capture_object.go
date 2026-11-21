package vm

import "github.com/go-quickjs/go-quickjs/internal/bytecode"

// These combinations fit the allocation classes their separate function
// objects and cell batches already occupy. The cells keep the function
// alive when another closure shares them.
type scriptFuncCells6 struct {
	scriptFuncObject
	captured [6]*upvalue
	owned    [3]upvalue
}

type scriptFuncCells15 struct {
	scriptFuncObject
	captured [15]*upvalue
	owned    [4]upvalue
}

func captureObjectFor(fn *bytecode.Function) uint8 {
	n := len(fn.Upvalues)
	if fn.Kind == bytecode.KindArrow || fn.CapturesByValue || !(n >= 4 && n <= 6 || n >= 12 && n <= 15) {
		return 0
	}
	count := 0
	for _, desc := range fn.Upvalues {
		if desc.FromParent {
			count++
		}
	}
	if n <= 6 && count == 3 || n >= 12 && count == 4 {
		return uint8(count)
	}
	return 0
}

func newCapturedFuncObject(proto *Object, n int, count uint8) (*scriptFuncObject, []*upvalue, []upvalue) {
	var fo *scriptFuncObject
	var captured []*upvalue
	var room []upvalue
	if count == 3 {
		x := &scriptFuncCells6{}
		fo, captured, room = &x.scriptFuncObject, x.captured[:n:n], x.owned[:]
	} else {
		x := &scriptFuncCells15{}
		fo, captured, room = &x.scriptFuncObject, x.captured[:n:n], x.owned[:]
	}
	fo.shape, fo.class, fo.flags = rootFor(ClassFunction, proto), ClassFunction, objExtensible
	fo.embedExt(&fo.side)
	fo.side.data = &fo.fn
	return fo, captured, room
}
