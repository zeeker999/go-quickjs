package ssa

import (
	"github.com/go-quickjs/go-quickjs/internal/arena"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

// A Workspace is the memory a runtime's compiles share, as V8 compiles in a
// zone: a Func built in one (BuildIn) takes its values, their argument
// lists, its frame states and blocks, and its passes' tables from the
// workspace's arenas, and Rewind takes them all back at once, keeping the
// chunks, so that a runtime that compiles often allocates little. Nothing
// built in a workspace may be used after its Rewind. A Func built without
// one (BuildWith) allocates its own, and may be kept.
type Workspace struct {
	values    arena.Arena[Value]
	refs      arena.Arena[*Value]
	states    arena.Arena[FrameState]
	blocks    arena.Arena[Block]
	blockRefs arena.Arena[*Block]
	bools     arena.Arena[bool]
	ints      arena.Arena[int]
	scratch   scratch
	ready     bool
}

// init has the workspace's arenas keep the big tables a compile makes.
func (w *Workspace) init() {
	if w.ready {
		return
	}
	w.ready = true
	w.values.KeepBig()
	w.refs.KeepBig()
	w.states.KeepBig()
	w.blocks.KeepBig()
	w.blockRefs.KeepBig()
	w.bools.KeepBig()
	w.ints.KeepBig()
}

// Rewind takes back everything built in the workspace.
func (w *Workspace) Rewind() {
	w.values.Rewind()
	w.refs.Rewind()
	w.states.Rewind()
	w.blocks.Rewind()
	w.blockRefs.Rewind()
	w.bools.Rewind()
	w.ints.Rewind()
}

// BuildIn is BuildWith, building in w.
func BuildIn(w *Workspace, p *ir.Program, fb Feedback) (*Func, error) {
	w.init()
	return build(w, p, fb)
}

// bools is n false flags: from the workspace, if the Func has one.
func (f *Func) bools(n int) []bool {
	if f.ws != nil {
		return f.ws.bools.Make(n)
	}
	return make([]bool, n)
}

// ints is n zeros: from the workspace, if the Func has one.
func (f *Func) ints(n int) []int {
	if f.ws != nil {
		return f.ws.ints.Make(n)
	}
	return make([]int, n)
}

// appendValue is append(s, v), growing s in the workspace if there is one.
func (f *Func) appendValue(s []*Value, v *Value) []*Value {
	if f.ws != nil {
		return f.ws.refs.Append(s, v)
	}
	return append(s, v)
}

// appendBlock is append(s, b), growing s in the workspace if there is one.
func (f *Func) appendBlock(s []*Block, b *Block) []*Block {
	if f.ws != nil {
		return f.ws.blockRefs.Append(s, b)
	}
	return append(s, b)
}
