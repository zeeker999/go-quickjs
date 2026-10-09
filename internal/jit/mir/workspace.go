package mir

import (
	"runtime"

	"github.com/go-quickjs/go-quickjs/internal/arena"
	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
	"github.com/go-quickjs/go-quickjs/internal/jit/ssa"
)

// A Workspace is the memory a runtime's compiles share, as ssa.Workspace is
// for building: a compile in one (CompileIn) takes its analysis and
// allocation tables from the workspace's arenas, and Rewind takes them all
// back at once, keeping the chunks. The Code's Bytes and Entries are its
// own; its Locations, for tests, may not be asked for after the Rewind.
type Workspace struct {
	bools     arena.Arena[bool]
	ints      arena.Arena[int]
	words     arena.Arena[uint64]
	sets      arena.Arena[valueSet]
	values    arena.Arena[*ssa.Value]
	blocks    arena.Arena[*ssa.Block]
	uses      arena.Arena[use]
	intervals arena.Arena[interval]
	ivRefs    arena.Arena[*interval]
	locs      arena.Arena[loc]
}

// Rewind takes back everything a compile in the workspace used.
func (w *Workspace) Rewind() {
	w.bools.Rewind()
	w.ints.Rewind()
	w.words.Rewind()
	w.sets.Rewind()
	w.values.Rewind()
	w.blocks.Rewind()
	w.uses.Rewind()
	w.intervals.Rewind()
	w.ivRefs.Rewind()
	w.locs.Rewind()
}

// CompileIn is Compile, in w.
func CompileIn(w *Workspace, f *ssa.Func, enc abi.Encoding) (*Code, error) {
	if runtime.GOARCH == "arm64" {
		return compileARM64(w, f, enc)
	}
	return compileAMD64(w, f, enc)
}

// The core's tables, from its workspace if it has one.

func (c *core) bools(n int) []bool {
	if c.ws != nil {
		return c.ws.bools.Make(n)
	}
	return make([]bool, n)
}

func (c *core) ints(n int) []int {
	if c.ws != nil {
		return c.ws.ints.Make(n)
	}
	return make([]int, n)
}

func (c *core) words(n int) valueSet {
	if c.ws != nil {
		return c.ws.words.Make(n)
	}
	return make(valueSet, n)
}

func (c *core) setList(n int) []valueSet {
	if c.ws != nil {
		return c.ws.sets.Make(n)
	}
	return make([]valueSet, n)
}

func (c *core) valueList(n int) []*ssa.Value {
	if c.ws != nil {
		return c.ws.values.Make(n)
	}
	return make([]*ssa.Value, n)
}

func (c *core) blockList(n int) []*ssa.Block {
	if c.ws != nil {
		return c.ws.blocks.Make(n)
	}
	return make([]*ssa.Block, n)
}

func (c *core) useList(n int) []use {
	if c.ws != nil {
		return c.ws.uses.Make(n)
	}
	return make([]use, n)
}

func (c *core) intervalList(n int) []interval {
	if c.ws != nil {
		return c.ws.intervals.Make(n)
	}
	return make([]interval, n)
}

func (c *core) intervalRefs(n int) []*interval {
	if c.ws != nil {
		return c.ws.ivRefs.Make(n)
	}
	return make([]*interval, n)
}

func (c *core) locList(n int) []loc {
	if c.ws != nil {
		return c.ws.locs.Make(n)
	}
	return make([]loc, n)
}
