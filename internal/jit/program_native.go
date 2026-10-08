//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package jit

import (
	"fmt"
	"os"
	"runtime"

	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

func compileProgram(p *ir.Program, limit int, dispatch bool) (*Code, error) {
	if err := p.Validate(); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrProgram, err)
	}
	if limit <= 0 {
		return nil, ErrCodeBudget
	}
	instructions, entries, err := programInstructions(p, dispatch)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrProgram, err)
	}
	c := &Code{entries: entries, maps: make([]ir.StateMap, len(p.Maps)), slots: p.Locals + p.StackSize, locals: p.Locals, dispatch: dispatch}
	copy(c.maps, p.Maps)
	if dispatch {
		c.dispatchEntries = append([]int(nil), entries...)
	}
	for pc, in := range p.Code {
		if in.Op == ir.Call && p.Maps[pc].Depth >= 0 {
			c.calls = append(c.calls, callEntry{pc: pc, in: in})
		}
		if in.Op == ir.Host && !in.Check && c.entries[pc] >= 0 {
			c.entries[pc] = hostProgramEntry
			if dispatch {
				c.dispatchEntries[pc] = hostProgramEntry
			}
		}
		if in.Op == ir.Call && !in.Check && c.entries[pc] >= 0 {
			c.entries[pc] = hostProgramEntry
		}
	}
	page := os.Getpagesize()
	allocation := (len(instructions) + page - 1) / page * page
	if allocation+c.metadataBytes() > limit {
		return nil, ErrCodeBudget
	}
	code, err := allocateCode(instructions)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrUnavailable, err)
	}
	c.code = code
	// Deterministic owner cleanup is primary; a forgotten Runtime must not
	// permanently retain mappings after its Go ownership disappears.
	runtime.SetFinalizer(c, func(owner *Code) { _ = owner.Close() })
	return c, nil
}

func runProgramCode(code []byte, offset int, state *programState, slots []ir.Value, arrays []ir.ArrayView) {
	var first *ir.Value
	if len(slots) != 0 {
		first = &slots[0]
	}
	var views *ir.ArrayView
	if len(arrays) != 0 {
		views = &arrays[0]
	}
	enterProgram(&code[offset], state, first, views)
	runtime.KeepAlive(arrays)
	runtime.KeepAlive(code)
	runtime.KeepAlive(state)
	runtime.KeepAlive(slots)
}

// enterProgram has the same leaf-only contract as enter: no native calls or
// stack changes, and every path is bounded by a checked instruction budget.
//
//go:noescape
func enterProgram(code *byte, state *programState, slots *ir.Value, arrays *ir.ArrayView)

func runDispatchCode(code *byte, state *dispatchState, slots *ir.Value, arrays []ir.ArrayView) {
	var views *ir.ArrayView
	if len(arrays) != 0 {
		views = &arrays[0]
	}
	enterDispatch(code, state, slots, views)
	runtime.KeepAlive(arrays)
	runtime.KeepAlive(state)
}

//go:noescape
func enterDispatch(code *byte, state *dispatchState, slots *ir.Value, arrays *ir.ArrayView)
