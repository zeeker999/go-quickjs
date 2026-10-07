package jit

import (
	"errors"
	"runtime"
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

// MaxCodeBytes bounds a single emitted allocation before page rounding.
const MaxCodeBytes = 1 << 20

// ErrProgram reports malformed or oversized IR, before executable allocation.
var ErrProgram = errors.New("invalid native program")

// ErrCodeBudget means optional code and its metadata cannot fit the owner budget.
var ErrCodeBudget = errors.New("native code memory budget exceeded")

// Code owns an immutable native program, entry offsets, and exit maps. Running
// and closing the same Code concurrently is unsupported. Separate owners may
// execute the same lowered IR concurrently with independent scratch storage.
type Code struct {
	code    []byte
	entries []int
	maps    []ir.StateMap
	slots   int
}

// A host instruction exits before doing native work. Mark its external entry
// separately from an unreachable PC (-1), avoiding a load/spill round trip.
const hostProgramEntry = -2

// programState is the native ABI. It contains no Go pointer. Scratch is passed
// separately as a typed pointer to scalar storage rooted by Run's slots slice.
type programState struct {
	remaining uint64
	reason    uint64
	pc        uint64
	value     ir.Value
}

// Compile validates and emits the slot IR. It never retains p or a runtime
// address. Unsupported builds and denied executable memory return ErrUnavailable.
func Compile(p *ir.Program) (*Code, error) {
	return CompileBudget(p, MaxCodeBytes+ir.MaxInstructions*32+1024)
}

// CompileBudget additionally bounds page-rounded executable memory and retained
// metadata together. A refusal occurs before OS allocation. Transient emission
// work is bounded by the IR and code-size limits regardless of this budget.
func CompileBudget(p *ir.Program, bytes int) (*Code, error) { return compileProgram(p, bytes) }

// EntryDepth reports the live operand depth at a reachable native entry.
// Closed code and unreachable or invalid PCs have no entry.
func (c *Code) EntryDepth(pc int) (int, bool) {
	if c == nil || len(c.code) == 0 || pc < 0 || pc >= len(c.entries) || c.entries[pc] == -1 {
		return 0, false
	}
	return c.maps[pc].Depth, true
}

// Run enters at a reachable bytecode PC and executes at most MaxIterations
// committed IR instructions before returning to Go. Guards commit no part of
// their failing instruction. Reference handles remain the caller's ownership.
func (c *Code) Run(slots []ir.Value, pc int, budget uint64) (ir.Exit, error) {
	return c.RunArrays(slots, nil, pc, budget)
}

// RunArrays borrows views only for this bounded entry. Native code may update
// numeric bits of existing numeric cells, but never Go references or slice state.
func (c *Code) RunArrays(slots []ir.Value, arrays []ir.ArrayView, pc int, budget uint64) (ir.Exit, error) {
	return c.runArrays(slots, arrays, pc, budget, true)
}

// RunEncodedArrays enters with scalars produced by the VM encoder or a previous
// native entry. The caller must preserve valid kinds and boolean bits when
// updating them. It skips the redundant scalar scan but still checks storage
// shape, reachable PC, closed code, and the instruction budget.
func (c *Code) RunEncodedArrays(slots []ir.Value, arrays []ir.ArrayView, pc int, budget uint64) (ir.Exit, error) {
	return c.runArrays(slots, arrays, pc, budget, false)
}

func (c *Code) runArrays(slots []ir.Value, arrays []ir.ArrayView, pc int, budget uint64, validate bool) (ir.Exit, error) {
	if len(arrays) != 0 && len(arrays) != ir.MaxSlots {
		return ir.Exit{}, ir.ErrState
	}
	if c == nil || len(c.code) == 0 {
		return ir.Exit{}, ErrClosed
	}
	if budget > MaxIterations {
		return ir.Exit{}, ErrIterations
	}
	if len(slots) != c.slots || pc < 0 || pc >= len(c.entries) || c.entries[pc] == -1 {
		return ir.Exit{}, ir.ErrState
	}
	if validate {
		for _, slot := range slots {
			if slot.Kind > ir.Opaque || slot.Kind == ir.Boolean && slot.Bits > 1 {
				return ir.Exit{}, ir.ErrState
			}
		}
	}
	if c.entries[pc] == hostProgramEntry {
		kind := ir.HostExit
		if budget == 0 {
			kind = ir.BudgetExit
		}
		return ir.Exit{Kind: kind, State: c.maps[pc]}, nil
	}
	s := programState{remaining: budget, pc: uint64(pc)}
	runProgramCode(c.code, c.entries[pc], &s, slots, arrays)
	runtime.KeepAlive(c)
	return ir.Exit{Kind: ir.ExitKind(s.reason), State: c.maps[s.pc], Value: s.value, Steps: budget - s.remaining}, nil
}

// Size reports page-rounded executable memory currently owned by c.
func (c *Code) Size() int {
	if c == nil {
		return 0
	}
	return len(c.code)
}

// MetadataSize reports the retained header and entry/exit capacities in bytes.
// Scratch is caller-owned and is not included. Closed code reports zero.
func (c *Code) MetadataSize() int {
	if c == nil || len(c.code) == 0 {
		return 0
	}
	return c.metadataBytes()
}

func (c *Code) metadataBytes() int {
	return int(unsafe.Sizeof(*c)) + cap(c.entries)*int(unsafe.Sizeof(int(0))) + cap(c.maps)*int(unsafe.Sizeof(ir.StateMap{}))
}

// Close releases code and metadata. An OS release failure retains ownership so
// a caller can retry. The owner must close only after native execution unwinds.
func (c *Code) Close() error {
	if c == nil || len(c.code) == 0 {
		return nil
	}
	if err := freeLoopCode(c.code); err != nil {
		return err
	}
	c.code, c.entries, c.maps = nil, nil, nil
	runtime.SetFinalizer(c, nil)
	return nil
}
