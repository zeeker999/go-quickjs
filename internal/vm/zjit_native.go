//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"errors"
	"math"
	"runtime"
	"weak"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/jit"
	jitcompile "github.com/go-quickjs/go-quickjs/internal/jit/compile"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

const jitCacheEntries = 128
const jitCacheBytes = 8 << 20
const jitMetadataBytes = 512 << 10

type jitFields struct {
	jitEnabled bool
	jit        *jitState
}

// A closure belongs to one runtime. Permanent bytecode refusals can be
// remembered here without repeatedly registering and looking up a weak key.
type jitClosureFields struct {
	jitRefused bool
}

type jitEntry struct {
	code   *jit.Code
	misses uint8
}

// Weak keys prevent a refusal or cached program from retaining a source graph.
// Native entry cannot call Go, so buffers are shared only until a guard exit.
type jitState struct {
	unavailable bool
	cache       map[weak.Pointer[bytecode.Function]]*jitEntry
	slots       [ir.MaxSlots]ir.Value
	roots       [ir.MaxSlots]Value
	rootCount   int
	entries     uint64
	guards      uint64
	budgets     uint64
}

func (r *Runtime) initJIT(enabled bool) { r.jitEnabled = enabled }

func (r *Runtime) jitCodeBytes() int64 {
	var n int64
	if r.jit != nil {
		for _, e := range r.jit.cache {
			n += int64(e.code.Size() + e.code.MetadataSize())
		}
	}
	return n
}

func (r *Runtime) releaseJIT() {
	if r.jit == nil {
		return
	}
	for key, e := range r.jit.cache {
		if e.code.Close() == nil {
			delete(r.jit.cache, key)
		}
	}
	clear(r.jit.roots[:])
	if len(r.jit.cache) == 0 {
		r.jit = nil
	}
}

// A compiler refusal must not stop a script merely because an optimization
// cannot fit. Account for conservative transient work before allocating IR.
func (r *Runtime) jitAllowance(fn *bytecode.Function) int {
	limit := jitCacheBytes - int(r.jitCodeBytes())
	if r.meter != nil {
		m := r.meter
		m.live = max(0, m.walk(r)-m.baseline)
		transient := int64(4*jit.MaxCodeBytes + len(fn.Code)*2048 + (32 << 10))
		available := m.limit - m.live - transient
		if available <= 0 {
			return 0
		}
		limit = int(min(int64(limit), available))
	}
	return limit
}

func (r *Runtime) jitFor(fn *bytecode.Function) *jitEntry {
	if r.jit != nil && r.jit.unavailable {
		return nil
	}
	if len(fn.Code) > jitcompile.MaxInstructions {
		return nil
	}
	if r.jit != nil {
		if e := r.jit.cache[weak.Make(fn)]; e != nil {
			return e
		}
		for key, e := range r.jit.cache {
			if key.Value() == nil && e.code.Close() == nil {
				delete(r.jit.cache, key)
			}
		}
		if len(r.jit.cache) >= jitCacheEntries {
			for key, e := range r.jit.cache {
				if e.code.Close() != nil {
					return nil
				}
				delete(r.jit.cache, key)
				break
			}
		}
	}
	limit := r.jitAllowance(fn)
	if limit <= 0 {
		return nil
	}
	if r.jit == nil {
		r.jit = &jitState{cache: make(map[weak.Pointer[bytecode.Function]]*jitEntry)}
	}
	s := r.jit
	var meta int
	for _, e := range s.cache {
		meta += e.code.MetadataSize()
	}
	// Entry maps and offsets are bounded before emission as well as afterwards.
	if meta+len(fn.Code)*32+1024 > jitMetadataBytes {
		return nil
	}
	p, err := jitcompile.Lower(fn)
	e := &jitEntry{}
	if err == nil {
		e.code, err = jit.CompileBudget(p, limit)
		if err != nil {
			if errors.Is(err, jit.ErrUnavailable) {
				s.unavailable = true
			}
			return nil
		}
	}
	s.cache[weak.Make(fn)] = e
	if r.meter != nil {
		// OS pages do not advance Go's allocation counter. Charge the new
		// owner now so a subsequent JS allocation sees the reduced allowance.
		r.meter.live = max(0, r.meter.walk(r)-r.meter.baseline)
	}
	return e
}

func (s *jitState) encode(v Value) ir.Value {
	switch v.Kind() {
	case KindNumber:
		return ir.Float(v.Number())
	case KindBool:
		return ir.Bool(v.Truthy())
	case KindUndefined:
		return ir.Value{Kind: ir.Undefined}
	case KindNull:
		return ir.Value{Kind: ir.Null}
	case KindUninitialized:
		return ir.Value{Kind: ir.Uninitialized}
	}
	i := s.rootCount
	s.roots[i], s.rootCount = v, i+1
	return ir.Value{Kind: ir.Opaque, Bits: uint64(i)}
}

func (s *jitState) decode(v ir.Value) Value {
	switch v.Kind {
	case ir.Number:
		return Float(math.Float64frombits(v.Bits))
	case ir.Boolean:
		return Bool(v.Bits != 0)
	case ir.Undefined:
		return Undefined
	case ir.Null:
		return Null
	case ir.Uninitialized:
		return uninitialized
	case ir.Opaque:
		return s.roots[v.Bits]
	}
	panic("invalid native scalar")
}

func (s *jitState) publish(f *frame, stack []Value, depth int) {
	for i := range f.locals {
		f.locals[i] = s.decode(s.slots[i])
	}
	for i := 0; i < depth; i++ {
		stack[f.base+i] = s.decode(s.slots[len(f.locals)+i])
	}
}

func (s *jitState) clearRoots() {
	clear(s.roots[:s.rootCount])
	s.rootCount = 0
}

func (r *Runtime) tryJITFrame(f *frame) (Value, error, bool) {
	if !r.jitEnabled || f.cl.jitRefused {
		return Undefined, nil, false
	}
	e := r.jitFor(f.cl.fn)
	if e == nil {
		return Undefined, nil, false
	}
	if e.code == nil {
		f.cl.jitRefused = true
		return Undefined, nil, false
	}
	if e.misses >= 8 {
		return Undefined, nil, false
	}
	s := r.jit
	n := len(f.locals) + f.cl.fn.MaxStack
	clear(s.slots[:n])
	for i, v := range f.locals {
		s.slots[i] = s.encode(v)
	}
	pc := 0
	for {
		s.entries++
		exit, err := e.code.Run(s.slots[:n], pc, jit.MaxIterations)
		runtime.KeepAlive(s)
		if err != nil {
			s.clearRoots()
			return Undefined, err, true
		}
		s.publish(f, r.stack, exit.State.Depth)
		f.pc = exit.State.PC
		switch exit.Kind {
		case ir.Returned:
			v := s.decode(exit.Value)
			s.clearRoots()
			return v, nil, true
		case ir.GuardExit:
			s.guards++
			e.misses++
			s.clearRoots()
			v, err := r.executeAt(f, f.base+exit.State.Depth, nil)
			return v, err, true
		case ir.BudgetExit:
			s.budgets++
			if err := r.checkInterruptNow(); err != nil {
				s.clearRoots()
				return Undefined, err, true
			}
			pc = int(exit.State.PC)
		}
	}
}
