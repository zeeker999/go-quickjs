//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"errors"
	"math"
	"runtime"
	"unsafe"
	"weak"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/jit"
	jitcompile "github.com/go-quickjs/go-quickjs/internal/jit/compile"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

const jitCacheEntries = 128
const jitCacheBytes = 8 << 20
const jitMetadataBytes = 512 << 10
const jitHotCalls = 8

type jitFields struct {
	jitEnabled       bool
	jitCallThreshold uint8
	jitDeoptDepth    uint32
	jit              *jitState
}

// A closure belongs to one runtime. Hotness and permanent selection refusals
// live here without registering a weak key on cold or refused calls.
type jitClosureFields struct {
	jitRefused   bool
	jitCalls     uint8
	jitLoopDelay uint8
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
	arrays      [ir.MaxSlots]ir.ArrayView
	hosts       uint64
	fastHosts   uint64
	rootCount   int
	entries     uint64
	guards      uint64
	budgets     uint64
	osrs        uint64
}

func (r *Runtime) initJIT(enabled bool) {
	// Debugger interrupt callbacks may evaluate code and mutate frames or arrays.
	// Native budget exits retain borrowed views, so debugger runtimes stay in Go.
	r.jitEnabled = enabled && r.debug == nil
	r.jitCallThreshold = jitHotCalls
}

// Loop promotion returns through a panic caught by the callee's runTree.
// A nested tree without its own recovery would return from its caller instead.
func (r *Runtime) jitTreeRecovery(f *frame) bool {
	return r.jitEnabled && !f.cl.jitRefused
}

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
	r.jit.clearRoots()
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
	if v.IsObject() {
		o := v.Object()
		if o.class == ClassArray {
			s.arrays[i] = ir.ArrayView{Data: unsafe.Pointer(unsafe.SliceData(o.elems)), DenseLength: uint64(len(o.elems)), Length: uint64(o.arrayLength()), NumberLimit: tagBase}
		}
	}
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
		stack[f.base+i] = s.decode(s.slots[len(f.locals)+len(f.cl.upvalues)+i])
	}
}

func (s *jitState) clearRoots() {
	clear(s.roots[:s.rootCount])
	clear(s.arrays[:s.rootCount])
	s.rootCount = 0
}

func (r *Runtime) tryJITFrame(f *frame) (Value, error, bool) {
	if !r.jitEnabled || f.cl.jitRefused {
		return Undefined, nil, false
	}
	// Cold calls stay in the existing tiers without allocating native state.
	// Saturating the counter keeps hot selection independent of call overflow.
	if f.cl.jitCalls < r.jitCallThreshold {
		f.cl.jitCalls++
		if f.cl.jitCalls < r.jitCallThreshold {
			return Undefined, nil, false
		}
	}
	return r.tryJITAt(f, 0, 0, false)
}

// The existing back-edge interrupt budget supplies coarse work feedback.
// Promotion starts at the completed branch's target, never at function entry.
func (r *Runtime) tryJITLoop(f *frame, pc uint32, sp int, fullBudget bool) (Value, error, bool) {
	if !r.jitEnabled || !fullBudget || f.cl.jitRefused || int(r.jitDeoptDepth) == r.frameDepth || f.cl.fn.TopLevel || f.cl.fn.IsModule {
		return Undefined, nil, false
	}
	if f.cl.jitLoopDelay != 0 {
		f.cl.jitLoopDelay--
		return Undefined, nil, false
	}
	f.cl.jitCalls = r.jitCallThreshold
	return r.tryJITAt(f, int(pc), sp-f.base, true)
}

func (r *Runtime) tryJITAt(f *frame, pc, depth int, osr bool) (Value, error, bool) {
	e := r.jitFor(f.cl.fn)
	if e == nil {
		// Resource and OS refusals may change. Retry after another warmup,
		// rather than charging every call for an unsuccessful compilation.
		f.cl.jitCalls = 0
		f.cl.jitLoopDelay = jitHotCalls
		return Undefined, nil, false
	}
	if e.code == nil {
		f.cl.jitRefused = true
		return Undefined, nil, false
	}
	if e.misses >= 8 {
		f.cl.jitRefused = true
		return Undefined, nil, false
	}
	wantDepth, ok := e.code.EntryDepth(pc)
	if !ok || depth != wantDepth {
		return Undefined, nil, false
	}
	f.cl.jitLoopDelay = 0
	s := r.jit
	n := len(f.locals) + len(f.cl.upvalues) + f.cl.fn.MaxStack
	s.encodeFrame(f, r.stack, depth)
	if osr {
		s.osrs++
	}
	budget := uint64(jit.MaxIterations)
	for {
		s.entries++
		// Encoding and the immutable native program preserve scalar validity.
		exit, err := e.code.RunEncodedArrays(s.slots[:n], s.arrays[:], pc, budget)
		runtime.KeepAlive(s)
		if err != nil {
			s.clearRoots()
			return Undefined, err, true
		}
		budget -= exit.Steps
		if exit.Kind == ir.HostExit {
			s.hosts++
			next, _ := r.jitHostFast(f, s, int(exit.State.PC), exit.State.Depth, int(min(budget, 16)))
			if next != int(exit.State.PC) {
				s.fastHosts++
				budget -= uint64(next - int(exit.State.PC))
				pc = next
				f.pc = uint32(pc)
				continue
			}
		}
		f.pc = exit.State.PC
		// Native returns carry a primitive value. Eligible locals cannot be
		// observed after this frame leaves: captures, mapped arguments, eval,
		// handlers, and debugger runtimes are excluded before native entry.
		if exit.Kind == ir.Returned {
			v := s.decode(exit.Value)
			s.clearRoots()
			return v, nil, true
		}
		s.publish(f, r.stack, exit.State.Depth)
		switch exit.Kind {
		case ir.GuardExit:
			s.guards++
			e.misses++
			s.clearRoots()
			// A guard has already selected interpretation for this invocation.
			// Suppress loop reentry even if coercion evicts the cached code.
			previous := r.jitDeoptDepth
			// Frame depth is bounded by maxGoRecursion and fits uint32.
			r.jitDeoptDepth = uint32(r.frameDepth)
			defer func() { r.jitDeoptDepth = previous }()
			v, err := r.executeAt(f, f.base+exit.State.Depth, nil)
			return v, err, true
		case ir.HostExit:
			s.clearRoots()
			sp, hostErr := r.jitHost(f, f.base+exit.State.Depth, int(min(budget, 16)))
			if hostErr != nil {
				return r.jitInterpret(f, sp, hostErr)
			}
			pc = int(f.pc)
			budget -= uint64(pc - int(exit.State.PC))
			// A reentrant callback can release this cache or evict the parent.
			// Reacquire it only after the callback has returned to Go.
			e = r.jitFor(f.cl.fn)
			if e == nil || e.code == nil {
				return r.jitInterpret(f, sp, nil)
			}
			s = r.jit
			s.encodeFrame(f, r.stack, sp-f.base)
		case ir.BudgetExit:
			s.budgets++
			if err := r.checkInterruptNow(); err != nil {
				s.clearRoots()
				return Undefined, err, true
			}
			pc = int(exit.State.PC)
			budget = jit.MaxIterations
		}
	}
}

// jitHostFast only reads or overwrites ordinary own data properties. These
// operations cannot call JavaScript or resize array storage, so scalar slots
// and borrowed views stay valid without publishing/reencoding the whole frame.
// Any accessor, proxy, exotic object, new property, or full root table takes
// the normal callback boundary at its original instruction and stack depth.
func (r *Runtime) jitHostFast(f *frame, s *jitState, pc, depth, limit int) (int, int) {
	n := len(f.locals) + len(f.cl.upvalues)
	for range limit {
		in := f.cl.fn.Code[pc]
		sp := n + depth
		var value Value
		dest, delta := sp, 1
		switch in.Op {
		case bytecode.OpSetLocalGet:
			s.slots[in.A] = s.slots[sp-1]
			s.slots[sp-1] = s.slots[in.B]
			pc++
			continue
		case bytecode.OpPushThis:
			var bound bool
			value, bound = f.thisValue()
			if !bound {
				return pc, depth
			}
		case bytecode.OpGetProp, bytecode.OpGetPropThis:
			obj := s.decode(s.slots[sp-1])
			if !obj.IsObject() {
				return pc, depth
			}
			var ok bool
			value, ok = plainOwn(obj.Object(), f.cl.names[in.A])
			if !ok {
				return pc, depth
			}
			if in.Op == bytecode.OpGetProp {
				dest, delta = sp-1, 0
			}
		case bytecode.OpSetProp:
			obj := s.decode(s.slots[sp-2])
			if !obj.IsObject() {
				return pc, depth
			}
			o := obj.Object()
			_, index, ok := plainOwnAt(o, f.cl.names[in.A])
			if !ok || o.props[index].flags&propWritable == 0 {
				return pc, depth
			}
			o.props[index].value = s.decode(s.slots[sp-1])
			depth -= 2
			pc++
			continue
		default:
			return pc, depth
		}
		if s.rootCount == ir.MaxSlots {
			return pc, depth
		}
		s.slots[dest] = s.encode(value)
		depth += delta
		pc++
	}
	return pc, depth
}

type jitTreeExit struct {
	value Value
	err   error
}

func (r *Runtime) tryJITTreeLoop(c *tctx, pc, depth int, fullBudget bool) {
	if v, err, done := r.tryJITLoop(c.f, c.cl.fn.Code[pc].A, c.f.base+depth, fullBudget); done {
		// Tree branch nodes return block indices. Unwind to runTree only
		// after native execution or its interpreter fallback has completed.
		panic(jitTreeExit{v, err})
	}
}

func jitTreeResult(p any) (Value, error, bool) {
	if exit, ok := p.(jitTreeExit); ok {
		return exit.value, exit.err, true
	}
	return Undefined, nil, false
}

func (s *jitState) encodeFrame(f *frame, stack []Value, depth int) {
	base := len(f.locals) + len(f.cl.upvalues)
	// Live slots are overwritten below; only inactive operands need clearing.
	clear(s.slots[base+depth : base+f.cl.fn.MaxStack])
	for i, v := range f.locals {
		s.slots[i] = s.encode(v)
	}
	for i, v := range f.cl.upvalues {
		s.slots[len(f.locals)+i] = s.encode(v.get())
	}
	for i, v := range stack[f.base : f.base+depth] {
		s.slots[base+i] = s.encode(v)
	}
}

// jitHost runs an explicitly lowered host operation with normal VM ordering.
// No borrowed view or native scalar root survives a callback. Consecutive host
// operations and intervening fused local copies share one frame publication;
// sixteen instructions bound the Go batch before native selection resumes.
func (r *Runtime) jitHost(f *frame, sp, limit int) (int, error) {
	for range limit {
		pc := int(f.pc)
		in := f.cl.fn.Code[pc]
		if in.Op == bytecode.OpSetLocalGet {
			f.locals[in.A] = r.stack[sp-1]
			r.stack[sp-1] = f.locals[in.B]
			f.pc++
			continue
		}
		switch in.Op {
		case bytecode.OpPushThis, bytecode.OpGetProp, bytecode.OpSetProp, bytecode.OpCall,
			bytecode.OpCallMethod, bytecode.OpGetGlobal, bytecode.OpGetPropThis:
		default:
			return sp, nil
		}
		f.pc++
		stack := r.stack
		var v Value
		var err error
		switch in.Op {
		case bytecode.OpPushThis:
			var bound bool
			v, bound = f.thisValue()
			if !bound {
				err = r.throwError(errReference, "\"this\" is not bound until super() has been called")
			}
		case bytecode.OpGetProp:
			sp--
			v, err = r.getValueProp(stack[sp], f.cl.names[in.A])
		case bytecode.OpSetProp:
			sp -= 2
			if err := r.setValueProp(stack[sp], f.cl.names[in.A], stack[sp+1], f.cl.fn.Strict); err != nil {
				return sp, err
			}
			continue
		case bytecode.OpCall, bytecode.OpCallMethod:
			argc := int(in.A)
			args := stack[sp-argc : sp]
			callee := stack[sp-argc-1]
			this := Undefined
			sp -= argc + 1
			if in.Op == bytecode.OpCallMethod {
				sp--
				this = stack[sp]
			}
			v, err = r.callDirect(callee, this, args)
		case bytecode.OpGetGlobal:
			c := tctx{r: r, f: f, cl: f.cl, locals: f.locals}
			v, err = r.getGlobalAt(&c, in, pc)
		case bytecode.OpGetPropThis:
			v, err = r.getValueProp(stack[sp-1], f.cl.names[in.A])
		default:
			panic("invalid JIT host operation")
		}
		if err != nil {
			return sp, err
		}
		r.stack[sp] = v
		sp++
	}
	return sp, nil
}

func (r *Runtime) jitInterpret(f *frame, sp int, pending error) (Value, error, bool) {
	previous := r.jitDeoptDepth
	r.jitDeoptDepth = uint32(r.frameDepth)
	defer func() { r.jitDeoptDepth = previous }()
	v, err := r.executeAt(f, sp, pending)
	return v, err, true
}
