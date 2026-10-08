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
	jitEntry     *jitEntry
}

type jitEntry struct {
	code          *jit.Code
	misses        uint8
	probes        uint8
	entrySlow     bool
	probeSteps    uint64
	probeHosts    uint64
	properties    bool
	this          bool
	globals       []bytecode.Instr
	shortCounter  uint16
	referenceKeys []uint32
	calls         bool
	calleeOnly    bool
}

// Weak keys prevent a refusal or cached program from retaining a source graph.
// Native entry cannot call Go, so buffers are shared only until a guard exit.
type jitState struct {
	unavailable     bool
	properties      bool
	this            bool
	referenceActive bool
	callActive      bool
	globals         []bytecode.Instr
	cache           map[weak.Pointer[bytecode.Function]]*jitEntry
	slots           [ir.MaxSlots]ir.Value
	roots           [ir.MaxSlots]Value
	arrays          [ir.MaxSlots]ir.ArrayView
	hosts           uint64
	fastHosts       uint64
	rootCount       int
	entries         uint64
	guards          uint64
	budgets         uint64
	osrs            uint64
	referenceKeys   []uint32
	references      *jitReferences
	callFrames      *jitCallFrames
	transfers       uint64
}

// Only selected own reference fields are prepared. The receiver bound keeps
// cycles and unusually large object graphs on the ordinary host path.
const jitReferenceReceivers = 32

type jitReferences struct {
	cells [jitReferenceReceivers][ir.MaxProperties]ir.ReferenceCell
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
			n += int64(e.code.Size() + e.code.MetadataSize() + cap(e.globals)*int(unsafe.Sizeof(bytecode.Instr{})) + cap(e.referenceKeys)*4)
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
	return r.jitForMode(fn, false)
}

func (r *Runtime) jitForMode(fn *bytecode.Function, callee bool) *jitEntry {
	if r.jit != nil && r.jit.unavailable {
		return nil
	}
	if len(fn.Code) > jitcompile.MaxInstructions {
		return nil
	}
	if r.jit != nil {
		if e := r.jit.cache[weak.Make(fn)]; e != nil && (!callee || e.calleeOnly || e.code != nil) {
			return e
		}
		for key, e := range r.jit.cache {
			if key.Value() == nil && e.code.Close() == nil {
				delete(r.jit.cache, key)
			}
		}
		if len(r.jit.cache) >= jitCacheEntries {
			for key, e := range r.jit.cache {
				if r.jit.callEntryActive(e) {
					continue
				}
				if e.code.Close() != nil {
					return nil
				}
				delete(r.jit.cache, key)
				break
			}
			if len(r.jit.cache) >= jitCacheEntries {
				return nil
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
		meta += e.code.MetadataSize() + cap(e.globals)*int(unsafe.Sizeof(bytecode.Instr{})) + cap(e.referenceKeys)*4
	}
	// Entry maps and offsets are bounded before emission as well as afterwards.
	if meta+len(fn.Code)*32+1024 > jitMetadataBytes {
		return nil
	}
	e := &jitEntry{}
	for _, in := range fn.Code {
		e.calls = e.calls || in.Op == bytecode.OpCall || in.Op == bytecode.OpCallMethod
	}
	lower := jitcompile.Lower
	if e.calls {
		lower = jitcompile.LowerCalls
	}
	if callee {
		lower = jitcompile.LowerCallee
		e.calleeOnly = true
		e.entrySlow = true
	}
	p, err := lower(fn)
	if err == nil {
		e.this = p.This
		e.shortCounter = p.ShortCounter
		for _, key := range p.Globals {
			for _, in := range fn.Code {
				if in.Op == bytecode.OpGetGlobal && in.A == key {
					e.globals = append(e.globals, in)
					break
				}
			}
		}
		// Lowering uses immutable source-name indices. Each runtime resolves
		// these to its own pointer-free atoms before emitting native searches.
		hasReferences := false
		for _, in := range p.Code {
			hasReferences = hasReferences || in.Op == ir.ReferenceRead
		}
		for i := range p.Code {
			in := &p.Code[i]
			if in.Op == ir.PropertyRead || in.Op == ir.PropertyWrite || in.Op == ir.BindingRead || in.Op == ir.ReferenceRead {
				e.properties = e.properties || in.Op != ir.BindingRead
				in.Key = uint32(r.atoms.intern(fn.Names[in.Key]))
				if hasReferences && in.Op != ir.BindingRead {
					found := false
					for _, key := range e.referenceKeys {
						found = found || key == in.Key
					}
					if !found && len(e.referenceKeys) < ir.MaxProperties {
						e.referenceKeys = append(e.referenceKeys, in.Key)
					}
				}
			}
		}
		bindingBytes := cap(e.globals)*int(unsafe.Sizeof(bytecode.Instr{})) + cap(e.referenceKeys)*4
		if len(e.referenceKeys) != 0 && s.references == nil {
			limit -= int(unsafe.Sizeof(jitReferences{}))
			if limit <= bindingBytes {
				return nil
			}
		}
		if meta+len(fn.Code)*32+1024+bindingBytes > jitMetadataBytes {
			return nil
		}
		e.code, err = jit.CompileBudget(p, limit-bindingBytes)
		if err != nil {
			if errors.Is(err, jit.ErrUnavailable) {
				s.unavailable = true
			}
			return nil
		}
		if len(e.referenceKeys) != 0 && s.references == nil {
			s.references = new(jitReferences)
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
	handleSlot := -1
	if s.callActive {
		handleSlot = int((uintptr(v.ref) >> 4) & 511)
		for {
			handle := s.callFrames.handles[handleSlot]
			if handle == 0 {
				break
			}
			i := int(handle) - 1
			root := s.roots[i]
			if root.ref == v.ref && math.Float64bits(root.num) == math.Float64bits(v.num) {
				return ir.Value{Kind: ir.Opaque, Bits: uint64(i)}
			}
			handleSlot = (handleSlot + 1) & 511
		}
	}
	i := s.rootCount
	s.roots[i], s.rootCount = v, i+1
	if handleSlot >= 0 {
		s.callFrames.handles[handleSlot] = uint16(i + 1)
	}
	if v.IsObject() {
		o := v.Object()
		if o.class == ClassArray {
			s.arrays[i] = jitArrayView(o)
		} else if s.properties && o.class == ClassObject && o.shapeIndex() == nil && len(o.props) <= ir.MaxProperties {
			s.arrays[i] = ir.ArrayView{Data: unsafe.Pointer(unsafe.SliceData(o.props)), DenseLength: uint64(len(o.props)), WritableHole: tagBase}
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
	base := len(f.locals) + len(f.cl.upvalues)
	base += len(s.globals)
	if s.this {
		base++
	}
	for i := 0; i < depth; i++ {
		stack[f.base+i] = s.decode(s.slots[base+i])
	}
}

func (s *jitState) clearRoots() {
	clear(s.roots[:s.rootCount])
	clear(s.arrays[:s.rootCount])
	if s.referenceActive {
		s.clearReferences()
	}
	s.rootCount = 0
}

// Keep optional reference cleanup outside the frequent numeric return path.
//
//go:noinline
func (s *jitState) clearReferences() {
	clear(s.references.cells[:min(s.rootCount, jitReferenceReceivers)])
}

func (r *Runtime) tryJITFrame(f *frame) (Value, error, bool) {
	if !r.jitEnabled || f.cl.jitRefused || f.cl.jitEntry != nil && f.cl.jitEntry.entrySlow {
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
	e := f.cl.jitEntry
	if e == nil || e.code != nil && e.code.Size() == 0 {
		e = r.jitFor(f.cl.fn)
		f.cl.jitEntry = e
	}
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
	if e.calleeOnly {
		return Undefined, nil, false
	}
	if e.shortCounter != 0 {
		counter := f.locals[e.shortCounter-1]
		if counter.IsNumber() && counter.Number() >= 0 && counter.Number() <= 1 {
			return Undefined, nil, false
		}
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
	n += len(e.globals)
	if e.this {
		n++
	}
	s.properties = e.properties
	s.this = e.this
	s.globals = e.globals
	s.referenceActive = len(e.referenceKeys) != 0
	if s.referenceActive {
		s.referenceKeys = e.referenceKeys
	}
	s.encodeFrame(r, f, r.stack, depth)
	if osr {
		s.osrs++
	}
	if e.calls && !s.referenceActive && (s.callFrames == nil || !s.callFrames.active) {
		if v, err, done := r.jitRunCalls(f, e, pc, depth, n); done {
			return v, err, true
		}
	}
	budget := uint64(jit.MaxIterations)
	var nativeSteps, hosts uint64
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
		nativeSteps += exit.Steps
		if exit.Kind == ir.HostExit {
			hosts++
			s.hosts++
			next, _, steps := r.jitHostFast(f, s, int(exit.State.PC), exit.State.Depth, int(min(budget, 16)))
			if steps != 0 {
				s.fastHosts++
				budget -= uint64(steps)
				pc = next
				f.pc = uint32(pc)
				continue
			}
		}
		f.pc = exit.State.PC
		// Decode a scalar or rooted handle before clearing its owner. Eligible
		// locals cannot be observed after this frame leaves: captures, mapped
		// arguments, eval, handlers, and debugger runtimes are excluded.
		if exit.Kind == ir.Returned {
			// Sample several calls so a short tail call cannot hide profitable
			// longer calls to the same kernel. Boundary-heavy entries stay in Go;
			// long invocations can still promote through the back-edge path.
			if e.probes < jitHotCalls {
				e.probes++
				e.probeSteps += nativeSteps
				e.probeHosts += hosts
				if e.probes == jitHotCalls && e.probeHosts >= 16 && e.probeSteps/e.probeHosts < 64 {
					e.entrySlow = true
				}
			}
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
			sp, steps, hostErr := r.jitHost(f, f.base+exit.State.Depth, int(min(budget, 16)))
			if hostErr != nil {
				return r.jitInterpret(f, sp, hostErr)
			}
			pc = int(f.pc)
			budget -= uint64(steps)
			// Most callbacks preserve this cache and executable owner. EntryDepth
			// rejects an owner closed by eviction; a released cache changes s.
			_, live := e.code.EntryDepth(pc)
			if r.jit != s || !live {
				e = r.jitFor(f.cl.fn)
				f.cl.jitEntry = e
				if e == nil || e.code == nil {
					return r.jitInterpret(f, sp, nil)
				}
			}
			s = r.jit
			s.properties = e.properties
			s.this = e.this
			s.globals = e.globals
			s.referenceActive = len(e.referenceKeys) != 0
			if s.referenceActive {
				s.referenceKeys = e.referenceKeys
			}
			s.encodeFrame(r, f, r.stack, sp-f.base)
		case ir.BudgetExit:
			s.budgets++
			if err := r.checkInterruptNow(); err != nil {
				s.clearRoots()
				return Undefined, err, true
			}
			if hosts >= 256 && nativeSteps/hosts < 64 {
				e.entrySlow = true
				s.clearRoots()
				return r.jitInterpret(f, f.base+exit.State.Depth, nil)
			}
			pc = int(exit.State.PC)
			budget = jit.MaxIterations
		}
	}
}

// jitHostFast handles comparisons and ordinary own data properties without
// JavaScript callbacks. Array growth refreshes every alias's borrowed view;
// scalar slots stay valid without publishing/reencoding the whole frame.
// Any accessor, proxy, exotic object, new property, or full root table takes
// the normal callback boundary at its original instruction and stack depth.
func (r *Runtime) jitHostFast(f *frame, s *jitState, pc, depth, limit int) (int, int, int) {
	// A reference write can invalidate frozen permissions. Reference programs
	// use the full publication and refresh boundary for every host operation.
	if s.referenceActive {
		return pc, depth, 0
	}
	n := len(f.locals) + len(f.cl.upvalues)
	n += len(s.globals)
	if s.this {
		n++
	}
	steps := 0
	for range limit {
		in := f.cl.fn.Code[pc]
		sp := n + depth
		if jitBinaryBoundary(in) {
			cmp, _ := jitBinaryAt(in, sp)
			left := s.decode(s.slots[cmp.left])
			right := cmp.literal
			if cmp.right >= 0 {
				right = s.decode(s.slots[cmp.right])
			}
			if cmp.op == bytecode.OpMod {
				if !left.IsNumber() || !right.IsNumber() {
					return pc, depth, steps
				}
				s.slots[cmp.dest] = ir.Float(jsMod(left.Number(), right.Number()))
				depth += cmp.delta
				pc++
				steps++
				continue
			}
			if !cmp.strict && !left.IsNullish() && !right.IsNullish() && left.Kind() != right.Kind() {
				return pc, depth, steps
			}
			result := left.StrictEquals(right)
			if !cmp.strict {
				// Same-kind and nullish comparisons never invoke coercion hooks,
				// including Annex B's special HTMLDDA/nullish comparison.
				result, _ = r.looseEquals(left, right)
			}
			result = result != cmp.negate
			depth += cmp.delta
			pc++
			if cmp.branch {
				if !result {
					pc = int(in.A)
				}
			} else {
				s.slots[cmp.dest] = ir.Bool(result)
			}
			steps++
			continue
		}
		var value Value
		dest, delta := sp, 1
		switch in.Op {
		case bytecode.OpSetLocalGet:
			s.slots[in.A] = s.slots[sp-1]
			s.slots[sp-1] = s.slots[in.B]
			pc++
			steps++
			continue
		case bytecode.OpPushThis:
			var bound bool
			value, bound = f.thisValue()
			if !bound {
				return pc, depth, steps
			}
		case bytecode.OpGetGlobal:
			name, env := f.cl.names[in.A], f.cl.scope()
			if f.evalVars != nil {
				return pc, depth, steps
			}
			if p := r.globalLexProp(env, name); p != nil {
				if p.value.IsUninitialized() {
					return pc, depth, steps
				}
				value = p.value
			} else {
				if env.class != ClassObject {
					return pc, depth, steps
				}
				site := &f.cl.ic[in.B]
				i := globalSlot(env, site, name)
				if i < 0 || env.props[i].flags&(propAccessor|propPrivate|propDeleted|propUninit) != 0 {
					return pc, depth, steps
				}
				if site.p1 != env {
					site.p1 = env
				}
				value = env.props[i].value
			}
		case bytecode.OpGetProp, bytecode.OpGetPropThis:
			obj := s.decode(s.slots[sp-1])
			if !obj.IsObject() {
				return pc, depth, steps
			}
			var ok bool
			value, ok = plainOwn(obj.Object(), f.cl.names[in.A])
			if !ok {
				return pc, depth, steps
			}
			if in.Op == bytecode.OpGetProp {
				dest, delta = sp-1, 0
			}
		case bytecode.OpSetProp:
			obj := s.decode(s.slots[sp-2])
			if !obj.IsObject() {
				return pc, depth, steps
			}
			o := obj.Object()
			_, index, ok := plainOwnAt(o, f.cl.names[in.A])
			if !ok || o.props[index].flags&propWritable == 0 {
				return pc, depth, steps
			}
			o.props[index].value = s.decode(s.slots[sp-1])
			depth -= 2
			pc++
			steps++
			continue
		case bytecode.OpSetIndex:
			obj, key, value := s.decode(s.slots[sp-3]), s.decode(s.slots[sp-2]), s.decode(s.slots[sp-1])
			if !obj.IsObject() || !key.IsNumber() {
				return pc, depth, steps
			}
			if !setElem(obj, key, value) {
				o := obj.Object()
				if !jitGrowElem(o, key.Number(), value) {
					return pc, depth, steps
				}
				// Encoding may hold multiple handles for the same receiver. A
				// reallocation invalidates all of their previous storage pointers.
				for i, root := range s.roots[:s.rootCount] {
					if root.IsObject() && root.Object() == o {
						s.arrays[i] = jitArrayView(o)
					}
				}
			}
			depth -= 3
			pc++
			steps++
			continue
		default:
			return pc, depth, steps
		}
		if s.rootCount == ir.MaxSlots {
			return pc, depth, steps
		}
		s.slots[dest] = s.encode(value)
		depth += delta
		pc++
		steps++
	}
	return pc, depth, steps
}

// A borrowed hole can be written natively only when an indexed assignment
// cannot consult a setter, a non-writable property, or an exotic prototype.
func jitArrayView(o *Object) ir.ArrayView {
	view := ir.ArrayView{Data: unsafe.Pointer(unsafe.SliceData(o.elems)), DenseLength: uint64(len(o.elems)), Length: uint64(o.arrayLength()), NumberLimit: tagBase}
	if jitDenseWritable(o) {
		view.WritableHole = holeBits
	}
	return view
}

func jitDenseWritable(o *Object) bool {
	const want = objExtensible | objArrayLengthWritable
	if o.class != ClassArray || o.flags&(want|objHasSparseElements|objMappedArguments) != want || !o.noIndexKeys() {
		return false
	}
	for p := o.proto; p != nil; p = p.proto {
		if p.class != ClassObject && p.class != ClassArray || len(p.elems) != 0 || !p.noIndexKeys() {
			return false
		}
	}
	return true
}

// The ordinary helper bounds the allocation caused by a single gap write.
func jitGrowElem(o *Object, n float64, value Value) bool {
	if n < 0 || n >= math.MaxUint32 || !jitDenseWritable(o) {
		return false
	}
	i := uint32(n)
	if float64(i) != n {
		return false
	}
	return o.setElem(i, value)
}

type jitBinary struct {
	left, right, dest, delta int
	literal                  Value
	strict, negate, branch   bool
	op                       bytecode.Op
}

func jitBinaryBoundary(in bytecode.Instr) bool {
	op := in.Op
	switch in.Op {
	case bytecode.OpBinLocal, bytecode.OpBinImm, bytecode.OpJumpIfCmpFalse:
		op = bytecode.Op(in.B)
	case bytecode.OpLocalBinImm:
		op = bytecode.Op(in.A >> 24)
	}
	return op == bytecode.OpMod || op >= bytecode.OpEq && op <= bytecode.OpStrictNe
}

// Operands use the native layout: locals, upvalues, then the live stack.
// The original bytecode retains strict/loose comparison semantics. Remainder
// also uses this layout, including fused local and immediate operands.
func jitBinaryAt(in bytecode.Instr, sp int) (jitBinary, bool) {
	c := jitBinary{left: sp - 2, right: sp - 1, dest: sp - 2, delta: -1}
	op := in.Op
	switch in.Op {
	case bytecode.OpEq, bytecode.OpNe, bytecode.OpStrictEq, bytecode.OpStrictNe, bytecode.OpMod:
	case bytecode.OpJumpIfCmpFalse:
		op = bytecode.Op(in.B)
		c.branch, c.delta = true, -2
	case bytecode.OpBinLocal:
		op = bytecode.Op(in.B)
		c.left, c.right, c.dest, c.delta = sp-1, int(in.A), sp-1, 0
	case bytecode.OpBinImm:
		op = bytecode.Op(in.B)
		c.left, c.right, c.dest, c.delta = sp-1, -1, sp-1, 0
		c.literal = Float(float64(int32(in.A)))
	case bytecode.OpLocalBinImm:
		op = bytecode.Op(in.A >> 24)
		c.left, c.right, c.dest, c.delta = int(in.A&0xffffff), -1, sp, 1
		c.literal = Float(float64(int32(in.B)))
	default:
		return c, false
	}
	c.op = op
	c.strict = op == bytecode.OpStrictEq || op == bytecode.OpStrictNe
	c.negate = op == bytecode.OpNe || op == bytecode.OpStrictNe
	return c, c.strict || op == bytecode.OpEq || op == bytecode.OpNe || op == bytecode.OpMod
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

func (s *jitState) encodeFrame(r *Runtime, f *frame, stack []Value, depth int) {
	base := len(f.locals) + len(f.cl.upvalues)
	if s.this {
		value, bound := f.thisValue()
		if !bound {
			value = uninitialized
		}
		s.slots[base] = s.encode(value)
		base++
	}
	for _, in := range s.globals {
		name, env := f.cl.names[in.A], f.cl.scope()
		owner := env
		var p *Property
		if f.evalVars == nil && env.class == ClassObject {
			if p = r.globalLexProp(env, name); p != nil {
				owner = r.globalLex
			} else {
				site := &f.cl.ic[in.B]
				if i := globalSlot(env, site, name); i >= 0 {
					p = &env.props[i]
					if site.p1 != env {
						site.p1 = env
					}
				}
			}
		}
		// A failed resolution is inert until the original read executes. No
		// getter, proxy trap, coercion or TDZ error runs during preparation.
		s.slots[base] = ir.Value{Kind: ir.Undefined}
		if p != nil && p.flags&^propDefault == 0 && p.value.IsNumber() {
			i := s.rootCount
			s.rootCount++
			s.roots[i] = Obj(owner)
			s.arrays[i] = ir.ArrayView{Data: unsafe.Pointer(p), DenseLength: 1, Length: 1, WritableHole: tagBase}
			s.slots[base] = ir.Value{Kind: ir.Opaque, Bits: uint64(i)}
		}
		base++
	}
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
	if s.referenceActive {
		s.prepareReferences()
	}
}

func (s *jitState) prepareReferences() {
	for i := 0; i < min(s.rootCount, jitReferenceReceivers); i++ {
		// Binding handles grant access to one numeric cell, not to their owner.
		view := s.arrays[i]
		if view.NumberLimit == 0 && view.WritableHole != 0 && view.Length == 1 {
			continue
		}
		root := s.roots[i]
		if !root.IsObject() || root.Object().class != ClassObject {
			continue
		}
		o := root.Object()
		var count int
		for _, key := range s.referenceKeys {
			value, index, ok := plainOwnAt(o, Atom(key))
			if !ok || o.props[index].flags&^propDefault != 0 {
				continue
			}
			handle := ir.MaxSlots
			if value.IsObject() {
				handle = 0
				for handle < s.rootCount {
					v := s.roots[handle]
					if v.IsObject() && v.Object() == value.Object() {
						break
					}
					handle++
				}
				if handle == s.rootCount {
					if handle == ir.MaxSlots {
						continue
					}
					s.encode(value)
				}
			}
			cell := &o.props[index]
			s.references.cells[i][count] = ir.ReferenceCell{Key: key, Bits: math.Float64bits(value.num), Reference: value.ref, Cell: (*ir.PropertyCell)(unsafe.Pointer(cell)), Handle: uint64(handle)}
			count++
		}
		if count != 0 {
			s.arrays[i] = ir.ArrayView{Data: unsafe.Pointer(&s.references.cells[i][0]), DenseLength: uint64(count), Length: tagBase}
		}
	}
}

// jitHost runs an explicitly lowered host operation with normal VM ordering.
// No borrowed view or native scalar root survives a callback. Consecutive host
// operations and intervening fused local copies share one frame publication;
// sixteen instructions bound the Go batch before native selection resumes.
func (r *Runtime) jitHost(f *frame, sp, limit int) (int, int, error) {
	steps := 0
	for range limit {
		pc := int(f.pc)
		in := f.cl.fn.Code[pc]
		n := len(f.locals) + len(f.cl.upvalues)
		if jitBinaryBoundary(in) {
			cmp, _ := jitBinaryAt(in, n+sp-f.base)
			left := r.jitFrameValue(f, cmp.left)
			right := cmp.literal
			if cmp.right >= 0 {
				right = r.jitFrameValue(f, cmp.right)
			}
			f.pc++
			sp += cmp.delta
			steps++
			if cmp.op == bytecode.OpMod {
				value, err := r.arith(cmp.op, left, right)
				if err != nil {
					return sp, steps, err
				}
				r.stack[f.base+cmp.dest-n] = value
				continue
			}
			result := left.StrictEquals(right)
			if !cmp.strict {
				var err error
				result, err = r.looseEquals(left, right)
				if err != nil {
					return sp, steps, err
				}
			}
			result = result != cmp.negate
			if cmp.branch {
				if !result {
					f.pc = in.A
				}
			} else {
				r.stack[f.base+cmp.dest-n] = Bool(result)
			}
			continue
		}
		if in.Op == bytecode.OpSetLocalGet {
			f.locals[in.A] = r.stack[sp-1]
			r.stack[sp-1] = f.locals[in.B]
			f.pc++
			steps++
			continue
		}
		switch in.Op {
		case bytecode.OpPushThis, bytecode.OpGetProp, bytecode.OpSetProp, bytecode.OpCall,
			bytecode.OpCallMethod, bytecode.OpGetGlobal, bytecode.OpGetPropThis, bytecode.OpSetIndex:
		default:
			return sp, steps, nil
		}
		f.pc++
		steps++
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
				return sp, steps, err
			}
			continue
		case bytecode.OpSetIndex:
			sp -= 3
			obj, key, value := stack[sp], stack[sp+1], stack[sp+2]
			if !(key.IsNumber() && setElem(obj, key, value)) {
				if err := r.setIndexed(obj, key, value, f.cl.fn.Strict); err != nil {
					return sp, steps, err
				}
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
			return sp, steps, err
		}
		r.stack[sp] = v
		sp++
	}
	return sp, steps, nil
}

func (r *Runtime) jitFrameValue(f *frame, index int) Value {
	if index < len(f.locals) {
		return f.locals[index]
	}
	index -= len(f.locals)
	if index < len(f.cl.upvalues) {
		return f.cl.upvalues[index].get()
	}
	return r.stack[f.base+index-len(f.cl.upvalues)]
}

func (r *Runtime) jitInterpret(f *frame, sp int, pending error) (Value, error, bool) {
	previous := r.jitDeoptDepth
	r.jitDeoptDepth = uint32(r.frameDepth)
	defer func() { r.jitDeoptDepth = previous }()
	v, err := r.executeAt(f, sp, pending)
	return v, err, true
}
