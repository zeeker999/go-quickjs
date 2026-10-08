//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/jit"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

// The coordinator keeps return PCs in Go-owned records, never in machine
// code. A callback may therefore close any mapping without leaving a native
// return address into it. The scalar arena and shared root table are bounded.
const jitCallDepth = 8

type jitCallFrame struct {
	f           *frame
	e           *jitEntry
	pc          int
	depth       int
	n           int
	steps       uint64
	hosts       uint64
	argSlot     int
	argsPending bool
	slots       [ir.MaxSlots]ir.Value
}

type jitCallFrames struct {
	active bool
	frames [jitCallDepth]jitCallFrame
	// Root index+1 occupies nine bits; bit 15 records String's kind.
	handles [512]uint16
}

func (s *jitState) callEntryActive(e *jitEntry) bool {
	a := s.callFrames
	if a == nil || !a.active {
		return false
	}
	for i := range a.frames {
		if a.frames[i].f != nil && a.frames[i].e == e {
			return true
		}
	}
	return false
}

func (s *jitState) callMode(e *jitEntry) {
	s.properties, s.this, s.globals = e.properties, e.this, e.globals
	s.strings = e.strings
	s.grows = e.grows
	s.referenceActive = false
}

// Ordinary inherited data is resolved only at the original read. Walking the
// live chain observes shadowing, deletions and prototype replacement; accessors
// and exotic receivers retain their original callback boundary.
func (r *Runtime) jitCallProperty(f *frame, s *jitState, pc, depth int) bool {
	in := f.cl.fn.Code[pc]
	if in.Op != bytecode.OpGetProp && in.Op != bytecode.OpGetPropThis {
		return false
	}
	base := len(f.locals) + len(f.cl.upvalues) + len(s.globals)
	if s.this {
		base++
	}
	sp := base + depth
	receiver := s.decode(s.slots[sp-1])
	if !receiver.IsObject() {
		return false
	}
	key := f.cl.names[in.A]
	for o, count := receiver.Object(), 0; o != nil && count < 8; o, count = o.proto, count+1 {
		if o.class != ClassObject {
			return false
		}
		p := o.getOwn(key)
		if p == nil {
			continue
		}
		if p.flags&(propAccessor|propPrivate|propDeleted|propUninit) != 0 || s.rootCount == ir.MaxSlots {
			return false
		}
		dest := sp - 1
		if in.Op == bytecode.OpGetPropThis {
			dest++
		}
		s.slots[dest] = s.encode(p.value)
		return true
	}
	return false
}

// jitRunCalls transfers encoded arguments and results without publishing the
// caller or encoding the callee's locals. Machine entries still return to this
// coordinator; all entries and host work consume the same instruction budget.
func (r *Runtime) jitRunCalls(f *frame, e *jitEntry, pc, depth, n int) (Value, error, bool) {
	s := r.jit
	if s.callFrames == nil {
		if r.meter != nil && r.jitAllowance(f.cl.fn) < int(unsafe.Sizeof(jitCallFrames{})) {
			return Undefined, nil, false
		}
		s.callFrames = new(jitCallFrames)
	}
	a := s.callFrames
	clear(a.handles[:])
	a.active, s.callActive = true, true
	initialDepth := r.frameDepth
	defer func() {
		for r.frameDepth > initialDepth {
			child := r.topFrame()
			r.popFrameOf(child, child.base-len(child.locals))
		}
		for i := range a.frames {
			a.frames[i].f, a.frames[i].e = nil, nil
		}
		clear(a.handles[:])
		a.active, s.callActive = false, false
		s.clearRoots()
	}()
	a.frames[0].f, a.frames[0].e = f, e
	a.frames[0].pc, a.frames[0].depth, a.frames[0].n = pc, depth, n
	a.frames[0].steps, a.frames[0].hosts = 0, 0
	top := 0
	budget := uint64(jit.MaxIterations)
	for {
		q := &a.frames[top]
		s.entries++
		if q.e.grows {
			s.prepareArrayGrowth()
		}
		exit, err := q.e.code.RunEncodedArrays(s.slots[:q.n], s.arrays[:], q.pc, budget)
		if q.e.grows {
			s.commitArrayGrowth()
		}
		if err != nil {
			return Undefined, err, true
		}
		budget -= exit.Steps
		q.steps += exit.Steps
		q.pc, q.depth = int(exit.State.PC), exit.State.Depth
		q.f.pc = exit.State.PC
		switch exit.Kind {
		case ir.Returned:
			if top == 0 {
				if q.e.probes < jitHotCalls {
					q.e.probes++
					q.e.probeSteps += q.steps
					q.e.probeHosts += q.hosts
					if q.e.probes == jitHotCalls && q.e.probeHosts >= 16 && q.e.probeSteps/q.e.probeHosts < 64 {
						q.e.entrySlow = true
					}
				}
				return s.decode(exit.Value), nil, true
			}
			steps, hosts := q.steps, q.hosts
			r.popFrameOf(q.f, q.f.base-len(q.f.locals))
			q.f, q.e = nil, nil
			top--
			q = &a.frames[top]
			copy(s.slots[:q.n], q.slots[:q.n])
			s.slots[q.n-q.f.cl.fn.MaxStack+q.depth] = exit.Value
			q.depth++
			q.steps += steps
			q.hosts += hosts
			s.callMode(q.e)
		case ir.GuardExit:
			s.guards++
			q.e.misses++
			r.jitPublishCalls(s, a, top)
			return r.jitInterpretCalls(a, top, nil)
		case ir.BudgetExit:
			s.budgets++
			if r.meter != nil {
				r.jitPublishCalls(s, a, top)
			}
			if err := r.checkInterruptNow(); err != nil {
				return Undefined, err, true
			}
			if r.meter != nil {
				var rebuilt bool
				s, rebuilt = r.jitRebuildCalls(s, a, top)
				if !rebuilt {
					return r.jitInterpretCalls(a, top, nil)
				}
			}
			if q.hosts >= 256 && q.steps/q.hosts < 64 {
				q.e.entrySlow = true
				r.jitPublishCalls(s, a, top)
				return r.jitInterpretCalls(a, top, nil)
			}
			budget = jit.MaxIterations
		case ir.HostExit:
			s.hosts++
			q.hosts++
			in := q.f.cl.fn.Code[q.pc]
			if budget != 0 && top+1 < jitCallDepth && (in.Op == bytecode.OpCall || in.Op == bytecode.OpCallMethod) {
				if r.jitPushCall(s, q, &a.frames[top+1], in, top) {
					copy(q.slots[:q.n], s.slots[:q.n])
					q.pc++
					q.depth -= int(in.A) + 1
					if in.Op == bytecode.OpCallMethod {
						q.depth--
					}
					q.f.pc = uint32(q.pc)
					top++
					r.jitEncodeCall(s, q, &a.frames[top], in)
					budget--
					q.steps++
					s.transfers++
					child := &a.frames[top]
					if first := child.f.cl.fn.Code[0].Op; first == bytecode.OpPushThis || first == bytecode.OpGetGlobal {
						next, depth, steps := r.jitHostFast(child.f, s, 0, 0, int(min(budget, 16)))
						child.pc, child.depth = next, depth
						child.steps += uint64(steps)
						if steps != 0 {
							child.hosts++
						}
						budget -= uint64(steps)
					}
					continue
				}
			}
			if budget != 0 && r.jitCallProperty(q.f, s, q.pc, q.depth) {
				q.pc++
				if in.Op == bytecode.OpGetPropThis {
					q.depth++
				}
				budget--
				q.steps++
				s.fastHosts++
				continue
			}
			next, nextDepth, steps := r.jitHostFast(q.f, s, q.pc, q.depth, int(min(budget, 16)))
			if steps != 0 {
				// Native string packing amortizes its bounded array-growth helpers;
				// count published callback boundaries for its entry policy.
				if q.e.strings {
					q.hosts--
				}
				q.pc, q.depth = next, nextDepth
				budget -= uint64(steps)
				q.steps += uint64(steps)
				s.fastHosts++
				continue
			}
			// Every caller is committed before any callback, including a callee
			// that reenters the VM, mutates a closure or releases the JIT cache.
			r.jitPublishCalls(s, a, top)
			sp, steps, hostErr := r.jitHost(q.f, q.f.base+q.depth, int(min(budget, 16)))
			q.pc, q.depth = int(q.f.pc), sp-q.f.base
			budget -= uint64(steps)
			q.steps += uint64(steps)
			if hostErr != nil {
				return r.jitInterpretCalls(a, top, hostErr)
			}
			var rebuilt bool
			s, rebuilt = r.jitRebuildCalls(s, a, top)
			if !rebuilt {
				return r.jitInterpretCalls(a, top, nil)
			}
		}
	}
}

// A transfer starts only with an already compiled, ordinary non-recursive
// callee. Recursion, lexical this, cross-realm calls and optional refusals use
// the existing call path. The Go frame reserves canonical storage for exits.
func (r *Runtime) jitPushCall(s *jitState, q, child *jitCallFrame, in bytecode.Instr, top int) bool {
	sp := q.n - q.f.cl.fn.MaxStack + q.depth
	value := s.decode(s.slots[sp-int(in.A)-1])
	if !value.IsObject() {
		return false
	}
	o := value.Object()
	fd := o.fn()
	if fd == nil || fd.closure == nil || fd.native != nil || fd.bound || fd.arrow || fd.extra != nil || fd.ctorKind == ctorDerived || fd.closure.realm != r.Realm {
		return false
	}
	cl := fd.closure
	fn, e := cl.fn, cl.jitEntry
	if !fn.DirectCall || fn.HasTailCall {
		return false
	}
	if e == nil || e.code == nil || e.code.Size() == 0 {
		// Memory walks require canonical frames. Compile cold limited-runtime
		// targets through the ordinary published call boundary instead.
		if r.meter != nil {
			return false
		}
		e = r.jitForMode(fn, true)
		cl.jitEntry = e
	}
	if e == nil || e.code == nil || cl.jitRefused && !e.calleeOnly || e.misses >= 8 || len(e.referenceKeys) != 0 {
		return false
	}
	if _, live := e.code.EntryDepth(0); !live {
		return false
	}
	for d := 0; d <= top; d++ {
		if s.callFrames.frames[d].f.cl == cl {
			return false
		}
	}
	n := fn.LocalCount + len(cl.upvalues) + len(e.globals) + fn.MaxStack
	if e.this {
		n++
	}
	if s.rootCount+n >= ir.MaxSlots || r.frameDepth >= r.maxFrames || r.stackTop+fn.LocalCount+fn.MaxStack > len(r.stack) {
		return false
	}
	this := Undefined
	if in.Op == bytecode.OpCallMethod {
		this = s.decode(s.slots[sp-int(in.A)-2])
	}
	if fn.CoerceThis && !this.IsObject() {
		if !this.IsNullish() {
			return false
		}
		this = r.globalThis
	}
	// tick can stop or enforce a memory limit. Leave that exceptional boundary
	// to the published ordinary call path when its check is due.
	if r.interruptCounter <= 1 || r.stopped != nil {
		return false
	}
	r.interruptCounter--
	base := r.stackTop
	r.stackTop += fn.LocalCount + fn.MaxStack
	r.stackHigh = max(r.stackHigh, r.stackTop)
	f := r.pushFrame()
	f.cl, f.locals, f.base, f.pc = cl, r.stack[base:base+fn.LocalCount:base+fn.LocalCount], base+fn.LocalCount, 0
	f.this, f.newTarget, f.callee, f.args = this, Undefined, o, nil
	child.argsPending = r.nodeQuirks
	if child.argsPending {
		argc := int(in.A)
		start := q.f.base + q.depth - argc
		f.args = r.stack[start : start+argc]
		child.argSlot = sp - argc
	}
	f.openUpvalues = f.openUpvalues[:0]
	f.handlers = f.handlers[:0]
	f.thisRef, f.withScopes, f.evalVars = nil, nil, nil
	f.native, f.savedSP = "", 0
	child.f, child.e, child.n, child.pc, child.depth = f, e, n, 0, 0
	child.steps, child.hosts = 0, 0
	return true
}

func (r *Runtime) jitEncodeCall(s *jitState, parent, child *jitCallFrame, in bytecode.Instr) {
	f := child.f
	s.callMode(child.e)
	clear(s.slots[:child.n])
	argc := int(in.A)
	argBase := parent.n - parent.f.cl.fn.MaxStack + parent.depth + 1
	if in.Op == bytecode.OpCallMethod {
		argBase++
	}
	for i := 0; i < min(argc, f.cl.fn.ParamCount); i++ {
		s.slots[i] = parent.slots[argBase+i]
	}
	// Canonical locals need not be initialized until publication. Upvalues,
	// receiver and binding permissions are encoded without an argument scan.
	base := len(f.locals)
	for i, up := range f.cl.upvalues {
		s.slots[base+i] = s.encode(up.get())
	}
	base += len(f.cl.upvalues)
	if s.this {
		s.slots[base] = s.encode(f.this)
		base++
	}
	for _, in := range s.globals {
		name, env := f.cl.names[in.A], f.cl.scope()
		owner := env
		var p *Property
		if env.class == ClassObject {
			if p = r.globalLexProp(env, name); p != nil {
				owner = r.globalLex
			} else {
				site := &f.cl.ic[in.B]
				if i := globalSlot(env, site, name); i >= 0 {
					p = &env.props[i]
					site.p1 = env
				}
			}
		}
		if p != nil && p.flags&^propDefault == 0 && p.value.IsNumber() {
			i := 0
			for i < s.rootCount {
				view := s.arrays[i]
				if view.Data == unsafe.Pointer(p) && view.Length == 1 && view.NumberLimit == 0 && view.WritableHole == tagBase {
					break
				}
				i++
			}
			if i == s.rootCount {
				s.rootCount++
				s.roots[i] = Obj(owner)
				s.arrays[i] = ir.ArrayView{Data: unsafe.Pointer(p), DenseLength: 1, Length: 1, WritableHole: tagBase}
			}
			s.slots[base] = ir.Value{Kind: ir.Opaque, Bits: uint64(i)}
		}
		base++
	}
	if s.strings {
		s.prepareStringMethods(r)
	}
}

func (r *Runtime) jitPublishCalls(s *jitState, a *jitCallFrames, top int) {
	copy(a.frames[top].slots[:a.frames[top].n], s.slots[:a.frames[top].n])
	for i := 1; i <= top; i++ {
		child, parent := &a.frames[i], &a.frames[i-1]
		if child.argsPending {
			for j := range child.f.args {
				child.f.args[j] = s.decode(parent.slots[child.argSlot+j])
			}
			child.argsPending = false
		}
	}
	for i := 0; i <= top; i++ {
		q := &a.frames[i]
		copy(s.slots[:q.n], q.slots[:q.n])
		s.callMode(q.e)
		q.f.pc = uint32(q.pc)
		s.publish(q.f, r.stack, q.depth)
	}
	s.clearRoots()
	clear(a.handles[:])
	s.callActive = false
}

func (r *Runtime) jitRebuildCalls(s *jitState, a *jitCallFrames, top int) (*jitState, bool) {
	for i := 0; i <= top; i++ {
		q := &a.frames[i]
		if _, live := q.e.code.EntryDepth(q.pc); !live || r.jit != s {
			q.e = r.jitForMode(q.f.cl.fn, q.e.calleeOnly)
			q.f.cl.jitEntry = q.e
			if q.e == nil || q.e.code == nil {
				return s, false
			}
		}
	}
	s = r.jit
	s.callFrames = a
	s.callActive = true
	for i := 0; i <= top; i++ {
		q := &a.frames[i]
		_, live := q.e.code.EntryDepth(q.pc)
		if !live || s.rootCount+q.n >= ir.MaxSlots {
			s.clearRoots()
			s.callActive = false
			return s, false
		}
		s.callMode(q.e)
		s.encodeFrame(r, q.f, r.stack, q.depth)
		copy(q.slots[:q.n], s.slots[:q.n])
	}
	return s, true
}

// Frames have already been published. Resume the deepest one, then commit
// each result at its caller's saved return PC. Earlier stores are never replayed.
func (r *Runtime) jitInterpretCalls(a *jitCallFrames, top int, pending error) (Value, error, bool) {
	var v Value
	for {
		q := &a.frames[top]
		v, pending, _ = r.jitInterpret(q.f, q.f.base+q.depth, pending)
		if top == 0 {
			return v, pending, true
		}
		r.popFrameOf(q.f, q.f.base-len(q.f.locals))
		q.f, q.e = nil, nil
		top--
		q = &a.frames[top]
		if pending == nil {
			r.stack[q.f.base+q.depth] = v
			q.depth++
		}
	}
}
