//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"slices"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	jitcompile "github.com/go-quickjs/go-quickjs/internal/jit/compile"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
	"github.com/go-quickjs/go-quickjs/internal/jit/ssa"
)

// Elements' kinds learned at register code's exits, as the stack form's
// code learns them (jitIndexSeen, jitKeyedWriteSeen, jitStringAddSeen):
// an element read that met a typed array, a BigInt typed array, a string
// or an object read by a name, an element write to an object native code
// does not store to, an addition that met a string -- kept by register pc
// in what the function's register code learned (jitState's regLearned),
// its lists the stack form's entry's (typedReads, stringReads, keyedReads,
// keyedWrites), and the lowering's hints (jitLearned's regHints), with
// which every lowering of its register code is made (jitRegLower).

// jitRegLower is cl's register code rc lowered as kind says -- for an entry
// (jitLowerSSA), hot (jitLowerHot), or for native callers and inlining
// (jitLowerInline) -- with the hints its code learned at its exits.
func (r *Runtime) jitRegLower(cl *closure, rc *bytecode.RegCode, kind jitLowerKind) (*ir.Program, error) {
	var h jitcompile.Hints
	if s := r.jit; s != nil {
		if l := s.regLearned[r.jitRegKeyOf(cl)]; l != nil {
			h = l.regHints
		}
	}
	switch kind {
	case jitLowerInline:
		return jitcompile.LowerRegInlineWith(rc, cl.fn, h)
	case jitLowerHot:
		return jitcompile.LowerRegHotWith(rc, cl.fn, h)
	case jitLowerForward:
		return jitcompile.LowerRegForwardWith(rc, cl.fn, h)
	}
	return jitcompile.LowerRegWith(rc, cl.fn, h)
}

// jitRegElemRead is the receiver and the key of the element read of the
// register instruction at rpc of c, on the frame whose window is at wb,
// as the instruction's start has them -- get_index's, get_index_imm's,
// get_index_pre's (its key stepped), get_index_post's, an add_index's --
// or false for any other instruction.
func (r *Runtime) jitRegElemRead(c *regCode, wb, rpc int) (o, k Value, ok bool) {
	if rpc < 0 || rpc >= len(c.rc.Code) {
		return Undefined, Undefined, false
	}
	w := c.rc.Code[rpc]
	at := func(n uint16) Value {
		if wb+int(n) < len(r.stack) {
			return r.stack[wb+int(n)]
		}
		return Undefined
	}
	switch w.Op() {
	case bytecode.RGetIndex, bytecode.RGetIndexPost:
		return at(w.B()), at(w.C()), true
	case bytecode.RGetIndexI:
		return at(w.B()), Float(float64(w.C())), true
	case bytecode.RGetIndexPre:
		k := at(w.C())
		if !k.IsNumber() {
			return Undefined, Undefined, false
		}
		return at(w.B()), Float(k.num + float64(int8(w.X()))), true
	case bytecode.RAddIndex, bytecode.RAddIndexI:
		return at(w.A()), at(w.B()), true
	case bytecode.RAddIndexK:
		return at(w.A()), Float(float64(w.C())), true
	}
	return Undefined, Undefined, false
}

// jitRegElemWrite is the receiver, the key and the value of the element
// write of the register instruction at rpc of c -- set_index's,
// set_index_imm's, an add_index's, whose value is its operand -- or false.
func (r *Runtime) jitRegElemWrite(c *regCode, wb, rpc int) (o, k, v Value, ok bool) {
	if rpc < 0 || rpc >= len(c.rc.Code) {
		return Undefined, Undefined, Undefined, false
	}
	w := c.rc.Code[rpc]
	at := func(n uint16) Value {
		if wb+int(n) < len(r.stack) {
			return r.stack[wb+int(n)]
		}
		return Undefined
	}
	switch w.Op() {
	case bytecode.RSetIndex:
		return at(w.A()), at(w.B()), at(w.C()), true
	case bytecode.RSetIndexI:
		return at(w.A()), Float(float64(w.B())), at(w.C()), true
	case bytecode.RAddIndex:
		return at(w.A()), at(w.B()), at(w.C()), true
	case bytecode.RAddIndexI:
		return at(w.A()), at(w.B()), Float(float64(w.SC())), true
	case bytecode.RAddIndexK:
		return at(w.A()), Float(float64(w.C())), at(w.B()), true
	}
	return Undefined, Undefined, Undefined, false
}

// jitRegElemSeen learns, in l, what the register code's element access at
// register pc rpc of c met, on the frame whose window is at wb, before Go
// runs it: at a host exit (reference), what a read made by its cell met
// -- a string's, a typed array's or a BigInt typed array's element, or an
// object's property by its name, which the next compile reads as such, or
// Go from native code (ssa's StringRead, TypedRead, TypedBigRead,
// KeyedRead) -- and a write to an object native code does not store to,
// which Go stores from native code (KeyedWrite); at a deoptimization, a
// read made as a number that met what is none, read as it is from the
// next lowering on (Hints' ElementReads, StringReads), as the stack form's
// code learns it (jitIndexSeen). It reports whether it learned anything.
func (r *Runtime) jitRegElemSeen(l *jitEntry, c *regCode, wb, rpc int, reference bool) bool {
	learned := false
	if o, k, ok := r.jitRegElemRead(c, wb, rpc); ok {
		learned = r.jitRegIndexSeen(l, rpc, o, k, reference)
	}
	if !reference {
		return learned
	}
	if o, k, v, ok := r.jitRegElemWrite(c, wb, rpc); ok && !slices.Contains(l.keyedWrites, int32(rpc)) &&
		o.IsObject() && o.Object().class != ClassArray && o.Object().class != ClassTypedArray && r.jitKeyedWrite(o, k, v, false) {
		l.keyedWrites = append(l.keyedWrites, int32(rpc))
		learned = true
	}
	return learned
}

// jitRegIndexSeen is jitIndexSeen for register code: the element read at
// register pc rpc, of receiver o at key k, learned in l.
func (r *Runtime) jitRegIndexSeen(l *jitEntry, rpc int, o, k Value, reference bool) bool {
	pc := int32(rpc)
	if slices.Contains(l.typedReads, pc) || jitBigReadAt(l.typedReads, rpc) != 0 || slices.Contains(l.stringReads, pc) || slices.Contains(l.keyedReads, pc) {
		return false
	}
	if !reference {
		keyed := o.IsObject() && (k.IsString() || k.IsNumber() && jitKeyedClass(o.Object()))
		chars := o.IsString() && k.IsNumber()
		big := int32(0)
		if !keyed && !chars && k.IsNumber() {
			big = jitBigTyped(o)
		}
		h := &l.regHints
		switch {
		case big != 0:
		case keyed:
			if _, ok := r.jitKeyedRead(o, k); !ok {
				return false
			}
		case o.IsObject() && k.IsNumber():
			// An array's element that is a string: read as it is, its
			// additions Go's (stringConcats).
			if v, ok := r.jitKeyedRead(o, k); !ok || !v.IsString() || slices.Contains(h.StringReads, pc) {
				return false
			}
		case !chars:
			return false
		}
		hint := &h.ElementReads
		if !keyed && big == 0 {
			hint = &h.StringReads
		}
		if !slices.Contains(*hint, pc) {
			*hint = append(*hint, pc)
		}
		switch {
		case big != 0:
			l.typedReads = append(l.typedReads, pc|big)
		case keyed:
			l.keyedReads = append(l.keyedReads, pc)
		case chars:
			l.stringReads = append(l.stringReads, pc)
		}
		return true
	}
	switch {
	case o.IsString() && k.IsNumber():
		l.stringReads = append(l.stringReads, pc)
	case k.IsNumber() && jitBigTyped(o) != 0:
		l.typedReads = append(l.typedReads, pc|jitBigTyped(o))
	case o.IsObject() && o.Object().class == ClassTypedArray:
		l.typedReads = append(l.typedReads, pc)
	case o.IsObject() && (k.IsString() || k.IsNumber() && jitKeyedClass(o.Object())):
		if _, ok := r.jitKeyedRead(o, k); !ok {
			return false
		}
		l.keyedReads = append(l.keyedReads, pc)
	default:
		return false
	}
	return true
}

// jitRegStringAddSeen learns, at a deoptimization of register code at
// register pc rpc of c, whose frame's window at wb holds the state at the
// instruction's start, an addition -- add's, add_imm's -- that met a
// string, which every lowering from here makes Go's (Hints' StringAdds),
// its operands read as they are, as the stack form's code learns it
// (jitStringAddSeen). It reports whether it learned one.
func (r *Runtime) jitRegStringAddSeen(l *jitEntry, c *regCode, wb, rpc int) bool {
	if rpc < 0 || rpc >= len(c.rc.Code) || slices.Contains(l.regHints.StringAdds, int32(rpc)) {
		return false
	}
	w := c.rc.Code[rpc]
	str := func(n uint16) bool { return wb+int(n) < len(r.stack) && r.stack[wb+int(n)].IsString() }
	switch w.Op() {
	case bytecode.RAdd:
		if !str(w.B()) && !str(w.C()) {
			return false
		}
	case bytecode.RAddI:
		if !str(w.B()) {
			return false
		}
	default:
		return false
	}
	l.regHints.StringAdds = append(l.regHints.StringAdds, int32(rpc))
	return true
}

// regElemPC is the register pc of the program's instruction pc, if it is
// of op.
func (f *jitRegFeedback) regElemPC(pc int, op ir.Op) (int32, bool) {
	if f.fb.e == nil || pc < 0 || pc >= len(f.p.Code) || f.p.Code[pc].Op != op {
		return 0, false
	}
	return int32(f.p.Origin[pc]), true
}

// StringRead is ssa.StringElementFeedback's (jitRegIndexSeen).
func (f *jitRegFeedback) StringRead(pc int) bool {
	rpc, ok := f.regElemPC(pc, ir.ArrayRead)
	return ok && slices.Contains(f.fb.e.stringReads, rpc)
}

// TypedRead is ssa.ElementFeedback's (jitRegIndexSeen).
func (f *jitRegFeedback) TypedRead(pc int) bool {
	rpc, ok := f.regElemPC(pc, ir.ArrayRead)
	return ok && slices.Contains(f.fb.e.typedReads, rpc)
}

// TypedBigRead is ssa.BigElementFeedback's (jitRegIndexSeen).
func (f *jitRegFeedback) TypedBigRead(pc int) (unsigned, ok bool) {
	rpc, ok := f.regElemPC(pc, ir.ArrayRead)
	if !ok {
		return false, false
	}
	m := jitBigReadAt(f.fb.e.typedReads, int(rpc))
	return m&jitBigReadUnsigned != 0, m != 0
}

// KeyedRead is ssa.KeyedFeedback's (jitRegIndexSeen).
func (f *jitRegFeedback) KeyedRead(pc int) bool {
	rpc, ok := f.regElemPC(pc, ir.ArrayRead)
	return ok && slices.Contains(f.fb.e.keyedReads, rpc)
}

// KeyedWrite is ssa.KeyedWriteFeedback's (jitRegElemSeen).
func (f *jitRegFeedback) KeyedWrite(pc int) bool {
	rpc, ok := f.regElemPC(pc, ir.ArrayWrite)
	return ok && slices.Contains(f.fb.e.keyedWrites, rpc)
}

// UpvalueTyped is ssa.UpvalueTypedFeedback's: whether the captured binding
// the read at the program's instruction pc reads (ir.UpvalueRead's Key)
// holds a typed array in the closure the code is compiled for, as the
// stack form's (jitFeedback's UpvalueTyped).
func (f *jitRegFeedback) UpvalueTyped(pc int) (uint8, bool) {
	cl := f.fb.cl
	if cl == nil || pc < 0 || pc >= len(f.p.Code) || f.p.Code[pc].Op != ir.UpvalueRead {
		return 0, false
	}
	k := int(f.p.Code[pc].Key)
	if k >= len(cl.upvalues) || cl.upvalues[k] == nil {
		return 0, false
	}
	v := cl.upvalues[k].get()
	if !jitTypedArray(v) {
		return 0, false
	}
	return jitTypedKind(true, v), true
}

var (
	_ ssa.ElementFeedback       = (*jitRegFeedback)(nil)
	_ ssa.BigElementFeedback    = (*jitRegFeedback)(nil)
	_ ssa.KeyedFeedback         = (*jitRegFeedback)(nil)
	_ ssa.KeyedWriteFeedback    = (*jitRegFeedback)(nil)
	_ ssa.StringElementFeedback = (*jitRegFeedback)(nil)
	_ ssa.UpvalueTypedFeedback  = (*jitRegFeedback)(nil)
)
