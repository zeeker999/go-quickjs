//go:build quickjs_young

package vm

import "github.com/go-quickjs/go-quickjs/internal/bytecode"

// youngBefore is a conservative escape barrier. Only operations whose
// references stay in the flat interpreter stack or ordinary young properties
// cross it without tenure. Calls, captures, exotic storage and coercions that
// may run user code release nursery ownership before entering that code.
func (r *Runtime) youngBefore(f *frame, in bytecode.Instr, stack []Value, sp int) {
	h := r.young
	if h == nil || h.stats.Young == 0 {
		return
	}
	switch in.Op {
	case bytecode.OpNop, bytecode.OpPushConst, bytecode.OpPushUndef,
		bytecode.OpPushNull, bytecode.OpPushTrue, bytecode.OpPushFalse,
		bytecode.OpPushInt, bytecode.OpPushEmptyString, bytecode.OpPushUninitialized,
		bytecode.OpDup, bytecode.OpDup2, bytecode.OpDrop, bytecode.OpSwap,
		bytecode.OpRot3, bytecode.OpRot4, bytecode.OpInsert2, bytecode.OpInsert3,
		bytecode.OpInsert4, bytecode.OpNipUnder,
		bytecode.OpGetLocal, bytecode.OpGetLocal2, bytecode.OpSetLocal,
		bytecode.OpPutLocal, bytecode.OpInitLocal, bytecode.OpSetLocalGet,
		bytecode.OpClearLocal, bytecode.OpGetUpvalue,
		bytecode.OpJump, bytecode.OpJumpIfTrue, bytecode.OpJumpIfFalse,
		bytecode.OpJumpIfTrueKeep, bytecode.OpJumpIfFalseKeep,
		bytecode.OpJumpIfNullish, bytecode.OpJumpIfNotNullish,
		bytecode.OpStrictEq, bytecode.OpStrictNe, bytecode.OpNot,
		bytecode.OpTypeOf, bytecode.OpIsNullish, bytecode.OpNewObject:
		return
	case bytecode.OpGetLocalCheck:
		if !f.locals[in.A].IsUninitialized() {
			return
		}
	case bytecode.OpSetLocalCheck:
		if !f.locals[in.A].IsUninitialized() && f.cl.fn.Locals[in.A].Mutable {
			return
		}
	case bytecode.OpAdd, bytecode.OpSub, bytecode.OpMul, bytecode.OpDiv,
		bytecode.OpMod, bytecode.OpPow, bytecode.OpBitAnd, bytecode.OpBitOr,
		bytecode.OpBitXor, bytecode.OpShl, bytecode.OpShr, bytecode.OpUShr,
		bytecode.OpEq, bytecode.OpNe, bytecode.OpLt, bytecode.OpLe,
		bytecode.OpGt, bytecode.OpGe, bytecode.OpJumpIfCmpFalse:
		if stack[sp-1].IsNumber() && stack[sp-2].IsNumber() {
			return
		}
	case bytecode.OpNeg, bytecode.OpPos, bytecode.OpInc, bytecode.OpDec,
		bytecode.OpBitNot, bytecode.OpBinImm:
		if stack[sp-1].IsNumber() {
			return
		}
	case bytecode.OpIncLocal, bytecode.OpDecLocal, bytecode.OpUpdateLocal:
		if f.locals[in.A].IsNumber() {
			return
		}
	case bytecode.OpLocalBinImm:
		if f.locals[in.A&(1<<24-1)].IsNumber() {
			return
		}
	case bytecode.OpBinLocal:
		if stack[sp-1].IsNumber() && f.locals[in.A].IsNumber() {
			return
		}
	case bytecode.OpGetProp, bytecode.OpGetPropThis:
		v := stack[sp-1]
		if v.IsObject() {
			o := v.Object()
			if o.class == ClassObject {
				if p := o.getOwnVisible(f.cl.names[in.A]); p != nil && p.flags&(propAccessor|propUninit) == 0 {
					return
				}
			}
		}
	case bytecode.OpDefineField, bytecode.OpSetProp:
		v := stack[sp-2]
		if v.IsObject() {
			o := v.Object()
			_, young := h.owners[objectWord(o)]
			if o.class == ClassObject && (young || stack[sp-1].ref == nil) {
				p := o.getOwnVisible(f.cl.names[in.A])
				if in.Op == bytecode.OpDefineField && o.flags&objExtensible != 0 && (p == nil || p.flags&propConfigurable != 0) {
					return
				}
				if in.Op == bytecode.OpSetProp && p != nil && p.flags&(propAccessor|propUninit|propWritable) == propWritable {
					return
				}
			}
		}
	}
	h.promoteAll()
}

func (r *Runtime) youngLiteral(f *frame, sp, n int, root *shape) *Object {
	if r.youngCapacity <= 0 || n != 2 || f.cl.fn.HasDirectEval || len(f.withScopes) != 0 {
		return nil
	}
	// Captured locals have references outside the flat frame, including
	// mapped arguments. Their stores need a separate escape barrier.
	for _, l := range f.cl.fn.Locals {
		if l.Captured {
			return nil
		}
	}
	if r.young == nil {
		r.young = newYoungHeap(r.youngCapacity)
	}
	h := r.young
	if len(h.free) == 0 {
		// Calls promote before suspending a frame, so every young root is
		// either in this frame's locals/operands or a young object's fields.
		// Include earlier stack windows conservatively for nested execution.
		for _, v := range r.stack[:sp] {
			h.markWord(valueWord(v))
		}
		for d := 0; d < r.frameDepth; d++ {
			fr := r.frameAt(d)
			for _, v := range fr.locals {
				h.markWord(valueWord(v))
			}
			for _, v := range fr.args {
				h.markWord(valueWord(v))
			}
			h.markWord(valueWord(fr.this))
		}
		h.collect()
	}
	// A full live nursery falls back to Go allocation rather than forcing
	// another collection with no work done between the two.
	return h.allocate(r.proto.object, root)
}
