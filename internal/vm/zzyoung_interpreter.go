//go:build quickjs_young

package vm

import "github.com/go-quickjs/go-quickjs/internal/bytecode"

// youngBefore is a conservative escape barrier. Only operations whose
// references stay in the flat interpreter stack or ordinary young properties
// cross it without tenure. Calls, captures, exotic storage and coercions that
// may run user code release nursery ownership before entering that code.
func (r *Runtime) youngBefore(f *frame, in bytecode.Instr, stack []Value, sp int) {
	if h := r.young; h != nil && h.stats.Young != 0 && !youngFlatOp[in.Op] {
		r.youngEscape(f, in, stack, sp)
	}
}

// These operations cannot run user code or expose references outside the flat
// frame. Keep their barrier inline rather than calling a helper per dispatch.
var youngFlatOp = [256]bool{
	bytecode.OpNop: true, bytecode.OpPushConst: true, bytecode.OpPushUndef: true,
	bytecode.OpPushNull: true, bytecode.OpPushTrue: true, bytecode.OpPushFalse: true,
	bytecode.OpPushInt: true, bytecode.OpPushEmptyString: true, bytecode.OpPushUninitialized: true,
	bytecode.OpDup: true, bytecode.OpDup2: true, bytecode.OpDrop: true, bytecode.OpSwap: true,
	bytecode.OpRot3: true, bytecode.OpRot4: true, bytecode.OpInsert2: true, bytecode.OpInsert3: true,
	bytecode.OpInsert4: true, bytecode.OpNipUnder: true,
	bytecode.OpGetLocal: true, bytecode.OpGetLocal2: true, bytecode.OpSetLocal: true,
	bytecode.OpPutLocal: true, bytecode.OpInitLocal: true, bytecode.OpSetLocalGet: true,
	bytecode.OpClearLocal: true, bytecode.OpGetUpvalue: true,
	bytecode.OpJump: true, bytecode.OpJumpIfTrue: true, bytecode.OpJumpIfFalse: true,
	bytecode.OpJumpIfTrueKeep: true, bytecode.OpJumpIfFalseKeep: true,
	bytecode.OpJumpIfNullish: true, bytecode.OpJumpIfNotNullish: true,
	bytecode.OpStrictEq: true, bytecode.OpStrictNe: true, bytecode.OpNot: true,
	bytecode.OpTypeOf: true, bytecode.OpIsNullish: true, bytecode.OpNewObject: true,
}

func (r *Runtime) youngEscape(f *frame, in bytecode.Instr, stack []Value, sp int) {
	h := r.young
	switch in.Op {
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
