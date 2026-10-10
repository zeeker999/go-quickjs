package jit

import (
	"math"
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
)

// ApplyExit writes the frame of an exit that left it to Go (abi.ExitTable)
// from its description and the registers it saved, as the exit's code
// would have written it, records applied, and puts the exit's own kind,
// PC, depth and site in ctx -- and its inlined callees' frames in the
// contexts past it. For any other exit it does nothing. Run and Resume
// apply it to the context they were given; one who runs native calls
// applies it to the deepest callee's, where that is the one that left.
//
// Every value is read before any slot is written, as Go applies an exit's
// records: a slot may be another's source. It returns how many of the
// slots it wrote held a reference, before or after: what an exit's code
// would have left to Go as records.
func ApplyExit(ctx *abi.Context) (refs int) {
	if ctx.ExitKind != abi.ExitTable {
		return 0
	}
	// The description, which the code keeps alive (Code.Exits), by its
	// address: a word native code writes, not a pointer the collector sees.
	d := (*abi.ExitDescriptor)(asPointer((*uint64)(unsafe.Pointer(&ctx.ExitDesc))))
	enc := d.Enc
	locals, stack := asPointer(&ctx.Regs[d.LocalsReg]), asPointer(&ctx.Regs[d.StackReg])
	size := int(enc.ValueSize)
	// slotAt is the address of slot i's value, as mir's slotSource has it.
	slotAt := func(i int32) unsafe.Pointer {
		switch {
		case i < d.FrameLocals:
			return unsafe.Add(locals, int(i)*size)
		case i == d.ThisSlot:
			return unsafe.Add(unsafe.Pointer(&ctx.This), -int(enc.NumOffset))
		case i < d.Locals:
			cell := *(*unsafe.Pointer)(unsafe.Add(ctx.Upvalues, int(i-d.FrameLocals)*8))
			return *(*unsafe.Pointer)(unsafe.Add(cell, int(enc.UpvalueSlot)))
		}
		return unsafe.Add(stack, int(i-d.Locals)*size)
	}
	num := func(p unsafe.Pointer) uint64 { return *(*uint64)(unsafe.Add(p, int(enc.NumOffset))) }
	ref := func(p unsafe.Pointer) unsafe.Pointer {
		return *(*unsafe.Pointer)(unsafe.Add(p, int(enc.RefOffset)))
	}
	word := func(l abi.ExitLoc) uint64 {
		switch l.Kind {
		case abi.ExitReg:
			return ctx.Regs[l.N]
		case abi.ExitSpill:
			return ctx.Spill[l.N]
		case abi.ExitConst:
			return l.Word
		case abi.ExitF64Reg, abi.ExitF64Spill:
			bits := ctx.XRegs[l.N&31]
			if l.Kind == abi.ExitF64Spill {
				bits = ctx.Spill[l.N]
			}
			if f := math.Float64frombits(bits); f != f {
				return enc.CanonicalNaN
			}
			return bits
		case abi.ExitBoolReg, abi.ExitBoolSpill:
			b := ctx.Regs[l.N&31]
			if l.Kind == abi.ExitBoolSpill {
				b = ctx.Spill[l.N]
			}
			if uint32(b) != 0 {
				return enc.True
			}
			return enc.False
		case abi.ExitSlotAddr:
			return uint64(uintptr(slotAt(l.N)))
		}
		return 0
	}
	type write struct {
		at  unsafe.Pointer
		num uint64
		ref unsafe.Pointer
	}
	var buf [64]write
	writes := buf[:0]
	for i := range d.Slots {
		s := &d.Slots[i]
		at := slotAt(s.Slot)
		w := word(s.Value)
		n, r := w, unsafe.Pointer(nil)
		switch {
		case s.Shadow.Kind != abi.ExitNone:
			// Where the reference came from is known only now: the slot's
			// own value is unchanged; a source's reference is copied.
			fw := word(s.Shadow)
			from := asPointer(&fw)
			if from == at {
				continue
			}
			if from != nil {
				if p := ref(from); p != nil {
					n, r = num(from), p
				}
			}
		case s.Origin >= 0:
			// The reference a slot held on entry, if it holds it still.
			o := slotAt(s.Origin)
			if p := ref(o); p != nil && (s.Load || w == num(o)) {
				if s.Origin == s.Slot {
					continue
				}
				n, r = num(o), p
			}
		}
		writes = append(writes, write{at, n, r})
	}
	for _, x := range writes {
		if x.ref != nil || ref(x.at) != nil {
			refs++
		}
		*(*uint64)(unsafe.Add(x.at, int(enc.NumOffset))) = x.num
		*(*unsafe.Pointer)(unsafe.Add(x.at, int(enc.RefOffset))) = x.ref
	}
	if len(d.Inline) != 0 {
		// The callers' frames' first slot, in the VM's stack.
		first := int64((uintptr(stack)-uintptr(ctx.StackBase))/uintptr(size)) - int64(d.Locals)
		next := ctx
		for _, in := range d.Inline {
			next = (*abi.Context)(unsafe.Add(unsafe.Pointer(next), abi.ContextSize))
			next.InlineClosure, next.InlineLocals, next.InlineThis = in.Closure, in.Locals, in.ThisSlot
			next.ExitKind, next.ExitPC, next.ExitDepth, next.ExitSite = in.Kind, in.PC, in.Depth, uint64(in.Site)
			next.Live, next.Base = abi.LiveInline, uint64(first+in.Base)
		}
	}
	ctx.Records = 0
	ctx.ExitKind, ctx.ExitPC, ctx.ExitDepth, ctx.ExitSite = d.Kind, d.PC, d.Depth, uint64(d.Site)
	return refs
}

// asPointer is the address a word holds, as a pointer: native code's words
// are addresses Go's memory holds still (ssa's origin.go).
func asPointer(w *uint64) unsafe.Pointer { return *(*unsafe.Pointer)(unsafe.Pointer(w)) }
