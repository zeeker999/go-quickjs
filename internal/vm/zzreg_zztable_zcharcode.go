package vm

import (
	"math"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
)

func init() {
	regTable[bytecode.RCharCodeCall], regSlowTable[bytecode.RCharCodeCall] = rhCharCodeCall, rsCharCodeCall
}

// The lookup and argument evaluation already happened: neither the fast
// path nor the fallback reads charCodeAt again, even if evaluating the
// argument changed the method, its receiver's binding or its prototype.
func rhCharCodeCall(regs *[1 << 16]Value, w bytecode.RInstr, pc int, st *regState) int {
	if o, a, m := regs[w.A()+1], regs[w.A()+2], regs[w.A()]; o.IsString() && a.IsNumber() && m.IsObject() {
		if fd := m.Object().fn(); fd != nil && fd.elemOp == elemStrCharCodeAt && !fd.bound {
			x := a.Number()
			if x != x {
				x = 0
			}
			v := Float(math.NaN())
			if s := o.String(); x > -1 && x < float64(s.Len()) && !s.IsEmpty() {
				v = Int(s.CharCodeAt(int(x)))
			}
			regs[w.B()] = v
			return pc + 1
		}
	}
	return ^pc
}

func rsCharCodeCall(regs *[1 << 16]Value, w bytecode.RInstr, pc int, st *regState) int {
	// Reuse ordinary call caching and entry for an overridden method,
	// instead of turning every call of script into a generic call.
	w = bytecode.MakeR(bytecode.RCallMW, 1, w.A(), w.B(), w.C())
	return rhCallW(regs, w, pc, st)
}
