//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package jit

import (
	"math"
	"math/bits"

	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

// A nonzero bound proves a finite number with abs(value) < 2^bound.
// Facts hold only within a precharged region. Arbitrary external entries and
// exact small-budget execution use the ordinary conversion path instead.
func (a *programAssembler) inferRanges(p *ir.Program) {
	a.ranges = make([][ir.MaxSlots]uint8, len(p.Code))
	var bounds [ir.MaxSlots]uint8
	read := func(o ir.Operand) uint8 {
		if o.Slot >= 0 {
			return bounds[o.Slot]
		}
		if o.Literal.Kind != ir.Number {
			return 0
		}
		v := math.Abs(math.Float64frombits(o.Literal.Bits))
		if math.IsInf(v, 0) || math.IsNaN(v) {
			return 0
		}
		_, exponent := math.Frexp(v)
		return boundedExponent(max(exponent, 1))
	}
	for pc, in := range p.Code {
		if p.Maps[pc].Depth < 0 {
			continue
		}
		if a.starts[pc] || pc > 0 && a.tails[pc-1] == 1 {
			clear(bounds[:])
		}
		a.ranges[pc] = bounds
		switch in.Op {
		case ir.Nop, ir.Jump, ir.Branch, ir.Return, ir.Host, ir.ArrayKey, ir.ArrayWrite, ir.PropertyWrite:
		case ir.Copy:
			bounds[in.Dest] = read(in.Left)
		case ir.CopyPair:
			l, r := read(in.Left), read(in.Right)
			bounds[in.Dest], bounds[in.Extra] = l, r
		case ir.StoreLoad:
			bounds[in.Dest] = read(in.Left)
			bounds[in.Extra] = read(in.Right)
		case ir.Swap:
			bounds[in.Dest], bounds[in.Extra] = bounds[in.Extra], bounds[in.Dest]
		case ir.Binary:
			l, r := read(in.Left), read(in.Right)
			bound := uint8(0)
			switch {
			case in.Operator >= ir.BitAnd && in.Operator <= ir.UShr:
				bound = 32
				if in.Right.Slot < 0 && in.Right.Literal.Kind == ir.Number {
					v := ir.ToUint32(math.Float64frombits(in.Right.Literal.Bits))
					if in.Operator == ir.BitAnd && v < 1<<31 {
						bound = uint8(max(bits.Len32(v), 1))
					} else if in.Operator == ir.UShr {
						bound = 32 - uint8(v&31)
					}
				}
			case l != 0 && r != 0:
				if in.Operator == ir.Add || in.Operator == ir.Sub {
					bound = boundedExponent(int(max(l, r)) + 1)
				} else if in.Operator == ir.Mul {
					bound = boundedExponent(int(l) + int(r))
				}
			}
			bounds[in.Dest] = bound
		case ir.Unary:
			bound := uint8(0)
			if in.Operator == ir.Int32 || in.Operator == ir.BitNot {
				bound = 32
			} else if in.Operator == ir.Pos || in.Operator == ir.Neg {
				bound = read(in.Left)
			}
			bounds[in.Dest] = bound
		case ir.Update:
			old := read(in.Left)
			bound := uint8(0)
			if old != 0 {
				bound = boundedExponent(int(old) + 1)
			}
			bounds[in.Dest] = bound
			if in.Extra >= 0 {
				if in.Postfix {
					bounds[in.Extra] = old
				} else {
					bounds[in.Extra] = bound
				}
			}
		case ir.ArrayRead, ir.ArrayLength, ir.PropertyRead, ir.BindingRead, ir.ReferenceRead:
			bounds[in.Dest] = 0
		case ir.StringMethod:
			bounds[in.Dest] = 0
		case ir.StringCode:
			bounds[in.Dest] = 16
		case ir.ArrayUpdate:
			bounds[in.Dest], bounds[in.Extra] = 0, 0
		case ir.Insert2, ir.Insert3:
			n, last := in.Dest, 2
			if in.Op == ir.Insert2 {
				last = 1
			}
			t := bounds[n+last]
			for i := last; i > 0; i-- {
				bounds[n+i] = bounds[n+i-1]
			}
			bounds[n], bounds[n+last+1] = t, t
		default:
			clear(bounds[:])
		}
	}
}

func boundedExponent(exponent int) uint8 {
	if exponent > 62 {
		return 0
	}
	return uint8(exponent)
}

func (a *programAssembler) boundedInteger(o ir.Operand) bool {
	return a.fast && o.Slot >= 0 && a.ranges[a.pc][o.Slot] != 0
}
