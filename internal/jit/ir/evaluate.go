package ir

import (
	"errors"
	"math"
)

// ErrState reports an invalid scratch size or a non-resumable entry PC.
var ErrState = errors.New("jit IR: invalid execution state")

// Evaluate executes a validated program in Go for at most budget instructions.
// Slots is caller-owned scratch, containing locals followed by operands. Exits
// leave all live slots in interpreter order; unused slots may hold stale values.
// It can resume at every reachable state map, including with budget zero.
func (p *Program) Evaluate(slots []Value, pc int, budget uint64) (Exit, error) {
	if p == nil || len(slots) != p.Locals+p.StackSize || pc < 0 || pc >= len(p.Code) ||
		len(p.Maps) != len(p.Code) || p.Maps[pc].Depth < 0 {
		return Exit{}, ErrState
	}
	read := func(o Operand) Value {
		if o.Slot == -1 {
			return o.Literal
		}
		return slots[o.Slot]
	}
	var steps uint64
	for {
		exit := Exit{Kind: GuardExit, State: p.Maps[pc], Steps: steps}
		if steps == budget {
			exit.Kind = BudgetExit
			return exit, nil
		}
		in := p.Code[pc]
		if in.Check && slots[in.CheckSlot].Kind == Uninitialized {
			return exit, nil
		}
		next := pc + 1
		switch in.Op {
		case Nop:
		case Copy:
			slots[in.Dest] = read(in.Left)
		case CopyPair:
			a, b := read(in.Left), read(in.Right)
			slots[in.Dest], slots[in.Extra] = a, b
		case StoreLoad:
			slots[in.Dest] = read(in.Left)
			slots[in.Extra] = read(in.Right)
		case Swap:
			slots[in.Dest], slots[in.Extra] = slots[in.Extra], slots[in.Dest]
		case Binary:
			v, ok := binary(in.Operator, read(in.Left), read(in.Right))
			if !ok {
				return exit, nil
			}
			slots[in.Dest] = v
		case Unary:
			a := read(in.Left)
			if in.Operator == Not {
				b, ok := truth(a)
				if !ok {
					return exit, nil
				}
				slots[in.Dest] = Bool(!b)
			} else {
				if a.Kind != Number {
					return exit, nil
				}
				if in.Operator == Neg {
					a.Bits ^= 1 << 63
				}
				slots[in.Dest] = a
			}
		case Update:
			old := read(in.Left)
			v, ok := binary(in.Operator, old, Float(1))
			if !ok {
				return exit, nil
			}
			slots[in.Dest] = v
			if in.Extra >= 0 {
				if in.Postfix {
					v = old
				}
				slots[in.Extra] = v
			}
		case Jump:
			next = in.Target
		case Branch:
			var condition bool
			var ok bool
			if in.Operator == Truth {
				condition, ok = truth(read(in.Left))
			} else {
				var v Value
				v, ok = binary(in.Operator, read(in.Left), read(in.Right))
				condition = v.Bits != 0
			}
			if !ok {
				return exit, nil
			}
			if condition == in.When {
				next = in.Target
			}
		case Return:
			v := read(in.Left)
			if v.Kind == Opaque || v.Kind == Uninitialized {
				return exit, nil
			}
			exit.Kind, exit.Value, exit.Steps = Returned, v, steps+1
			return exit, nil
		default:
			return Exit{}, ErrState
		}
		steps++
		pc = next
	}
}

func binary(op Operator, a, b Value) (Value, bool) {
	if a.Kind != Number || b.Kind != Number {
		return Value{}, false
	}
	x, y := math.Float64frombits(a.Bits), math.Float64frombits(b.Bits)
	switch op {
	case Add:
		return Float(x + y), true
	case Sub:
		return Float(x - y), true
	case Mul:
		return Float(x * y), true
	case Div:
		return Float(x / y), true
	case Lt:
		return Bool(x < y), true
	case Le:
		return Bool(x <= y), true
	case Gt:
		return Bool(x > y), true
	case Ge:
		return Bool(x >= y), true
	case Eq:
		return Bool(x == y), true
	case Ne:
		return Bool(x != y), true
	}
	return Value{}, false
}

func truth(v Value) (bool, bool) {
	switch v.Kind {
	case Undefined, Null:
		return false, true
	case Boolean:
		return v.Bits != 0, true
	case Number:
		n := math.Float64frombits(v.Bits)
		return n != 0 && !math.IsNaN(n), true
	}
	return false, false
}
