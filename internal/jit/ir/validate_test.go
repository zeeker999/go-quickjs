package ir

import "testing"

func TestValidateNativeSafety(t *testing.T) {
	fixture := func() *Program {
		return &Program{Locals: 1, StackSize: 1,
			Code: []Instruction{{Op: Copy, Left: Slot(0), Dest: 1}, {Op: Return, Left: Slot(1)}},
			Maps: []StateMap{{PC: 0}, {PC: 1, Depth: 1}}}
	}
	for _, tc := range []struct {
		name string
		edit func(*Program)
	}{
		{"source", func(p *Program) { p.Code[0].Left.Slot = -2 }},
		{"inactive", func(p *Program) { p.Code[0].Left.Slot = 1 }},
		{"destination", func(p *Program) { p.Code[0].Dest = 2 }},
		{"literal", func(p *Program) { p.Code[0].Left = Literal(Value{Kind: Kind(99)}) }},
		{"boolean", func(p *Program) { p.Code[0].Left = Literal(Value{Kind: Boolean, Bits: 2}) }},
		{"guard", func(p *Program) { p.Code[0].Check, p.Code[0].CheckSlot = true, 1 }},
		{"initial-depth", func(p *Program) { p.Maps[0].Depth = 1 }},
		{"map", func(p *Program) { p.Maps[1].PC = 2 }},
		{"capacity", func(p *Program) { p.StackSize = MaxSlots }},
		{"missing-map", func(p *Program) { p.Maps = nil }},
		{"fallthrough", func(p *Program) { p.Code[1].Op = Nop }},
		{"target", func(p *Program) { p.Code[0].Op, p.Code[0].Target = Jump, 2 }},
		{"unreachable", func(p *Program) { p.Maps[1].Depth = -1 }},
		{"binary", func(p *Program) { p.Code[0].Op, p.Code[0].Operator = Binary, Truth }},
		{"unary", func(p *Program) { p.Code[0].Op, p.Code[0].Operator = Unary, Add }},
		{"update", func(p *Program) { p.Code[0].Op, p.Code[0].Operator = Update, Div }},
		{"swap", func(p *Program) { p.Code[0].Op, p.Code[0].Extra = Swap, 0 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := fixture()
			if err := p.Validate(); err != nil {
				t.Fatal(err)
			}
			tc.edit(p)
			if err := p.Validate(); err == nil {
				t.Fatal("accepted unsafe native IR")
			}
		})
	}
	if err := (*Program)(nil).Validate(); err == nil {
		t.Fatal("accepted nil program")
	}
}
