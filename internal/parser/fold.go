package parser

import (
	"github.com/go-quickjs/go-quickjs/internal/ast"
	"github.com/go-quickjs/go-quickjs/internal/fold"
)

// unaryOf builds op of operand -- !, ~, + or - -- or, where operand is a
// literal the operator folds (fold.Unary), the literal of its value, as
// V8's parser does (Parser::BuildUnaryExpression): ! of any literal, -, +
// and ~ of a number, and - of a BigInt. The operand was folded as it was
// built, so that a nest of them folds from the bottom up, once each.
func (p *parser) unaryOf(op string, operand ast.Expr, start int) ast.Expr {
	switch op {
	case "!":
		if t, ok := fold.Not(operand); ok {
			n := p.nodes.bools.alloc()
			n.Value, n.Start = t, start
			return n
		}
	case "-", "+", "~":
		if !fold.On(fold.Unary) {
			break
		}
		switch x := operand.(type) {
		case *ast.NumberLit:
			v := x.Value
			switch op {
			case "-":
				v = -v
			case "~":
				v = fold.BitNot(v)
			}
			return p.nodes.number(v, start)
		case *ast.BigIntLit:
			if op == "-" {
				if raw, ok := fold.Negate(x.Raw); ok {
					return &ast.BigIntLit{Raw: raw, Start: start}
				}
			}
		}
	}
	return p.nodes.unaryOp(op, operand, start)
}

// binaryOf builds op of left and right, or, where both are numbers and op
// one of the arithmetic, shift and bitwise operators (fold.Arith), the
// number of its value, as V8's parser does
// (Parser::ShortcutLiteralBinaryExpression).
func (p *parser) binaryOf(op string, left, right ast.Expr) ast.Expr {
	if x, ok := left.(*ast.NumberLit); ok {
		if y, ok := right.(*ast.NumberLit); ok && fold.On(fold.Numeric) {
			if v, ok := fold.Arith(op, x.Value, y.Value); ok {
				return p.nodes.number(v, x.Start)
			}
		}
	}
	return p.nodes.binaryOp(op, left, right)
}
