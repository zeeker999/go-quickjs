package regcomp

import (
	"github.com/go-quickjs/go-quickjs/internal/ast"
	"github.com/go-quickjs/go-quickjs/internal/bytecode"
)

// A try statement is compiled to handler tables (docs/register-interpreter.md,
// "Exceptions"): the code of its block is in the ranges of a handler, and an
// exception an instruction there throws goes on at the handler's pc, its
// value in the handler's register. Nothing is run as the block is entered
// or left. A finally clause is compiled once for each way out of what it
// protects: in place after the block's normal completion, before each
// break, continue or return that leaves it, and as a handler for a throw,
// which throws the value again at its end. A clause's copy before an exit
// is outside the ranges of the handlers the exit leaves, and inside those
// of the handlers around them.

// prot is a handler while its code is compiled.
type prot struct {
	// ranges are the [start, end) pairs of pcs it covers so far, and open
	// the start of the one being compiled, or -1.
	ranges []uint32
	open   int
	to     *mark
	reg    int
	// finally is the finally clause's statements, nil for a catch clause.
	finally []ast.Stmt
	// cursor is, for a for-of loop's handler, the register of its cursor,
	// which an exit closes, and iterOf the loop, which a continue of it
	// does not leave; -1 and nil otherwise.
	cursor int
	iterOf *loop
	// async marks a for await's cursor, closed as AsyncIteratorClose
	// closes it: its return method's result awaited (asyncClose).
	async bool
	// loops is how many loops were around the statement: a break or a
	// continue to one of them leaves the handler.
	loops int
}

// protect puts p in force from here.
func (g *gen) protect(p *prot) {
	p.open = len(g.code)
	p.loops = len(g.loops)
	g.prots = append(g.prots, p)
}

// unprotect ends p, the innermost handler in force, and adds it to the
// table.
func (g *gen) unprotect(p *prot) {
	g.suspend(p)
	g.prots = g.prots[:len(g.prots)-1]
	if len(p.ranges) > 0 {
		g.handlers = append(g.handlers, p)
	}
}

// suspend ends p's open range, and resume opens another.
func (g *gen) suspend(p *prot) {
	if p.open >= 0 && p.open < len(g.code) {
		p.ranges = append(p.ranges, uint32(p.open), uint32(len(g.code)))
	}
	p.open = -1
}

func (g *gen) resume(p *prot) { p.open = len(g.code) }

// finallyDepth bounds how deep finally clauses' copies nest: each clause
// is compiled once for each way out of it, so that a clause inside one is
// compiled for each of the outer's copies.
const finallyDepth = 4

// leave compiles an exit that leaves the handlers from g.prots[from:]: the
// finally clauses among them, innermost first, each outside the ranges of
// the handlers it is past and inside those of the rest, and then the exit
// itself, by last.
func (g *gen) leave(from int, last func()) {
	crossed := g.prots[from:]
	fin := false
	for _, p := range crossed {
		fin = fin || p.finally != nil || p.cursor >= 0
	}
	if !fin {
		last()
		return
	}
	saved, savedLoops := g.prots, g.loops
	for i := len(crossed) - 1; i >= 0; i-- {
		p := crossed[i]
		g.suspend(p)
		if p.cursor >= 0 {
			// A loop left closes its iterator, which may throw.
			if p.async {
				g.asyncClose(p.cursor)
			} else {
				g.emit(bytecode.MakeR(bytecode.RCold, uint8(bytecode.RColdIterClose), 0, r16(p.cursor), 0))
			}
			continue
		}
		if p.finally == nil {
			continue
		}
		g.prots = saved[: from+i : from+i]
		g.loops = savedLoops[:p.loops:p.loops]
		g.finallyCopy(p.finally)
		g.prots, g.loops = saved, savedLoops
		if g.err != "" {
			return
		}
	}
	last()
	for _, p := range crossed {
		g.resume(p)
	}
}

// finallyCopy compiles a copy of a finally clause's statements.
func (g *gen) finallyCopy(list []ast.Stmt) {
	if g.copies >= finallyDepth {
		g.fail("finally clauses nested too deep")
		return
	}
	g.copies++
	mark := g.top
	if g.completion > 0 {
		// What the clause produces is not the statement's value: the try
		// block's is, kept meanwhile; a break or a continue inside the
		// clause carries the clause's own, undefined where it produced
		// none, as the stack form's compileFinallyClause has it.
		save := g.alloc()
		release := g.hold()
		g.emit(bytecode.MakeR(bytecode.RMove, 0, r16(save), r16(g.completion), 0))
		g.resetCompletion()
		g.stmts(list)
		g.emit(bytecode.MakeR(bytecode.RMove, 0, r16(g.completion), r16(save), 0))
		release()
	} else {
		g.stmts(list)
	}
	g.top = mark
	g.copies--
}

// protsOutside is how many handlers in force are around loop l: the ones
// past it a break or a continue to l leaves. A continue does not leave
// the loop's own iterator.
func (g *gen) protsOutside(l *loop, cont bool) int {
	i := 0
	for i < len(g.prots) && g.prots[i].loops <= g.loopIndex(l) {
		i++
	}
	if cont && i < len(g.prots) && g.prots[i].iterOf == l {
		i++
	}
	return i
}

func (g *gen) loopIndex(l *loop) int {
	for i, x := range g.loops {
		if x == l {
			return i
		}
	}
	return len(g.loops)
}

// tryStmt compiles a try statement.
func (g *gen) tryStmt(n *ast.TryStmt) {
	if n.Inline {
		g.fail("a using declaration's scope")
		return
	}
	var fin, cat *prot
	if n.Finally != nil {
		fin = &prot{to: g.newMark(), reg: g.alloc(), finally: n.Finally, cursor: -1}
		defer g.hold()()
		g.protect(fin)
	}
	after := g.newMark()
	if n.Catch == nil {
		g.stmts(n.Block)
	} else {
		reg, ok := g.catchReg(n.Catch)
		if !ok {
			return
		}
		cat = &prot{to: g.newMark(), reg: reg, cursor: -1}
		defer g.hold()()
		g.protect(cat)
		g.stmts(n.Block)
		g.unprotect(cat)
		g.jump(bytecode.MakeR(bytecode.RJmp, 0, 0, 0, 0), after, false)
		g.bind(cat.to)
		if pat := n.Catch.Param; pat != nil {
			if _, ok := pat.(*ast.Ident); !ok {
				g.patternBindings(pat)
				g.pattern(pat, cat.reg, true)
			}
		}
		g.stmts(n.Catch.Body)
	}
	g.bind(after)
	if fin == nil {
		return
	}
	g.unprotect(fin)
	end := g.newMark()
	g.finallyCopy(n.Finally)
	g.jump(bytecode.MakeR(bytecode.RJmp, 0, 0, 0, 0), end, false)
	g.bind(fin.to)
	g.finallyCopy(n.Finally)
	g.at(n.Start)
	g.emit(bytecode.MakeR(bytecode.RThrow, 0, r16(fin.reg), 0, 0))
	g.bind(end)
}

// catchReg is the register a catch clause's value goes to: its binding's,
// or a temporary where it has none or nothing reads it.
func (g *gen) catchReg(c *ast.CatchClause) (int, bool) {
	if c.Param == nil {
		return g.alloc(), true
	}
	if _, ok := c.Param.(*ast.Ident); !ok {
		// A pattern, which the clause's body unpacks first.
		return g.alloc(), true
	}
	slot, ok := g.rec.Catches[c]
	if !ok {
		// bindCatchParam records every identifier's: none is the stack
		// compiler's bug, not a form to refuse.
		panic("regcomp: a catch clause's binding the record does not name")
	}
	if int(slot) >= len(g.slotReg) || g.slotReg[slot] < 0 {
		return g.alloc(), true
	}
	if g.fn.Locals[slot].Captured && g.inLoop() {
		// Each turn's binding is a new one a closure keeps.
		g.fail("a captured catch binding in a loop")
		return 0, false
	}
	return int(g.slotReg[slot]), true
}

// handlerTable is the code's handlers, innermost first.
func (g *gen) handlerTable() []bytecode.RHandler {
	var hs []bytecode.RHandler
	for _, p := range g.handlers {
		hs = append(hs, bytecode.RHandler{Ranges: p.ranges, To: uint32(p.to.at), Reg: r16(p.reg)})
	}
	return hs
}

// finallyAround reports whether a finally clause, or a for-of loop's
// iterator, is around the code: what a return runs on its way out.
func (g *gen) finallyAround() bool {
	for _, p := range g.prots {
		if p.finally != nil || p.cursor >= 0 {
			return true
		}
	}
	return false
}

// forOfStmt compiles a for-of loop, or with in a for-in loop: a cursor in
// a register of its own, made by for_of_start or for_in_start (cold) and
// stepped by iter_next, a direct operation, at the head of each turn, which jumps out of the loop where
// it has no value. A for-of loop's body is in the ranges of a handler of
// its own, which closes the iterator for a throw and throws again; a
// break, continue or return that leaves the loop closes it on its way out
// (leave). A for-in loop's cursor needs no closing.
func (g *gen) forOfStmt(left ast.Node, right ast.Expr, body ast.Stmt, await, in bool, label string, pos int) {
	var target ast.Expr
	lexical := false
	switch l := left.(type) {
	case *ast.VarDecl:
		if l.Kind != ast.DeclVar && l.Kind != ast.DeclLet && l.Kind != ast.DeclConst || len(l.Decls) != 1 || l.Decls[0].Init != nil {
			g.fail("a for-of or for-in head's declaration")
			return
		}
		target, lexical = l.Decls[0].Target, l.Kind != ast.DeclVar
	case ast.Expr:
		target = l
	}
	id, ok := target.(*ast.Ident)
	if !ok {
		g.forOfPattern(target, right, body, await, in, label, pos, lexical, left)
		return
	}
	res, ok := g.rec.Write[id]
	if !ok {
		g.fail("a store the record does not resolve")
		return
	}
	dst := -1
	switch res.Kind {
	case ResLocal, ResLocalInit:
		slot := int(res.Index)
		if lexical && g.fn.Locals[slot].Captured {
			// Each turn's binding is a new one a closure keeps.
			g.fail("a captured lexical binding in a loop")
			return
		}
		dst = int(g.slotReg[slot])
		if lexical {
			// The head's expression has a binding of the name of its own,
			// in its dead zone for good: a closure there that captures it
			// keeps the stack form's.
			for i, l := range g.fn.Locals {
				if i != slot && l.Name == id.Name && l.Captured {
					g.fail("a closure of a for-of or for-in head's binding")
					return
				}
			}
		}
		if lexical && g.tdz[slot] {
			// The head's expression sees the binding in its dead zone.
			g.emit(bytecode.MakeR(bytecode.RLoadUninit, 0, r16(dst), 0, 0))
		}
	case ResUpval, ResGlobal:
	default:
		g.fail("a for-of or for-in loop's store the backend does not compile")
		return
	}
	src := g.expr(right, -1)
	cursor := g.alloc()
	g.at(pos)
	g.emit(bytecode.MakeR(bytecode.RCold, uint8(cursorStart(await, in)), r16(cursor), r16(src), 0))
	l := g.pushLoop(label)
	g.bind(l.cont)
	val := dst
	if val < 0 {
		val = g.alloc()
	}
	g.at(pos)
	g.iterStep(val, cursor, l.brk, await)
	var p *prot
	if !in {
		p = &prot{to: g.newMark(), reg: g.alloc(), cursor: cursor, iterOf: l, async: await}
		g.protect(p)
	}
	defer g.hold()()
	if dst < 0 {
		g.storeValue(id, val)
	}
	g.stmt(body)
	g.jump(bytecode.MakeR(bytecode.RJmp, 0, 0, 0, 0), l.cont, false)
	if p != nil {
		g.unprotect(p)
		g.bind(p.to)
		g.closeThrow(p)
	}
	g.bind(l.brk)
	g.popLoop()
}

// forOfPattern is forOfStmt with a pattern, or an assignment's property,
// for its target: each turn's value unpacked into it in the loop's body,
// where a throw closes the iterator.
func (g *gen) forOfPattern(target ast.Expr, right ast.Expr, body ast.Stmt, await, in bool, label string, pos int, lexical bool, left ast.Node) {
	_, declaring := left.(*ast.VarDecl)
	if lexical {
		// The head's expression sees the bindings in their dead zone; a
		// closure's binding is each turn's own.
		for _, id := range patternNames(target, nil) {
			if res, ok := g.rec.Write[id]; ok && res.Kind == ResLocalInit && g.fn.Locals[res.Index].Captured {
				g.fail("a captured lexical binding in a loop")
				return
			}
		}
		g.patternBindings(target)
	}
	src := g.expr(right, -1)
	cursor := g.alloc()
	g.at(pos)
	g.emit(bytecode.MakeR(bytecode.RCold, uint8(cursorStart(await, in)), r16(cursor), r16(src), 0))
	l := g.pushLoop(label)
	g.bind(l.cont)
	val := g.alloc()
	g.at(pos)
	g.iterStep(val, cursor, l.brk, await)
	var p *prot
	if !in {
		p = &prot{to: g.newMark(), reg: g.alloc(), cursor: cursor, iterOf: l, async: await}
		g.protect(p)
	}
	defer g.hold()()
	mark := g.top
	g.pattern(target, val, declaring)
	g.top = mark
	g.stmt(body)
	g.jump(bytecode.MakeR(bytecode.RJmp, 0, 0, 0, 0), l.cont, false)
	if p != nil {
		g.unprotect(p)
		g.bind(p.to)
		g.closeThrow(p)
	}
	g.bind(l.brk)
	g.popLoop()
}

// cursorStart is the operation that makes a loop's cursor.
func cursorStart(await, in bool) bytecode.RColdOp {
	switch {
	case await:
		return bytecode.RColdForAwaitOfStart
	case in:
		return bytecode.RColdForInStart
	}
	return bytecode.RColdForOfStart
}

// iterStep steps a loop's cursor into val, jumping to brk where it has no
// value: iter_next, or for a for await, the next method's result awaited
// and unpacked, in a register of its own, which code that runs meanwhile
// could otherwise see in the binding.
func (g *gen) iterStep(val, cursor int, brk *mark, await bool) {
	if !await {
		g.jump(bytecode.MakeR(bytecode.RIterNext, 0, r16(val), r16(cursor), 0), brk, true)
		return
	}
	t := g.alloc()
	g.cold(bytecode.RColdAsyncIterNext, t, cursor, 0)
	g.cold(bytecode.RColdAwait, t, t, 0)
	g.cold(bytecode.RColdIterResult, val, t, 0)
	g.jump(bytecode.MakeR(bytecode.RJmp, 0, 0, 0, 0), brk, false)
	g.top = t
}

// closeThrow is the handler of a for-of loop's body, p, whose exception,
// in p's register, leaves the loop: the iterator closed, what closing it
// throws dropped, and the exception thrown again. A for await's return
// method's result is awaited, as AsyncIteratorClose awaits it.
func (g *gen) closeThrow(p *prot) {
	if !p.async {
		g.emit(bytecode.MakeR(bytecode.RCold, uint8(bytecode.RColdIterCloseThrow), 0, r16(p.cursor), 0))
		g.emit(bytecode.MakeR(bytecode.RThrow, 0, r16(p.reg), 0, 0))
		return
	}
	q := &prot{to: g.newMark(), reg: g.alloc(), cursor: -1}
	g.protect(q)
	v := g.alloc()
	g.cold(bytecode.RColdAsyncIterClose, v, p.cursor, 0)
	g.jump(bytecode.MakeR(bytecode.RJmp, 0, 0, 0, 0), q.to, false)
	g.cold(bytecode.RColdAwait, v, v, 0)
	g.unprotect(q)
	g.bind(q.to)
	g.emit(bytecode.MakeR(bytecode.RThrow, 0, r16(p.reg), 0, 0))
}

// asyncClose closes a for await's cursor as a break, a continue or a
// return that leaves the loop does: its return method called, where it has
// one, and the result awaited and checked to be an object.
func (g *gen) asyncClose(cursor int) {
	v := g.alloc()
	skip := g.newMark()
	g.cold(bytecode.RColdAsyncIterClose, v, cursor, 0)
	g.jump(bytecode.MakeR(bytecode.RJmp, 0, 0, 0, 0), skip, false)
	g.cold(bytecode.RColdAwait, v, v, 0)
	g.cold(bytecode.RColdAsyncIterCheck, v, 0, 0)
	g.bind(skip)
}
