package regcomp

import (
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/go-quickjs/go-quickjs/internal/ast"
	"github.com/go-quickjs/go-quickjs/internal/bytecode"
)

// The register form's frame: r0 the function, r1 its this, the parameters
// from r2, then the function's bindings, then the temporaries
// (docs/register-interpreter.md, "The frame").
const (
	regCallee = 0
	regThis   = 1
	regParams = 2
)

// limit16 is where a 16-bit operand ends.
const limit16 = 1 << 16

// Compile compiles lit, whose stack form is fn and whose identifiers' resolution
// is rec, into register code. It returns nil where the function has anything
// the backend does not compile yet; the function then runs as before.
func Compile(lit *ast.FuncLit, fn *bytecode.Function, rec *Record) *bytecode.RegCode {
	if w := why(fn); w != "" && !(w == "arguments" && simpleArguments(fn)) {
		Refuse(fn, w)
		return nil
	}
	g := &gen{lit: lit, fn: fn, rec: rec}
	if !g.layout() {
		Refuse(fn, "its bindings' layout")
		return nil
	}
	g.body()
	if g.err != "" {
		Refuse(fn, g.err)
		return nil
	}
	return g.finish()
}

// trace prints why a function has no register code (QJS_REGTRACE).
var trace = os.Getenv("QJS_REGTRACE") != ""

type gen struct {
	lit *ast.FuncLit
	fn  *bytecode.Function
	rec *Record

	code []bytecode.RInstr
	pos  []int32
	cur  int32

	// slotReg is the register of each stack slot the function's bindings
	// are in, or -1.
	slotReg []int32
	inits   int
	// lexical marks the slots of let and const bindings; tdz those whose
	// dead zone some read checks or a closure may see.
	tdz   []bool
	top   int
	max   int
	loops []*loop
	label string

	consts   map[any]uint32
	names    map[string]uint32
	caps     [][]uint16
	siteName []uint32
	auxes    map[int]uint32
	calls    int
	err      string
}

type loop struct {
	label        string
	brk, cont    *mark
	isLoop       bool
	scopeTopMark int
}

// mark is a place in the code jumps go to, and the jumps waiting for it.
type mark struct {
	at   int
	refs []jumpRef
}

type jumpRef struct {
	at    int
	short bool
}

func (g *gen) fail(why string) {
	if g.err == "" {
		g.err = why
	}
}

// layout gives each binding of the function a register: the parameters
// theirs, and every other slot the record, a hoisted function or a closure
// names, in slot order.
func (g *gen) layout() bool {
	fn := g.fn
	n := fn.LocalCount
	g.slotReg = make([]int32, n)
	g.tdz = make([]bool, n)
	used := make([]bool, n)
	note := func(res Res) bool {
		switch res.Kind {
		case ResLocal, ResLocalCheck, ResLocalInit:
			if int(res.Index) >= n {
				return false
			}
			used[res.Index] = true
			if res.Kind == ResLocalCheck {
				g.tdz[res.Index] = true
			}
		}
		return true
	}
	for _, res := range g.rec.Read {
		if !note(res) {
			return false
		}
	}
	for _, res := range g.rec.Write {
		if !note(res) {
			return false
		}
	}
	for _, slot := range g.rec.Funcs {
		if int(slot) >= n {
			return false
		}
		used[slot] = true
	}
	for _, c := range fn.Constants {
		if c.Kind != bytecode.ConstFunction || c.Fn == nil {
			continue
		}
		for _, u := range c.Fn.Upvalues {
			if u.FromParent {
				if int(u.Index) >= n {
					return false
				}
				used[u.Index] = true
			}
		}
	}
	params := fn.ParamCount
	next := regParams + params
	for i := range g.slotReg {
		switch {
		case i < params:
			g.slotReg[i] = int32(regParams + i)
		case used[i]:
			g.slotReg[i] = int32(next)
			next++
		default:
			g.slotReg[i] = -1
		}
	}
	g.inits = next - regParams - params
	g.top, g.max = next, next
	if next >= limit16 {
		return false
	}
	g.consts = map[any]uint32{}
	g.names = map[string]uint32{}
	for i, name := range fn.Names {
		if _, ok := g.names[name]; !ok {
			g.names[name] = uint32(i)
		}
	}
	g.caps = make([][]uint16, len(fn.Constants))
	g.siteName = make([]uint32, fn.PropSites+1)
	return true
}

// --- emission -------------------------------------------------------------

func (g *gen) emit(w bytecode.RInstr) int {
	g.code = append(g.code, w)
	g.pos = append(g.pos, g.cur)
	return len(g.code) - 1
}

func (g *gen) at(pos int) { g.cur = int32(pos) }

func r16(r int) uint16 { return uint16(r) }

func (g *gen) alloc() int {
	r := g.top
	g.top++
	if g.top > g.max {
		g.max = g.top
	}
	if g.top >= limit16 {
		g.fail("more than 65,536 registers")
	}
	return r
}

func (g *gen) allocN(n int) int {
	r := g.top
	for i := 0; i < n; i++ {
		g.alloc()
	}
	return r
}

// dest is dst, or a fresh temporary for -1.
func (g *gen) dest(dst int) int {
	if dst >= 0 {
		return dst
	}
	return g.alloc()
}

func (g *gen) newMark() *mark { return &mark{at: -1} }

func (g *gen) bind(m *mark) {
	m.at = len(g.code)
	for _, ref := range m.refs {
		g.patch(ref, m.at)
	}
	m.refs = nil
}

func (g *gen) patch(ref jumpRef, target int) {
	off := int64(target) - int64(ref.at+1)
	if ref.short {
		if off < -1<<15 || off >= 1<<15 {
			g.fail("a jump past 16 bits")
			return
		}
		g.code[ref.at] = g.code[ref.at].WithSC(int32(off))
		return
	}
	g.code[ref.at] = g.code[ref.at].WithBC(uint32(int32(off)))
}

// jump emits w, a jump to m: short says its offset is SC, else BC.
func (g *gen) jump(w bytecode.RInstr, m *mark, short bool) {
	at := g.emit(w)
	ref := jumpRef{at: at, short: short}
	if m.at >= 0 {
		g.patch(ref, m.at)
		return
	}
	m.refs = append(m.refs, ref)
}

func (g *gen) site(name uint32) uint16 {
	g.fn.PropSites++
	s := g.fn.PropSites
	if s >= limit16 {
		g.fail("a site past 16 bits")
		return 0
	}
	g.siteName = append(g.siteName, 0)
	g.siteName[s] = name
	return uint16(s)
}

func (g *gen) nameIdx(name string) uint32 {
	if i, ok := g.names[name]; ok {
		return i
	}
	i := uint32(len(g.fn.Names))
	g.fn.Names = append(g.fn.Names, name)
	g.names[name] = i
	return i
}

func (g *gen) name16(name string) uint16 {
	i := g.nameIdx(name)
	if i >= limit16 {
		g.fail("a name past 16 bits")
	}
	return uint16(i)
}

func (g *gen) constIdx(c bytecode.Constant) uint32 {
	var key any
	switch c.Kind {
	case bytecode.ConstNumber:
		key = math.Float64bits(c.Num)
	case bytecode.ConstString:
		key = c.Str
	default:
		key = len(g.fn.Constants)
	}
	if i, ok := g.consts[key]; ok {
		return i
	}
	i := uint32(len(g.fn.Constants))
	g.fn.Constants = append(g.fn.Constants, c)
	g.caps = append(g.caps, nil)
	g.consts[key] = i
	return i
}

// --- functions and statements ---------------------------------------------

func (g *gen) body() {
	lit := g.lit
	// The bindings whose dead zone a read checks or a closure may see start
	// in it; a self name a closure captures is the function.
	for slot, need := range g.tdz {
		if need && g.slotReg[slot] >= 0 {
			g.emit(bytecode.MakeR(bytecode.RLoadUninit, 0, r16(int(g.slotReg[slot])), 0, 0))
		}
	}
	if lit.Name != nil && !declares(lit.Body, lit.Name.Name) {
		for slot, l := range g.fn.Locals {
			if slot < len(g.slotReg) && g.slotReg[slot] >= 0 && l.Name == lit.Name.Name && !g.declared(slot) {
				g.emit(bytecode.MakeR(bytecode.RMove, 0, r16(int(g.slotReg[slot])), regCallee, 0))
			}
		}
	}
	// The arguments object, made as the function starts, in its binding.
	if g.fn.UsesArguments || g.fn.MakesArguments {
		made := false
		for slot, l := range g.fn.Locals {
			if l.Name == "arguments" && slot < len(g.slotReg) && g.slotReg[slot] >= 0 {
				g.emit(bytecode.MakeR(bytecode.RArguments, 0, r16(int(g.slotReg[slot])), 0, 0))
				made = true
			}
		}
		if !made {
			g.fail("an arguments binding the record does not name")
			return
		}
	}
	// Hoisted function declarations of the body; one in a block is left to
	// the stack form.
	for _, s := range lit.Body {
		if fd, ok := s.(*ast.FuncDecl); ok {
			slot, ok := g.rec.Funcs[fd]
			if !ok {
				g.fail("a function declaration without a binding")
				return
			}
			g.closure(fd.Fn, int(g.slotReg[slot]))
		}
	}
	g.stmts(lit.Body)
	g.emit(bytecode.MakeR(bytecode.RRetUndef, 0, 0, 0, 0))
}

// simpleArguments reports whether fn's use of arguments is the one the
// backend compiles: an object made as the function starts, in the binding
// the stack form gives it, of a function with simple parameters.
func simpleArguments(fn *bytecode.Function) bool {
	if fn.ForwardsArguments {
		// Only handed on to apply, which the stack form does without the
		// object: making it would cost every call an allocation.
		return false
	}
	if fn.HasRest || !fn.HasSimpleParams || fn.ParamsAreLexical || fn.Generator || fn.Async ||
		fn.HasDirectEval || fn.Debug != nil || fn.TopLevel || fn.PrivateSites != 0 || fn.HasTailCall {
		return false
	}
	switch fn.Kind {
	case bytecode.KindNormal, bytecode.KindMethod:
	default:
		return false
	}
	for _, l := range fn.Locals {
		if l.Name == "arguments" {
			return true
		}
	}
	return false
}

// switchStmt compiles a switch: the discriminant once, each case's test
// with ===, in order, jumping to its body, and default's body where no test
// matched; the bodies in order, each falling into the next.
func (g *gen) switchStmt(n *ast.SwitchStmt, label string) {
	for _, c := range n.Cases {
		for _, s := range c.Body {
			switch d := s.(type) {
			case *ast.VarDecl:
				if d.Kind != ast.DeclVar {
					g.fail("a lexical declaration in a switch")
					return
				}
			case *ast.FuncDecl, *ast.ClassDecl:
				g.fail("a declaration in a switch")
				return
			}
		}
	}
	d := g.expr(n.Disc, -1)
	if g.isBinding(d) {
		t := g.alloc()
		g.emit(bytecode.MakeR(bytecode.RMove, 0, r16(t), r16(d), 0))
		d = t
	}
	l := &loop{label: label, brk: g.newMark()}
	if label != "" {
		l.cont = nil
	}
	g.loops = append(g.loops, l)
	bodies := make([]*mark, len(n.Cases))
	def := -1
	for i, c := range n.Cases {
		bodies[i] = g.newMark()
		if c.Test == nil {
			def = i
			continue
		}
		mark := g.top
		if imm, ok := imm16(c.Test); ok {
			g.jump(bytecode.MakeR(bytecode.RJSEqI, 0, r16(d), uint16(int16(imm)), 0), bodies[i], true)
		} else {
			v := g.expr(c.Test, -1)
			g.jump(bytecode.MakeR(bytecode.RJSEq, 0, r16(d), r16(v), 0), bodies[i], true)
		}
		g.top = mark
	}
	if def >= 0 {
		g.jump(bytecode.MakeR(bytecode.RJmp, 0, 0, 0, 0), bodies[def], false)
	} else {
		g.jump(bytecode.MakeR(bytecode.RJmp, 0, 0, 0, 0), l.brk, false)
	}
	for i, c := range n.Cases {
		g.bind(bodies[i])
		for _, s := range c.Body {
			if g.err != "" {
				return
			}
			g.stmt(s)
		}
	}
	g.bind(l.brk)
	g.loops = g.loops[:len(g.loops)-1]
}

// declares reports whether the statements declare name, with var, let,
// const or a function declaration, anywhere but in a nested function: what
// shadows a function expression's own name.
func declares(list []ast.Stmt, name string) bool {
	for _, s := range list {
		switch n := s.(type) {
		case *ast.VarDecl:
			for _, d := range n.Decls {
				if id, ok := d.Target.(*ast.Ident); !ok || id.Name == name {
					return true
				}
			}
		case *ast.FuncDecl:
			if n.Fn.Name != nil && n.Fn.Name.Name == name {
				return true
			}
		case *ast.BlockStmt:
			if declares(n.Body, name) {
				return true
			}
		case *ast.IfStmt:
			if declares([]ast.Stmt{n.Cons}, name) || n.Alt != nil && declares([]ast.Stmt{n.Alt}, name) {
				return true
			}
		case *ast.ForStmt:
			if n.Init != nil && declares([]ast.Stmt{n.Init}, name) || declares([]ast.Stmt{n.Body}, name) {
				return true
			}
		case *ast.WhileStmt:
			if declares([]ast.Stmt{n.Body}, name) {
				return true
			}
		case *ast.DoWhileStmt:
			if declares([]ast.Stmt{n.Body}, name) {
				return true
			}
		case *ast.LabeledStmt:
			if declares([]ast.Stmt{n.Body}, name) {
				return true
			}
		case *ast.ExprStmt, *ast.ReturnStmt, *ast.EmptyStmt, *ast.BreakStmt, *ast.ContinueStmt:
		default:
			// A statement the backend does not compile anyway.
			return true
		}
	}
	return false
}

// declared reports whether slot is a binding the record says the body
// stores to or reads, which a self name is not.
func (g *gen) declared(slot int) bool {
	for _, res := range g.rec.Write {
		if (res.Kind == ResLocal || res.Kind == ResLocalInit || res.Kind == ResLocalCheck) && int(res.Index) == slot {
			return true
		}
	}
	for _, s := range g.rec.Funcs {
		if int(s) == slot {
			return true
		}
	}
	return slot < g.fn.ParamCount
}

func (g *gen) stmts(list []ast.Stmt) {
	g.enterBlock(list)
	for _, s := range list {
		if g.err != "" {
			return
		}
		g.stmt(s)
	}
}

// enterBlock puts the lexical bindings a block declares that start in their
// dead zone back in it, as each entry of the block makes them anew.
func (g *gen) enterBlock(list []ast.Stmt) {
	for _, s := range list {
		vd, ok := s.(*ast.VarDecl)
		if !ok || vd.Kind == ast.DeclVar {
			continue
		}
		for _, d := range vd.Decls {
			id, ok := d.Target.(*ast.Ident)
			if !ok {
				continue
			}
			res := g.rec.Write[id]
			if res.Kind != ResLocalInit {
				continue
			}
			slot := int(res.Index)
			if g.inLoop() && g.fn.Locals[slot].Captured {
				// Each turn's binding is a new one a closure keeps.
				g.fail("a captured lexical binding in a loop")
				return
			}
			if g.tdz[slot] || g.fn.Locals[slot].Captured {
				// A closure may read it before its declaration runs.
				g.emit(bytecode.MakeR(bytecode.RLoadUninit, 0, r16(int(g.slotReg[slot])), 0, 0))
			}
		}
	}
}

func (g *gen) inLoop() bool {
	for _, l := range g.loops {
		if l.isLoop {
			return true
		}
	}
	return false
}

func (g *gen) stmt(s ast.Stmt) {
	mark := g.top
	defer func() { g.top = mark }()
	label := g.label
	g.label = ""
	switch n := s.(type) {
	case *ast.ExprStmt:
		g.discard(n.X)
	case *ast.EmptyStmt:
	case *ast.BlockStmt:
		g.stmts(n.Body)
	case *ast.VarDecl:
		g.varDecl(n)
	case *ast.FuncDecl:
		// Hoisted to the body's start; one in a block is the stack form's.
		if !g.isTopLevel(n) {
			g.fail("a function declaration in a block")
		}
	case *ast.ReturnStmt:
		if n.Arg == nil {
			g.emit(bytecode.MakeR(bytecode.RRetUndef, 0, 0, 0, 0))
			return
		}
		g.ret(n.Arg)
	case *ast.IfStmt:
		els := g.newMark()
		g.cond(n.Test, false, els)
		g.stmt(n.Cons)
		if n.Alt == nil {
			g.bind(els)
			return
		}
		end := g.newMark()
		g.jump(bytecode.MakeR(bytecode.RJmp, 0, 0, 0, 0), end, false)
		g.bind(els)
		g.stmt(n.Alt)
		g.bind(end)
	case *ast.ForStmt:
		g.forStmt(n, label)
	case *ast.WhileStmt:
		l := g.pushLoop(label)
		test := g.newMark()
		body := g.newMark()
		l.cont = test
		g.jump(bytecode.MakeR(bytecode.RJmp, 0, 0, 0, 0), test, false)
		g.bind(body)
		g.stmt(n.Body)
		g.bind(test)
		g.cond(n.Test, true, body)
		g.bind(l.brk)
		g.popLoop()
	case *ast.DoWhileStmt:
		l := g.pushLoop(label)
		body := g.newMark()
		g.bind(body)
		g.stmt(n.Body)
		g.bind(l.cont)
		g.cond(n.Test, true, body)
		g.bind(l.brk)
		g.popLoop()
	case *ast.BreakStmt:
		l := g.findLoop(n.Label, false)
		if l == nil {
			g.fail("a break without its target")
			return
		}
		g.jump(bytecode.MakeR(bytecode.RJmp, 0, 0, 0, 0), l.brk, false)
	case *ast.ContinueStmt:
		l := g.findLoop(n.Label, true)
		if l == nil {
			g.fail("a continue without its loop")
			return
		}
		g.jump(bytecode.MakeR(bytecode.RJmp, 0, 0, 0, 0), l.cont, false)
	case *ast.ThrowStmt:
		r := g.expr(n.Arg, -1)
		g.at(n.Start)
		g.emit(bytecode.MakeR(bytecode.RThrow, 0, r16(r), 0, 0))
	case *ast.SwitchStmt:
		g.switchStmt(n, label)
	case *ast.LabeledStmt:
		switch n.Body.(type) {
		case *ast.ForStmt, *ast.WhileStmt, *ast.DoWhileStmt:
			g.label = n.Label
			g.stmt(n.Body)
		default:
			g.fail("a labelled statement that is not a loop")
		}
	default:
		g.fail(fmt.Sprintf("a statement the backend does not compile (%T)", s))
	}
}

// ret returns e: a conditional's each arm returns itself, with no move to
// a register the two share and no jump to the return.
func (g *gen) ret(e ast.Expr) {
	if c, ok := e.(*ast.Conditional); ok {
		els := g.newMark()
		g.cond(c.Test, false, els)
		mark := g.top
		g.ret(c.Cons)
		g.top = mark
		g.bind(els)
		g.ret(c.Alt)
		return
	}
	r := g.expr(e, -1)
	g.emit(bytecode.MakeR(bytecode.RRet, 0, r16(r), 0, 0))
}

func (g *gen) isTopLevel(fd *ast.FuncDecl) bool {
	for _, s := range g.lit.Body {
		if s == ast.Stmt(fd) {
			return true
		}
	}
	return false
}

func (g *gen) pushLoop(label string) *loop {
	l := &loop{label: label, brk: g.newMark(), cont: g.newMark(), isLoop: true}
	g.loops = append(g.loops, l)
	return l
}

func (g *gen) popLoop() { g.loops = g.loops[:len(g.loops)-1] }

func (g *gen) findLoop(label string, cont bool) *loop {
	for i := len(g.loops) - 1; i >= 0; i-- {
		l := g.loops[i]
		if cont && !l.isLoop {
			// A switch is no continue's target.
			if label != "" && l.label == label {
				return nil
			}
			continue
		}
		if label == "" || l.label == label {
			return l
		}
	}
	return nil
}

func (g *gen) forStmt(n *ast.ForStmt, label string) {
	// The head's bindings are each turn's own where a closure captures
	// them; the check is made as the loop is entered.
	l := g.pushLoop(label)
	if n.Init != nil {
		switch init := n.Init.(type) {
		case *ast.VarDecl:
			g.enterBlock([]ast.Stmt{init})
			g.varDecl(init)
		case *ast.ExprStmt:
			g.discard(init.X)
		default:
			g.fail("a for loop's head")
		}
	}
	g.top = g.topAfterBindings()
	test := g.newMark()
	body := g.newMark()
	g.jump(bytecode.MakeR(bytecode.RJmp, 0, 0, 0, 0), test, false)
	g.bind(body)
	g.stmt(n.Body)
	g.bind(l.cont)
	if n.Update != nil {
		mark := g.top
		g.discard(n.Update)
		g.top = mark
	}
	g.bind(test)
	if n.Test != nil {
		mark := g.top
		g.cond(n.Test, true, body)
		g.top = mark
	} else {
		g.jump(bytecode.MakeR(bytecode.RJmp, 0, 0, 0, 0), body, false)
	}
	g.bind(l.brk)
	g.popLoop()
}

// topAfterBindings is the first temporary register.
func (g *gen) topAfterBindings() int { return regParams + g.fn.ParamCount + g.inits }

func (g *gen) varDecl(n *ast.VarDecl) {
	if n.Kind != ast.DeclVar && n.Kind != ast.DeclLet && n.Kind != ast.DeclConst {
		g.fail("a using declaration")
		return
	}
	for _, d := range n.Decls {
		id, ok := d.Target.(*ast.Ident)
		if !ok {
			g.fail("a destructuring declaration")
			return
		}
		if d.Init == nil {
			if n.Kind == ast.DeclVar {
				continue
			}
			res, ok := g.rec.Write[id]
			if !ok || res.Kind != ResLocalInit {
				g.fail("a declaration without a binding")
				return
			}
			g.emit(bytecode.MakeR(bytecode.RLoadUndef, 0, r16(int(g.slotReg[res.Index])), 0, 0))
			continue
		}
		g.storeIdent(id, d.Init, true)
		g.top = g.topAfterBindings()
	}
}

// --- expressions ----------------------------------------------------------

// discard compiles e for its effects alone.
func (g *gen) discard(e ast.Expr) {
	switch n := e.(type) {
	case *ast.Update:
		g.update(n, false, -1)
		return
	case *ast.Assign:
		g.assign(n, -1, false)
		return
	case *ast.Sequence:
		for _, x := range n.Exprs {
			g.discard(x)
		}
		return
	}
	g.expr(e, -1)
}

// isBinding reports whether r is a binding's register rather than a
// temporary.
func (g *gen) isBinding(r int) bool { return r >= regParams && r < g.topAfterBindings() }

// slotOf is the stack slot whose register is r, or -1.
func (g *gen) slotOf(r int) int {
	for slot, reg := range g.slotReg {
		if int(reg) == r {
			return slot
		}
	}
	return -1
}

// guard copies a binding's register r to a temporary where evaluating rest
// could change the binding before r is used: an assignment or an update of
// the binding in rest, or anything that runs code where a closure may
// write it.
func (g *gen) guard(r int, rest ast.Expr) int {
	if !g.isBinding(r) || rest == nil {
		return r
	}
	slot := g.slotOf(r)
	if slot < 0 {
		return r
	}
	captured := g.fn.Locals[slot].Captured
	if g.writes(rest, slot) || captured && !pure(rest) {
		t := g.alloc()
		g.emit(bytecode.MakeR(bytecode.RMove, 0, r16(t), r16(r), 0))
		return t
	}
	return r
}

// pure reports whether evaluating e runs no code of the program's.
func pure(e ast.Expr) bool {
	switch e.(type) {
	case *ast.Ident, *ast.NumberLit, *ast.StringLit, *ast.BoolLit, *ast.NullLit, *ast.This:
		return true
	}
	return false
}

// writes reports whether e assigns or updates the binding in slot.
func (g *gen) writes(e ast.Expr, slot int) bool {
	found := false
	var walk func(n ast.Node)
	walk = func(n ast.Node) {
		if found || n == nil {
			return
		}
		switch x := n.(type) {
		case *ast.Assign:
			if id, ok := x.Target.(*ast.Ident); ok {
				if res, ok := g.rec.Write[id]; ok && (res.Kind == ResLocal || res.Kind == ResLocalInit || res.Kind == ResLocalCheck) && int(res.Index) == slot {
					found = true
					return
				}
			} else {
				walk(x.Target)
			}
			walk(x.Value)
		case *ast.Update:
			if id, ok := x.Operand.(*ast.Ident); ok {
				if res, ok := g.rec.Write[id]; ok && int(res.Index) == slot && res.Kind != ResGlobal {
					found = true
					return
				}
			}
			walk(x.Operand)
		case *ast.Binary:
			walk(x.Left)
			walk(x.Right)
		case *ast.Logical:
			walk(x.Left)
			walk(x.Right)
		case *ast.Unary:
			walk(x.Operand)
		case *ast.Conditional:
			walk(x.Test)
			walk(x.Cons)
			walk(x.Alt)
		case *ast.Call:
			walk(x.Callee)
			for _, a := range x.Args {
				walk(a)
			}
		case *ast.New:
			walk(x.Callee)
			for _, a := range x.Args {
				walk(a)
			}
		case *ast.Member:
			walk(x.Object)
			if x.Computed {
				walk(x.Property)
			}
		case *ast.Sequence:
			for _, y := range x.Exprs {
				walk(y)
			}
		case *ast.ArrayLit:
			for _, y := range x.Elements {
				walk(y)
			}
		case *ast.ObjectLit:
			for _, p := range x.Props {
				walk(p.Value)
			}
		}
	}
	walk(e)
	return found
}

// direct reports whether e may be compiled into a binding's register
// itself: an operation that reads all its operands before it writes its
// result, rather than one that builds its result in the register first.
func direct(e ast.Expr) bool {
	switch e.(type) {
	case *ast.ObjectLit, *ast.ArrayLit, *ast.Logical, *ast.Conditional, *ast.Assign, *ast.Sequence, *ast.Update:
		return false
	}
	return true
}

// into compiles e into register dst.
func (g *gen) into(e ast.Expr, dst int) {
	if g.isBinding(dst) && !direct(e) {
		r := g.expr(e, -1)
		g.emit(bytecode.MakeR(bytecode.RMove, 0, r16(dst), r16(r), 0))
		return
	}
	r := g.expr(e, dst)
	if r != dst {
		g.emit(bytecode.MakeR(bytecode.RMove, 0, r16(dst), r16(r), 0))
	}
}

// expr compiles e and returns the register holding its value: dst, or,
// for -1, wherever it is -- a binding's own register, read in place.
func (g *gen) expr(e ast.Expr, dst int) int {
	if g.err != "" {
		if dst >= 0 {
			return dst
		}
		return 0
	}
	switch n := e.(type) {
	case *ast.NumberLit:
		return g.number(n.Value, dst)
	case *ast.StringLit:
		d := g.dest(dst)
		if n.Value == "" {
			g.emit(bytecode.MakeR(bytecode.RLoadEmpty, 0, r16(d), 0, 0))
		} else {
			g.emit(bytecode.MakeRBC(bytecode.RLoadConst, 0, r16(d), g.constIdx(bytecode.Constant{Kind: bytecode.ConstString, Str: n.Value})))
		}
		return d
	case *ast.BoolLit:
		d := g.dest(dst)
		op := bytecode.RLoadFalse
		if n.Value {
			op = bytecode.RLoadTrue
		}
		g.emit(bytecode.MakeR(op, 0, r16(d), 0, 0))
		return d
	case *ast.NullLit:
		d := g.dest(dst)
		g.emit(bytecode.MakeR(bytecode.RLoadNull, 0, r16(d), 0, 0))
		return d
	case *ast.This:
		return g.place(regThis, dst)
	case *ast.Ident:
		return g.ident(n, dst)
	case *ast.Binary:
		return g.binary(n, dst)
	case *ast.Logical:
		return g.logical(n, dst)
	case *ast.Unary:
		return g.unary(n, dst)
	case *ast.Update:
		return g.update(n, true, dst)
	case *ast.Assign:
		return g.assign(n, dst, true)
	case *ast.Conditional:
		d := g.dest(dst)
		els, end := g.newMark(), g.newMark()
		g.cond(n.Test, false, els)
		g.into(n.Cons, d)
		g.jump(bytecode.MakeR(bytecode.RJmp, 0, 0, 0, 0), end, false)
		g.bind(els)
		g.into(n.Alt, d)
		g.bind(end)
		return d
	case *ast.Sequence:
		for _, x := range n.Exprs[:len(n.Exprs)-1] {
			g.discard(x)
		}
		return g.expr(n.Exprs[len(n.Exprs)-1], dst)
	case *ast.Member:
		return g.member(n, dst)
	case *ast.Call:
		return g.call(n, dst)
	case *ast.New:
		return g.newExpr(n, dst)
	case *ast.ObjectLit:
		return g.object(n, dst)
	case *ast.ArrayLit:
		return g.array(n, dst)
	case *ast.FuncLit:
		d := g.dest(dst)
		g.closure(n, d)
		return d
	}
	g.fail(fmt.Sprintf("an expression the backend does not compile (%T)", e))
	return g.dest(dst)
}

// place is a value in register r, moved to dst if there is one.
func (g *gen) place(r, dst int) int {
	if dst < 0 || dst == r {
		return r
	}
	g.emit(bytecode.MakeR(bytecode.RMove, 0, r16(dst), r16(r), 0))
	return dst
}

// isNullish reports whether e is the literal null.
func isNullish(e ast.Expr) bool {
	_, ok := e.(*ast.NullLit)
	return ok
}

func smallInt(v float64) (int32, bool) {
	i := int32(v)
	return i, float64(i) == v && !(v == 0 && math.Signbit(v))
}

func imm16(e ast.Expr) (int32, bool) {
	n, ok := e.(*ast.NumberLit)
	if !ok {
		return 0, false
	}
	i, ok := smallInt(n.Value)
	return i, ok && i >= -1<<15 && i < 1<<15
}

func (g *gen) number(v float64, dst int) int {
	d := g.dest(dst)
	if i, ok := smallInt(v); ok {
		g.emit(bytecode.MakeRBC(bytecode.RLoadInt, 0, r16(d), uint32(i)))
		return d
	}
	g.emit(bytecode.MakeRBC(bytecode.RLoadConst, 0, r16(d), g.constIdx(bytecode.Constant{Kind: bytecode.ConstNumber, Num: v})))
	return d
}

func (g *gen) ident(n *ast.Ident, dst int) int {
	res, ok := g.rec.Read[n]
	if !ok {
		g.fail("a name the record does not resolve")
		return g.dest(dst)
	}
	switch res.Kind {
	case ResLocal, ResLocalCheck:
		r := int(g.slotReg[res.Index])
		if res.Kind == ResLocalCheck {
			g.at(int(n.Start))
			g.emit(bytecode.MakeR(bytecode.RCheckRef, 0, r16(r), uint16(res.Index), 0))
		}
		return g.place(r, dst)
	case ResUpval:
		d := g.dest(dst)
		g.emit(bytecode.MakeRBC(bytecode.RGetUpval, 0, r16(d), res.Index))
		return d
	case ResGlobal:
		d := g.dest(dst)
		g.at(int(n.Start))
		name := g.name16(n.Name)
		g.emit(bytecode.MakeR(bytecode.RGetGlobal, 0, r16(d), name, g.globalSite()))
		return d
	case ResCallee:
		return g.place(regCallee, dst)
	case ResLiteral:
		if res.Lit == nil {
			d := g.dest(dst)
			g.emit(bytecode.MakeR(bytecode.RLoadUndef, 0, r16(d), 0, 0))
			return d
		}
		return g.expr(res.Lit, dst)
	}
	g.fail("a name the backend does not read")
	return g.dest(dst)
}

// globalSite is a cache site of a global's read or write, which keeps no
// name of the site table's.
func (g *gen) globalSite() uint16 { return g.site(0) }

var binOps = map[string]bytecode.Op{
	"+": bytecode.OpAdd, "-": bytecode.OpSub, "*": bytecode.OpMul, "/": bytecode.OpDiv, "%": bytecode.OpMod,
	"**": bytecode.OpPow, "&": bytecode.OpBitAnd, "|": bytecode.OpBitOr, "^": bytecode.OpBitXor,
	"<<": bytecode.OpShl, ">>": bytecode.OpShr, ">>>": bytecode.OpUShr,
	"==": bytecode.OpEq, "!=": bytecode.OpNe, "===": bytecode.OpStrictEq, "!==": bytecode.OpStrictNe,
	"<": bytecode.OpLt, "<=": bytecode.OpLe, ">": bytecode.OpGt, ">=": bytecode.OpGe,
}

// regOps are the operators of two registers with an operation of their own.
var regOps = map[string]bytecode.ROp{
	"+": bytecode.RAdd, "-": bytecode.RSub, "*": bytecode.RMul,
	"&": bytecode.RBitAnd, "|": bytecode.RBitOr, "^": bytecode.RBitXor,
	"<<": bytecode.RShl, ">>": bytecode.RShr, ">>>": bytecode.RUShr,
}

// immOps are the operators of a register and an immediate with one.
var immOps = map[string]bytecode.ROp{
	"+": bytecode.RAddI, "-": bytecode.RSubI,
	"&": bytecode.RAndI, "|": bytecode.ROrI, "^": bytecode.RXorI,
	"<<": bytecode.RShlI, ">>": bytecode.RShrI, ">>>": bytecode.RUShrI,
}

func (g *gen) binary(n *ast.Binary, dst int) int {
	switch n.Op {
	case "instanceof", "in":
		a := g.guard(g.expr(n.Left, -1), n.Right)
		b := g.expr(n.Right, -1)
		d := g.dest(dst)
		g.at(n.Start)
		if n.Op == "in" {
			g.emit(bytecode.MakeR(bytecode.RIn, 0, r16(d), r16(a), r16(b)))
			return d
		}
		at := g.emit(bytecode.MakeR(bytecode.RInstanceOf, 0, r16(d), r16(a), r16(b)))
		g.aux(at, uint32(g.site(0)))
		return d
	}
	op, ok := binOps[n.Op]
	if !ok {
		g.fail("the operator " + n.Op)
		return g.dest(dst)
	}
	if _, isCmp := g.compareJump(n.Op, true); isCmp {
		// A comparison's value, through its compare-and-branch, whose fast
		// path the generic operator has not: load true, branch over the
		// load of false.
		d := g.dest(dst)
		if g.isBinding(d) {
			d = g.alloc()
		}
		done := g.newMark()
		yes := g.newMark()
		g.cond(n, true, yes)
		g.emit(bytecode.MakeR(bytecode.RLoadFalse, 0, r16(d), 0, 0))
		g.jump(bytecode.MakeR(bytecode.RJmp, 0, 0, 0, 0), done, false)
		g.bind(yes)
		g.emit(bytecode.MakeR(bytecode.RLoadTrue, 0, r16(d), 0, 0))
		g.bind(done)
		return g.place(d, dst)
	}
	return g.operate(n.Op, op, n.Left, n.Right, -1, dst, n.Start)
}

// operate is left op right into dst; a is left's register, where it has
// been evaluated already (a compound assignment's target), or -1.
func (g *gen) operate(text string, op bytecode.Op, left, right ast.Expr, a, dst, pos int) int {
	if a < 0 {
		a = g.guard(g.expr(left, -1), right)
	}
	if imm, ok := imm16(right); ok {
		if rop, ok := immOps[text]; ok {
			d := g.dest(dst)
			g.at(pos)
			g.emit(bytecode.MakeR(rop, 0, r16(d), r16(a), uint16(int16(imm))))
			return d
		}
	}
	b := g.expr(right, -1)
	d := g.dest(dst)
	g.at(pos)
	if rop, ok := regOps[text]; ok {
		g.emit(bytecode.MakeR(rop, 0, r16(d), r16(a), r16(b)))
	} else {
		g.emit(bytecode.MakeR(bytecode.RBin, uint8(op), r16(d), r16(a), r16(b)))
	}
	return d
}

func (g *gen) logical(n *ast.Logical, dst int) int {
	d := g.dest(dst)
	end := g.newMark()
	g.into(n.Left, d)
	switch n.Op {
	case "&&":
		g.jump(bytecode.MakeR(bytecode.RJmpF, 0, r16(d), 0, 0), end, false)
	case "||":
		g.jump(bytecode.MakeR(bytecode.RJmpT, 0, r16(d), 0, 0), end, false)
	default:
		g.fail("??")
		return d
	}
	g.into(n.Right, d)
	g.bind(end)
	return d
}

func (g *gen) unary(n *ast.Unary, dst int) int {
	switch n.Op {
	case "-":
		if lit, ok := n.Operand.(*ast.NumberLit); ok {
			return g.number(-lit.Value, dst)
		}
		return g.unaryOp(bytecode.OpNeg, n.Operand, dst, n.Start)
	case "+":
		return g.unaryOp(bytecode.OpPos, n.Operand, dst, n.Start)
	case "!":
		return g.unaryOp(bytecode.OpNot, n.Operand, dst, n.Start)
	case "~":
		return g.unaryOp(bytecode.OpBitNot, n.Operand, dst, n.Start)
	case "typeof":
		if id, ok := n.Operand.(*ast.Ident); ok {
			if res := g.rec.Read[id]; res.Kind == ResGlobal || res.Kind == ResNone {
				// typeof of an undeclared global does not throw.
				g.fail("typeof of a global")
				return g.dest(dst)
			}
		}
		return g.unaryOp(bytecode.OpTypeOf, n.Operand, dst, n.Start)
	case "delete":
		m, ok := n.Operand.(*ast.Member)
		if !ok || m.Optional {
			if !ok {
				// delete of anything but a property is true, its operand
				// evaluated: a name is the stack form's.
				if _, isName := n.Operand.(*ast.Ident); !isName {
					g.discard(n.Operand)
					d := g.dest(dst)
					g.emit(bytecode.MakeR(bytecode.RLoadTrue, 0, r16(d), 0, 0))
					return d
				}
			}
			g.fail("delete of a name or an optional chain")
			return g.dest(dst)
		}
		if _, isSuper := m.Object.(*ast.Super); isSuper {
			g.fail("delete of a super property")
			return g.dest(dst)
		}
		o := g.expr(m.Object, -1)
		var k int
		if m.Computed {
			o = g.guard(o, m.Property)
			k = g.expr(m.Property, -1)
		} else {
			name, ok := propName(m)
			if !ok {
				g.fail("a property name the backend does not compile")
				return g.dest(dst)
			}
			k = g.alloc()
			g.emit(bytecode.MakeRBC(bytecode.RLoadConst, 0, r16(k), g.constIdx(bytecode.Constant{Kind: bytecode.ConstString, Str: name})))
		}
		d := g.dest(dst)
		g.at(n.Start)
		g.emit(bytecode.MakeR(bytecode.RDelete, 0, r16(d), r16(o), r16(k)))
		return d
	case "void":
		g.discard(n.Operand)
		d := g.dest(dst)
		g.emit(bytecode.MakeR(bytecode.RLoadUndef, 0, r16(d), 0, 0))
		return d
	}
	g.fail("the unary operator " + n.Op)
	return g.dest(dst)
}

func (g *gen) unaryOp(op bytecode.Op, operand ast.Expr, dst, pos int) int {
	a := g.expr(operand, -1)
	d := g.dest(dst)
	g.at(pos)
	g.emit(bytecode.MakeR(bytecode.RUnary, uint8(op), r16(d), r16(a), 0))
	return d
}

// cond jumps to target where e's truthiness is when, and falls through
// otherwise.
func (g *gen) cond(e ast.Expr, when bool, target *mark) {
	mark := g.top
	defer func() { g.top = mark }()
	switch n := e.(type) {
	case *ast.Unary:
		if n.Op == "!" {
			g.cond(n.Operand, !when, target)
			return
		}
	case *ast.BoolLit:
		if n.Value == when {
			g.jump(bytecode.MakeR(bytecode.RJmp, 0, 0, 0, 0), target, false)
		}
		return
	case *ast.Logical:
		switch {
		case n.Op == "&&" && when:
			skip := g.newMark()
			g.cond(n.Left, false, skip)
			g.cond(n.Right, true, target)
			g.bind(skip)
			return
		case n.Op == "&&":
			g.cond(n.Left, false, target)
			g.cond(n.Right, false, target)
			return
		case n.Op == "||" && when:
			g.cond(n.Left, true, target)
			g.cond(n.Right, true, target)
			return
		case n.Op == "||":
			skip := g.newMark()
			g.cond(n.Left, true, skip)
			g.cond(n.Right, false, target)
			g.bind(skip)
			return
		}
	case *ast.Binary:
		if n.Op == "==" || n.Op == "!=" {
			// == null and != null: whether the value is null or undefined.
			other, isNull := n.Left, isNullish(n.Right)
			if !isNull {
				other, isNull = n.Right, isNullish(n.Left)
			}
			if isNull {
				a := g.expr(other, -1)
				op := bytecode.RJNull
				if (n.Op == "==") != when {
					op = bytecode.RJNotNull
				}
				g.at(n.Start)
				g.jump(bytecode.MakeR(op, 0, r16(a), 0, 0), target, true)
				return
			}
		}
		if op, ok := g.compareJump(n.Op, when); ok {
			a := g.guard(g.expr(n.Left, -1), n.Right)
			g.at(n.Start)
			if imm, ok := imm16(n.Right); ok {
				g.jump(bytecode.MakeR(op+immOffset(op), 0, r16(a), uint16(int16(imm)), 0), target, true)
				return
			}
			b := g.expr(n.Right, -1)
			g.at(n.Start)
			g.jump(bytecode.MakeR(op, 0, r16(a), r16(b), 0), target, true)
			return
		}
	}
	r := g.expr(e, -1)
	op := bytecode.RJmpF
	if when {
		op = bytecode.RJmpT
	}
	g.jump(bytecode.MakeR(op, 0, r16(r), 0, 0), target, false)
}

// compareJump is the register form of the compare-and-branch that jumps
// when the comparison text is when.
func (g *gen) compareJump(text string, when bool) (bytecode.ROp, bool) {
	type pair struct{ yes, no bytecode.ROp }
	ops := map[string]pair{
		"<": {bytecode.RJLt, bytecode.RJNLt}, "<=": {bytecode.RJLe, bytecode.RJNLe},
		">": {bytecode.RJGt, bytecode.RJNGt}, ">=": {bytecode.RJGe, bytecode.RJNGe},
		"===": {bytecode.RJSEq, bytecode.RJSNe}, "!==": {bytecode.RJSNe, bytecode.RJSEq},
		"==": {bytecode.RJEq, bytecode.RJNe}, "!=": {bytecode.RJNe, bytecode.RJEq},
	}
	p, ok := ops[text]
	if !ok {
		return 0, false
	}
	if when {
		return p.yes, true
	}
	return p.no, true
}

// immOffset is how far a compare-and-branch's immediate form is from its
// register form.
func immOffset(op bytecode.ROp) bytecode.ROp {
	switch op {
	case bytecode.RJSEq, bytecode.RJSNe:
		return bytecode.RJSEqI - bytecode.RJSEq
	case bytecode.RJEq, bytecode.RJNe:
		return bytecode.RJEqI - bytecode.RJEq
	}
	return bytecode.RJLtI - bytecode.RJLt
}

// --- stores ---------------------------------------------------------------

// storeIdent stores value into the binding id names; init says it is the
// binding's declaration.
func (g *gen) storeIdent(id *ast.Ident, value ast.Expr, init bool) int {
	res, ok := g.rec.Write[id]
	if !ok {
		g.fail("a store the record does not resolve")
		return 0
	}
	switch res.Kind {
	case ResLocal, ResLocalInit:
		r := int(g.slotReg[res.Index])
		g.into(value, r)
		return r
	case ResUpval:
		v := g.expr(value, -1)
		g.emit(bytecode.MakeRBC(bytecode.RSetUpval, 0, r16(v), res.Index))
		return v
	case ResGlobal:
		if g.fn.Strict {
			// Whether the name resolves is settled before the value is
			// evaluated, which the stack form's checks do.
			g.fail("a strict global store")
			return 0
		}
		v := g.expr(value, -1)
		g.at(int(id.Start))
		g.emit(bytecode.MakeR(bytecode.RSetGlobal, 0, r16(v), g.name16(id.Name), g.globalSite()))
		return v
	case ResIgnore:
		return g.expr(value, -1)
	}
	g.fail("a store the backend does not compile")
	return 0
}

func (g *gen) assign(n *ast.Assign, dst int, keep bool) int {
	if n.Op == "=" {
		switch t := n.Target.(type) {
		case *ast.Ident:
			r := g.storeIdent(t, n.Value, false)
			return g.place(r, dst)
		case *ast.Member:
			o := g.guard(g.expr(t.Object, -1), n.Value)
			if t.Computed {
				if k, ok := imm16(t.Property); ok && k >= 0 {
					v := g.expr(n.Value, -1)
					g.at(int(n.Start))
					g.emit(bytecode.MakeR(bytecode.RSetIndexI, 0, r16(o), uint16(k), r16(v)))
					return g.place(v, dst)
				}
				k := g.guard(g.expr(t.Property, -1), n.Value)
				v := g.expr(n.Value, -1)
				g.at(int(n.Start))
				g.emit(bytecode.MakeR(bytecode.RSetIndex, 0, r16(o), r16(k), r16(v)))
				return g.place(v, dst)
			}
			name, ok := propName(t)
			if !ok {
				g.fail("a property name the backend does not compile")
				return 0
			}
			v := g.expr(n.Value, -1)
			g.at(int(n.Start))
			g.emit(bytecode.MakeR(bytecode.RSetProp, 0, r16(o), r16(v), g.site(g.nameIdx(name))))
			return g.place(v, dst)
		}
		g.fail("an assignment target the backend does not compile")
		return 0
	}
	text := n.Op[:len(n.Op)-1]
	op, ok := binOps[text]
	if !ok {
		g.fail("a logical assignment")
		return 0
	}
	switch t := n.Target.(type) {
	case *ast.Ident:
		res := g.rec.Write[t]
		if res.Kind == ResLocal {
			r := int(g.slotReg[res.Index])
			if rr := g.rec.Read[t]; rr.Kind != ResLocal || rr.Index != res.Index {
				g.fail("a compound assignment's read")
				return 0
			}
			a := g.guard(r, n.Value)
			g.operate(text, op, nil, n.Value, a, r, int(n.Start))
			return g.place(r, dst)
		}
		old := g.guard(g.readForWrite(t, -1), n.Value)
		v := g.operate(text, op, nil, n.Value, old, -1, int(n.Start))
		g.storeValue(t, v)
		return g.place(v, dst)
	case *ast.Member:
		o := g.guard(g.expr(t.Object, -1), n.Value)
		var k int
		var name uint32
		if t.Computed {
			k = g.guard(g.expr(t.Property, -1), n.Value)
		} else {
			s, ok := propName(t)
			if !ok {
				g.fail("a property name the backend does not compile")
				return 0
			}
			name = g.nameIdx(s)
		}
		cur := g.alloc()
		g.at(t.Property.Pos())
		if t.Computed {
			g.emit(bytecode.MakeR(bytecode.RGetIndex, 0, r16(cur), r16(o), r16(k)))
		} else {
			g.emit(bytecode.MakeR(bytecode.RGetProp, 0, r16(cur), r16(o), g.site(name)))
		}
		v := g.operate(text, op, nil, n.Value, cur, cur, int(n.Start))
		g.at(int(n.Start))
		if t.Computed {
			g.emit(bytecode.MakeR(bytecode.RSetIndex, 0, r16(o), r16(k), r16(v)))
		} else {
			g.emit(bytecode.MakeR(bytecode.RSetProp, 0, r16(o), r16(v), g.site(name)))
		}
		return g.place(v, dst)
	}
	g.fail("a compound assignment target")
	return 0
}

// readForWrite reads the binding a compound assignment or an update of
// id names, which the stack form may have read with the store's own
// resolution rather than a read of its own.
func (g *gen) readForWrite(id *ast.Ident, dst int) int {
	if _, ok := g.rec.Read[id]; ok {
		return g.ident(id, dst)
	}
	res := g.rec.Write[id]
	d := g.dest(dst)
	switch res.Kind {
	case ResUpval:
		g.emit(bytecode.MakeRBC(bytecode.RGetUpval, 0, r16(d), res.Index))
	case ResGlobal:
		g.at(int(id.Start))
		g.emit(bytecode.MakeR(bytecode.RGetGlobal, 0, r16(d), g.name16(id.Name), g.globalSite()))
	default:
		g.fail("a compound assignment's read")
	}
	return d
}

// storeValue stores register v into the binding id names.
func (g *gen) storeValue(id *ast.Ident, v int) {
	res := g.rec.Write[id]
	switch res.Kind {
	case ResLocal:
		g.emit(bytecode.MakeR(bytecode.RMove, 0, r16(int(g.slotReg[res.Index])), r16(v), 0))
	case ResUpval:
		g.emit(bytecode.MakeRBC(bytecode.RSetUpval, 0, r16(v), res.Index))
	case ResGlobal:
		if g.fn.Strict {
			g.fail("a strict global store")
			return
		}
		g.at(int(id.Start))
		g.emit(bytecode.MakeR(bytecode.RSetGlobal, 0, r16(v), g.name16(id.Name), g.globalSite()))
	default:
		g.fail("a store the backend does not compile")
	}
}

func (g *gen) update(n *ast.Update, keep bool, dst int) int {
	step := bytecode.RInc
	if n.Op == "--" {
		step = bytecode.RDec
	}
	switch t := n.Operand.(type) {
	case *ast.Ident:
		res := g.rec.Write[t]
		if res.Kind == ResLocal {
			r := int(g.slotReg[res.Index])
			g.at(n.Start)
			if !keep || n.Prefix {
				g.emit(bytecode.MakeR(step, 0, r16(r), r16(r), 0))
				return g.place(r, dst)
			}
			old := g.dest(dst)
			if old == r {
				old = g.alloc()
			}
			post := bytecode.RPostInc
			if step == bytecode.RDec {
				post = bytecode.RPostDec
			}
			g.emit(bytecode.MakeR(post, 0, r16(old), r16(r), 0))
			return g.place(old, dst)
		}
		cur := g.readForWrite(t, g.alloc())
		return g.stepped(n, cur, step, keep, dst, func(v int) { g.storeValue(t, v) })
	case *ast.Member:
		o := g.expr(t.Object, -1)
		cur := g.alloc()
		if t.Computed {
			k := g.expr(t.Property, -1)
			g.at(t.Property.Pos())
			g.emit(bytecode.MakeR(bytecode.RGetIndex, 0, r16(cur), r16(o), r16(k)))
			return g.stepped(n, cur, step, keep, dst, func(v int) {
				g.at(n.Start)
				g.emit(bytecode.MakeR(bytecode.RSetIndex, 0, r16(o), r16(k), r16(v)))
			})
		}
		s, ok := propName(t)
		if !ok {
			g.fail("a property name the backend does not compile")
			return 0
		}
		name := g.nameIdx(s)
		g.at(t.Property.Pos())
		g.emit(bytecode.MakeR(bytecode.RGetProp, 0, r16(cur), r16(o), g.site(name)))
		return g.stepped(n, cur, step, keep, dst, func(v int) {
			g.at(n.Start)
			g.emit(bytecode.MakeR(bytecode.RSetProp, 0, r16(o), r16(v), g.site(name)))
		})
	}
	g.fail("an update target")
	return 0
}

// stepped is an update of the value in cur, stored by store.
func (g *gen) stepped(n *ast.Update, cur int, step bytecode.ROp, keep bool, dst int, store func(int)) int {
	g.at(n.Start)
	if keep && !n.Prefix {
		old := g.alloc()
		g.emit(bytecode.MakeR(bytecode.RUnary, uint8(bytecode.OpToNumeric), r16(old), r16(cur), 0))
		g.emit(bytecode.MakeR(step, 0, r16(cur), r16(old), 0))
		store(cur)
		return g.place(old, dst)
	}
	g.emit(bytecode.MakeR(step, 0, r16(cur), r16(cur), 0))
	store(cur)
	return g.place(cur, dst)
}

// --- properties, calls, literals ------------------------------------------

func propName(m *ast.Member) (string, bool) {
	if m.Computed || m.Optional {
		return "", false
	}
	id, ok := m.Property.(*ast.Ident)
	if !ok {
		return "", false
	}
	return id.Name, true
}

func (g *gen) member(n *ast.Member, dst int) int {
	if n.Optional {
		g.fail("an optional chain")
		return g.dest(dst)
	}
	o := g.expr(n.Object, -1)
	if n.Computed {
		if k, ok := imm16(n.Property); ok && k >= 0 {
			d := g.dest(dst)
			g.at(n.Property.Pos())
			g.emit(bytecode.MakeR(bytecode.RGetIndexI, 0, r16(d), r16(o), uint16(k)))
			return d
		}
		o = g.guard(o, n.Property)
		k := g.expr(n.Property, -1)
		d := g.dest(dst)
		g.at(n.Property.Pos())
		g.emit(bytecode.MakeR(bytecode.RGetIndex, 0, r16(d), r16(o), r16(k)))
		return d
	}
	name, ok := propName(n)
	if !ok {
		g.fail("a property the backend does not compile")
		return g.dest(dst)
	}
	d := g.dest(dst)
	g.at(n.Property.Pos())
	if name == "length" {
		g.emit(bytecode.MakeR(bytecode.RGetLength, 0, r16(d), r16(o), 0))
		return d
	}
	g.emit(bytecode.MakeR(bytecode.RGetProp, 0, r16(d), r16(o), g.site(g.nameIdx(name))))
	return d
}

func (g *gen) args(args []ast.Expr, at int) {
	for i, a := range args {
		if _, ok := a.(*ast.Spread); ok {
			g.fail("a spread argument")
			return
		}
		g.into(a, at+i)
	}
}

func (g *gen) call(n *ast.Call, dst int) int {
	if n.Optional || len(n.Args) > 255 {
		g.fail("an optional call or too many arguments")
		return g.dest(dst)
	}
	argc := len(n.Args)
	w := g.allocN(2 + argc)
	op := bytecode.RCallW
	switch c := n.Callee.(type) {
	case *ast.Member:
		if c.Optional {
			g.fail("an optional chain")
			return w
		}
		if _, ok := c.Object.(*ast.Super); ok {
			g.fail("super")
			return w
		}
		g.into(c.Object, w+1)
		if c.Computed {
			k := g.expr(c.Property, -1)
			g.at(c.Property.Pos())
			g.emit(bytecode.MakeR(bytecode.RGetIndex, 0, r16(w), r16(w+1), r16(k)))
		} else {
			name, ok := propName(c)
			if !ok {
				g.fail("a method name the backend does not compile")
				return w
			}
			g.at(c.Property.Pos())
			g.emit(bytecode.MakeR(bytecode.RGetMethod, 0, r16(w), r16(w+1), g.site(g.nameIdx(name))))
		}
		op = bytecode.RCallMW
	case *ast.Super:
		g.fail("super()")
		return w
	case *ast.Ident:
		if c.Name == "eval" {
			if res := g.rec.Read[c]; res.Kind == ResGlobal {
				g.fail("a call of eval")
				return w
			}
		}
		g.into(c, w)
	default:
		g.into(n.Callee, w)
	}
	g.args(n.Args, w+2)
	d := dst
	if d < 0 {
		d = w
	}
	g.at(int(n.Start))
	g.emit(bytecode.MakeR(op, uint8(argc), r16(w), r16(d), g.callSite()))
	return d
}

// aux gives the instruction at pc its extra operand (RegCode.Aux).
func (g *gen) aux(pc int, v uint32) {
	if g.auxes == nil {
		g.auxes = map[int]uint32{}
	}
	g.auxes[pc] = v
}

// callSite is a call's cache: the function it called last, and how.
func (g *gen) callSite() uint16 {
	s := g.calls
	g.calls++
	if s >= limit16 {
		g.fail("a call site past 16 bits")
	}
	return uint16(s)
}

func (g *gen) newExpr(n *ast.New, dst int) int {
	if len(n.Args) > 255 {
		g.fail("too many arguments")
		return g.dest(dst)
	}
	argc := len(n.Args)
	w := g.allocN(2 + argc)
	g.into(n.Callee, w)
	g.args(n.Args, w+2)
	d := dst
	if d < 0 {
		d = w
	}
	g.at(n.Start)
	g.emit(bytecode.MakeR(bytecode.RNewW, uint8(argc), r16(w), r16(d), 0))
	return d
}

func (g *gen) object(n *ast.ObjectLit, dst int) int {
	if len(n.Props) >= limit16 {
		g.fail("too many properties")
		return g.dest(dst)
	}
	d := g.dest(dst)
	g.at(n.Start)
	g.emit(bytecode.MakeR(bytecode.RNewObject, 0, r16(d), uint16(len(n.Props)), g.site(0)))
	seen := map[string]bool{}
	for _, p := range n.Props {
		if p.Kind != ast.PropInit || p.Computed || p.Method {
			g.fail("an object literal's property the backend does not compile")
			return d
		}
		var key string
		switch k := p.Key.(type) {
		case *ast.Ident:
			key = k.Name
		case *ast.StringLit:
			key = k.Value
		default:
			g.fail("an object literal's key the backend does not compile")
			return d
		}
		if key == "__proto__" || seen[key] {
			g.fail("an object literal's __proto__ or a key twice")
			return d
		}
		seen[key] = true
		v := g.expr(p.Value, -1)
		g.emit(bytecode.MakeR(bytecode.RDefineField, 0, r16(d), r16(v), g.name16(key)))
	}
	return d
}

func (g *gen) array(n *ast.ArrayLit, dst int) int {
	if len(n.Elements) >= limit16 {
		g.fail("too many elements")
		return g.dest(dst)
	}
	base := g.allocN(len(n.Elements))
	for i, e := range n.Elements {
		if e == nil {
			g.fail("a hole")
			return base
		}
		if _, ok := e.(*ast.Spread); ok {
			g.fail("a spread element")
			return base
		}
		g.into(e, base+i)
	}
	d := g.dest(dst)
	g.emit(bytecode.MakeR(bytecode.RNewArray, 0, r16(d), r16(base), uint16(len(n.Elements))))
	return d
}

// closure makes a closure of the nested function lit into register d.
func (g *gen) closure(lit *ast.FuncLit, d int) {
	k := -1
	for i, c := range g.fn.Constants {
		if c.Kind == bytecode.ConstFunction && c.Fn != nil && int(c.Fn.Start) == lit.Start {
			k = i
			break
		}
	}
	if k < 0 {
		g.fail("a function the stack form has no template of")
		return
	}
	tmpl := g.fn.Constants[k].Fn
	caps := make([]uint16, len(tmpl.Upvalues))
	for i, u := range tmpl.Upvalues {
		if !u.FromParent {
			continue
		}
		r := g.slotReg[u.Index]
		if r < 0 {
			g.fail("a capture of a binding without a register")
			return
		}
		caps[i] = uint16(r)
	}
	g.caps[k] = caps
	g.emit(bytecode.MakeRBC(bytecode.RClosureW, 0, r16(d), uint32(k)))
}

// --- the result -----------------------------------------------------------

func (g *gen) finish() *bytecode.RegCode {
	// Each instruction's position, as the stack form's instruction at the
	// same place: what a frame's pc names for a stack trace.
	type entry struct {
		pos int32
		pc  uint32
	}
	var table []entry
	for i, l := range g.fn.Lines {
		end := uint32(len(g.fn.Code))
		if i+1 < len(g.fn.Lines) {
			end = g.fn.Lines[i+1].PC
		}
		if l.PC < end {
			table = append(table, entry{l.Pos, l.PC})
		}
	}
	sort.SliceStable(table, func(i, j int) bool { return table[i].pos < table[j].pos })
	src := make([]uint32, len(g.code))
	for i, p := range g.pos {
		j := sort.Search(len(table), func(j int) bool { return table[j].pos > p }) - 1
		if j >= 0 && int(table[j].pc) < len(g.fn.Code) {
			src[i] = table[j].pc
		}
	}
	for len(g.siteName) <= int(g.fn.PropSites) {
		g.siteName = append(g.siteName, 0)
	}
	var aux []uint32
	if len(g.auxes) > 0 {
		aux = make([]uint32, len(g.code))
		for pc, v := range g.auxes {
			aux[pc] = v
		}
	}
	rc := &bytecode.RegCode{
		Aux: aux, UsesArguments: g.fn.UsesArguments || g.fn.MakesArguments,
		Code: g.code, NumRegs: g.max + 1, SrcPC: src, SiteName: g.siteName, This: -1,
		Wide: true, Params: g.fn.ParamCount, Inits: g.inits, Caps: g.caps, CallSites: g.calls,
	}
	if why := Verify(rc, g.fn); why != "" {
		Refuse(g.fn, why)
		return nil
	}
	return rc
}
