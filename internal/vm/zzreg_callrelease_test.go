package vm

import (
	"testing"
	"weak"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/compiler"
)

// TestRegCallCachesReleasedAtTurnEnd: a call site's cache in a template
// the program shares -- a library's each -- holds the closure it called
// last, and that closure what it captured. Once the turn that made them
// has ended, and the program let go of what it captured, it is
// collected: the turn's end empties the caches (releaseRegCalls). Held,
// a finished benchmark's whole graph had stayed live through the next.
func TestRegCallCachesReleasedAtTurnEnd(t *testing.T) {
	steps := []string{
		`function each(f) { return f(); }
		 var keep = { graph: [1, 2, 3] };
		 function use() { var o = keep; return each(() => o); }
		 use(); use(); use();
		 keep`,
		`keep = undefined; 0`,
	}
	compiler.SetRegisterForm(true)
	fns := make([]*bytecode.Function, len(steps))
	for i, s := range steps {
		fns[i] = compileForTest(t, s)
	}
	compiler.SetRegisterForm(bytecode.RegDefault())
	SetRegTier(true)
	defer SetRegTier(bytecode.RegDefault())
	before := RegsBuilt()
	r := New(Config{})
	defer r.Close()
	v, err := r.Run(fns[0])
	if err != nil {
		t.Fatal(err)
	}
	if !v.IsObject() {
		t.Fatalf("got %v, want the object", v)
	}
	ref := weak.Make(v.Object())
	v = Undefined
	if built := RegsBuilt() - before; built < 2 {
		t.Fatalf("only %d functions ran as register code", built)
	}
	held := false
	for _, w := range r.weakMaps.calls.codes {
		if cc := w.Value(); cc != nil {
			for _, e := range cc.regCalls {
				held = held || e.callee != nil
			}
		}
	}
	if held {
		t.Errorf("a call cache holds a callee after its turn ended")
	}
	if _, err := r.Run(fns[1]); err != nil {
		t.Fatal(err)
	}
	if !collected(r, ref) {
		t.Errorf("the object the last turn's closure captured survived it")
	}
}

// TestRegCallFeedbackOutlivesTurn: what a call site called is kept, held
// weakly, past the turn whose end let its cache go (regCallSeen): a
// function warmed in one turn and compiled in the next finds its calls'
// targets where they were (regCallsRestored), as V8's call feedback holds
// a target while it lives -- and a site whose callee is gone finds none.
func TestRegCallFeedbackOutlivesTurn(t *testing.T) {
	steps := []string{
		`function g() { return 1; }
		 var h = function () { return 2; };
		 function f() { return g() + h(); }
		 for (var i = 0; i < 10; i++) f();
		 f`,
		`h = undefined; 0`,
	}
	compiler.SetRegisterForm(true)
	fns := make([]*bytecode.Function, len(steps))
	for i, s := range steps {
		fns[i] = compileForTest(t, s)
	}
	compiler.SetRegisterForm(bytecode.RegDefault())
	SetRegTier(true)
	defer SetRegTier(bytecode.RegDefault())
	r := New(Config{})
	defer r.Close()
	v, err := r.Run(fns[0])
	if err != nil {
		t.Fatal(err)
	}
	cl := v.Object().fn().closure
	if len(cl.regCalls) < 2 {
		t.Fatalf("f has %d call caches, want 2", len(cl.regCalls))
	}
	for i, e := range cl.regCalls {
		if e.callee != nil {
			t.Fatalf("site %d's cache holds its callee after its turn ended", i)
		}
	}
	gv, err := r.getProp(r.global, r.atoms.intern("g"), Obj(r.global))
	if err != nil || !gv.IsObject() {
		t.Fatalf("g = %v, %v", gv, err)
	}
	hv, err := r.getProp(r.global, r.atoms.intern("h"), Obj(r.global))
	if err != nil || !hv.IsObject() {
		t.Fatalf("h = %v, %v", hv, err)
	}
	h := weak.Make(hv.Object())
	hv = Undefined
	if _, err := r.Run(fns[1]); err != nil {
		t.Fatal(err)
	}
	if !collected(r, h) {
		t.Fatal("h, which nothing but f's call site held, survived")
	}
	calls := r.regCallsRestored(cl, -1)
	var got []*Object
	for _, e := range calls {
		if e.callee != nil {
			got = append(got, e.callee)
		}
	}
	if len(got) != 1 || got[0] != gv.Object() {
		t.Errorf("f's call sites restored to %v, want g's alone", got)
	}
	if e := calls[0]; e.callee == gv.Object() && (e.cl == nil || e.rc == nil || e.kind&^regCallMarks != regCallThis) {
		t.Errorf("g's site restored as %+v, want its closure, code and a plain call's kind", e)
	}
}
