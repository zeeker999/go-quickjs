package vm

import (
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
)

// TestPureByTree pins which pure bodies are called as their register code:
// a closure register code makes is given its code's call at once (regPlan),
// a pure body's however short, as the register tier gave it before the old
// tiers went -- the tree tier's choice (pureByTree), which left a short one
// or one that calls frameless (pureCall), was the tree's alone. A call from
// inside another pure body is still evaluated frameless (pureInvoke).
func TestPureByTree(t *testing.T) {
	t.Parallel()
	const src = `
		var HELD = 4, RUN = 2;
		function T() { this.state = 0; this.v = 1 }
		T.prototype.held = function () { return (this.state & HELD) != 0 || this.state == RUN };
		T.prototype.mark = function () { this.state = this.state | HELD };
		T.prototype.one = function () { return 1 };
		T.prototype.viaHeld = function () { return this.held() };
		var t = new T(), n = 0;
		for (var i = 0; i < 300; i++) {
			if (t.held()) n++;
			if (t.viaHeld()) n += 2;
			n += t.one();
			if (i == 150) t.mark();
		}
		[n, T.prototype.held, T.prototype.mark, T.prototype.one, T.prototype.viaHeld]`
	r := New(Config{})
	v, err := r.Run(compileForTest(t, src))
	if err != nil {
		t.Fatal(err)
	}
	res := v.Object().elemsOf()
	if got := res[0].Number(); got != 300+149*3 {
		t.Errorf("n = %v, want %v", got, 300+149*3)
	}
	// Which of them are given their register code's call.
	for i, want := range []bool{true, true, true, true} {
		fd := res[i+1].Object().fn()
		name := []string{"held", "mark", "one", "viaHeld"}[i]
		if fd.closure.fn.RegCompiled().Leaf != bytecode.LeafPure {
			t.Fatalf("%s is not a pure body", name)
		}
		if got := fd.regCall != nil; got != want {
			t.Errorf("%s: given its register code's call = %v, want %v", name, got, want)
		}
	}
	r.Close()
}

// TestPureByTreeComparison pins that a pure body returning a comparison --
// six register instructions, the branch and the two loads its value takes
// besides the return -- is evaluated frameless (pureByTree false), as it
// was when the stack bytecode's four instructions were counted: called with
// a frame, BigInt64Array's every of x => x <= last ran natively at the
// interpreter's speed. A body of a dozen instructions goes to its code.
func TestPureByTreeComparison(t *testing.T) {
	t.Parallel()
	top := compileForTest(t, `function f(last) { return [(x) => x <= last, (x, y) => (x * 3 + y * 5 + last) * (x - y) - (y + 7) * (x + 9) * last] }`)
	var fs []*bytecode.Function
	for _, k := range top.Constants {
		if k.Fn != nil && k.Fn.Name == "f" {
			for _, c := range k.Fn.Constants {
				if c.Fn != nil {
					fs = append(fs, c.Fn)
				}
			}
		}
	}
	if len(fs) != 2 {
		t.Fatalf("%d arrows", len(fs))
	}
	for i, want := range []bool{false, true} {
		rc := fs[i].RegCode()
		if rc == nil || rc.Leaf != bytecode.LeafPure || rc.PureCalls {
			t.Fatalf("arrow %d is not a pure body that calls nothing: %v", i, rc)
		}
		if got := pureByTree(rc); got != want {
			t.Errorf("arrow %d, %d instructions: pureByTree %v, want %v", i, len(rc.Code), got, want)
		}
	}
}
