//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"fmt"
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/jit/mir"
)

// regLoopSrc's loops test at their end in register code, entered by a
// jump to the test: a sum stored once the loop is done, a float carried
// round, a test of two comparisons, one that reads a property and calls
// a method, a loop never entered, and nested loops.
const regLoopSrc = `
var total = 0;
var box = { n: 7, more(i) { return i < this.n * 40 } };
function sumTo(n) { var j, s = 0; for (j = 0; j < n; j++) s += j; total = s; return n }
function floats(n) { var a = 0.1, s = 0; for (var i = 0; i < n; i++) { s += a * a; a += 1.1 } return s }
function both(a, n, k) { var i = 0; while (i < n && a[i & 7] !== k) i++; return i }
function calls(o) { var s = 0; for (var i = 0; o.more(i); i++) s += i & 3; return s }
function never(n) { var s = 1.5; for (var i = n; i < 0; i++) s += i; return s + n }
function nested(n) { var s = 0; for (var i = 0; i < n; i++) for (var j = 0; j < 8; j++) s += i * j; return s }
function main(n) {
	return sumTo(n) + ":" + total + ":" + floats(n) + ":" + both([1, 2, 3, 4, 5, 6, 7, 8], n, 9) + ":" + both([1, 2, 3], n, 3) + ":" +
		calls(box) + ":" + never(n) + ":" + nested(n);
}
`

// A register loop is built with its body's first instruction its one
// header, its test copied before it (the compile package's loopTest), as
// the stack form's loops are: the numbers it carries are not boxed inside
// it -- a sum stored once it is done had been boxed at every turn, the
// test's block no loop's header -- and every answer is the interpreter's,
// with the collector idle and marking.
func TestJITRegLoopsOneHeader(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			listings := map[string]string{}
			mir.Annotate = true
			JITCodeHook = func(f JITFunction, _ []byte, c *mir.Code) { listings[f.Name] = c.Listing }
			defer func() { JITCodeHook, mir.Annotate = nil, false }()
			rts := regNativeRuntimes(t, regLoopSrc)
			var got [2]string
			for k, r := range rts {
				fn, _ := r.GetProp(Obj(r.Global()), r.Intern("main"))
				for i := range 30 {
					v, err := r.Call(fn, Undefined, []Value{Float(float64(200 + i%3))})
					got[k] = regNativeShow(r, v, err)
				}
			}
			const want = "202:20301:3304260.449999981:202:2:420:203.5:568428" // Node's
			if got[0] != got[1] || got[1] != want {
				t.Errorf("with the JIT %q, without %q, want %q", got[0], got[1], want)
			}
			for _, name := range []string{"sumTo", "floats", "nested"} {
				l, ok := listings[name]
				if !ok {
					t.Errorf("%s was not compiled", name)
					continue
				}
				if n := innermostLoopBoxes(l); n != 0 {
					t.Errorf("%s makes %d boxes inside its innermost loops:\n%s", name, n, l)
				}
			}
		})
	}
}

// A loop that carries an object escape analysis keeps unmade has no entry
// in its function's code (ssa.Func's Unentered), and a frame that waits
// there long goes on in code compiled for it apart (jitRegLoopCode), its
// entry kept (ssa.Func's FrameAt): both named by the header's register pc,
// as frame states name it. The register form had read Unentered's as the
// program's instructions and given FrameAt one, and so had never compiled
// code with the entry: a long loop called from Go ran in the interpreter
// to its end, every call.
func TestJITRegLoopEntryUnmade(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			rts := regNativeRuntimes(t, `var res;
function props(n) { var j, o = { x: 1, y: 2 }, s = 0; for (j = 0; j < n; j++) s += o.x + o.y; res = s; return n }`)
			// Compiled as the code asks, not as stress configures it.
			rts[0].jitStress = jitStressConfig{}
			var got [2]string
			for k, r := range rts {
				fn, _ := r.GetProp(Obj(r.Global()), r.Intern("props"))
				for range 4 {
					v, err := r.Call(fn, Undefined, []Value{Float(300000)})
					res, _ := r.GetProp(Obj(r.Global()), r.Intern("res"))
					got[k] = regNativeShow(r, v, err) + ":" + regNativeShow(r, res, nil)
				}
			}
			if want := "300000:900000"; got[0] != got[1] || got[1] != want {
				t.Errorf("with the JIT %q, without %q, want %q", got[0], got[1], want)
			}
			r := rts[0]
			fn, _ := r.GetProp(Obj(r.Global()), r.Intern("props"))
			cl := fn.Object().fn().closure
			e := r.jit.regCache[r.jitKeyOf(cl.fn)]
			if e == nil || e.code == nil {
				t.Fatal("props has no register code")
			}
			if len(e.unentered) != 1 {
				t.Fatalf("props' code dropped the entries of %v, want its loop's", e.unentered)
			}
			o := e.osr
			if o == nil || o.code == nil {
				t.Fatalf("no code compiled for the frame waiting at the loop at %d", e.unentered[0])
			}
			if _, ok := o.loopEntry(e.unentered[0]); !ok || o.life.ssaStats.entries == 0 {
				t.Errorf("the loop's code has entries %v, entered %d times", o.loops, o.life.ssaStats.entries)
			}
		})
	}
}
