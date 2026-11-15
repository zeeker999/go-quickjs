//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"fmt"
	"strings"
	"testing"
)

// regLexThisSetup's functions make arrows that read their this, the maker's
// register 1: one called in a loop, one given to a method calling it in a
// loop (Vec.forEach), each inlined where it is called and never made
// (ssa's calledClosures, InlineSite's LexThis) -- the maker's this a
// method's receiver, then another object's -- and each meets a throw inside
// the inlined arrow at its at, where Go makes the arrow, its this the
// maker's, for the frame it was in; arrows of a sloppy and a strict maker
// called plainly, their this the global object and undefined; and
// callbacks made just before a
// built-in's forEach, map or reduce, inlined in the loop and never made
// (ssa.MadeCallback), one reading an object a turn may be no object.
const regLexThisSetup = `
	class Vec {
		constructor(n) { this.storage = []; for (let i = 0; i < n; i++) this.storage.push(i) }
		forEach(fn) { for (let i = 0; i < this.storage.length; i += 1) { fn(this.storage[i]) } }
	}
	var vec = new Vec(12)
	var holder = {
		k: 3, bad: null,
		sum(n, at) { let s = 0; const f = (x) => x === at ? this.bad.v : x * this.k; for (let i = 0; i < n; i++) s += f(i); return s },
		each(at) { let s = 0; vec.forEach((x) => { s += x === at ? this.bad.v : x + this.k }); return s },
	}
	var other = { k: 7, bad: { v: -1000 }, sum: holder.sum, each: holder.each }
	function arrows(n, at) {
		let r = ''
		for (const h of [holder, other]) {
			try { let e = 0; for (let j = 0; j < n; j++) e = h.each(at); r += h.sum(n, at) + ':' + e + ';' } catch (e) { r += e.constructor.name + ';' }
		}
		return r
	}
	function loose(n) { const g = (x) => x + (this === globalThis ? 1 : 2); let s = 0; for (let i = 0; i < n; i++) s += g(i); return s }
	function tight(n) { 'use strict'; const g = (x) => x + (this === undefined ? 3 : 4); let s = 0; for (let i = 0; i < n; i++) s += g(i); return s }
	function plainThis(n) { let s = 0; for (let i = 0; i < 20; i++) s += loose(n) * 1000 + tight(n); return s }
	var boxes = [], plain = []
	for (let i = 0; i < 20; i++) { boxes.push({ q: i }); plain.push(i) }
	function builtins(n, at) {
		let s = 0, k = 3
		for (let j = 0; j < n; j++) {
			plain.forEach((x) => { s += x === at ? boxes[x].q.r.t : x * k })
			s += plain.map((x) => x + k).reduce((a, x) => a + x, 0)
		}
		return s
	}
`

// regCalledRounds are the rounds a register test of calledSetup, givenSetup
// or regLexThisSetup runs: warm ones, then ones whose exits are inside the
// inlined closures.
// Their loops run ten times the stack form's tests' turns, which the
// register tier's samples need to find them hot -- a driver's loop a
// hundred times, its back edges what samples it: the interpreter's back
// edges are counted across the runtime, and those of loops native code
// runs are not, so the more of the other rounds run natively the later a
// driver of 300 turns was sampled.
func regCalledRounds(setup string) []string {
	var rounds []string
	switch setup {
	case calledSetup:
		rounds = calledRounds()
	case givenSetup:
		rounds = givenRounds(true)
	default:
		// Ten warm turns: builtins, called once a round, learns its
		// callbacks at its calls of Go after its first compile and is
		// compiled again for them once it has spent a budget since
		// (jitRegLearnedDue), at a call some rounds later.
		for i := range 16 {
			at := []string{"-1", "5"}[i/10]
			rounds = append(rounds, "arrows(30, "+at+")", "builtins(40, "+at+")", "plainThis(30)")
		}
	}
	for i, r := range rounds {
		r = strings.Replace(r, "(40, ", "(400, ", 1)
		r = strings.Replace(r, "(30, ", "(300, ", 1)
		r = strings.Replace(r, "(10, ", "(100, ", 1)
		rounds[i] = strings.Replace(r, "i < 30;", "i < 3000;", 1)
	}
	return rounds
}

// TestJITRegCalledClosuresPolled runs calledSetup's and regLexThisSetup's
// rounds with loops polling every 3 back edges and every 7th poll finishing
// the invocation in the interpreter (QJS_JIT_STRESS's budget and deopt), the
// collector idle and marking: the frame the interpreter goes on with names
// the closures the code only calls by their templates, which Go makes
// before it does, as at any exit -- left unmade, squares threw "object is
// not a function" at its call of sq.
func TestJITRegCalledClosuresPolled(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, setup := range []struct{ name, src string }{{"called", calledSetup}, {"lexThis", regLexThisSetup}} {
		for _, marking := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/marking=%v", setup.name, marking), func(t *testing.T) {
				if marking {
					jitMarkingForTest(t)
				}
				rts := regNativeRuntimes(t, setup.src)
				r := rts[0]
				r.SetJITWarmUp(1, 1)
				r.jitStress = jitStressConfig{budget: 3, deopt: 7}
				for _, round := range regCalledRounds(setup.src) {
					prog := compileForTest(t, round)
					var got [2]string
					for k, r := range rts {
						v, err := r.Run(prog)
						got[k] = regNativeShow(r, v, err)
					}
					if got[0] != got[1] {
						t.Fatalf("%s: with the JIT %q, without %q", round, got[0], got[1])
					}
				}
				if r.jit.materialized == 0 {
					t.Error("Go made no closure the code did not make")
				}
			})
		}
	}
}
