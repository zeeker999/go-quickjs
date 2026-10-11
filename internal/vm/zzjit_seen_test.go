//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"runtime"
	"runtime/debug"
	"testing"
	"weak"
)

// A write that adds its property, and a read that finds its property, along
// a prototype chain deeper than a cache holds are noted by the existing
// tiers (jitNoteDeepAdd, jitNoteDeepRead), and code compiled after they ran
// them makes them natively from its first compile: deep's objects are of
// three classes whose chains are four to six deep, as DeltaBlue's
// constraints' are. Before, the adds were learned one exit, and one
// compile, at a time, and the reads, which no cache could hold, left
// native code at every one. Each answer is the interpreter's.
func TestJITSSADeepChainsSeenByTheInterpreter(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	// Native code adds no property while the collector marks, leaving the
	// add to Go, which would count as its exits.
	runtime.GC()
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	setup := `function K0(){} function K1(){} K1.prototype=Object.create(K0.prototype);
		function K2(){} K2.prototype=Object.create(K1.prototype);
		function K3(){} K3.prototype=Object.create(K2.prototype);
		function K4(){} K4.prototype=Object.create(K3.prototype);
		K0.prototype.w=5;
		var ks=[K2,K3,K4,K2],objs=[];
		function deep(n){let t=0;for(let i=0;i<n;i++){const o=objs[i];o.strength=i;t=(t+o.strength+o.w)|0}return t}`
	// The objects, made at the top level, which is never compiled.
	src := `objs=[];for(var i=0;i<6000;i++)objs.push(new ks[i&3]());String(deep(6000))`
	want := New(Config{})
	defer func() { want.Close(); want.ReleaseClosed() }()
	r := New(Config{JIT: true})
	defer func() { r.Close(); r.ReleaseClosed() }()
	r.jitSSA = true
	for _, rt := range []*Runtime{want, r} {
		if _, err := rt.Run(compileForTest(t, setup)); err != nil {
			t.Fatal(err)
		}
	}
	cl := r.global.getOwn(r.atoms.intern("deep")).value.Object().fn().closure
	for i := range 2 {
		wv, err := want.Run(compileForTest(t, src))
		if err != nil {
			t.Fatal(err)
		}
		gv, err := r.Run(compileForTest(t, src))
		if err != nil {
			t.Fatal(err)
		}
		if got, want := gv.String().Go(), wv.String().Go(); got != want {
			t.Fatalf("round %d: got %s, interpreter %s", i, got, want)
		}
		e := r.jit.cache[weak.Make(cl.fn)]
		if e == nil || e.ssa == nil || e.entrySlow {
			t.Fatalf("round %d: deep did not run natively", i)
		}
		if r.jit.reoptimized != 0 {
			t.Fatalf("round %d: deep was compiled %d times more", i, r.jit.reoptimized)
		}
		if n := e.ssaStats.hosts + e.ssaStats.guards; n != 0 {
			t.Fatalf("round %d: deep left native code %d times", i, n)
		}
	}
}
