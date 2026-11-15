//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"fmt"
	"runtime"
	"testing"
	"unsafe"
	"weak"

	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
)

func TestJITLiteralArrayRoomWeakPools(t *testing.T) {
	pool := &abi.ObjectPool{Source: unsafe.Pointer(&jitPoolSource{array: true, room: 4096})}
	s := &jitState{arrayRooms: []weak.Pointer[abi.ObjectPool]{weak.Make(pool)}}
	pool = nil
	runtime.GC()
	jitLiteralArraySweep[struct{}](s)
	if len(s.arrayRooms) != 0 {
		t.Fatal("the sample list retained a discarded native pool")
	}
}

func TestJITLiteralArrayRoomNative(t *testing.T) {
	jitNoInlineForTest(t, "fill")
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprint(marking), func(t *testing.T) {
			if marking {
				jitMarkingForTest(t)
			}
			r := jitRuntimeForTest(t, Config{JIT: true})
			r.jitStress = jitStressConfig{threshold: true, budget: 3, deopt: 7}
			if _, err := r.Run(compileForTest(t, `
				function fill(n) {
					var a = [];
					for (var i = 0; i < n; i++) a.push(i);
					return a;
				}
				function render(n) {
					var a = fill(n);
					return a.length + ':' + a[0] + ':' + a[n - 1];
				}
			`)); err != nil {
				t.Fatal(err)
			}
			cl := r.global.getOwn(r.atoms.intern("fill")).value().Object().fn().closure
			cl.jitRefused = true
			for range 4 {
				if _, err := r.Run(compileForTest(t, "fill(4096)")); err != nil {
					t.Fatal(err)
				}
			}
			if len(cl.arrays) != 1 || cl.arrays[0].room != 4096 {
				t.Fatal("tree tier did not confirm the grown length")
			}
			cl.jitRefused = false
			for range 8 {
				if _, err := r.Run(compileForTest(t, "render(4096)")); err != nil {
					t.Fatal(err)
				}
			}
			e := jitNativeForTest(t, r, "fill")
			inherited := false
			for _, l := range e.literals {
				s := (*jitPoolSource)(l.pool.Source)
				inherited = inherited || l.array && s.room == 4096 && l.pool.Size == 1
				if l.array && s.room >= 256 && s.sample != nil && l.pool.Count == 0 {
					t.Fatal("native hint retained its completed array after the turn")
				}
			}
			if !inherited {
				t.Fatal("native literal did not inherit the tree tier's capacity")
			}
			for i := range e.literals {
				l := &e.literals[i]
				s := (*jitPoolSource)(l.pool.Source)
				if !l.array || s.room != 4096 {
					continue
				}
				r.jitFillLiteral(l)
				jitLiteralArraysLetGo[struct{}](r)
				if l.pool.Count != 1 || s.room != 4096 || s.sample == nil {
					t.Fatal("an unused native array was observed as a short result")
				}
			}
			// Refilling releases the old sample before it allocates. A
			// native caller's live arrays must remain independently rooted.
			stop := jitCollectingForTest(t)
			v, err := r.Run(compileForTest(t, `function pairs() {
				var sum = 0;
				for (var i = 0; i < 80; i++) {
					var x = fill(4096);
					x[0] = i;
					var y = fill(4096);
					y[0] = i + 1;
					sum += x[0] + y[0] + x[4095] + y[4095];
				}
				return sum;
			}
			pairs()`))
			stop()
			if err != nil || !v.IsNumber() || v.Number() != 661600 {
				t.Fatalf("collecting: got %v, %v; want 661600", v, err)
			}
			for _, n := range []int{1024, 1024, 3, 1024, 0, 2048, 2048, arrayRoomLimit + 1} {
				want := fmt.Sprintf("%d:0:%d", n, n-1)
				if n == 0 {
					want = "0:undefined:undefined"
				}
				v, err := r.Run(compileForTest(t, fmt.Sprintf("render(%d)", n)))
				if err != nil || !v.IsString() || v.String().Go() != want {
					t.Fatalf("length %d: got %v, %v; want %s", n, v, err, want)
				}
			}
		})
	}
}

func TestJITLiteralArrayRoomInlined(t *testing.T) {
	t.Parallel()
	r := jitRuntimeForTest(t, Config{JIT: true})
	r.jitStress = jitStressConfig{threshold: true, budget: 3, deopt: 7}
	if _, err := r.Run(compileForTest(t, `
		function make() { return []; }
		function fill(n) {
			var a = make();
			for (var i = 0; i < n; i++) a.push(i);
			return a;
		}
	`)); err != nil {
		t.Fatal(err)
	}
	maker := r.global.getOwn(r.atoms.intern("make")).value().Object().fn().closure
	filler := r.global.getOwn(r.atoms.intern("fill")).value().Object().fn().closure
	maker.jitRefused, filler.jitRefused = true, true
	call := compileForTest(t, "fill(4096)")
	for range 4 {
		if _, err := r.Run(call); err != nil {
			t.Fatal(err)
		}
	}
	if len(maker.arrays) != 1 || maker.arrays[0].room != 4096 || !maker.arrays[0].acrossTurns {
		t.Fatal("callee did not confirm its caller's grown array across turns")
	}
	maker.jitRefused, filler.jitRefused = false, false
	for range 8 {
		v, err := r.Run(call)
		if err != nil {
			t.Fatal(err)
		}
		els := v.Object().elemsOf()
		if len(els) != 4096 || els[0].Number() != 0 || els[4095].Number() != 4095 {
			t.Fatal("inlined literal changed its caller's result")
		}
	}
	e := jitNativeForTest(t, r, "fill")
	for _, l := range e.literals {
		if l.array && l.fn == maker.fn {
			src := (*jitPoolSource)(l.pool.Source)
			if src.room != 4096 || l.pool.Size != 1 || src.sample != nil && l.pool.Count == 0 {
				t.Fatal("inlined literal lost its callee's hint or retained a completed sample")
			}
			return
		}
	}
	t.Fatal("native caller did not inline its callee's literal")
}
