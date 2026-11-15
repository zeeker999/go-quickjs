//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"runtime/debug"
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
)

// A native call's state is described statically (abi.CallState): Go
// expands it into the records it makes the caller's frame from when the
// callee leaves (jit.ExpandCall), each value from where the caller's code
// keeps it -- a spill slot, a constant, a slot that held it on entry -- or
// from where the call wrote it: a register dead past the call, a pointer
// word read from a cell the callee may change. Here the callee throws, so
// the caller goes on from its frame, not natively after the call, and
// every kind of value is live across it: numbers, doubles, booleans,
// strings, an object from a parameter, one read from a property the callee
// replaces, a constant, an object made in the loop.
func TestJITCallStateRebuildsTheFrame(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	const setup = `
		var holder = {p: {v: 1}, tag: "h"}, other = {tag: "m"}
		function thrower(x, h) {
			let w = 0
			for (let j = 0; j < 3; j++) w = (w + j) | 0
			h.p = {v: x + w - 3}
			if (x % 50 === 49) throw new Error("e" + x)
			return x
		}
		function caller(o, n, m) {
			let out = []
			for (let i = 0; i < n; i++) {
				const a = o.p, b = i * 0.5 + 0.25, c = i > 10, d = o, e = "s" + (i & 3), f = {q: i}, k = 7, g = i & 1 ? m : o
				let r
				try { r = thrower(i, o) } catch (err) { r = err.message }
				out.push([a.v, b, c, d === o, d.tag, e, f.q, k, r, o.p.v, g.tag].join())
			}
			return out.join(";")
		}`
	want := New(Config{})
	t.Cleanup(func() { want.Close(); want.ReleaseClosed() })
	r := jitRuntimeForTest(t, Config{JIT: true})
	for _, rt := range []*Runtime{want, r} {
		if _, err := rt.Run(compileForTest(t, setup+`; caller(holder, 200, other); caller(holder, 200, other)`)); err != nil {
			t.Fatal(err)
		}
	}
	fnOf := func(name string) *bytecode.Function {
		return r.global.getOwn(r.atoms.intern(name)).value().Object().fn().closure.fn
	}
	ce, te := r.jit.cache[r.jitKeyOf(fnOf("caller"))], r.jit.cache[r.jitKeyOf(fnOf("thrower"))]
	if ce == nil || ce.ssa == nil || te == nil || te.ssa == nil {
		t.Fatalf("not compiled: caller %+v, thrower %+v", ce, te)
	}
	in, records := te.link.in, r.jit.ssaRecords
	for range 3 {
		wv, err := want.Run(compileForTest(t, `caller(holder, 400, other)`))
		if err != nil {
			t.Fatal(err)
		}
		gv, err := r.Run(compileForTest(t, `caller(holder, 400, other)`))
		if err != nil {
			t.Fatal(err)
		}
		if gv.String().Go() != wv.String().Go() {
			t.Fatalf("caller = %.300s...; interpreter %.300s...", gv.String().Go(), wv.String().Go())
		}
	}
	if te.link.in == in || r.jit.ssaRecords == records {
		t.Fatalf("thrower called natively %d times, %d records applied", te.link.in-in, r.jit.ssaRecords-records)
	}
}
