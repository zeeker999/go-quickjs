//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"fmt"
	"runtime"
	"runtime/debug"
	"strings"
	"testing"
)

// typedLayoutNames are the element types typedLayoutSetup has functions
// for: those native code reads and writes as numbers, and the two BigInt
// types, whose elements it reads and writes as 64-bit words.
var typedLayoutNames = append(append([]string(nil), typedKindNames...), "BigInt64Array", "BigUint64Array")

// typedLayoutSetup makes, for each element type, functions of their own,
// each site meeting that type alone: reads and writes bounded by a count
// that may pass the view's length, and by the view's length, read at each
// turn; and loops that detach the view's buffer, or resize it, as they
// run. layoutRound(name) runs them over views of every kind of buffer --
// its own, a subarray's, at a byte offset, a resizable one's tracking its
// length and of a fixed length, as it shrinks, grows within its capacity
// and past it, an immutable one's, a growable shared one's -- and reads
// each view by Go's methods too, which read the buffer as it is.
var typedLayoutSetup = func() string {
	var b strings.Builder
	for _, n := range typedLayoutNames {
		zero, op, w := "0", "+", "x + 0.25"
		if strings.HasPrefix(n, "Big") {
			zero, op, w = "0n", "^", "BigInt(x)"
		}
		fmt.Fprintf(&b, `
	function W_%[1]s(x) { return %[4]s }
	function rd_%[1]s(a, n) { const out = []; for (let i = 0; i < n; i++) out.push(a[i]); return out.join() }
	function sum_%[1]s(a) { let s = %[2]s; for (let i = 0; i < a.length; i++) s %[3]s= a[i]; return String(s) }
	function wr_%[1]s(a, n, k) { for (let i = 0; i < n; i++) a[i] = W_%[1]s(i * 3 + k); return a.length }
	function wrl_%[1]s(a, k) { for (let i = 0; i < a.length; i++) a[i] = W_%[1]s(i * 5 + k) }
	function rdx_%[1]s(a, n, at) { const out = []; for (let i = 0; i < n; i++) { if (i === at) a.buffer.transfer(); out.push(a[i]) } return out.join() }
	function wrx_%[1]s(a, n, at) { for (let i = 0; i < n; i++) { if (i === at) a.buffer.transfer(); a[i] = W_%[1]s(i) } return a.length }
	function rdr_%[1]s(a, n, at, to) { const out = []; for (let i = 0; i < n; i++) { if (i === at) a.buffer.resize(to); out.push(i < a.length ? a[i] : '-') } return out.join() }
	function ea_%[1]s(a) { let s = 0; try { a.forEach(x => { s += Number(x) }) } catch (e) { return e.constructor.name } return s }
	function re_%[1]s(a) { try { return a.reduce((s, x) => s + Number(x), 0) } catch (e) { return e.constructor.name } }
	function ma_%[1]s(a) { try { return a.map(x => x).join() } catch (e) { return e.constructor.name } }
	function sumr_%[1]s(a, at, to) { let s = %[2]s; for (let i = 0; i < a.length; i++) { if (i === at) a.buffer.resize(to); s %[3]s= a[i] } return String(s) }
`, n, zero, op, w)
	}
	b.WriteString(`
	function layoutRound(name) {
		const K = globalThis[name], z = K.BYTES_PER_ELEMENT, out = []
		const f = k => globalThis[k + '_' + name]
		const ea = f('ea'), re = f('re'), ma = f('ma'), loops = a => ea(a) + ':' + re(a) + ':' + ma(a)
		const rd = f('rd'), sum = f('sum'), wr = f('wr'), wrl = f('wrl'), rdx = f('rdx'), wrx = f('wrx'), rdr = f('rdr'), sumr = f('sumr')
		// A view of its own buffer, a subarray of it, one at a byte offset.
		const a = new K(24)
		out.push(wr(a, 26, 1), rd(a, 26), sum(a))
		const s = a.subarray(3, 11)
		out.push(wr(s, 9, 100), rd(a, 24), rd(s, 9), sum(s), s.length, a.join())
		const o = new K(a.buffer, 5 * z, 7)
		wrl(o, 9)
		out.push(rd(o, 8), a.join(), sum(o))
		// Detached by a transfer as the loop runs, read and written.
		const d1 = new K(16)
		wr(d1, 16, 2)
		out.push(rdx(d1, 16, 5), d1.length, sum(d1), rd(d1, 2))
		const d2 = new K(16)
		out.push(wrx(d2, 16, 7), d2.length, rd(d2, 3), sum(d2))
		// Views of a buffer nothing changed, after another's was.
		out.push(rd(a, 24), sum(s), a.join())
		// A resizable buffer's views: tracking its length, and fixed.
		const rab = new ArrayBuffer(8 * z, { maxByteLength: 64 * z })
		const t = new K(rab), fx = new K(rab, 2 * z, 4), f2 = new K(rab, 0, 8)
		wrl(t, 3)
		out.push(rd(t, 9), sum(t), sum(fx), t.length, fx.length)
		out.push(loops(t), loops(fx))
		rab.resize(4 * z)
		// The fixed views out of bounds: their methods throw.
		out.push(rd(t, 9), rd(fx, 5), sum(t), sum(fx), t.length, fx.length, f2.length, loops(t), loops(fx), loops(f2))
		out.push(wr(fx, 4, 50), wrl(t, 60), t.join(), new K(rab).join())
		// Past its capacity: the bytes move.
		rab.resize(40 * z)
		wrl(t, 70)
		out.push(rd(t, 41), rd(fx, 5), sum(t), sum(fx), t.length, fx.length, f2.length, f2.join(), new K(rab, 0, 40).join())
		// Within it: the same bytes, shorter, then longer, read as zeros.
		rab.resize(10 * z)
		out.push(rd(t, 12), sum(fx), sum(t), f2.length)
		rab.resize(30 * z)
		out.push(rd(t, 31), sum(t), t.join())
		// Resized as the loop runs.
		out.push(rdr(t, 30, 6, 12 * z), t.length, sumr(t, 4, 3 * z), t.length, rdr(fx, 4, 1, z), fx.length)
		rab.resize(64 * z)
		out.push(rdr(fx, 4, 2, 5 * z), fx.length)
		for (let r = 0; r < 30; r++) {
			const b = new ArrayBuffer(16 * z, { maxByteLength: 32 * z }), v = new K(b), w = new K(b, z, 8)
			wrl(v, r)
			out.push(rdr(v, 16, 3 + r % 6, (8 - r % 6) * z), v.length, sumr(v, 2, (10 + 2 * (r % 6)) * z), v.length, rdr(w, 8, r % 6, (2 + r % 6) * z), w.length, sum(w), new K(b).join())
		}
		// A tracking view whose offset the buffer shrinks past.
		const t2 = new K(rab, 4 * z)
		rab.resize(2 * z)
		out.push(t2.length, rd(t2, 2), sum(t2), loops(t2))
		rab.resize(6 * z)
		wrl(t2, 9)
		out.push(t2.length, rd(t2, 3), sum(t2), t.join())
		// The resizable buffer transferred: every view of it empty.
		const moved = rab.transfer()
		out.push(t.length, rd(t, 3), fx.length, sum(t2), new K(moved).join(), loops(t), loops(fx), loops(new K(moved)))
		// Views of buffers of their own, detached, and empty.
		const e0 = new K(0), e1 = new K(6)
		wrl(e1, 2)
		out.push(loops(e0), loops(e1), e0.length)
		e1.buffer.transfer()
		out.push(loops(e1), e1.length)
		// An immutable buffer's view: read, never written.
		const src = new K(12)
		wr(src, 12, 4)
		const imm = new K(src.buffer.transferToImmutable())
		let threw = false
		try { wr(imm, 12, 9) } catch (e) { threw = e instanceof TypeError }
		out.push(rd(imm, 12), threw, rd(imm, 12), sum(imm), src.length)
		// A growable shared buffer's views, tracking and fixed.
		const sab = new SharedArrayBuffer(4 * z, { maxByteLength: 32 * z }), g = new K(sab), gf = new K(sab, 0, 4)
		wrl(g, 1)
		out.push(rd(g, 6), sum(g), g.length)
		sab.grow(20 * z)
		wrl(g, 2)
		out.push(rd(g, 21), sum(g), g.length, sum(gf), gf.join())
		return out.join(' | ')
	}
	// forEach's, reduce's and map's loops at sites that meet every type,
	// their callbacks made at each call, over views a resize leaves in
	// their buffers, out of them, and back.
	function collect(a) { const out = []; a.forEach(n => { out.push(n) }); return out.map(Number).join() }
	function total(a) { return Number(a.reduce((s, n) => s + Number(n), 0)) }
	function mapped(a) { return a.map(n => n).join() }
	function attempt(f, a) { try { return f(a) } catch (e) { return e.constructor.name } }
	function resizeRound(name) {
		const K = globalThis[name], z = K.BYTES_PER_ELEMENT, out = []
		const rab = new ArrayBuffer(4 * z, { maxByteLength: 8 * z })
		const views = [new K(rab, 0, 4), new K(rab, 2 * z, 2), new K(rab, 0), new K(rab, 2 * z)]
		wrl_` + typedLayoutNames[0] + `(new Int8Array(rab), 1)
		const all = () => { for (const v of views) for (const f of [collect, total, mapped]) out.push(attempt(f, v)) }
		all()
		rab.resize(3 * z)
		all()
		rab.resize(z)
		all()
		rab.resize(0)
		all()
		rab.resize(6 * z)
		all()
		rab.transfer()
		all()
		return out.join(' | ')
	}
	function layoutAll() { return typedLayoutNames.split(',').map(layoutRound).join(' || ') }
	var typedLayoutNames = '` + strings.Join(typedLayoutNames, ",") + `'
`)
	return b.String()
}()

// A typed array's element read or written natively, and its length read,
// reach the element through the view's copy of where its elements are and
// how many (typedArrayData's data and length), which holds while its epoch
// is the runtime's (abi.Encoding's TypedEpoch): every change of where a
// buffer's bytes are or how long they are -- a transfer as a loop runs, a
// resizable buffer shrunk, grown within its capacity and past it, which
// moves its bytes, a shared buffer grown -- leaves every view's copy
// stale, and native code has Go make it anew (ssa.GoTypedView) before it
// reads it. Views of a buffer's own, a subarray, a byte offset, tracking
// a resizable buffer's length or of a fixed length in it, of an immutable
// buffer, which takes no store: every answer the interpreter's, read by
// native code and by Go's methods alike, with the collector idle and
// while it marks.
func TestJITTypedLayout(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	for _, marking := range []bool{false, true} {
		t.Run(fmt.Sprintf("marking=%v", marking), func(t *testing.T) {
			runtime.GC()
			gc := debug.SetGCPercent(-1)
			t.Cleanup(func() { debug.SetGCPercent(gc) })
			if marking {
				jitMarkingForTest(t)
			}
			rounds := []string{}
			for _, name := range typedLayoutNames {
				for range 8 {
					rounds = append(rounds, fmt.Sprintf("layoutRound('%s')", name))
				}
			}
			r := jitAgreesForTest(t, typedLayoutSetup, rounds...)
			// Compiled at once, the callbacks' loops too -- forEach's,
			// reduce's, map's over views detached or out of their buffers,
			// which throw -- as CI's runs of test262 with the JIT compile.
			stress := rounds[: 3*8 : 3*8]
			for range 3 {
				for _, name := range typedLayoutNames {
					stress = append(stress, fmt.Sprintf("resizeRound('%s')", name))
				}
			}
			jitAgreesForTestWith(t, jitStressConfig{threshold: true, budget: 3, deopt: 7}, typedLayoutSetup, stress...)
			for _, name := range typedLayoutNames {
				for _, f := range []string{"rd_", "sum_", "wr_", "wrl_", "rdx_", "wrx_", "rdr_", "sumr_"} {
					cl := r.global.getOwn(r.atoms.intern(f + name)).value().Object().fn().closure
					if e := r.jit.hint(cl); e == nil || e.ssa == nil || e.ssaStats.entries == 0 {
						t.Errorf("%s never ran natively %+v", f+name, e)
					}
				}
			}
			// A view read natively after another buffer was resized: the
			// view's copy is made anew by one call of Go, then read with
			// none.
			var src strings.Builder
			for _, name := range typedLayoutNames {
				zero, mask := "0", "s"
				if strings.HasPrefix(name, "Big") {
					zero, mask = "0n", "s & 255n"
				}
				el := "i * 3 - 50"
				if zero == "0n" {
					el = "BigInt(i * 3)"
				}
				fmt.Fprintf(&src, `
	var keep_%[1]s = new %[1]s(64); for (let i = 0; i < 64; i++) keep_%[1]s[i] = %[2]s
	function tot_%[1]s(a) { let s = %[3]s; for (let i = 0; i < a.length; i++) s += a[i]; return %[4]s }
	for (let i = 0; i < 300; i++) tot_%[1]s(keep_%[1]s)
`, name, el, zero, mask)
			}
			src.WriteString("var other = new ArrayBuffer(8, { maxByteLength: 16 })")
			r2 := jitRuntimeForTest(t, Config{JIT: true})
			if _, err := r2.Run(compileForTest(t, src.String())); err != nil {
				t.Fatal(err)
			}
			made := 0
			jitTypedViewMade = func() { made++ }
			t.Cleanup(func() { jitTypedViewMade = nil })
			readMaking := func(src string) (string, int) {
				t.Helper()
				made = 0
				v, err := r2.Run(compileForTest(t, src))
				if err != nil {
					t.Fatal(err)
				}
				return v.String().Go(), made
			}
			for _, name := range typedLayoutNames {
				read := fmt.Sprintf("'' + tot_%[1]s(keep_%[1]s)", name)
				// The resize of the type before's has left this view's copy
				// stale too.
				readMaking(read)
				want, n := readMaking(read)
				if n != 0 {
					t.Errorf("%s: %d copies made reading a view nothing changed (%s)", name, n, want)
				}
				if _, err := r2.Run(compileForTest(t, "other.resize(12)")); err != nil {
					t.Fatal(err)
				}
				if got, n := readMaking(read); n != 1 || got != want {
					t.Errorf("%s: %d copies made reading a view after another buffer was resized, got %s, want 1, %s", name, n, got, want)
				}
				if got, n := readMaking(read); n != 0 || got != want {
					t.Errorf("%s: %d copies made reading the view again, got %s, want 0, %s", name, n, got, want)
				}
				jitNativeForTest(t, r2, "tot_"+name)
			}
		})
	}
}

// A view tracking a growable shared buffer's length, which another agent
// may grow with no call of this runtime's, has a copy of its length that
// may be less than the length (typedArrayData's noLength): native code
// reads its elements up to the copy's, but never takes it for the view's
// length, which Go measures.
func TestJITTypedViewGrownElsewhere(t *testing.T) {
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	r := jitRuntimeForTest(t, Config{JIT: true})
	run := func(src string) string {
		t.Helper()
		v, err := r.Run(compileForTest(t, src))
		if err != nil {
			t.Fatalf("%s: %v", src, err)
		}
		return v.String().Go()
	}
	run(`
	var sab = new SharedArrayBuffer(8, { maxByteLength: 64 }), g = new Uint8Array(sab)
	function lengths(a) { let n = 0; for (let i = 0; i < 50; i++) n += a.length; return n }
	function sum(a) { let s = 0; for (let i = 0; i < 8; i++) s += a[i]; return s }
	for (let i = 0; i < 8; i++) g[i] = i
	for (let i = 0; i < 300; i++) lengths(g), sum(g)
	''`)
	mem, _, err := r.SharedMemoryOf(r.global.getOwn(r.atoms.intern("sab")).value())
	if err != nil {
		t.Fatal(err)
	}
	if got := run("'' + lengths(g) + ' ' + sum(g)"); got != "400 28" {
		t.Fatalf("before the growth: %s, want 400 28", got)
	}
	// Another agent's growth: the memory's, not this runtime's buffer's.
	if !mem.grow(32) {
		t.Fatal("the memory did not grow")
	}
	if got := run("'' + lengths(g) + ' ' + sum(g) + ' ' + g.length"); got != "1600 28 32" {
		t.Errorf("after another agent grew the buffer: %s, want 1600 28 32", got)
	}
}
