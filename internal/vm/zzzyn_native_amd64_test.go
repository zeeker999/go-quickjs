package vm

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/compiler"
)

// The native handlers' tests (zzzyn_native_amd64.go): each runs code by
// the Go table's loop and by the native handlers, with the collector idle
// and as while it marks, and checks the answers are the same, and that
// the native code did the work it is for -- the same exits to Go marking
// or not, the calls made natively where rhCallW enters a frame itself.

// nativeKernels are micro-benchmarks' functions by name, what each needs
// besides, and the work to give it.
var nativeKernels = []struct {
	name string
	deps []string
	n    int
}{
	{"empty_loop", nil, 1000},
	{"loop_locals", nil, 1000},
	{"call_fib", []string{"call_fib_rec"}, 177 * 20},
	{"crypto_am3", []string{"am3"}, 32 * 20},
	{"prop_read", nil, 1000},
	{"prop_write", nil, 1000},
	{"bigint64_arith", []string{"bigint_arith"}, 3},
	{"typed_f64_read", []string{"extra_f64s"}, 1000},
	{"generator_busy", []string{"extra_busy_gen"}, 3},
	{"prop_in", nil, 1000},
	{"call_mutual", []string{"call_even", "call_odd"}, 65 * 20},
	{"method_call", []string{"ExtraPoint", "=ExtraPoint.prototype.sum", "=ExtraPoint.prototype.same"}, 1000},
	{"forward_call0", []string{"extra_fwd0", "extra_fwd_target0"}, 1000},
	{"forward_call2", []string{"extra_fwd2", "extra_fwd_target2"}, 1000},
	{"forward_call6", []string{"extra_fwd6", "extra_fwd_target6"}, 1000},
}

// nativeKernelSource is the source of the named functions of the
// micro-benchmarks; a name "=x" is the statement x = function ... };.
func nativeKernelSource(t testing.TB, names ...string) string {
	var all string
	for _, f := range []string{"../v8bench/microbench.js", "../v8bench/microbench_extra.js"} {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		all += strings.ReplaceAll(string(b), "\r\n", "\n")
	}
	var out strings.Builder
	out.WriteString("var global_res;\n")
	for _, name := range names {
		if strings.HasPrefix(name, "=") {
			i := strings.Index(all, name[1:]+" = function")
			if i < 0 {
				t.Fatalf("no %s", name)
			}
			j := strings.Index(all[i:], "\n};\n")
			out.WriteString(all[i : i+j+4])
			continue
		}
		i := strings.Index(all, "function "+name+"(")
		if i < 0 {
			i = strings.Index(all, "function* "+name+"(")
		}
		if i < 0 {
			t.Fatalf("no function %s", name)
		}
		j := strings.Index(all[i:], "\n}\n")
		out.WriteString(all[i : i+j+3])
	}
	return out.String()
}

// nativeCounting turns the native handlers' counters on for a test.
func nativeCounting(t testing.TB) {
	nativeCount = true
	t.Cleanup(func() { nativeCount = false })
}

// nativeMarking has the contexts made after it store as while the
// collector marks, or not.
func nativeMarking(t testing.TB, on bool) {
	if on {
		m := new(uint8)
		*m = 1
		nativeMarkFlag.Store(m)
	} else {
		nativeMarkFlag.Store(nil)
	}
	t.Cleanup(func() { nativeMarkFlag.Store(nil) })
}

// nativeExitOps is the exits to Go so far, by operation.
func nativeExitOps() (n [256]int64) {
	for i := range n {
		n[i] = atomic.LoadInt64(&nativeExits[i])
	}
	return n
}

// nativeRunSrc compiles src in the register form and runs it on a new
// runtime, by the native handlers or by the Go table's loop, and returns
// its value as a string.
func nativeRunSrc(t testing.TB, src string, native bool) string {
	t.Helper()
	compiler.SetRegisterForm(true)
	fn := compileForTest(t, src)
	compiler.SetRegisterForm(false)
	SetRegTier(true)
	was := regNativeOn
	regNativeOn = native
	defer func() { SetRegTier(false); regNativeOn = was }()
	r := New(Config{})
	defer r.Close()
	v, err := r.Run(fn)
	if err != nil {
		t.Fatalf("native %v: %v", native, err)
	}
	if r.frameDepth != 0 {
		t.Fatalf("native %v: frame depth %d left", native, r.frameDepth)
	}
	return v.String().Go()
}

// nativeRunKernel runs a kernel and returns its answer and how many
// operations it did.
func nativeRunKernel(t testing.TB, native bool, name string, deps []string, n int) (string, int64) {
	src := nativeKernelSource(t, append([]string{name}, deps...)...) +
		fmt.Sprintf("var ops = %s(%d); String(ops) + ' ' + String(global_res)", name, n)
	s := nativeRunSrc(t, src, native)
	var ops int64
	fmt.Sscan(s, &ops)
	return s, ops
}

// TestRegNativeKernels runs each kernel by the native handlers, with the
// collector idle and as while it marks, every pointer stored by the
// barrier's routine: the answers of the Go table's loop, and the same
// exits to Go marking or not.
func TestRegNativeKernels(t *testing.T) {
	nativeCounting(t)
	for _, k := range nativeKernels {
		want, _ := nativeRunKernel(t, false, k.name, k.deps, k.n)
		var exits [2]int64
		for i, on := range []bool{false, true} {
			nativeMarking(t, on)
			before := atomic.LoadInt64(&nativeEnters)
			if got, _ := nativeRunKernel(t, true, k.name, k.deps, k.n); got != want {
				t.Errorf("%s (marking %v): %s, want %s", k.name, on, got, want)
			}
			exits[i] = atomic.LoadInt64(&nativeEnters) - before
		}
		nativeMarking(t, false)
		if exits[0] != exits[1] {
			t.Errorf("%s: %d exits, %d while marking", k.name, exits[0], exits[1])
		}
	}
}

// TestRegNativeExits logs each kernel's entries of the native code an
// operation, and its exits to Go by the operation left: where the native
// handlers leave work to Go.
func TestRegNativeExits(t *testing.T) {
	nativeCounting(t)
	for _, k := range nativeKernels {
		before, exits := atomic.LoadInt64(&nativeEnters), nativeExitOps()
		_, ops := nativeRunKernel(t, true, k.name, k.deps, k.n*50)
		var top []string
		for op, c := range nativeExitOps() {
			if d := c - exits[op]; d*20 > ops {
				top = append(top, fmt.Sprintf("%s %.2f", bytecode.ROp(op), float64(d)/float64(ops)))
			}
		}
		t.Logf("%-16s %6.3f entries an operation; exits by op: %v", k.name,
			float64(atomic.LoadInt64(&nativeEnters)-before)/float64(ops), top)
	}
}

// nativeCallsSrc is call-heavy code for the native call handlers: a
// recursion to the stack's limit and its RangeError, a throw from a deep
// callee caught several frames out, by a frame the code entered itself,
// a generator called and run from native frames, methods, closures that
// capture a frame's locals, and arguments missing and extra.
const nativeCallsSrc = `
	function deep(n) { return n === 0 ? 0 : 1 + deep(n - 1) }
	function boom(n) { if (n === 0) throw new TypeError("boom at " + n); return boom(n - 1) + 1 }
	function catcher(n) { try { return boom(n) } catch (e) { return e.message + "|" + (e instanceof TypeError) } }
	function mid(n) { return catcher(n) + "!" }
	function* gen(n) { for (var i = 0; i < n; i++) yield i * 2 }
	function sumGen(n) { var s = 0; for (var v of gen(n)) s += v; return s }
	function callsGen(n) { return sumGen(n) + sumGen(n + 1) }
	function P(x) { this.x = x }
	P.prototype.get = function () { return this.x };
	P.prototype.add = function (y) { return this.get() + y };
	function methods(n) { var p = new P(3), s = 0; for (var i = 0; i < n; i++) s += p.add(i); return s }
	function counter() { var c = 0; return function () { c++; return c } }
	function closures(n) { var f = counter(), s = 0; for (var i = 0; i < n; i++) s += f(); return s }
	function args3(a, b, c) { return String(a) + String(b) + String(c) }
	function argsMissing() { return args3(1) + args3(1, 2, 3, 4) }
	function overflow() { try { return deep(1e6) } catch (e) { return e instanceof RangeError } }
	function stack() { try { boom(3) } catch (e) { return e.stack.split(String.fromCharCode(10)).filter(l => /boom/.test(l)).length } }
	var out = [deep(500), mid(50), callsGen(10), methods(100), closures(50), argsMissing(), overflow(), stack()];
	out.join(",")`

// nativeFastSrc is code for the native handlers that answer what used to
// exit: typeof of every kind of value, the length of arrays and strings
// and of what has another, and typed arrays' elements of every kind read
// and written, at the edges -- out of range, a NaN, a fraction, a number
// too large for an integer kind, a detached and a shrunk view -- and a
// dense array's elements of any value.
const nativeFastSrc = `
	var out = [];
	var vals = [1, -0, NaN, undefined, null, true, false, "s", "", Symbol("y"), 1n, 2n ** 70n, -(2n ** 64n),
		{}, [], function () {}, () => 1, class {}, Math.max.bind(null), new Proxy(function () {}, {}),
		new Proxy({}, {}), new Date(0), new Uint8Array(2), /x/];
	for (var round = 0; round < 3; round++) {
		var t = [];
		for (var i = 0; i < vals.length; i++) t.push(typeof vals[i]);
		out.push(t.join(" "));
	}
	var sparse = []; sparse[100000] = 1;
	var holey = [1, , 3];
	var lens = [[], [1, 2, 3], sparse, holey, "", "abc", String.fromCharCode(0xe9, 0xd83d, 0xde00), "ab" + "cd".repeat(3),
		{ length: 7 }, new Uint8Array(5), function (a, b) {}, new String("xy")];
	for (var round = 0; round < 3; round++) {
		var l = [];
		for (var i = 0; i < lens.length; i++) l.push(lens[i].length);
		out.push(l.join(" "));
	}
	var mixed = [1, "s", {}, undefined, null, 2.5, [7]];
	for (var round = 0; round < 3; round++) {
		var m = [];
		for (var i = 0; i < mixed.length + 2; i++) { var v = mixed[i]; m.push(typeof v === "object" && v ? "o" : String(v)); }
		m.push(String(mixed[0]), String(mixed[1]), String(mixed[3]));
		out.push(m.join(" "));
	}
	var kinds = [Int8Array, Uint8Array, Uint8ClampedArray, Int16Array, Uint16Array, Int32Array, Uint32Array,
		Float32Array, Float64Array, typeof Float16Array === "function" ? Float16Array : Float64Array];
	var ins = [0, 1, -1, 127, 128, 255, 256, -129, 65535, 65536, 2147483647, 2147483648, -2147483649, 4294967296,
		1.5, -1.5, 0.1, NaN, Infinity, -Infinity, -0, 2 ** 53, 2 ** 63, -(2 ** 63), 2 ** 64, 1e300, 3.4e38, 1e-45];
	for (var k = 0; k < kinds.length; k++) {
		var a = new kinds[k](ins.length + 1);
		for (var round = 0; round < 2; round++) {
			for (var i = 0; i < ins.length; i++) a[i] = ins[i];
			var r = [];
			for (var i = 0; i < ins.length + 2; i++) r.push(Object.is(a[i], -0) ? "-0" : String(a[i]));
			r.push(String(a[0]), String(a[1]), String(a[2]), String(a[3]));
			a[0] = 3; a[1] = 300; a[2] = -7.5; a[3] = NaN;
			r.push(String(a[0]), String(a[1]), String(a[2]), String(a[3]));
			a[1.5] = 9; a[-1] = 9; a[ins.length + 5] = 9;
			r.push(String(a[1.5]), String(a[-1]), String(a[ins.length + 5]));
			out.push(kinds[k].name + ": " + r.join(" "));
		}
	}
	var big = new BigInt64Array(2); big[0] = 5n; big[1] = -(2n ** 63n);
	out.push(String(big[0]) + " " + String(big[1]));
	var buf = new ArrayBuffer(16), view = new Float64Array(buf);
	view[0] = 1.25; view[1] = 2.5;
	var before = view[0] + view[1];
	buf.transfer();
	view[0] = 7;
	out.push(before + " " + String(view[0]) + " " + view.length);
	var rbuf = new ArrayBuffer(16, { maxByteLength: 32 }), rview = new Int32Array(rbuf);
	rview[3] = 9;
	var r3 = rview[3];
	rbuf.resize(8);
	rview[3] = 4;
	out.push(r3 + " " + String(rview[3]) + " " + rview.length);
	var proto = { inherited: 1 }, obj = Object.create(proto), sym = Symbol("k");
	obj.a = 1; obj[sym] = 2;
	var keys = ["a", "z", "inherited", "toString", 0, "0", sym, "b"];
	for (var round = 0; round < 6; round++) {
		var hs = [];
		for (var i = 0; i < keys.length; i++) hs.push(keys[i] in obj ? 1 : 0);
		hs.push("length" in [], 1 in [5, 6], "x" in new Proxy({ x: 1 }, {}));
		try { hs.push("a" in 5) } catch (e) { hs.push(e instanceof TypeError) }
		out.push(hs.join(""));
		if (round === 1) { delete obj.a; obj.b = 1 }
		if (round === 2) { proto.z = 1 }
		if (round === 3) { Object.setPrototypeOf(obj, null) }
		if (round === 4) { obj.a = 3; Object.setPrototypeOf(obj, proto) }
	}
	var bigs = [0n, 1n, -1n, 7n, -9n, 2n ** 31n, 2n ** 48n, -(2n ** 48n), 2n ** 48n - 1n, 2n ** 49n, 2n ** 62n,
		2n ** 63n - 1n, -(2n ** 63n), 2n ** 63n, 2n ** 64n - 1n, -(2n ** 64n) + 1n, 2n ** 64n, 3n ** 50n];
	for (var round = 0; round < 2; round++) {
		var bs = [];
		for (var i = 0; i < bigs.length; i++)
			for (var j = 0; j < bigs.length; j++) {
				var x = bigs[i], y = bigs[j];
				bs.push(String(x + y), String(x - y), String(x * y));
			}
		try { bs.push(String(1n + 1)) } catch (e) { bs.push(e instanceof TypeError) }
		var acc = 0n;
		for (var i = 0; i < 2000; i++) { acc += 2n ** 53n * 3n; acc -= 5n }
		bs.push(String(acc));
		out.push(bs.join(" "));
	}
	var o1 = {}, s1 = Symbol("q");
	var eqs = [0, -0, 1, NaN, Infinity, undefined, null, true, false, "ab", "a" + "b", "", o1, {}, s1, Symbol("q"),
		1n, 2n ** 70n, 2n ** 69n * 2n, "1", 1];
	for (var round = 0; round < 2; round++) {
		var es = [];
		for (var i = 0; i < eqs.length; i++)
			for (var j = 0; j < eqs.length; j++) {
				var x = eqs[i], y = eqs[j];
				if (x === y) es.push(1); else es.push(0);
				if (x !== y) es.push(1); else es.push(0);
			}
		es.push(o1 === o1, s1 === s1, undefined === null);
		out.push(es.join(""));
	}
	function self(f) { return this === undefined ? 0 : 1 }
	out.push(String(self()) + String(self.call(5)));
	var leafer = { arg: function (k) { return k }, two: function (a, b) { return b }, konst: function () { return "c" },
		none: function () {} };
	for (var round = 0; round < 3; round++) {
		var ls = [];
		for (var i = 0; i < 3; i++) ls.push(leafer.arg(i), leafer.arg(), leafer.two(i, { v: i }).v, leafer.two(i),
			leafer.konst(), leafer.none(i), leafer.arg("s" + i));
		out.push(ls.join(","));
	}
	out.join(String.fromCharCode(10))`

// nativeCallCases are calls each of which rhCallW answers one way: the
// frame-entry path the native call handler takes itself (native), or a
// path of Go's -- a built-in, a generator, a frame that keeps its
// arguments or reads new.target, a bound function -- which it must leave
// to Go. Each runs 64 times in a loop; the first call of a site, which
// fills its cache, is Go's in any case.
var nativeCallCases = []struct {
	name, src string
	native    bool
}{
	{"plain", `function f(a, b) { var t = a; return t + b + 1 } function run(n) { var s = 0; for (var i = 0; i < n; i++) s += f(i, 2); return s }`, true},
	{"missing", `function f(a, b, c) { var t = a; return t + (c === undefined ? 1 : 2) + (b === undefined ? 4 : 8) } function run(n) { var s = 0; for (var i = 0; i < n; i++) s += f(i); return s }`, true},
	{"method", `var o = { k: 3, m(x) { var t = this.k; return t + x } }; function run(n) { var s = 0; for (var i = 0; i < n; i++) s += o.m(i); return s }`, true},
	{"leafArg", `var o = { id(x) { return x } }; function run(n) { var s = 0; for (var i = 0; i < n; i++) s += o.id(i); return s }`, true},
	{"leafConst", `function k() { return 7 } function run(n) { var s = 0; for (var i = 0; i < n; i++) s += k(); return s }`, true},
	{"builtin", `function run(n) { var s = 0; for (var i = 0; i < n; i++) s += Math.max(i, 3); return s }`, false},
	{"generator", `function* g() { yield 1 } function run(n) { var s = 0; for (var i = 0; i < n; i++) s += g().next().value; return s }`, false},
	{"arguments", `function f() { return arguments.length } function run(n) { var s = 0; for (var i = 0; i < n; i++) s += f(i, i); return s }`, false},
	{"newTarget", `function f() { return new.target === undefined ? 1 : 2 } function run(n) { var s = 0; for (var i = 0; i < n; i++) s += f(); return s }`, false},
	{"bound", `var b = function (x) { return x + 1 }.bind(null); function run(n) { var s = 0; for (var i = 0; i < n; i++) s += b(i); return s }`, false},
}

// TestRegNativeCalls runs nativeCallsSrc by the Go table's loop and by
// the native handlers, with the collector idle and as while it marks: the
// same answers, and the code made calls of its own.
func TestRegNativeCalls(t *testing.T) {
	nativeCounting(t)
	want := nativeRunSrc(t, nativeCallsSrc, false)
	for _, marking := range []bool{false, true} {
		nativeMarking(t, marking)
		before := atomic.LoadInt64(&nativeCalls)
		if got := nativeRunSrc(t, nativeCallsSrc, true); got != want {
			t.Errorf("native (marking %v): %s, want %s", marking, got, want)
		}
		if atomic.LoadInt64(&nativeCalls) == before {
			t.Errorf("native (marking %v): no call made natively", marking)
		}
	}
}

// TestRegNativeFastPaths runs nativeFastSrc by the Go table's loop and by
// the native handlers, with the collector idle and as while it marks: the
// same answers, and the same exits to Go marking or not.
func TestRegNativeFastPaths(t *testing.T) {
	nativeCounting(t)
	want := nativeRunSrc(t, nativeFastSrc, false)
	var exits [2]int64
	for i, marking := range []bool{false, true} {
		nativeMarking(t, marking)
		before := atomic.LoadInt64(&nativeEnters)
		if got := nativeRunSrc(t, nativeFastSrc, true); got != want {
			t.Errorf("native (marking %v):\n%s\nwant\n%s", marking, got, want)
		}
		exits[i] = atomic.LoadInt64(&nativeEnters) - before
	}
	if exits[0] != exits[1] {
		t.Errorf("%d exits, %d while marking", exits[0], exits[1])
	}
}

// TestRegNativeCallExits checks the native call handler against rhCallW,
// the Go call path, case by case: the same answer, and the call made
// natively exactly where rhCallW enters a frame itself -- at most the
// first calls of the site left to Go -- and left to Go every time where
// rhCallW takes a path of its own. A condition rhCallW gains, or one the
// handler checks that rhCallW does not, shows here.
func TestRegNativeCallExits(t *testing.T) {
	nativeCounting(t)
	const n = 64
	for _, c := range nativeCallCases {
		src := c.src + fmt.Sprintf("; String(run(%d))", n)
		want := nativeRunSrc(t, src, false)
		before := nativeExitOps()
		got := nativeRunSrc(t, src, true)
		after := nativeExitOps()
		exits := after[bytecode.RCallW] - before[bytecode.RCallW] + after[bytecode.RCallMW] - before[bytecode.RCallMW]
		if got != want {
			t.Errorf("%s: %s, want %s", c.name, got, want)
		}
		// The script's call of run and the site's first two calls, which
		// fill its cache, are Go's.
		if c.native && exits > 3 {
			t.Errorf("%s: %d calls left to Go, want the call made natively", c.name, exits)
		} else if !c.native && exits < n {
			t.Errorf("%s: %d calls left to Go of %d, want every one: rhCallW answers it by a path of its own", c.name, exits, n)
		}
	}
}

// TestRegNativeDeep runs recursions past the native frame stack's room,
// by calls the code makes and by calls Go makes between them, to the
// stack's limit: the answers of the Go table's loop.
func TestRegNativeDeep(t *testing.T) {
	src := `
	function deep(n) { return n === 0 ? 0 : 1 + deep(n - 1) }
	function viaGo(n) { return n === 0 ? 0 : 1 + [n - 1].map(viaGo)[0] }
	function mixed(n) { return n === 0 ? 0 : n % 3 === 0 ? 1 + [n - 1].map(mixed)[0] : 1 + mixed(n - 1) }
	function T(n) { this.d = n === 0 ? 0 : 1 + new T(n - 1).d }
	function thrower(n) { if (n === 0) throw new Error("at the bottom"); return mixed(2) + thrower(n - 1) }
	function caught() { try { return thrower(700) } catch (e) { return e.message } }
	function over() { try { return mixed(1e6) } catch (e) { return e instanceof RangeError } }
	[deep(1000), viaGo(300), mixed(2000), new T(500).d, caught(), over()].join()`
	want := nativeRunSrc(t, src, false)
	for _, marking := range []bool{false, true} {
		nativeMarking(t, marking)
		if got := nativeRunSrc(t, src, true); got != want {
			t.Errorf("native (marking %v): %s, want %s", marking, got, want)
		}
	}
}

// TestRegNativeConcurrent runs nativeCallsSrc, the call kernels and
// nativeFastSrc by the native handlers on two runtimes at once, each on a
// goroutine of its own, the code shared: each gives the answers it gives
// alone.
func TestRegNativeConcurrent(t *testing.T) {
	compiler.SetRegisterForm(true)
	fn := compileForTest(t, nativeCallsSrc)
	kern := nativeKernelSource(t, "call_fib", "call_fib_rec", "call_mutual", "call_even", "call_odd") +
		"call_fib(177 * 20); var a = global_res; call_mutual(65 * 20); a + ' ' + global_res"
	kfn := compileForTest(t, kern)
	ffn := compileForTest(t, nativeFastSrc)
	compiler.SetRegisterForm(false)
	SetRegTier(true)
	was := regNativeOn
	regNativeOn = true
	defer func() { SetRegTier(false); regNativeOn = was }()
	run := func() (string, error) {
		r := New(Config{})
		defer r.Close()
		var out []string
		for i := 0; i < 20; i++ {
			for _, f := range []*bytecode.Function{fn, kfn, ffn} {
				v, err := r.Run(f)
				if err != nil {
					return "", err
				}
				out = append(out, v.String().Go())
			}
		}
		if r.frameDepth != 0 {
			return "", fmt.Errorf("frame depth %d left", r.frameDepth)
		}
		return strings.Join(out, ";"), nil
	}
	want, err := run()
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	got := make([]string, 2)
	errs := make([]error, 2)
	for i := range got {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got[i], errs[i] = run()
		}()
	}
	wg.Wait()
	for i := range got {
		if errs[i] != nil {
			t.Errorf("goroutine %d: %v", i, errs[i])
		} else if got[i] != want {
			t.Errorf("goroutine %d: answers differ from one runtime's alone", i)
		}
	}
}

// TestRegNativeOpMix, with QJS_NATIVE_MIX naming micro-benchmarks
// (comma-separated, each name:n), logs each one's operation mix: the
// instructions dispatched an operation by their operation, run by the Go
// table, and how many of them the native handlers left to Go. It is how
// a test slower natively shows which native paths it lacks.
func TestRegNativeOpMix(t *testing.T) {
	only := os.Getenv("QJS_NATIVE_MIX")
	if only == "" {
		t.Skip("QJS_NATIVE_MIX names no test")
	}
	var defs string
	for _, f := range []string{"../v8bench/microbench.js", "../v8bench/microbench_extra.js"} {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		s := strings.ReplaceAll(string(b), "\r\n", "\n")
		if i := strings.LastIndex(s, "\nif (typeof scriptArgs === \"undefined\")"); i >= 0 {
			s = s[:i+1]
		}
		defs += s
	}
	compiler.SetRegisterForm(true)
	defFn := compileForTest(t, "if (typeof console === 'undefined') globalThis.console = { log: function () {} };\n"+defs)
	compiler.SetRegisterForm(false)
	SetRegTier(true)
	was := regNativeOn
	defer func() { SetRegTier(false); regNativeOn = was }()
	nativeCounting(t)
	var counts [256]int64
	orig := regTable
	defer func() { regTable = orig }()
	for _, spec := range strings.Split(only, ",") {
		name, n, _ := strings.Cut(spec, ":")
		if n == "" {
			n = "1000"
		}
		compiler.SetRegisterForm(true)
		call := compileForTest(t, fmt.Sprintf("String(%s(%s))", name, n))
		compiler.SetRegisterForm(false)
		var before, after [256]int64
		run := func(native bool) int64 {
			regNativeOn = native
			r := New(Config{})
			defer r.Close()
			if _, err := r.Run(defFn); err != nil {
				t.Fatal(err)
			}
			counts = [256]int64{}
			before = nativeExitOps()
			v, err := r.Run(call)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			after = nativeExitOps()
			var ops int64
			fmt.Sscan(v.String().Go(), &ops)
			return max(ops, 1)
		}
		for i := range regTable {
			h, op := orig[i], i
			regTable[i] = func(regs *[1 << 16]Value, w bytecode.RInstr, pc int, st *regState) int {
				counts[op]++
				return h(regs, w, pc, st)
			}
		}
		ops := run(false)
		regTable = orig
		mix := counts
		run(true)
		var total, exited int64
		type row struct {
			op       bytecode.ROp
			n, exits int64
		}
		var rows []row
		for op := range mix {
			total += mix[op]
			e := after[op] - before[op]
			exited += e
			if mix[op] > 0 || e > 0 {
				rows = append(rows, row{bytecode.ROp(op), mix[op], e})
			}
		}
		for i := range rows {
			for j := i + 1; j < len(rows); j++ {
				if rows[j].n > rows[i].n {
					rows[i], rows[j] = rows[j], rows[i]
				}
			}
		}
		var parts []string
		for _, r := range rows {
			if r.n*50 < total && r.exits*50 < total {
				continue
			}
			parts = append(parts, fmt.Sprintf("%s %.2f/%.2f", r.op, float64(r.n)/float64(ops), float64(r.exits)/float64(ops)))
		}
		t.Logf("%-24s %7.2f instructions an op, %6.2f exited (%4.1f%%): %s", name,
			float64(total)/float64(ops), float64(exited)/float64(ops), 100*float64(exited)/float64(max(total, 1)), strings.Join(parts, ", "))
	}
}
