package v8bench

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	quickjs "github.com/go-quickjs/go-quickjs"
	"github.com/go-quickjs/go-quickjs/internal/compiler"
	"github.com/go-quickjs/go-quickjs/internal/vm"
)

// The register tier's differential tests (docs/register-interpreter.md,
// "Raw int32 registers"): the same programs run with the register tier off
// and on, each in runtimes of their own, and every result compared. The
// register code's correctness rests on its verifier for memory and on these
// for meaning: what a raw int32 register holds must be what the Value the
// stack form keeps would be, at every overflow, -0, NaN and shift.

// withRegTier runs f with the register backend and tier set to on, and puts
// back what the environment says after.
func withRegTier(on bool, f func()) {
	compiler.SetRegisterForm(on)
	vm.SetRegTier(on)
	defer func() {
		env := os.Getenv("QJS_REG") != ""
		compiler.SetRegisterForm(env)
		vm.SetRegTier(env)
	}()
	f()
}

// regDiffRun evaluates src in a new runtime and then each of calls, and
// returns each call's result as a string, or its error's.
func regDiffRun(t *testing.T, src string, calls []string) []string {
	t.Helper()
	rt := quickjs.New()
	defer rt.Close()
	if _, err := rt.Eval(src); err != nil {
		t.Fatalf("eval: %v", err)
	}
	out := make([]string, len(calls))
	for i, c := range calls {
		v, err := rt.Eval(c)
		if err != nil {
			out[i] = "error: " + err.Error()
			continue
		}
		out[i] = v.String()
	}
	return out
}

// regDiff runs src and calls with the register tier off, twice, and on, and
// fails where the tier on answers otherwise. A call whose two answers with
// the tier off differ -- a clock, Math.random -- is not compared. It
// returns how many functions the tier gave register code.
func regDiff(t *testing.T, src string, calls []string) int64 {
	t.Helper()
	var off, again, on []string
	withRegTier(false, func() { off = regDiffRun(t, src, calls) })
	withRegTier(false, func() { again = regDiffRun(t, src, calls) })
	before := vm.RegsBuilt()
	withRegTier(true, func() { on = regDiffRun(t, src, calls) })
	built := vm.RegsBuilt() - before
	bad, skipped := 0, 0
	defer func() {
		t.Logf("%d calls compared, %d not (their answers vary), %d functions as register code",
			len(calls)-skipped, skipped, built)
	}()
	for i := range calls {
		if off[i] != again[i] {
			skipped++
			continue
		}
		if off[i] == on[i] {
			continue
		}
		if bad++; bad <= 20 {
			t.Errorf("%s:\n  register tier off: %.300s\n  register tier on:  %.300s", calls[i], off[i], on[i])
		}
	}
	if bad > 20 {
		t.Errorf("and %d more", bad-20)
	}
	return built
}

// regDiffResult is the expression a call is compared by: what it returns
// and what it left in global_res, as text.
func regDiffResult(call string) string {
	return fmt.Sprintf(`(function () { var r = %s; var g = global_res; try { g = JSON.stringify(g) } catch (e) { g = String(g) } return String(r) + " / " + g })()`, call)
}

func TestRegDiffMicrobenchmarks(t *testing.T) {
	if testing.Short() {
		t.Skip("runs every micro-benchmark three times")
	}
	end := strings.LastIndex(microbench, "\nif (typeof scriptArgs === \"undefined\")")
	if end < 0 {
		t.Fatal("microbench.js: no call of main to leave out")
	}
	src := `if (typeof console === "undefined") globalThis.console = { log: function () {} };
if (typeof performance === "undefined") globalThis.performance = { now: function () { return 0 } };
var global_res;
` + microbench[:end+1] + microbenchExtra
	calls := make([]string, 0, len(microWork))
	for _, w := range microWork {
		// A little of each test's work: enough for its loops to turn and
		// its functions to run as each tier runs them.
		n := max(w.n/4096, 1)
		calls = append(calls, regDiffResult(fmt.Sprintf("%s(%d)", w.name, n)))
	}
	if built := regDiff(t, src, calls); built < 100 {
		t.Errorf("only %d functions ran as register code", built)
	}
}

func TestRegDiffV8Suite(t *testing.T) {
	if testing.Short() {
		t.Skip("runs the V8 suite three times")
	}
	dir := os.Getenv("QUICKJS_V8_DIR")
	if dir == "" {
		t.Skip("QUICKJS_V8_DIR names no checkout of the V8 suite (v8bench -fetch)")
	}
	var src strings.Builder
	for _, f := range suites["v8"].files {
		b, err := os.ReadFile(filepath.Join(dir, f))
		if err != nil {
			t.Skip(err)
		}
		src.Write(b)
		src.WriteString("\n")
	}
	// Each benchmark run three times, its own checks of its answers on --
	// Richards' counts, DeltaBlue's projections, Crypto's decryption,
	// NavierStokes' checksum, Splay's tree -- and what it leaves to see
	// compared: the error a check throws, if any.
	src.WriteString(`
function regDiffBench(name) {
	var out = [];
	BenchmarkSuite.suites.forEach(function (s) {
		s.benchmarks.forEach(function (b) {
			if (b.name !== name) return;
			try {
				b.Setup();
				for (var i = 0; i < 3; i++) out.push(String(b.run()));
				b.TearDown();
			} catch (e) { out.push("threw " + e) }
		});
	});
	return out.join(",");
}
var global_res;
`)
	var calls []string
	for _, name := range []string{"Richards", "DeltaBlue", "Encrypt", "Decrypt", "RayTrace", "EarleyBoyer", "RegExp", "Splay", "NavierStokes"} {
		calls = append(calls, fmt.Sprintf("regDiffBench(%q)", name))
	}
	if built := regDiff(t, src.String(), calls); built < 100 {
		t.Errorf("only %d functions ran as register code", built)
	}
}

// TestRegDiffInt32 compares generated integer arithmetic: functions of
// loops over locals that bitwise operators, shifts and | 0 keep int32s,
// and sums, differences and products that leave them -- past 2^31, to -0,
// to NaN -- with shifts by 32 and more, >>> 0, and doubles mixed in.
func TestRegDiffInt32(t *testing.T) {
	rnd := rand.New(rand.NewSource(1))
	consts := []string{"0", "1", "-1", "2", "7", "31", "32", "33", "64", "255", "0x3fff", "0xfffffff",
		"0x7fffffff", "-0x80000000", "0x80000000", "0xffffffff", "4294967296", "-0", "0.5", "-1.5",
		"1e10", "NaN", "Infinity", "65535", "65536", "-65536", "46341", "-46341"}
	ops := []string{"+", "-", "*", "&", "|", "^", "<<", ">>", ">>>", "/", "%"}
	var src strings.Builder
	var calls []string
	vars := []string{"a", "b", "c", "d"}
	operand := func() string {
		if rnd.Intn(3) == 0 {
			return consts[rnd.Intn(len(consts))]
		}
		return vars[rnd.Intn(len(vars))]
	}
	var expr func(depth int) string
	expr = func(depth int) string {
		if depth == 0 || rnd.Intn(4) == 0 {
			return operand()
		}
		e := "(" + expr(depth-1) + " " + ops[rnd.Intn(len(ops))] + " " + expr(depth-1) + ")"
		switch rnd.Intn(6) {
		case 0:
			e = "(" + e + " | 0)"
		case 1:
			e = "(" + e + " >>> 0)"
		case 2:
			e = "(" + e + " & 0x3fff)"
		}
		return e
	}
	cmps := []string{"<", "<=", ">", ">=", "==", "!=", "===", "!=="}
	for i := 0; i < 300; i++ {
		name := fmt.Sprintf("g%d", i)
		fmt.Fprintf(&src, "function %s(a, b, n) {\n\tvar c = %s, d = %s, t = [], out = [];\n", name, operand(), operand())
		fmt.Fprintf(&src, "\tfor (var i = 0; i < 8; i++) t[i] = i * 3;\n")
		fmt.Fprintf(&src, "\tfor (var j = 0; j < n; j++) {\n")
		for k := 0; k < 3; k++ {
			v := vars[rnd.Intn(len(vars))]
			switch rnd.Intn(5) {
			case 0:
				fmt.Fprintf(&src, "\t\t%s = %s;\n", v, expr(3))
			case 1:
				fmt.Fprintf(&src, "\t\t%s = (%s + j) | 0;\n", v, v)
			case 2:
				fmt.Fprintf(&src, "\t\t%s = t[j & 7] + %s;\n", v, expr(2))
			case 3:
				fmt.Fprintf(&src, "\t\tif (%s %s %s) %s++; else %s--;\n", expr(2), cmps[rnd.Intn(len(cmps))], expr(1), v, v)
			default:
				fmt.Fprintf(&src, "\t\tt[(j + %d) & 7] = %s;\n", rnd.Intn(8), expr(2))
			}
		}
		fmt.Fprintf(&src, "\t\tout.push(a, b, c, d);\n\t}\n")
		fmt.Fprintf(&src, "\treturn out.map(function (x) { return Object.is(x, -0) ? \"-0\" : String(x) }).join() + \"|\" + t.join();\n}\n")
		for _, args := range [][2]string{{"1", "2"}, {"0x7ffffff0", "-0x7ffffff0"}, {"-0", "0.25"}, {"46340", "46341"}, {"NaN", "3"}} {
			calls = append(calls, fmt.Sprintf("%s(%s, %s, 6)", name, args[0], args[1]))
		}
	}
	if built := regDiff(t, src.String(), calls); built < 300 {
		t.Errorf("only %d functions ran as register code", built)
	}
}

// TestRegDiffStacksAndNames compares what a stack trace says of register
// frames -- each call's position, its function's name, whether it is a
// construct, its receiver's type -- after the call sites' caches are warm,
// and the names a function's body resolves: its own name shadowed in a
// block, the arguments object an apply other than the built-in is given.
func TestRegDiffStacksAndNames(t *testing.T) {
	src := `
function frames(e) { return e.stack.split("\n").slice(1).map(function (s) { return s.trim() }).join(" | ") }
function thrower() { return null.x }
function calls(n) { return n > 0 ? calls(n - 1) + 1 : thrower() }
function made() { return new Error("m") }
function viaArrow() { return (() => made())() }
function L() { return Error.call(this) }
function M() { this.e = made() }
var o = { m: function () { return made() }, n: function named() { return made() } };
function Foo() {} Foo.prototype.m = function () { return made() };
var a = {}; a.b = function () { return made() };
function self() { return (function g() { { let g = 1 } return typeof g })() }
function selfVar() { return (function g() { var g; return typeof g })() }
function own(a) { var f = { apply: function (t, args) { return args } }; var x = f.apply(null, arguments); a = 2; var y = f.apply(null, arguments); return [x === y, x[0], x.length] }
function warm(f) { for (var i = 0; i < 4; i++) f() }
function stackOf(f) { try { var v = f(); return v && v.stack ? frames(v) : String(v) } catch (e) { return frames(e) } }
`
	var calls []string
	for _, f := range []string{
		"function () { return calls(3) }",
		"made", "viaArrow", "function () { return new L() }", "function () { return new M().e }",
		"function () { return o.m() }", "function () { return o.n() }", "function () { return new Foo().m() }",
		"function () { return a.b() }", "self", "selfVar", "function () { return own(1, 2) }",
	} {
		calls = append(calls, fmt.Sprintf("(function () { var f = %s; warm(function () { try { f() } catch (e) {} }); return stackOf(f) })()", f))
	}
	regDiff(t, src, calls)
}

// TestRegDiffChainsAndClosures compares string chains -- when each part
// is converted, against what the parts after it do, numbers, -0, symbols
// that throw, chains past maxChainParts -- and closures made once in a
// loop or taking what they capture by value.
func TestRegDiffChainsAndClosures(t *testing.T) {
	src := `
var log = [];
function o(n) { return { toString: function () { log.push("s" + n); return "<" + n + ">" }, valueOf: function () { log.push("v" + n); return n } } }
function e(n) { log.push("e" + n); return n }
function chain(a, b, c) { return a + b + "|" + c + e(1) + "|" + o(3) + -0 + 1.5 + e("x") }
function pre(a, b) { return a + b + "s" + o(1) + "" + b }
function long(a) { var s = ""; for (var i = 0; i < 3; i++) s = "a" + a + 1 + a + 2 + a + 3 + a + 4 + a + 5 + a + 6 + a + 7 + a + 8 + a + 9 + a + 10 + a + 11 + a + 12 + a + 13 + a + 14 + a + 15 + a + 16 + a + 17 + a + 18 + a + i; return s }
function sym(a) { try { return "x" + a + Symbol() } catch (x) { return x.constructor.name } }
function reuse(n) { 'use strict'; var t = 0; for (var i = 0; i < n; i++) { var g = function () { return i }; t += g() } return t }
function byValue(a, b, c) { 'use strict'; return () => a + b + c }
function mixed(a, b, c) { 'use strict'; b++; var f = function () { return a + b + c }; b++; return f() }
function run(f) { log = []; var v = f(); return v + " / " + log.join() }
function keys() { var o = { "<1>": 1 }, k = o(1); o[k] += 1; o[k]++; ++o[k]; return o["<1>"] }
function bad(b) { var k = o(2); b[k] += 1 }
function bad2(b) { var k = o(2); b[k]++ }
function upd(t) { t.x++ }
function updLoop(t) { for (var i = 0; i < 2; i++) t[0] += 1 }
function made(t, i) { t[i - 1] += 1; t[i + ""] |= 2; t[` + "`k${i}`" + `] += "s"; t[-i]++; t[i++] += 1; t[!i] = 1; return JSON.stringify(t) + i }
function post(t, i) { var s = []; for (var n = 0; n < 4; n++) s.push(t[i++], t[i--]); return s.join() + " " + typeof i + " " + i }
function id(a, b) { return b }
function none() {}
var ido = { id: function (k) { return k }, none: none, arrow: (k) => k };
class Ret { constructor(k) { return k } }
function ctors(n) { var r = []; for (var i = 0; i < n; i++) { try { r.push(Ret(i)) } catch (e) { r.push(e.name) } r.push(typeof new id(i, i), typeof new Ret({}), typeof new Ret(i)) } return r.join() }
function trivial(n) { var s = []; for (var i = 0; i < n; i++) s.push(id(i, i * 2), id(i), none(i), ido.id(i), ido.id(), ido.none(), ido.arrow(i), id.call(3, 1, i), ido.id.apply("s", [i])); return s.join() }
function madeObj(t) { var k = o(5); t[k + ""] += 1; t["" + e(6)] += 1; return JSON.stringify(t) }
function caught(f, a) { try { f(a) } catch (x) { return x.message + " " + x.stack.split(String.fromCharCode(10))[1] } }
`
	calls := []string{
		`run(function () { return chain(1, 2, o(2)) })`, `run(function () { return chain("a", o(1), 3) })`,
		`run(function () { return pre(1, 2) })`, `run(function () { return pre(o(4), "z") })`,
		`run(function () { return long(7) })`, `run(function () { return long(o(9)) })`,
		`run(function () { return sym(1) })`, `run(function () { return reuse(5) + reuse(1) })`,
		`run(function () { var f = byValue(1, 2, 3); return f() + f() })`, `run(function () { return mixed(1, 2, 3) })`,
		`run(keys)`, `run(function () { return caught(bad, null) })`, `run(function () { return caught(bad2, undefined) })`,
		`caught(upd, null)`, `caught(upd, undefined)`, `caught(updLoop, null)`,
		`made({}, 3)`, `run(function () { return madeObj({}) })`, `run(function () { return caught(madeObj, null) })`,
		`caught(made, null)`, `caught(made, undefined)`,
		`trivial(5)`, `ctors(4)`, `post([1, 2, , 4, 5], 0)`, `post([1, 2, 3], "1")`, `post("abcdef", 1)`, `post({ 1: "a", 2: "b" }, 1)`,
		`run(function () { return post([7, 8, 9], o(1)) })`, `caught(post, null)`, `post([1, 2], 1.5)`, `post([1, 2], -1)`,
	}
	for i := range calls {
		calls[i] = fmt.Sprintf("(function () { var r; for (var k = 0; k < 4; k++) r = %s; return r })()", calls[i])
	}
	regDiff(t, src, calls)
}

// TestRegDiffSites compares the register operations given a site or a fast
// path of their own after the stage's first measurements, each with its
// slow paths.
func TestRegDiffSites(t *testing.T) {
	src := `
function grow(n, m) { var out = []; for (var j = 0; j < m; j++) { var a = []; for (var i = 0; i < n; i++) a.push(i, " "); out.push(a.length, a[0], a[a.length - 1]); if (j == 2) a.x = 1; } return out.join() }
function fresh() { var a = [], b = []; a.push(1); return [a === b, a.length, b.length, Array.isArray(b), Object.getPrototypeOf(b) === Array.prototype].join() }
function negs(x) { var r = []; r.push(x === -0, x !== -1, x + -1, x > -2, x < -2, x <= -32768, x >= -32769, x - -1, x & -1, x | -2, x ^ -3, x << -31, x >> -1, x >>> -1, -2147483648, x === -2147483648, Object.is(x * -0, -0), 1 / -0, [-1, -0, x].map(function (v) { return 1 / v }).join("/"), "" + -1 + x + -0); var o = { a: x }; o.a += -1; r.push(o.a); switch (x) { case -1: r.push("m1"); break; case -0: r.push("z"); break; case -32768: r.push("big"); break; case -2147483648: r.push("min"); break; default: r.push("d") } return r.join() }
function konst() { return -1 }
function konstF() { return 1.5 }
function konstT() { return true }
function konstN() { return null }
function konsts(n) { var s = 0, t = []; for (var i = 0; i < n; i++) { s += konst() + konstF(i) + konst(1, 2); t.push(konstT(), konstN(), konst.call("x"), new konst() instanceof konst) } return s + t.join() }
function kept(n) { var all = []; for (var j = 0; j < n; j++) { var a = []; for (var i = 0; i < 300; i++) a.push(j); all.push(a) } return all.map(function (a) { return a.length + ":" + a[299] }).join() }
`
	calls := []string{
		`negs(-1)`, `negs(0)`, `negs(-0)`, `negs(-2)`, `negs(-32768)`, `negs(-2147483648)`, `negs("-1")`, `negs(undefined)`, `negs(5)`, `negs(-1.5)`, `negs({ valueOf: function () { return -1 } })`,
		`konsts(5)`,
		`grow(300, 6)`, `grow(3, 4)`, `grow(0, 3)`, `grow(700, 3)`, `fresh()`, `kept(4)`,
	}
	for i := range calls {
		calls[i] = fmt.Sprintf("(function () { var r; for (var k = 0; k < 4; k++) r = %s; return r })()", calls[i])
	}
	regDiff(t, src, calls)
}
