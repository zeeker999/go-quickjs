package oracle

import (
	"math"
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/compiler"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
	"github.com/go-quickjs/go-quickjs/internal/parser"
)

func TestNumberString(t *testing.T) {
	for x, want := range map[float64]string{1: "1", -1.5: "-1.5", 1e21: "1e+21", 1e20: "100000000000000000000",
		0.000001: "0.000001", 1e-7: "1e-7", 123.456: "123.456", 0.30000000000000004: "0.30000000000000004", -2.5e-10: "-2.5e-10",
		math.MaxFloat64: "1.7976931348623157e+308", 4294967295: "4294967295"} {
		if got := NumberString(x); got != want {
			t.Errorf("%v: got %s, want %s", x, got, want)
		}
	}
}

// TestMachine runs register code through the evaluator and the machine:
// closures and their captured bindings, objects, methods and
// constructions, arrays and strings, recursion, globals the host gives.
func TestMachine(t *testing.T) {
	src := `
var K = 10;
function fib(n) { return n < 2 ? n : fib(n - 1) + fib(n - 2) }
function counter(n) { var c = 0; var inc = function (d) { c += d; return c }; for (var i = 0; i < n; i++) inc(i); return c }
function Point(x, y) { this.x = x; this.y = y }
function points(n) {
	Point.prototype.norm = function () { return this.x * this.x + this.y * this.y };
	var s = 0;
	for (var i = 0; i < n; i++) { var p = new Point(i, i + 1); s += p.norm() }
	return s;
}
function arrays(n) { var a = []; for (var i = 0; i < n; i++) a[i] = i * K; var s = 0; for (var j = 0; j < a.length; j++) s += a[j]; return s + a.length }
function strs(n) { var s = ""; for (var i = 0; i < n; i++) s += i + ","; return s + s.length + typeof s }
function objs(n) { var o = { a: n, b: [1, 2, 3], c: "x" }; o.a += 2; o.d = o.b.length; return o.a * 100 + o.d + (o.c === "x" ? 0.5 : 0) + (o.e === undefined ? 1000 : 0) }
function builtins(n) { var a = [3, 1, 2]; a.push(n, -n); var b = a.map(function (x) { return x * 2 }); var s = 0; for (var x of b) s += x; return Math.max(a.pop(), Math.floor(n / 3)) + s + a.indexOf(n) + a.join("-").length + String.fromCharCode(65 + n).charCodeAt(0) }
function scopes(n) { var fs = []; for (let i = 0; i < n; i++) fs.push(function () { return i }); var s = 0; for (var j = 0; j < fs.length; j++) s = s * 10 + fs[j](); return s }
function klass(n) { var c = new Point(n, 1); return c.x + (c instanceof Point ? 100 : 0) + ({} instanceof Point ? 1000 : 0) }
function catches(n) { var r = 0; try { var o; r = o.x } catch (e) { r = typeof e } try { if (n > 2) throw n * 2 } catch (e) { r += e } return r + inner(n) }
function inner(n) { try { thrower(n) } catch (e) { return e.message.length } finally { n++ } return -1 }
function thrower(n) { throw new RangeError("abc") }
function shapes(n) { var o = { a: 1, b: 2 }; var [x, y, ...z] = [n, n + 1, n + 2, n + 3]; var s = ""; for (var k in o) s += k; return s + x + y + z.length }
function codes(s) { var h = 0; for (var i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0; return h }
`
	compiler.SetRegisterForm(true)
	prog, err := parser.Parse(src, parser.Options{})
	if err != nil {
		t.Fatal(err)
	}
	top, err := compiler.Compile(prog, compiler.Options{Text: src})
	compiler.SetRegisterForm(false)
	if err != nil {
		t.Fatal(err)
	}
	m := New()
	fns := map[string]*bytecode.Function{}
	for _, c := range top.Constants {
		if c.Kind == bytecode.ConstFunction {
			fns[c.Fn.Name] = c.Fn
			m.SetGlobal(c.Fn.Name, m.Function(c.Fn))
		}
	}
	m.Global = func(name string) (ir.Value, bool) {
		if name == "K" {
			return ir.Float(10), true
		}
		return ir.Value{}, false
	}
	for _, tc := range []struct {
		name string
		arg  ir.Value
		want string
	}{
		{"fib", ir.Float(15), "610"},
		{"counter", ir.Float(10), "45"},
		{"points", ir.Float(4), "44"},
		{"arrays", ir.Float(5), "105"},
		{"strs", ir.Float(3), `"0,1,2,6string"`},
		{"objs", ir.Float(1), "1303.5"},
		{"codes", m.String("hello"), "99162322"},
		{"builtins", ir.Float(4), "92"},
		{"catches", ir.Float(4), `"object83"`},
		{"shapes", ir.Float(4), `"ab452"`},
		{"scopes", ir.Float(4), "123"},
		{"klass", ir.Float(4), "104"},
	} {
		v, ok, err := m.Call(m.globals[tc.name], ir.Value{}, []ir.Value{tc.arg})
		if err != nil || !ok {
			t.Errorf("%s: %v, stopped at %q", tc.name, err, m.Stop)
			continue
		}
		if got, _ := m.Text(v); got != tc.want {
			t.Errorf("%s: got %s, want %s", tc.name, got, tc.want)
		}
	}
}
