package main

import (
	"strings"
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/compiler"
)

// count is the tally of one script.
func count(t *testing.T, src string) *tally {
	t.Helper()
	compiler.SetRegisterForm(true)
	c := newTally()
	c.countSource("case.js", src, false)
	if c.failed != 0 {
		t.Fatalf("%q did not compile", src)
	}
	if c.reserved != 0 {
		t.Errorf("%d instructions of the reserved operations", c.reserved)
	}
	return c
}

// The step pairs of each loop shape, by the branch's kind and the step's,
// and those no fused instruction takes.
func TestStepPairs(t *testing.T) {
	c := count(t, `function f(n, a) {
		for (let i = 0; i < n; i++) {}
		for (var k = 0; k < 10; k += 2) {}
		var j = 10;
		while (--j >= 0) {}
		do {} while (--n > 0);
		for (var m = 0; m <= n; m += 100) {}
		var q = 0;
		while (++q < q) {}
		return function () { for (var u = 0; u <= a; u++) {} };
	}
	f(1, 2)();`)
	want := map[string]map[string]int{
		"jlt":        {"+1": 1},
		"jlt_imm":    {"+k": 1},
		"jge_imm":    {"-1": 1},
		"jgt_imm":    {"-1": 1},
		"jle_upval":  {"+1": 1},
		"jnlt":       nil,
		"jle_imm":    nil,
		"jle":        nil,
		"jlt_upval":  nil,
		"jnlt_upval": nil,
	}
	for k, m := range want {
		for s, n := range m {
			if got := c.step[k][s]; got != n {
				t.Errorf("%s %s: %d pairs, want %d", k, s, got, n)
			}
		}
		if m == nil && sum(c.step[k]) != 0 {
			t.Errorf("%s: %d pairs, want none", k, sum(c.step[k]))
		}
	}
	for _, k := range []string{"jlt", "jlt_imm", "jge_imm", "jgt_imm", "jle_upval"} {
		if c.back[k] != 1 {
			t.Errorf("%s: %d backward, want 1", k, c.back[k])
		}
	}
	if c.stepMiss["step past 7 bits (jle)"] != 1 {
		t.Errorf("i += 100: misses %v, want one past 7 bits", c.stepMiss)
	}
	if c.stepMiss["bound is the counter"] != 1 {
		t.Errorf("++q < q: misses %v, want one whose bound is the counter", c.stepMiss)
	}
}

// The operators' patterns, each counted where regcomp's binary would
// match it, and the near forms that are not.
func TestOperatorPatterns(t *testing.T) {
	c := count(t, `function f(x, y, a, b, n, s) {
		var r = 0;
		r = x * 3; r = 3 * x.p; r = x * 70000; r = x * 1.5; r *= 4; r = x * y;
		r = (x + y) | 0; r = (x - y) | 0; r = (x + 1) | 0; r = (x - 4) | 0; r = (x - -32768) | 0;
		r = ("a" + x) | 0; r = (x + y) | 1; r = (x * y) | 0;
		r = (b ^ (a >>> 7)) | 0; r = (x & 255) | 0; r = (a ^ b) | 0; r = x | 0; r |= 0;
		if ((a | 0) < (n | 0)) r++;
		if ((a | 0) <= 10) r++;
		if ((a | 0) > n) r++;
		return r;
	}`)
	want := map[string]int{
		"mul: x * k":                           2,
		"mul: k * x":                           1,
		"mul: x * k, k not 16-bit int":         2,
		"or0: (x + y) | 0  -> add_i32":         1,
		"or0: (x - y) | 0  -> sub_i32":         2,
		"or0: (x + k) | 0  -> add_imm_i32":     1,
		"or0: (x - k) | 0  -> add_imm_i32":     1,
		"or0: (string chain) | 0":              1,
		"or0: (x * y) | 0":                     1,
		"or0: (int32 bitwise) | 0, identity":   2,
		"or0: (x bitwise y) | 0, not provable": 1,
		"cmp: (a|0) < (b|0)":                   1,
		"cmp: (a|0) <= k":                      1,
		"cmp: (a|0) > b":                       1,
	}
	// The | 0s: the ten of the parentheses above, x | 0 itself, r |= 0
	// and the four of the comparisons.
	want["or0: all x | 0"] = 16
	for k, n := range want {
		if got := c.expr[k]; got != n {
			t.Errorf("%s: %d, want %d", k, got, n)
		}
	}
	for k, n := range c.expr {
		if _, ok := want[k]; !ok {
			t.Errorf("%s: %d, want none", k, n)
		}
	}
}

// Code a file evals from a long string literal, as Octane's zlib does, is
// counted as a file of its own.
func TestEvaledCode(t *testing.T) {
	code := "function g(x, y) { return (x + y) | 0 }" + strings.Repeat(" ", 1000)
	c := count(t, "var myEval = eval; myEval('"+code+"'); eval('(1 + 2) | 0');")
	if c.files != 2 || c.evals != 1 || c.expr["or0: (x + y) | 0  -> add_i32"] != 1 {
		t.Errorf("%d files, %d evaled, %v; want 2, 1 and one add_i32", c.files, c.evals, c.expr)
	}
}

// The report names every kind it counted.
func TestReport(t *testing.T) {
	c := count(t, `function f(n) { for (let i = 0; i < n; i++) {} return (n + 1) | 0 }`)
	var sb strings.Builder
	c.report(&sb, "case")
	for _, s := range []string{"case: 1 files", "jlt ", "add_imm_i32"} {
		if !strings.Contains(sb.String(), s) {
			t.Errorf("report has no %q:\n%s", s, sb.String())
		}
	}
}
