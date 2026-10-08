package quickjs_test

import (
	"testing"

	"github.com/go-quickjs/go-quickjs"
)

// The opt-in is valid in ordinary builds too, where it falls back. Pin its
// independence from code-generation policy and Node compatibility options.
func TestJITOption(t *testing.T) {
	for _, node := range []bool{false, true} {
		opts := []quickjs.Option{quickjs.WithJIT(), quickjs.WithoutCodeGeneration()}
		if node {
			opts = append(opts, quickjs.WithNodeQuirks())
		}
		r := quickjs.New(opts...)
		defer r.Close()
		for _, tc := range []struct{ source, want string }{
			{`function f(n) { let s=0; for(let i=0;i<n;i++) s+=i; return s } f(10000)`, "49995000"},
			{`let count=0; let x=f({valueOf() { count++; return 3 }}); x+":"+count`, "3:4"},
			{`typeof eval`, "undefined"},
			{`function z(n) { let a=0; for(let i=0;i<n;i++) a=-a; return a } Object.is(z(1),-0)`, "true"},
		} {
			if got := evalString(t, r, tc.source); got != tc.want {
				t.Fatalf("node=%v: %s = %s, want %s", node, tc.source, got, tc.want)
			}
		}
		if _, err := r.Eval(`Function("return 1")()`); err == nil {
			t.Fatal("JIT enabled blocked Function constructor")
		}
	}
}
