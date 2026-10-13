package quickjs_test

import (
	"testing"

	quickjs "github.com/go-quickjs/go-quickjs"
)

func TestYoungGenerationOption(t *testing.T) {
	for _, capacity := range []int{0, 1, 8} {
		r := quickjs.New(quickjs.WithYoungGeneration(capacity))
		if st := r.YoungGenerationStats(); st != (quickjs.YoungGenerationStats{}) {
			t.Fatalf("eager nursery: %+v", st)
		}
		if got := evalString(t, r, `function f(){let o;for(let i=0;i<100;i++)o={a:i,b:i+1};return o}let o=f();f();o.a+":"+o.b`); got != "99:100" {
			t.Fatal(got)
		}
		r.Close()
		if st := r.YoungGenerationStats(); st != (quickjs.YoungGenerationStats{}) {
			t.Fatalf("closed nursery: %+v", st)
		}
	}
}
