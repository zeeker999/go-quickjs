package vm

import (
	"runtime"
	"testing"
)

// TestConstStringShared pins string constants of one text as one string
// of the runtime's, which === then finds equal by identity: two functions'
// literals, and a script's run later, share it while it lives; one no
// constant holds any more is swept out of the table.
func TestConstStringShared(t *testing.T) {
	r := New(Config{})
	defer r.Close()
	a, b := r.constString("forward"), r.constString("forward")
	if a != b {
		t.Fatal("two constants of one text made two strings")
	}
	if c := r.constString("backward"); c == a || c.Go() != "backward" {
		t.Fatal("a constant of another text shares its string")
	}
	for i := 0; i < 2000; i++ {
		r.constString("x" + string(rune('a'+i%26)) + string(rune('a'+i/26%26)) + string(rune('a'+i/676)))
		if i%500 == 0 {
			runtime.GC()
		}
	}
	runtime.GC()
	r.constString("last")
	if n := len(r.strConsts); n > 1500 {
		t.Errorf("%d strings kept of constants no value holds", n)
	}
	runtime.KeepAlive(a)
}
