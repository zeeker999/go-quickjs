package jit

import (
	"errors"
	"os"
	"runtime"
	"testing"
	"unsafe"
)

func TestRequiredBackend(t *testing.T) {
	if os.Getenv("QUICKJS_REQUIRE_JIT") != "" && !Supported() {
		t.Fatalf("native backend required but excluded on %s/%s", runtime.GOOS, runtime.GOARCH)
	}
}

func TestLoopStateLayout(t *testing.T) {
	var state loopState
	if unsafe.Sizeof(state) != 32 || unsafe.Offsetof(state.next) != 0 ||
		unsafe.Offsetof(state.step) != 8 || unsafe.Offsetof(state.sum) != 16 ||
		unsafe.Offsetof(state.count) != 24 {
		t.Fatal("loopState no longer matches the native ABI")
	}
}

func TestUninitializedLoop(t *testing.T) {
	for _, loop := range []*Loop{nil, {}} {
		if _, err := loop.Run(0, 0, 0, 0); !errors.Is(err, ErrClosed) {
			t.Fatalf("Run: %v, want ErrClosed", err)
		}
		if loop.Size() != 0 {
			t.Fatal("uninitialized loop owns code")
		}
		if err := loop.Close(); err != nil {
			t.Fatal(err)
		}
	}
}
