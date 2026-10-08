//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package jit

import (
	"bytes"
	"compress/gzip"
	"errors"
	"io"
	"math"
	"os"
	"runtime/pprof"
	"testing"
	"time"

	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

func sameFloat(a, b float64) bool {
	return math.Float64bits(a) == math.Float64bits(b) || math.IsNaN(a) && math.IsNaN(b)
}

func TestNativeInvalidCodeSize(t *testing.T) {
	for _, instructions := range [][]byte{nil, make([]byte, MaxCodeBytes+1)} {
		code, err := allocateCode(instructions)
		if code != nil || err == nil {
			t.Fatalf("allocateCode(%d bytes) = %v, %v", len(instructions), code, err)
		}
	}
}

// This can run alone in a hardened executable. Allocation denial must be an
// ordinary, identifiable refusal, before any attempt to enter native code.
func TestExecutableMemoryPolicy(t *testing.T) {
	c, err := Compile(constantProgram())
	if err != nil {
		if c != nil || !errors.Is(err, ErrUnavailable) {
			t.Fatalf("unavailable executable memory: %v, %v", c, err)
		}
		t.Logf("backend refused safely: %v", err)
		return
	}
	defer func() {
		if err := c.Close(); err != nil {
			t.Error(err)
		}
	}()
	if os.Getenv("QUICKJS_EXPECT_JIT_DENIED") != "" {
		t.Fatal("executable-memory policy unexpectedly allowed native code")
	}
	exit, err := c.Run(nil, 0, 1)
	if err != nil || exit.Kind != ir.Returned || exit.Value != ir.Float(42) {
		t.Fatalf("allowed native code: %+v, %v", exit, err)
	}
}

// Programs differing only in an immediate, compiled and released in turn,
// exercise instruction-cache publication when the OS reuses freed addresses.
func TestNativeReusedAllocations(t *testing.T) {
	for i := 0; i < 128; i++ {
		p := &ir.Program{Code: []ir.Instruction{{Op: ir.Return, Left: ir.Literal(ir.Float(float64(i)))}}, Maps: []ir.StateMap{{PC: 0}}}
		c, err := Compile(p)
		if err != nil {
			t.Fatal(err)
		}
		exit, runErr := c.Run(nil, 0, 1)
		closeErr := c.Close()
		if runErr != nil || closeErr != nil || exit.Value != ir.Float(float64(i)) {
			t.Fatalf("allocation %d: %+v, %v, %v", i, exit, runErr, closeErr)
		}
	}
}

func TestNativeCPUProfile(t *testing.T) {
	p := &ir.Program{Locals: 1, Code: []ir.Instruction{
		{Op: ir.Binary, Operator: ir.Add, Left: ir.Slot(0), Right: ir.Literal(ir.Float(1)), Dest: 0},
		{Op: ir.Jump, Target: 0},
	}, Maps: []ir.StateMap{{PC: 0}, {PC: 1}}}
	c := newTestCode(t, p)
	var profile bytes.Buffer
	if err := pprof.StartCPUProfile(&profile); err != nil {
		t.Fatal(err)
	}
	func() {
		defer pprof.StopCPUProfile()
		slots := []ir.Value{ir.Float(0)}
		deadline := time.Now().Add(250 * time.Millisecond)
		for time.Now().Before(deadline) {
			if exit, err := c.Run(slots, 0, MaxIterations); err != nil || exit.Kind != ir.BudgetExit {
				t.Fatalf("profiled loop: %+v, %v", exit, err)
			}
		}
	}()
	zr, err := gzip.NewReader(bytes.NewReader(profile.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	defer zr.Close()
	data, err := io.ReadAll(zr)
	if err != nil || len(data) == 0 {
		t.Fatalf("CPU profile: %d bytes, %v", len(data), err)
	}
}

// A panic while emitting is a refused program, not a crashed host.
func TestNativeEmitPanicRefuses(t *testing.T) {
	previous := emitInstructions
	defer func() { emitInstructions = previous }()
	emitInstructions = func(*ir.Program) ([]byte, []int, error) { panic("emitter bug") }
	c, err := Compile(constantProgram())
	if c != nil || !errors.Is(err, ErrProgram) {
		t.Fatalf("Compile after an emitter panic = %v, %v; want nil, ErrProgram", c, err)
	}
}
