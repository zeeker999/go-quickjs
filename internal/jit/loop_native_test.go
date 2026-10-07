//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package jit

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func newTestLoop(t testing.TB) *Loop {
	t.Helper()
	if !Supported() {
		t.Fatal("native tests require a compiled backend")
	}
	loop, err := NewLoop()
	if err != nil {
		t.Fatalf("native tests must execute code, not fall back: %v", err)
	}
	t.Cleanup(func() {
		if err := loop.Close(); err != nil {
			t.Error(err)
		}
	})
	return loop
}

func goLoop(next, step, sum float64, count uint64) LoopResult {
	for i := uint64(0); i < count; i++ {
		sum += next
		next += step
	}
	return LoopResult{Next: next, Sum: sum}
}

func sameFloat(a, b float64) bool {
	return math.Float64bits(a) == math.Float64bits(b) || math.IsNaN(a) && math.IsNaN(b)
}

func TestNativeLoop(t *testing.T) {
	loop := newTestLoop(t)
	values := []float64{0, math.Copysign(0, -1), 1, -1, 0.1, -1.5,
		math.SmallestNonzeroFloat64, math.MaxFloat64, math.Inf(1), math.Inf(-1), math.NaN()}
	for _, next := range values {
		for _, step := range values {
			for _, sum := range values {
				for _, count := range []uint64{0, 1, 2, 31, MaxIterations} {
					want := goLoop(next, step, sum, count)
					got, err := loop.Run(next, step, sum, count)
					if err != nil || !sameFloat(got.Next, want.Next) || !sameFloat(got.Sum, want.Sum) {
						t.Fatalf("Run(%g, %g, %g, %d) = %+v, %v; want %+v", next, step, sum, count, got, err, want)
					}
				}
			}
		}
	}
	// An exact result also pins the order of the two additions.
	got, err := loop.Run(1, 2, 0, 4)
	if err != nil || got != (LoopResult{Next: 9, Sum: 16}) {
		t.Fatalf("Run = %+v, %v; want Next=9, Sum=16", got, err)
	}
}

func TestNativeLoopBudgetAndClose(t *testing.T) {
	loop := newTestLoop(t)
	if loop.Size() != os.Getpagesize() {
		t.Fatalf("allocation size %d, want %d", loop.Size(), os.Getpagesize())
	}
	for _, count := range []uint64{MaxIterations + 1, math.MaxUint64} {
		if _, err := loop.Run(0, 1, 0, count); !errors.Is(err, ErrIterations) {
			t.Fatalf("Run count %d: %v, want ErrIterations", count, err)
		}
	}
	if err := loop.Close(); err != nil {
		t.Fatal(err)
	}
	if loop.Size() != 0 {
		t.Fatal("closed loop retains an executable allocation")
	}
	if err := loop.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := loop.Run(0, 0, 0, 0); !errors.Is(err, ErrClosed) {
		t.Fatalf("Run after close: %v, want ErrClosed", err)
	}
}

func TestNativeInvalidCodeSize(t *testing.T) {
	for _, instructions := range [][]byte{nil, make([]byte, os.Getpagesize()+1)} {
		code, err := allocateCode(instructions)
		if code != nil || err == nil {
			t.Fatalf("allocateCode(%d bytes) = %v, %v", len(instructions), code, err)
		}
	}
}

// This can run alone in a hardened executable. Allocation denial must be an
// ordinary, identifiable refusal, before any attempt to enter native code.
func TestExecutableMemoryPolicy(t *testing.T) {
	loop, err := NewLoop()
	if err != nil {
		if loop != nil || !errors.Is(err, ErrUnavailable) {
			t.Fatalf("unavailable executable memory: %v, %v", loop, err)
		}
		t.Logf("backend refused safely: %v", err)
		return
	}
	defer func() {
		if err := loop.Close(); err != nil {
			t.Error(err)
		}
	}()
	if os.Getenv("QUICKJS_EXPECT_JIT_DENIED") != "" {
		t.Fatal("executable-memory policy unexpectedly allowed native code")
	}
	got, err := loop.Run(1, 2, 0, 4)
	if err != nil || got != (LoopResult{Next: 9, Sum: 16}) {
		t.Fatalf("allowed native code: %+v, %v", got, err)
	}
}

func TestNativeReusedAllocations(t *testing.T) {
	for i := 0; i < 128; i++ {
		instructions := loopInstructions()
		want := 16.0
		if i%2 == 1 {
			// Alternate addition and subtraction in the same kernel, exercising
			// instruction-cache publication when the OS reuses freed addresses.
			if runtime.GOARCH == "amd64" {
				at := bytes.Index(instructions, []byte{0xf2, 0x0f, 0x58, 0xd0})
				if at < 0 {
					t.Fatal("sum instruction not found")
				}
				instructions[at+2] = 0x5c
			} else {
				at := bytes.Index(instructions, []byte{0x42, 0x28, 0x60, 0x1e})
				if at < 0 {
					t.Fatal("sum instruction not found")
				}
				binary.LittleEndian.PutUint32(instructions[at:], 0x1e603842)
			}
			want = -16
		}
		code, err := allocateCode(instructions)
		if err != nil {
			t.Fatal(err)
		}
		loop := &Loop{code: code}
		got, runErr := loop.Run(1, 2, 0, 4)
		closeErr := loop.Close()
		if runErr != nil || closeErr != nil || got != (LoopResult{Next: 9, Sum: want}) {
			t.Fatalf("allocation %d: %+v, %v, %v", i, got, runErr, closeErr)
		}
	}
}

// Grow the Go stack before entry and use its slots after return, so a broken
// bridge cannot pass simply by returning a value with corrupted caller state.
//
//go:noinline
func runOnGrowingStack(loop *Loop, depth int) (float64, error) {
	var slots [128]float64
	for i := range slots {
		slots[i] = float64(i + depth)
	}
	var sum float64
	if depth == 0 {
		got, err := loop.Run(1, 1, 0, MaxIterations)
		if err != nil {
			return 0, err
		}
		sum = got.Sum
	} else {
		var err error
		sum, err = runOnGrowingStack(loop, depth-1)
		if err != nil {
			return 0, err
		}
	}
	for _, slot := range slots {
		sum += slot
	}
	return sum, nil
}

func TestNativeBoundary(t *testing.T) {
	loop := newTestLoop(t)
	for _, procs := range []int{1, 2} {
		t.Run(fmt.Sprintf("%d processors", procs), func(t *testing.T) {
			previous := runtime.GOMAXPROCS(procs)
			defer runtime.GOMAXPROCS(previous)
			stop := make(chan struct{})
			done := make(chan struct{})
			var collections atomic.Int64
			go func() {
				defer close(done)
				for {
					select {
					case <-stop:
						return
					default:
						runtime.GC()
						var stacks [16 << 10]byte
						runtime.Stack(stacks[:], true)
						collections.Add(1)
					}
				}
			}()
			defer func() { close(stop); <-done }()
			want := float64(MaxIterations*(MaxIterations+1)) / 2
			for depth := 0; depth <= 64; depth++ {
				for i := 0; i < 128; i++ {
					want += float64(i + depth)
				}
			}
			deadline := time.Now().Add(100 * time.Millisecond)
			for i := 0; i < 256 || time.Now().Before(deadline); i++ {
				got, err := runOnGrowingStack(loop, 64)
				if err != nil || got != want {
					t.Fatalf("growing stack: %g, %v; want %g", got, err, want)
				}
				if i%128 == 0 {
					runtime.Gosched()
				}
			}
			if collections.Load() == 0 {
				t.Fatal("GC stress worker did not run")
			}
		})
	}
}

func TestNativeCPUProfile(t *testing.T) {
	loop := newTestLoop(t)
	var profile bytes.Buffer
	if err := pprof.StartCPUProfile(&profile); err != nil {
		t.Fatal(err)
	}
	func() {
		defer pprof.StopCPUProfile()
		deadline := time.Now().Add(250 * time.Millisecond)
		for time.Now().Before(deadline) {
			if _, err := loop.Run(1, 1, 0, MaxIterations); err != nil {
				t.Fatal(err)
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

func TestNativeIndependentOwners(t *testing.T) {
	const workers = 8
	results := make(chan error, workers)
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 32; j++ {
				loop, err := NewLoop()
				if err != nil {
					results <- err
					return
				}
				got, err := loop.Run(1, 2, 0, 4)
				closeErr := loop.Close()
				if err != nil || closeErr != nil || got != (LoopResult{Next: 9, Sum: 16}) {
					results <- fmt.Errorf("independent owner: %+v, %v, %v", got, err, closeErr)
					return
				}
			}
		}()
	}
	wg.Wait()
	close(results)
	for err := range results {
		t.Error(err)
	}
}

func BenchmarkNativeLoop(b *testing.B) {
	loop := newTestLoop(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := loop.Run(1, 1, 0, MaxIterations); err != nil {
			b.Fatal(err)
		}
	}
}

var loopSink LoopResult

func BenchmarkGoLoop(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		loopSink = goLoop(1, 1, 0, MaxIterations)
	}
}

func BenchmarkNewLoop(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		loop, err := NewLoop()
		if err != nil {
			b.Fatal(err)
		}
		if err := loop.Close(); err != nil {
			b.Fatal(err)
		}
	}
}
