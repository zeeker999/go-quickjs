// Package jit contains the optional native executor infrastructure. It is
// experimental and is not yet connected to JavaScript execution.
package jit

import (
	"errors"
	"runtime"
)

var (
	// ErrUnavailable means this build or host cannot execute native code.
	ErrUnavailable = errors.New("native execution is unavailable")
	// ErrClosed means the executable allocation has been released.
	ErrClosed = errors.New("native code is closed")
	// ErrIterations means a native batch exceeds its execution budget.
	ErrIterations = errors.New("native iteration budget exceeded")
)

// MaxIterations bounds the work done before returning to a Go safe point.
const MaxIterations uint64 = 4096

// Supported reports whether this build includes a backend. It does not probe
// executable-memory permissions or allocate memory.
func Supported() bool { return supported }

// Loop owns a bounded numeric kernel and its executable allocation. Like a
// VM runtime, it must not be used concurrently, including calls to Close.
type Loop struct {
	code []byte
}

// LoopResult is the state after repeatedly adding Next to Sum and Step to Next.
type LoopResult struct {
	Next float64
	Sum  float64
}

// loopState is the native ABI: four eight-byte, pointer-free slots. Neither
// backend modifies SP, the Go goroutine register, or this layout.
type loopState struct {
	next  float64
	step  float64
	sum   float64
	count uint64
}

// NewLoop emits and seals a numeric kernel. The kernel proves the execution
// boundary; it is not a compiler for JavaScript bytecode.
func NewLoop() (*Loop, error) {
	code, err := newLoopCode()
	if err != nil {
		return nil, err
	}
	return &Loop{code: code}, nil
}

// Run executes at most MaxIterations iterations, without calling Go or
// accessing reference values from native code. Zero iterations preserve inputs.
func (l *Loop) Run(next, step, sum float64, count uint64) (LoopResult, error) {
	if l == nil || len(l.code) == 0 {
		return LoopResult{}, ErrClosed
	}
	if count > MaxIterations {
		return LoopResult{}, ErrIterations
	}
	s := loopState{next: next, step: step, sum: sum, count: count}
	runLoopCode(l.code, &s)
	runtime.KeepAlive(l)
	return LoopResult{Next: s.next, Sum: s.sum}, nil
}

// Size is the page-rounded executable allocation size, or zero after Close.
func (l *Loop) Size() int {
	if l == nil {
		return 0
	}
	return len(l.code)
}

// Close releases the executable allocation. It is idempotent; a release
// failure leaves the allocation owned by l so that the caller can retry.
func (l *Loop) Close() error {
	if l == nil || len(l.code) == 0 {
		return nil
	}
	if err := freeLoopCode(l.code); err != nil {
		return err
	}
	l.code = nil
	return nil
}
