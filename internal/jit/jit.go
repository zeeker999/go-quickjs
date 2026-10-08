// Package jit contains the optional native executor infrastructure. It is
// experimental; the VM selects it only with a tagged build and runtime opt-in.
package jit

import "errors"

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
