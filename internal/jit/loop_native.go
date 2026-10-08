//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package jit

import (
	"fmt"
	"runtime"
)

const supported = true

func newLoopCode() ([]byte, error) {
	code, err := allocateCode(loopInstructions())
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrUnavailable, err)
	}
	return code, nil
}

func runLoopCode(code []byte, state *loopState) {
	enter(&code[0], state)
	runtime.KeepAlive(code)
	runtime.KeepAlive(state)
}

// enter tail-jumps to a leaf kernel: it retains the Go return address and
// stack, and the kernel returns directly to the Go caller. The kernel cannot
// make calls, allocate stack space, or run beyond the checked batch budget.
//
//go:noescape
func enter(code *byte, state *loopState)
