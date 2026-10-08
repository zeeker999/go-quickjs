//go:build !quickjs_jit || android || ios || !((linux && amd64) || (windows && amd64) || (darwin && arm64))

package jit

import "github.com/go-quickjs/go-quickjs/internal/jit/ir"

func compileProgram(*ir.Program, int, bool) (*Code, error) { return nil, ErrUnavailable }

func runProgramCode([]byte, int, *programState, []ir.Value, []ir.ArrayView) {
	panic("jit: native program entry in an unsupported build")
}

func runDispatchCode(*byte, *dispatchState, *ir.Value, []ir.ArrayView) {
	panic("jit: native dispatch in an unsupported build")
}
