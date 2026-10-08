//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package jit

import (
	"fmt"

	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

// Emit returns the machine code arch's emitter makes for p, "amd64" or
// "arm64", without allocating executable memory. Both emitters build on every
// supported target, so that a test can check either's encodings anywhere
// (internal/jit/verify does, against a disassembler).
func Emit(arch string, p *ir.Program) (code []byte, err error) {
	if err := p.Validate(); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrProgram, err)
	}
	defer func() {
		if v := recover(); v != nil {
			code, err = nil, fmt.Errorf("%w: internal error: %v", ErrProgram, v)
		}
	}()
	switch arch {
	case "amd64":
		code, _, err = amd64Instructions(p)
	case "arm64":
		code, _, err = arm64Instructions(p)
	default:
		return nil, fmt.Errorf("jit: no emitter for %s", arch)
	}
	return code, err
}
