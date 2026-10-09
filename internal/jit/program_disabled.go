//go:build !quickjs_jit || android || ios || !((linux && amd64) || (windows && amd64) || (darwin && arm64))

package jit

import "github.com/go-quickjs/go-quickjs/internal/jit/ir"

const supported = false

func executablePolicy() error { return ErrUnavailable }

func mapCode(int) ([]byte, error) { return nil, ErrUnavailable }

func protectCode([]byte, bool) error { return ErrUnavailable }

func flushCode([]byte) error { return ErrUnavailable }

func unmapCode([]byte) error { return nil }

func compileProgram(*Arena, *ir.Program, int) (*Code, error) { return nil, ErrUnavailable }

func runProgramCode([]byte, int, *programState, []ir.Value, []ir.ArrayView) {
	panic("jit: native program entry in an unsupported build")
}
