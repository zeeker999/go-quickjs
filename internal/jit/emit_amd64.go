//go:build quickjs_jit && !android && !ios && (linux || windows)

package jit

import "github.com/go-quickjs/go-quickjs/internal/jit/ir"

// programInstructions is this target's emitter. Both emitters build on every
// supported target, so that either's encodings can be tested anywhere; the
// linker keeps only the one called here.
func programInstructions(p *ir.Program) ([]byte, []int, error) { return amd64Instructions(p) }
