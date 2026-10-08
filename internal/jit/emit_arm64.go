//go:build quickjs_jit && !android && !ios && darwin

package jit

import "github.com/go-quickjs/go-quickjs/internal/jit/ir"

// programInstructions is this target's emitter; see emit_amd64.go.
func programInstructions(p *ir.Program) ([]byte, []int, error) { return arm64Instructions(p) }
