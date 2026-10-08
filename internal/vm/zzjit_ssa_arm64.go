//go:build quickjs_jit && !android && !ios && darwin && arm64

package vm

import (
	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/jit"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

// The new pipeline has no arm64 backend yet (P3 of docs/jit-progress.md):
// every function goes to the slot IR emitters.

const (
	jitSSADefault = false
	jitSSABackend = false
)

func (r *Runtime) compileSSA(*bytecode.Function, *closure, *ir.Program, int) (*jit.SSACode, []*shape) {
	return nil, nil
}

func (r *Runtime) runSSA(*frame, *jitEntry, int, int) (Value, error, bool) {
	return Undefined, nil, false
}
