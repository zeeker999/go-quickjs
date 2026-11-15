//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package verify

import (
	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	jitcompile "github.com/go-quickjs/go-quickjs/internal/jit/compile"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

// regLowerForTest lowers fn's register code for an entry.
func regLowerForTest(fn *bytecode.Function) (*ir.Program, error) {
	return jitcompile.LowerReg(fn.RegCode(), fn)
}

// regLowerInlineForTest lowers fn's register code for a native caller or
// inlining.
func regLowerInlineForTest(fn *bytecode.Function) (*ir.Program, error) {
	return jitcompile.LowerRegInline(fn.RegCode(), fn)
}
