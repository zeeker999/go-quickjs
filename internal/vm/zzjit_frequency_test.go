//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
)

// jitFnForTest is the function of the global name.
func jitFnForTest(t *testing.T, r *Runtime, name string) *bytecode.Function {
	t.Helper()
	return r.global.getOwn(r.atoms.intern(name)).value().Object().fn().closure.fn
}
