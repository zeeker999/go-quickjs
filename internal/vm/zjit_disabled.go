//go:build !quickjs_jit || android || ios || !((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

// jitBuilt is false: the hooks it guards are compiled out, so the
// interpreter's code is what it is without the JIT.
const jitBuilt = false

type jitFields struct{}
type jitClosureFields struct{}
type jitRealmFields struct{}

// The memory meter names these types; builds without the JIT have none.
type jitState struct{}
type jitEntry struct{}

func (*Runtime) recordJITStringIntrinsic() {}

func (*Runtime) initJIT(bool)                                         {}
func (*Runtime) tryJITFrame(*frame) (Value, error, bool)              { return Value{}, nil, false }
func (*Runtime) jitBackEdge(*frame, uint32, int) (Value, error, bool) { return Value{}, nil, false }
func (*Runtime) tryJITTreeLoop(*tctx, int, int, bool)                 {}
func (*Runtime) jitTreeRecovery(*frame) bool                          { return false }
func jitTreeResult(any) (Value, error, bool)                          { return Value{}, nil, false }
func (*Runtime) jitCodeBytes() int64                                  { return 0 }
func (*Runtime) releaseJIT()                                          {}
