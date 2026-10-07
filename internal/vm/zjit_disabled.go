//go:build !quickjs_jit || android || ios || !((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

type jitFields struct{}
type jitClosureFields struct{}

func (*Runtime) initJIT(bool)                            {}
func (*Runtime) tryJITFrame(*frame) (Value, error, bool) { return Undefined, nil, false }
func (*Runtime) tryJITLoop(*frame, uint32, int, bool) (Value, error, bool) {
	return Undefined, nil, false
}
func (*Runtime) tryJITTreeLoop(*tctx, int, int, bool) {}
func (*Runtime) jitTreeRecovery(*frame) bool          { return false }
func jitTreeResult(any) (Value, error, bool)          { return Undefined, nil, false }
func (*Runtime) jitCodeBytes() int64                  { return 0 }
func (*Runtime) releaseJIT()                          {}
