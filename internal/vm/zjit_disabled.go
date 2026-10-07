//go:build !quickjs_jit || android || ios || !((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

type jitFields struct{}

func (*Runtime) initJIT(bool)                            {}
func (*Runtime) tryJITFrame(*frame) (Value, error, bool) { return Undefined, nil, false }
func (*Runtime) jitCodeBytes() int64                     { return 0 }
func (*Runtime) releaseJIT()                             {}
