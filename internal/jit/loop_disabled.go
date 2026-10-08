//go:build !quickjs_jit || android || ios || !((linux && amd64) || (windows && amd64) || (darwin && arm64))

package jit

const supported = false

func newLoopCode() ([]byte, error) { return nil, ErrUnavailable }
func freeLoopCode([]byte) error    { return nil }

func runLoopCode([]byte, *loopState) {
	panic("jit: native entry in an unsupported build")
}
