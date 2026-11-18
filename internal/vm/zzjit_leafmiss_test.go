//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import "testing"

func TestJITRegLeafMiss(t *testing.T) {
	regWrongCheck(t, nativeLeafMissSource, []regWrongCase{
		{"leafMisses", []float64{100}, 3, "4950"},
	})
}
