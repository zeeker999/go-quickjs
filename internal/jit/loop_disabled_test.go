//go:build !quickjs_jit || android || ios || !((linux && amd64) || (windows && amd64) || (darwin && arm64))

package jit

import (
	"errors"
	"testing"
)

func TestBackendExcluded(t *testing.T) {
	if Supported() {
		t.Fatal("unsupported build reports native support")
	}
	loop, err := NewLoop()
	if loop != nil || !errors.Is(err, ErrUnavailable) {
		t.Fatalf("NewLoop = %v, %v; want nil, ErrUnavailable", loop, err)
	}
}
