//go:build !quickjs_jit || android || ios || !((linux && amd64) || (windows && amd64) || (darwin && arm64))

package jit

import (
	"errors"
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

func TestBackendExcluded(t *testing.T) {
	if Supported() {
		t.Fatal("unsupported build reports native support")
	}
	code, err := Compile(&ir.Program{})
	if code != nil || !errors.Is(err, ErrUnavailable) {
		t.Fatalf("Compile = %v, %v; want nil, ErrUnavailable", code, err)
	}
}
