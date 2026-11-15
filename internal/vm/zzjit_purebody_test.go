//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
)

