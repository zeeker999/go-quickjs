//go:build quickjs_jit && !android && !ios && darwin

package jit

import "github.com/go-quickjs/go-quickjs/internal/jit/abi"

// SSACode is the new pipeline's compiled code, which has no arm64 backend
// yet; nothing makes one here.
type SSACode struct{}

// HasEntry reports no entries.
func (*SSACode) HasEntry(int) bool { return false }

// Run is never reached: nothing makes an SSACode on arm64.
func (*SSACode) Run(int, *abi.Context) error { return ErrUnavailable }

// Size is zero.
func (*SSACode) Size() int { return 0 }

// Close does nothing.
func (*SSACode) Close() error { return nil }
