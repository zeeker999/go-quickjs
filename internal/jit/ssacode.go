//go:build quickjs_jit && !android && !ios && (((linux || windows) && amd64) || (darwin && arm64))

package jit

import (
	"fmt"
	"runtime"

	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
	"github.com/go-quickjs/go-quickjs/internal/jit/mir"
)

// SSACode owns a function the new pipeline compiled (internal/jit/mir) and
// its executable memory. Like Code, it belongs to one runtime.
type SSACode struct {
	code []byte
	// entries is each slot IR PC's code offset, or -1: a slice, since every
	// exit to Go looks one up.
	entries []int32
}

// NewSSACode publishes compiled code: written, then sealed executable.
func NewSSACode(c *mir.Code) (*SSACode, error) {
	code, err := allocateCode(c.Bytes)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrUnavailable, err)
	}
	n := 0
	for pc := range c.Entries {
		n = max(n, pc+1)
	}
	entries := make([]int32, n)
	for i := range entries {
		entries[i] = -1
	}
	for pc, off := range c.Entries {
		entries[pc] = int32(off)
	}
	s := &SSACode{code: code, entries: entries}
	runtime.SetFinalizer(s, func(s *SSACode) { _ = s.Close() })
	return s, nil
}

// HasEntry reports whether the code can be entered at a slot IR PC.
func (s *SSACode) HasEntry(pc int) bool {
	return pc >= 0 && pc < len(s.entries) && s.entries[pc] >= 0
}

// Run enters at a slot IR PC with ctx, which must hold the frame's addresses
// and the back-edge counter; it returns when native code exits, with the
// exit record in ctx.
func (s *SSACode) Run(pc int, ctx *abi.Context) error {
	if s == nil || len(s.code) == 0 {
		return ErrClosed
	}
	if !s.HasEntry(pc) {
		return fmt.Errorf("jit: no entry at pc %d", pc)
	}
	enterSSA(&s.code[s.entries[pc]], ctx)
	runtime.KeepAlive(s)
	runtime.KeepAlive(ctx)
	return nil
}

// Size is the executable memory s owns.
func (s *SSACode) Size() int {
	if s == nil {
		return 0
	}
	return len(s.code)
}

// Close releases the executable memory; it is idempotent.
func (s *SSACode) Close() error {
	if s == nil || len(s.code) == 0 {
		return nil
	}
	if err := freeCode(s.code); err != nil {
		return err
	}
	s.code = nil
	runtime.SetFinalizer(s, nil)
	return nil
}

//go:noescape
func enterSSA(code *byte, ctx *abi.Context)
