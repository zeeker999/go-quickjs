//go:build quickjs_jit && !android && !ios && (((linux || windows) && amd64) || (darwin && arm64))

package jit

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
	"github.com/go-quickjs/go-quickjs/internal/jit/mir"
)

// SSACode owns a function the new pipeline compiled (internal/jit/mir) and
// its executable memory. Like Code, it belongs to one runtime.
type SSACode struct {
	code  []byte
	arena *Arena
	// entries is each slot IR PC's code offset, or -1: a slice, since every
	// exit to Go looks one up.
	entries []int32
	// exits are the descriptions the code's exits name by address.
	exits []*abi.ExitDescriptor
}

// NewSSACode publishes compiled code in a, a runtime's arena (nil gives it
// one of its own): written, then sealed executable.
func NewSSACode(a *Arena, c *mir.Code) (*SSACode, error) {
	if a == nil {
		a = NewArena()
	}
	code, err := a.place(c.Bytes)
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
	s := &SSACode{code: code, arena: a, entries: entries, exits: c.Exits}
	runtime.SetFinalizer(s, func(s *SSACode) { _ = s.Close() })
	return s, nil
}

// HasEntry reports whether the code can be entered at a slot IR PC.
func (s *SSACode) HasEntry(pc int) bool {
	return pc >= 0 && pc < len(s.entries) && s.entries[pc] >= 0
}

// EntryAddress is the address of the code's entry at a slot IR PC, which a
// caller's native code jumps to (mir's native calls), or 0 for none.
func (s *SSACode) EntryAddress(pc int) uintptr {
	if s == nil || len(s.code) == 0 || !s.HasEntry(pc) {
		return 0
	}
	return uintptr(unsafe.Pointer(&s.code[s.entries[pc]]))
}

// Run enters at a slot IR PC with ctx, which must hold the frame's addresses
// and the back-edge counter; it returns when native code exits, with the
// exit record in ctx -- or, for an exit that left its frame to Go
// (abi.ExitTable), in the context that left, the caller's to apply
// (ApplyExit): ctx, or a native callee's, which only the caller knows.
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

// Resume goes on in s's code at addr, where a native call returns to after
// its callee (abi.Context.ReturnTo), with ctx, the callee's context, which
// holds the call's result (RetValue) and where its frame began (Base): a
// caller's code goes on after a callee Go finished. It returns as Run does.
func (s *SSACode) Resume(addr uintptr, ctx *abi.Context) error {
	if s == nil || len(s.code) == 0 {
		return ErrClosed
	}
	base := uintptr(unsafe.Pointer(&s.code[0]))
	if addr < base || addr >= base+uintptr(len(s.code)) {
		return fmt.Errorf("jit: %#x is not in the code", addr)
	}
	enterSSA(&s.code[addr-base], ctx)
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
	if err := s.arena.release(s.code); err != nil {
		return err
	}
	s.code, s.arena = nil, nil
	runtime.SetFinalizer(s, nil)
	return nil
}

//go:noescape
func enterSSA(code *byte, ctx *abi.Context)
