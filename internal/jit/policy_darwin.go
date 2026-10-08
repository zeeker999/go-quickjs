//go:build quickjs_jit && !android && !ios && arm64

package jit

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"

	"golang.org/x/sys/unix"
)

func executablePolicy() error {
	// CS_OPS_STATUS writes p_csflags. Hardened/enforced executables can allow
	// mprotect yet kill the process when it enters an unsigned code page.
	// This backend does not implement the entitled MAP_JIT path, so refuse
	// such hosts before allocating or executing, even if they have entitlements.
	const forbidden = 0x00000100 | 0x00001000 | 0x00010000 // CS_HARD, CS_ENFORCEMENT, CS_RUNTIME
	var flags uint32
	_, _, err := syscall.Syscall6(unix.SYS_CSOPS, uintptr(os.Getpid()), 0,
		uintptr(unsafe.Pointer(&flags)), unsafe.Sizeof(flags), 0, 0)
	if err != 0 {
		return fmt.Errorf("read executable code-signing policy: %w", err)
	}
	if flags&forbidden != 0 {
		return fmt.Errorf("executable code-signing policy %#x requires a MAP_JIT backend", flags)
	}
	return nil
}
