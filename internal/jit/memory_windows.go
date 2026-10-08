//go:build quickjs_jit && !android && !ios && amd64

package jit

import (
	"errors"
	"fmt"
	"os"
	"unsafe"

	"golang.org/x/sys/windows"
)

var flushInstructionCache = windows.NewLazySystemDLL("kernel32.dll").NewProc("FlushInstructionCache")

func allocateCode(instructions []byte) ([]byte, error) {
	if len(instructions) == 0 || len(instructions) > MaxCodeBytes {
		return nil, fmt.Errorf("invalid native kernel size: %d", len(instructions))
	}
	page := os.Getpagesize()
	size := (len(instructions) + page - 1) / page * page
	address, err := windows.VirtualAlloc(0, uintptr(size),
		windows.MEM_COMMIT|windows.MEM_RESERVE, windows.PAGE_READWRITE)
	if err != nil {
		return nil, fmt.Errorf("allocate code: %w", err)
	}
	code := foreignBytes(address, size)
	copy(code, instructions)
	var oldProtect uint32
	if err = windows.VirtualProtect(address, uintptr(len(code)), windows.PAGE_EXECUTE_READ, &oldProtect); err != nil {
		return nil, fmt.Errorf("seal code: %w", errors.Join(err, freeLoopCode(code)))
	}
	ok, _, err := flushInstructionCache.Call(uintptr(windows.CurrentProcess()), address, uintptr(len(code)))
	if ok == 0 {
		return nil, fmt.Errorf("flush instruction cache: %w", errors.Join(err, freeLoopCode(code)))
	}
	return code, nil
}

func freeLoopCode(code []byte) error {
	return windows.VirtualFree(uintptr(unsafe.Pointer(&code[0])), 0, windows.MEM_RELEASE)
}

// foreignBytes represents a VirtualAlloc allocation as a slice. Its address
// refers to OS-owned memory, not a Go object recovered from a uintptr.
//
//go:noescape
func foreignBytes(address uintptr, size int) []byte
