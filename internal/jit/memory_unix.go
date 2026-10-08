//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (darwin && arm64))

package jit

import (
	"errors"
	"fmt"
	"os"

	"golang.org/x/sys/unix"
)

func allocateCode(instructions []byte) ([]byte, error) {
	if len(instructions) == 0 || len(instructions) > MaxCodeBytes {
		return nil, fmt.Errorf("invalid native kernel size: %d", len(instructions))
	}
	if err := executablePolicy(); err != nil {
		return nil, err
	}
	page := os.Getpagesize()
	size := (len(instructions) + page - 1) / page * page
	code, err := unix.Mmap(-1, 0, size, unix.PROT_READ|unix.PROT_WRITE,
		unix.MAP_PRIVATE|unix.MAP_ANON)
	if err != nil {
		return nil, fmt.Errorf("allocate code: %w", err)
	}
	copy(code, instructions)
	if err = unix.Mprotect(code, unix.PROT_READ|unix.PROT_EXEC); err != nil {
		return nil, fmt.Errorf("seal code: %w", errors.Join(err, unix.Munmap(code)))
	}
	flushCode(code)
	return code, nil
}

func freeCode(code []byte) error { return unix.Munmap(code) }
