//go:build quickjs_jit && !android && !ios && amd64

package jit

import (
	"os"
	"strconv"
	"strings"
	"testing"
	"unsafe"
)

func mappingAt(t *testing.T, address uint64) string {
	t.Helper()
	maps, err := os.ReadFile("/proc/self/maps")
	if err != nil {
		t.Fatal(err)
	}
	for line := range strings.SplitSeq(string(maps), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		startText, endText, ok := strings.Cut(fields[0], "-")
		if !ok {
			t.Fatalf("invalid mapping: %s", line)
		}
		start, startErr := strconv.ParseUint(startText, 16, 64)
		end, endErr := strconv.ParseUint(endText, 16, 64)
		if startErr != nil || endErr != nil {
			t.Fatalf("invalid mapping: %s", line)
		}
		if address >= start && address < end {
			return fields[1]
		}
	}
	return ""
}

func TestLinuxExecutableProtectionAndRelease(t *testing.T) {
	loop := newTestLoop(t)
	address := uint64(uintptr(unsafe.Pointer(&loop.code[0])))
	if permissions := mappingAt(t, address); permissions != "r-xp" {
		t.Fatalf("published code has permissions %q, want r-xp", permissions)
	}
	if err := loop.Close(); err != nil {
		t.Fatal(err)
	}
	if permissions := mappingAt(t, address); permissions != "" {
		t.Fatalf("released code is still mapped: %s", permissions)
	}
}
