//go:build quickjs_jit && !android && !ios && amd64

package jit

import (
	"testing"
	"unsafe"

	"golang.org/x/sys/windows"
)

func TestWindowsExecutableProtectionAndRelease(t *testing.T) {
	loop, err := Compile(constantProgram())
	if err != nil {
		t.Fatal(err)
	}
	address := uintptr(unsafe.Pointer(&loop.code[0]))
	var info windows.MemoryBasicInformation
	if err := windows.VirtualQuery(address, &info, unsafe.Sizeof(info)); err != nil {
		t.Fatal(err)
	}
	if info.Protect != windows.PAGE_EXECUTE_READ || info.State != windows.MEM_COMMIT {
		t.Fatalf("published code: protection %#x, state %#x", info.Protect, info.State)
	}
	if err := loop.Close(); err != nil {
		t.Fatal(err)
	}
	if err := windows.VirtualQuery(address, &info, unsafe.Sizeof(info)); err != nil {
		t.Fatal(err)
	}
	const memFree = 0x10000 // MEMORY_BASIC_INFORMATION.State's MEM_FREE value.
	if info.State != memFree {
		t.Fatalf("released code has state %#x, want MEM_FREE", info.State)
	}
}
