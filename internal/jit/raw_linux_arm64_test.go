//go:build quickjs_jit && !android && !ios

package jit

import (
	"encoding/binary"
	"testing"
	"unsafe"
)

// On linux/arm64 RawCode runs and the JavaScript tier is not built: code
// placed in an arena is entered, writes its context and returns, and code
// placed again over released memory runs as its new bytes say.
func TestRawWithoutSSA(t *testing.T) {
	if Supported() {
		t.Fatal("the JavaScript tier reports support on linux/arm64")
	}
	if !RawSupported() {
		t.Fatal("RawCode is not supported on linux/arm64")
	}
	a := NewArena()
	for i := uint64(1); i <= 3; i++ {
		// movz x1, #i; str x1, [x0]; ret
		var code []byte
		for _, w := range []uint32{0xD2800001 | uint32(i)<<5, 0xF9000001, 0xD65F03C0} {
			code = binary.LittleEndian.AppendUint32(code, w)
		}
		raw, err := PlaceRaw(a, code)
		if err != nil {
			t.Fatal(err)
		}
		var ctx uint64
		raw.Enter(unsafe.Pointer(&ctx))
		if ctx != i {
			t.Fatalf("placement %d wrote %d", i, ctx)
		}
		if err := raw.Close(); err != nil {
			t.Fatal(err)
		}
	}
}
