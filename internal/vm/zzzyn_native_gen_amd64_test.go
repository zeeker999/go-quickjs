package vm

import (
	"bytes"
	"os"
	"runtime/debug"
	"strings"
	"sync"
	"testing"
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/vm/internal/nativegen"
)

// The native handlers are generated offline by
// internal/vm/internal/nativegen (go generate ./internal/vm), from the
// package's layout, which its helper reads in a build with the
// quickjs_nativelayout tag (zzzyn_native_layout_amd64.go).

// nativeBuildTags are the build tags this test binary was built with.
func nativeBuildTags() []string {
	if bi, ok := debug.ReadBuildInfo(); ok {
		for _, s := range bi.Settings {
			if s.Key == "-tags" && s.Value != "" {
				return strings.Split(s.Value, ",")
			}
		}
	}
	return nil
}

// nativeGenerated is what the generator makes from this build's layout:
// the build its helper runs has this test binary's tags.
var nativeGenerated = sync.OnceValues(func() (*nativegen.Output, error) {
	return nativegen.Build("amd64", nativeBuildTags()...)
})

// TestRegNativeGenerated checks that the generated files are what the
// generator makes now from this build's layout, and that the block and
// the table the binary links are its. One block serves every build: what
// the code reads of a layout that a tag changes, it reads relative to a
// base the context gives (rtBase, ccBase), so that the block made under
// any tag is the one file; the test fails under a tag whose layouts would
// make it another. A change to the emitter or to the layouts its offsets
// come from fails it until the files are written again (go generate
// ./internal/vm).
func TestRegNativeGenerated(t *testing.T) {
	if testing.Short() {
		t.Skip("builds the package to read its layout")
	}
	out, err := nativeGenerated()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("layout of a build with tags %q", nativeBuildTags())
	for _, f := range out.Files {
		old, err := os.ReadFile(f.Name)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(bytes.ReplaceAll(old, []byte("\r\n"), []byte("\n")), f.Data) {
			t.Errorf("%s is not what nativegen makes: go generate ./internal/vm", f.Name)
		}
	}
	// The block the binary holds is the generator's.
	base := nativeHandlersAddr()
	if got := unsafe.Slice((*byte)(nativePtr(&base)), len(out.Code)); !bytes.Equal(got, out.Code) {
		t.Fatal("the linked block differs from the generator's code")
	}
	at := nativeTableAddr()
	tab := (*[256]uintptr)(nativePtr(&at))
	for op := range tab {
		if want := base + uintptr(out.Table[op]); tab[op] != want {
			t.Fatalf("table entry %d is %#x, want %#x", op, tab[op], want)
		}
	}
}
