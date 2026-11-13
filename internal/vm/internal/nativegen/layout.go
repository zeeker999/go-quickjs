package nativegen

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"sort"
	"strings"
)

// Layout is what the emitters read of package vm's private layout, as
// plain numbers by name: the offsets of the fields of the types the
// handlers read and write, by "Type.field"; the types' sizes; and the
// constants the handlers test values and states against. One layout
// serves every architecture of a word size, whose structs Go lays out
// alike.
type Layout struct {
	Offsets map[string]int64  `json:"offsets"`
	Sizes   map[string]int64  `json:"sizes"`
	Consts  map[string]uint64 `json:"consts"`
}

// layoutTag is the build tag that alone builds vm's NativeLayout
// (zzzyn_native_layout_amd64.go) and the helper that prints it.
const layoutTag = "quickjs_nativelayout"

// helper is the helper's package, which imports vm.
const helper = "github.com/go-quickjs/go-quickjs/internal/vm/internal/nativegen/cmd/layout"

// ReadLayout reads vm's layout in a build for goarch with the build tags
// tags, by running the helper (go run, with vm's NativeLayout built in):
// the host must run goarch's programs.
func ReadLayout(goarch string, tags ...string) (*Layout, error) {
	gotool, err := exec.LookPath("go")
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(gotool, "run", "-tags", strings.Join(append([]string{layoutTag}, tags...), ","), helper)
	cmd.Env = append(os.Environ(), "GOARCH="+goarch)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("nativegen: reading vm's %s layout (tags %q): %v\n%s", goarch, tags, err, stderr.Bytes())
	}
	l := new(Layout)
	dec := json.NewDecoder(bytes.NewReader(out))
	dec.DisallowUnknownFields()
	if err := dec.Decode(l); err != nil {
		return nil, fmt.Errorf("nativegen: vm's %s layout (tags %q): %v", goarch, tags, err)
	}
	return l, nil
}

// reader looks a layout's numbers up and records what the emitters read,
// so that a name vm no longer has fails the generation rather than read
// as zero.
type reader struct {
	l      *Layout
	failed []string
}

// lay is the layout being generated from, which the emitters read by off
// and size and loadOffsets and loadConsts load: set only while Generate
// holds genMu.
var lay *reader

// off is field name's offset, "Type.field", in the layout being
// generated from; size is type name's size.
func off(name string) int32 { return lay.off(name) }

func size(name string) int32 { return lay.size(name) }

func (r *reader) miss(kind, name string) {
	r.failed = append(r.failed, kind+" "+name)
}

// off is field name's offset, "Type.field".
func (r *reader) off(name string) int32 {
	v, ok := r.l.Offsets[name]
	if !ok {
		r.miss("offset", name)
	}
	return int32(v)
}

// size is type name's size.
func (r *reader) size(name string) int32 {
	v, ok := r.l.Sizes[name]
	if !ok {
		r.miss("size", name)
	}
	return int32(v)
}

// konst is constant name.
func (r *reader) konst(name string) uint64 {
	v, ok := r.l.Consts[name]
	if !ok {
		r.miss("constant", name)
	}
	return v
}

func (r *reader) err() error {
	if len(r.failed) == 0 {
		return nil
	}
	sort.Strings(r.failed)
	return fmt.Errorf("nativegen: vm's layout has no %s (NativeLayout, zzzyn_native_layout_amd64.go)", strings.Join(r.failed, ", "))
}
