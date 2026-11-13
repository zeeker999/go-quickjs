package nativegen

import (
	"fmt"
	"sync"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
)

// File is a file the generator writes in internal/vm.
type File struct {
	Name string
	Data []byte
}

// block is an emitter's code: its bytes, where its exits and each
// handler start, and the operations in the code's order.
type block struct {
	code          []byte
	exit, exitRet int
	at            map[bytecode.ROp]int
	order         []bytecode.ROp
}

// entry is op's table entry, its handler's offset in b, the exit stub's
// where it has none.
func (b *block) entry(op bytecode.ROp) int {
	if o, ok := b.at[op]; ok {
		return o
	}
	return b.exit
}

// arch is an architecture's emitter: the handlers' block and its file.
type arch struct {
	goarch string
	// file is the handlers' file in internal/vm.
	file string
	// emit makes the block from the layout loaded (lay).
	emit func() (*block, error)
}

// arches are the architectures whose handlers are generated.
var arches = []*arch{
	{goarch: "amd64", file: "zzzyn_native_handlers_amd64.s", emit: nativeEmit},
}

// genMu holds lay while an emitter runs.
var genMu sync.Mutex

// emitFrom runs ar's emitter on layout l.
func (ar *arch) emitFrom(l *Layout) (*block, error) {
	genMu.Lock()
	defer genMu.Unlock()
	lay = &reader{l: l}
	defer func() { lay = nil }()
	// The constants first: the offsets relative to rt and cc are by
	// rtBase and ccBase.
	loadConsts()
	loadOffsets()
	if err := lay.err(); err != nil {
		return nil, err
	}
	b, err := ar.emit()
	if err != nil {
		return nil, fmt.Errorf("nativegen: %s: %v", ar.goarch, err)
	}
	if err := lay.err(); err != nil {
		return nil, err
	}
	if err := handlerOrder(ar.goarch, b.order); err != nil {
		return nil, err
	}
	return b, nil
}

// Output is what the generator makes for one architecture from the
// layout of one build.
type Output struct {
	// Files are the files it writes.
	Files []File
	// Code is the handlers' block, and Table each operation's entry in
	// the native table, an offset in Code.
	Code  []byte
	Table [256]int
}

func lookup(goarch string) (*arch, error) {
	for _, ar := range arches {
		if ar.goarch == goarch {
			return ar, nil
		}
	}
	return nil, fmt.Errorf("nativegen: no emitter for %s", goarch)
}

// Build generates goarch's handlers from the layout of a build for goarch
// with the tags tags (ReadLayout).
func Build(goarch string, tags ...string) (*Output, error) {
	ar, err := lookup(goarch)
	if err != nil {
		return nil, err
	}
	l, err := ReadLayout(goarch, tags...)
	if err != nil {
		return nil, err
	}
	b, err := ar.emitFrom(l)
	if err != nil {
		return nil, err
	}
	out := &Output{Files: []File{{Name: ar.file, Data: handlersSource(b)}}, Code: b.code}
	for op := range out.Table {
		out.Table[op] = b.entry(bytecode.ROp(op))
	}
	return out, nil
}

// Generate makes every architecture's files.
func Generate() ([]File, error) {
	var files []File
	for _, ar := range arches {
		out, err := Build(ar.goarch)
		if err != nil {
			return nil, err
		}
		files = append(files, out.Files...)
	}
	return files, nil
}
