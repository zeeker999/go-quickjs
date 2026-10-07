//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package jit

import (
	"encoding/binary"
	"fmt"
)

type relocation struct {
	offset, label int
	conditional   bool
}

type programAssembler struct {
	code   []byte
	labels []int
	fixups []relocation
}

func (a *programAssembler) label() int {
	a.labels = append(a.labels, -1)
	return len(a.labels) - 1
}

func (a *programAssembler) mark(label int)      { a.labels[label] = len(a.code) }
func (a *programAssembler) bytes(bytes ...byte) { a.code = append(a.code, bytes...) }
func (a *programAssembler) word(word uint32)    { a.code = binary.LittleEndian.AppendUint32(a.code, word) }
func (a *programAssembler) quad(word uint64)    { a.code = binary.LittleEndian.AppendUint64(a.code, word) }

func (a *programAssembler) valid() error {
	if len(a.code) == 0 || len(a.code) > MaxCodeBytes {
		return fmt.Errorf("native code budget: %d bytes", len(a.code))
	}
	for _, fixup := range a.fixups {
		if a.labels[fixup.label] < 0 {
			return fmt.Errorf("unbound native branch")
		}
	}
	return nil
}
