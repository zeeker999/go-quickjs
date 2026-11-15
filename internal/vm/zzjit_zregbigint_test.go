//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
)

// TestRegSoleRegisters pins which registers hold their values alone, of a
// function of one parameter (r2), four bindings (r3 to r6) and a
// temporary (r7): the bindings an operator reads -- an accumulator, a
// step -- or an append adds or a str_freeze reads do; the parameter, the
// temporary, a binding a move, a call's window, a return, a global's
// store, a property's read or a postfix update reads, and one a closure
// captures do not; nor any where the function has an instruction the
// analysis does not know.
func TestRegSoleRegisters(t *testing.T) {
	R := bytecode.MakeR
	loop := []bytecode.RInstr{
		R(bytecode.RLoadInt, 0, 3, 0, 1),
		R(bytecode.RMul, 0, 3, 3, 4),
		R(bytecode.RAdd, 0, 5, 3, 4),
		R(bytecode.RMul, 0, 7, 5, 3),
		R(bytecode.RAdd, 0, 4, 4, 7),
		R(bytecode.RInc, 0, 6, 6, 0),
		R(bytecode.RJLt, 0, 6, 2, 0),
	}
	all := uint64(1<<3 | 1<<4 | 1<<5 | 1<<6)
	for _, c := range []struct {
		name     string
		code     []bytecode.RInstr
		captured int
		want     uint64
	}{
		{"operators", nil, -1, all},
		{"move", []bytecode.RInstr{R(bytecode.RMove, 0, 4, 3, 0)}, -1, all &^ (1 << 3)},
		{"window", []bytecode.RInstr{R(bytecode.RCallW, 1, 4, 3, 0)}, -1, all &^ (1<<4 | 1<<5 | 1<<6)},
		{"return", []bytecode.RInstr{R(bytecode.RRet, 0, 5, 0, 0)}, -1, all &^ (1 << 5)},
		{"global", []bytecode.RInstr{R(bytecode.RSetGlobal, 0, 3, 0, 0)}, -1, all &^ (1 << 3)},
		{"property", []bytecode.RInstr{R(bytecode.RGetProp, 0, 4, 3, 0)}, -1, all &^ (1 << 3)},
		{"postfix", []bytecode.RInstr{R(bytecode.RPostInc, 0, 4, 3, 0)}, -1, all &^ (1 << 3)},
		{"captured", nil, 3, all &^ (1 << 3)},
		{"append", []bytecode.RInstr{R(bytecode.RAppend, 0, 5, 4, 0), R(bytecode.RStrFreeze, 0, 5, 0, 0)}, -1, all},
		{"unknown", []bytecode.RInstr{R(bytecode.RGetIndex, 0, 4, 3, 5)}, -1, 0},
	} {
		rc := &bytecode.RegCode{Code: append(append([]bytecode.RInstr{}, loop...), c.code...), NumRegs: 8, Params: 1, Inits: 4}
		fn := &bytecode.Function{ParamCount: 1}
		p := &ir.Program{Regs: 8}
		if c.captured >= 0 {
			p.Captured = make([]bool, 8)
			p.Captured[c.captured] = true
		}
		if got := regSoleRegisters(fn, rc, p); got != c.want {
			t.Errorf("%s: registers holding their values alone %b, want %b", c.name, got, c.want)
		}
	}
}

// regBigRuntime is a runtime with the JIT whose code is register code.
func regBigRuntime(t *testing.T) *Runtime {
	t.Helper()
	r := New(Config{JIT: true})
	t.Cleanup(func() { r.Close(); r.ReleaseClosed() })
	return r
}

