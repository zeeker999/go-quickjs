//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package jit

import (
	"errors"
	"math"
	"testing"
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/jit/abi"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
	"github.com/go-quickjs/go-quickjs/internal/jit/mir"
	"github.com/go-quickjs/go-quickjs/internal/jit/ssa"
)

// valueProgram returns v; a long one runs through n instructions first, so
// that its code spans pages.
func valueProgram(v float64, n int) *ir.Program {
	p := &ir.Program{Code: make([]ir.Instruction, n), Maps: make([]ir.StateMap, n)}
	for pc := range p.Maps {
		p.Maps[pc].PC = uint32(pc)
	}
	p.Code[n-1] = ir.Instruction{Op: ir.Return, Left: ir.Literal(ir.Float(v))}
	return p
}

// An arena packs a runtime's functions into a few mappings: each placed
// after the last, a long one across pages, all running, and running still
// as others next to them are released and more are placed, which opens
// and seals the pages they share. Releasing the last unmaps everything.
func TestArenaPacksCode(t *testing.T) {
	a := NewArena()
	var codes []*Code
	place := func(i int) {
		n := 1
		if i%50 == 25 {
			n = 1000
		}
		c, err := CompileIn(a, valueProgram(float64(i), n), MaxCodeBytes)
		if err != nil {
			t.Fatal(err)
		}
		codes = append(codes, c)
	}
	runAll := func(when string) {
		t.Helper()
		for i, c := range codes {
			if c.Size() == 0 {
				continue
			}
			exit, err := c.Run(nil, 0, MaxIterations)
			if err != nil || exit.Kind != ir.Returned || exit.Value != ir.Float(float64(i)) {
				t.Fatalf("%s: code %d returned %+v, %v", when, i, exit, err)
			}
		}
	}
	bytes := func() (n int) {
		for _, c := range codes {
			n += c.Size()
		}
		return n
	}
	for i := 0; i < 200; i++ {
		place(i)
	}
	if want := (bytes()+chunkBytes-1)/chunkBytes + 1; a.Chunks() > want {
		t.Fatalf("%d bytes of code in %d mappings, want at most %d", bytes(), a.Chunks(), want)
	}
	if next := uintptr(unsafe.Pointer(&codes[1].code[0])) - uintptr(unsafe.Pointer(&codes[0].code[0])); next != uintptr(codes[0].Size()) {
		t.Fatalf("the second function is %d bytes after the first, which takes %d", next, codes[0].Size())
	}
	runAll("placed")
	for i := 1; i < len(codes); i += 2 {
		if err := codes[i].Close(); err != nil {
			t.Fatal(err)
		}
	}
	runAll("half released")
	for i := 200; i < 300; i++ {
		place(i)
	}
	runAll("placed among released")
	for _, c := range codes {
		if err := c.Close(); err != nil {
			t.Fatal(err)
		}
	}
	if a.Chunks() != 0 {
		t.Fatalf("an empty arena holds %d mappings", a.Chunks())
	}
	// An emptied arena maps again.
	codes = codes[:0]
	place(0)
	runAll("placed again")
	if err := codes[0].Close(); err != nil || a.Chunks() != 0 {
		t.Fatalf("released: %v, %d mappings", err, a.Chunks())
	}
}

// The new pipeline's code shares an arena with the old one's.
func TestArenaHoldsBothPipelines(t *testing.T) {
	a := NewArena()
	old, err := CompileIn(a, constantProgram(), MaxCodeBytes)
	if err != nil {
		t.Fatal(err)
	}
	f, err := ssa.Build(constantProgram())
	if err != nil {
		t.Fatal(err)
	}
	mc, err := mir.Compile(f, testEncoding)
	if err != nil {
		t.Fatal(err)
	}
	c, err := NewSSACode(a, mc)
	if err != nil {
		t.Fatal(err)
	}
	if a.Chunks() != 1 {
		t.Fatalf("two functions in %d mappings", a.Chunks())
	}
	var ctx abi.Context
	if err := c.Run(0, &ctx); err != nil || ctx.ExitKind != abi.ExitReturn || ctx.Ret != math.Float64bits(42) {
		t.Fatalf("new pipeline: %v, %+v", err, ctx)
	}
	if exit, err := old.Run(nil, 0, 1); err != nil || exit.Value != ir.Float(42) {
		t.Fatalf("old pipeline: %+v, %v", exit, err)
	}
	if err := errors.Join(old.Close(), c.Close()); err != nil || a.Chunks() != 0 {
		t.Fatalf("released: %v, %d mappings", err, a.Chunks())
	}
}
