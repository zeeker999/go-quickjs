package main

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// regInner gives the eight placements each count of runReg's filler once,
// and -k 4's four placements four different ones. (Four distinct counts
// cannot sum alike at both phases of every target: the filler is a factor
// varied beside the design, not one balanced within it.)
func TestRegInnerBalanced(t *testing.T) {
	seen := map[int]bool{}
	for n := range 8 {
		m := regInner(n)
		if m < 0 || m >= regInnerMax || seen[m] {
			t.Fatalf("regInner(%d) = %d: not a permutation of 0..%d", n, m, regInnerMax-1)
		}
		seen[m] = true
		if 3*m+3 > padKeysLen || inner*n > padKeysLen {
			t.Errorf("placement %d reads past padKeys' %d keys", n, padKeysLen)
		}
	}
	four := map[int]bool{}
	for _, n := range placementsOf(4) {
		four[regInner(n)] = true
	}
	if len(four) != 4 {
		t.Errorf("-k 4's placements %v share a count of runReg's filler", placementsOf(4))
	}
}

// regFill puts the calls in the loop's nop case, and the loop still
// parses; with none it is the loop as it is.
func TestRegFill(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "..", "..", "..", regGo))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(string(src), regNop) != 1 {
		t.Fatalf("%s has no single case for register nop", regGo)
	}
	if got := regFill(src, 0); got != string(src) {
		t.Error("regFill with no calls changed the loop")
	}
	got := regFill(src, 3)
	for _, call := range []string{"padRegInner(uint32(w.A()), 0)", "padRegInner(uint32(w.A()), 3)", "padRegInner(uint32(w.A()), 6)"} {
		if !strings.Contains(got, regNop+"\t\t\t") || !strings.Contains(got, call) {
			t.Errorf("regFill's loop has no %s after the nop case", call)
		}
	}
	if _, err := parser.ParseFile(token.NewFileSet(), regGo, got, 0); err != nil {
		t.Errorf("regFill's loop does not parse: %v", err)
	}
	if pad := string(padFile(0, 0, 0, 0)); !strings.Contains(pad, "func padRegInner(a uint32, i int)") {
		t.Error("the first pad has no padRegInner")
	}
}
