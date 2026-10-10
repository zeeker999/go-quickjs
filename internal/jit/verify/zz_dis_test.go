package verify

import (
	"fmt"
	"os"
	"testing"

	"golang.org/x/arch/x86/x86asm"
)

func TestTmpDisassemble(t *testing.T) {
	b, err := os.ReadFile(os.Getenv("DIS_BIN"))
	if err != nil {
		t.Skip()
	}
	var from, to int
	fmt.Sscan(os.Getenv("DIS_FROM"), &from)
	fmt.Sscan(os.Getenv("DIS_TO"), &to)
	for pc := 0; pc < len(b) && pc <= to; {
		inst, err := x86asm.Decode(b[pc:], 64)
		n := inst.Len
		if err != nil {
			n = 1
		}
		if pc >= from {
			t.Logf("%6d %v", pc, x86asm.GNUSyntax(inst, uint64(pc), nil))
		}
		pc += n
	}
}
