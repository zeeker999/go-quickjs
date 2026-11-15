package arm64

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"testing"
)

// narrowReach has the assembler keep branches within reaches shift bits
// narrower than the instructions', until the test ends: pools, veneers
// of veneers and jumps back past a B's reach, in code of a few megabytes.
func narrowReach(t *testing.T, shift uint) {
	near, far := nearReach, farReach
	set := func(n, f int) {
		nearReach, nearDue, nearVeneer = n, n/2, n/4
		farReach, farDue, farVeneer = f, f/2, f/4
	}
	set(near>>shift, far>>shift)
	t.Cleanup(func() { set(near, far) })
}

// poolBranch is a branch a test made: where it began, where the caller's
// next instruction went, its label and what it was.
type poolBranch struct {
	at, next int
	label    Label
	word     uint32 // the branch as made, its offset 0
}

// follow is where code goes from pc, the branch br's condition holding or
// not: through every jump, conditional branch, ADR, subtraction and BR the
// assembler made for it, to the first instruction that is the branch's
// target or the caller's next (stop), or -1 and why.
func follow(code []uint32, pc int, br poolBranch, holds bool, stop func(int) bool) (int, int, string) {
	var x [32]int64
	sext := func(v uint32, bits uint) int { return int(int32(v<<(32-bits)) >> (32 - bits)) }
	for step := 0; step < 10000; step++ {
		if pc < 0 || pc >= len(code) {
			return -1, step, fmt.Sprintf("left the code at %d", pc)
		}
		if step > 0 && stop(pc) {
			return pc, step, ""
		}
		w := code[pc]
		switch {
		case w&0xFC000000 == 0x14000000:
			pc += sext(w&(1<<26-1), 26)
		case w&0xFF000010 == 0x54000000:
			// The branch's own condition, or its opposite.
			taken := holds
			if w&0xF != br.word&0xF {
				taken = !holds
			}
			if taken {
				pc += sext(w>>5&(1<<19-1), 19)
			} else {
				pc++
			}
		case w&0x7E000000 == 0x34000000:
			// The register is zero if a CBZ's condition holds, or a
			// CBNZ's does not.
			zero := holds == (br.word&(1<<24) == 0)
			if zero == (w&(1<<24) == 0) {
				pc += sext(w>>5&(1<<19-1), 19)
			} else {
				pc++
			}
		case w&0x9F000000 == 0x10000000:
			x[w&31] = int64(4*pc + sext(w>>5&(1<<19-1), 19)*4 + int(w>>29&3))
			pc++
		case w&0xFFC00000 == 0xD1400000:
			x[w&31] = x[w>>5&31] - int64(w>>10&0xFFF)<<12
			pc++
		case w&0xFFFFFC1F == 0xD61F0000:
			at := x[w>>5&31]
			if at%4 != 0 {
				return -1, step, fmt.Sprintf("BR to %d", at)
			}
			pc = int(at / 4)
		default:
			return -1, step, fmt.Sprintf("ran into %#08x at %d", w, pc)
		}
	}
	return -1, 0, "looped"
}

// poolProgram is code of n instructions, plain ones and, at random, a
// branch of every kind to a label bound behind or still to be bound, the
// labels bound at random, some data among them; each branch goes where
// its label is when taken, and where the caller's next instruction is
// when not, whatever pools came between.
func poolProgram(t *testing.T, seed int64, n, every int) {
	rng := rand.New(rand.NewSource(seed))
	var a Asm
	labels := make([]Label, 64)
	for i := range labels {
		labels[i] = a.NewLabel()
	}
	var brs []poolBranch
	for a.Len()/4 < n {
		switch r := rng.Intn(every); {
		case r == 0:
			l := labels[rng.Intn(len(labels))]
			at := a.Len() / 4
			var word uint32
			switch rng.Intn(4) {
			case 0:
				a.B(l)
				word = 0x14000000
			case 1:
				c := Cond(rng.Intn(14))
				a.BCond(c, l)
				word = 0x54000000 | uint32(c)
			case 2:
				a.Cbz(3, l, true)
				word = 0x34000000 | sf(true) | 3
			default:
				a.Cbnz(3, l, false)
				word = 0x35000000 | 3
			}
			brs = append(brs, poolBranch{at, a.Len() / 4, l, word})
		case r == 1:
			// A label bound here, and a new one in its place.
			k := rng.Intn(len(labels))
			if a.labels[labels[k]] < 0 {
				a.Bind(labels[k])
			}
			labels[k] = a.NewLabel()
		case r == 2:
			a.Data(make([]byte, 4*rng.Intn(64)))
		default:
			a.MovRR(4, 5)
		}
	}
	for _, l := range labels {
		if a.labels[l] < 0 {
			a.Bind(l)
		}
	}
	a.MovRR(4, 5)
	b, err := a.Finish()
	if err != nil {
		t.Fatal(err)
	}
	code := make([]uint32, len(b)/4)
	for i := range code {
		code[i] = binary.LittleEndian.Uint32(b[4*i:])
	}
	hops := 0
	for _, br := range brs {
		to := a.labels[br.label]
		got, steps, why := follow(code, br.at, br, true, func(pc int) bool { return pc == to })
		if steps > 1 {
			hops++
		}
		if got != to {
			t.Fatalf("branch at %d to %d (%#08x): taken, %d: %s", br.at, to, br.word, got, why)
		}
		if br.word&0xFC000000 == 0x14000000 {
			continue
		}
		got, _, why = follow(code, br.at, br, false, func(pc int) bool { return pc == br.next })
		if got != br.next {
			t.Fatalf("branch at %d to %d (%#08x): not taken, %d, want %d: %s", br.at, to, br.word, got, br.next, why)
		}
	}
	if hops == 0 {
		t.Fatalf("no branch went through another")
	}
	t.Logf("%d instructions, %d branches, %d through others", n, len(brs), hops)
}

// Code longer than every reach: each branch, forward and back, of every
// kind, goes where it should -- through veneers, veneers of veneers, the
// opposite condition over a B, and a jump through the temporary.
func TestPoolsNarrowed(t *testing.T) {
	narrowReach(t, 8)
	for seed := range int64(4) {
		poolProgram(t, seed, 3<<20, 50)
	}
}

// At the instructions' own reaches: 12 MB of code with conditional
// branches across all of it.
func TestPoolsLong(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	poolProgram(t, 9, 3<<20, 2000)
}

// Code shorter than a quarter of a conditional branch's reach has no
// pool: it is as it was without them.
func TestNoPoolInShortCode(t *testing.T) {
	var a Asm
	end := a.NewLabel()
	n := nearVeneer - 16
	for range n {
		a.BCond(NE, end)
	}
	a.Bind(end)
	b, err := a.Finish()
	if err != nil {
		t.Fatal(err)
	}
	if len(b) != 4*n {
		t.Fatalf("%d words, want %d", len(b)/4, n)
	}
	for i := range n {
		if w := binary.LittleEndian.Uint32(b[4*i:]); w != 0x54000001|uint32(n-i)<<5 {
			t.Fatalf("word %d: %#08x", i, w)
		}
	}
}
