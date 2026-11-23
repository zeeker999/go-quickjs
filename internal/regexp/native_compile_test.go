//go:build quickjs_jit

package regexp

import (
	"bytes"
	"fmt"
	"slices"
	"testing"
)

var nativeCompileCases = []struct {
	pattern, flags string
	accepted       bool
}{
	{`^\s*|\s*$`, "", true},
	{`([a-zA-Z]|\s)+`, "", true},
	{`^(\w+|\*)$`, "", true},
	{`(\d*)(\D*)`, "", true},
	{`(?:^|;)\s*lang=([^;]*)`, "", true},
	{`^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?$`, "", true},
	{`\b(active|hidden|selected)\b`, "", true},
	{`[a-z\u0100-\u0200\u1000-\u2000]+`, "i", true},
	{`[\p{L}\p{N}_]+`, "u", true},
	{`(ab|cd){2,8}?`, "", true},
	{`(a*)*b`, "", true},
	{`(a+)\1`, "", true},
	{`a(?=b)`, "", false},
	{`(?<=a)b`, "", false},
}

func TestNativeCompileReuse(t *testing.T) {
	for _, arch := range []string{"amd64", "arm64"} {
		for _, wide := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/wide=%v", arch, wide), func(t *testing.T) {
				t.Parallel()
				var kept, want []nativeProgram
				for round := range 8 {
					for i, tc := range nativeCompileCases {
						re, err := Compile(tc.pattern, tc.flags)
						if err != nil {
							t.Fatal(err)
						}
						p, ok := genNative(arch, re.prog, 2*(re.groupCount+1), wide, re.flags&FlagUnicode != 0)
						if ok != tc.accepted {
							t.Fatalf("/%s/%s: accepted=%v, want %v", tc.pattern, tc.flags, ok, tc.accepted)
						}
						if round == 0 {
							kept = append(kept, p)
							p.bytes = slices.Clone(p.bytes)
							want = append(want, p)
						} else if !bytes.Equal(p.bytes, want[i].bytes) || p.text != want[i].text || p.maxPush != want[i].maxPush {
							t.Fatalf("/%s/%s: compile changed after reuse", tc.pattern, tc.flags)
						}
					}
					// A host may keep code while other runtimes compile and reuse
					// the assembler; those compiles must not overwrite its bytes.
					for i, p := range kept {
						if !bytes.Equal(p.bytes, want[i].bytes) {
							t.Fatalf("/%s/: retained code overwritten", nativeCompileCases[i].pattern)
						}
					}
				}
			})
		}
	}
}

func TestNativeMasmReset(t *testing.T) {
	for _, arch := range []string{"amd64", "arm64"} {
		t.Run(arch, func(t *testing.T) {
			m := newMasm(arch)
			m.Table(make([]byte, 128))
			m.Tables()
			// Force the discard path so the test can inspect the reset
			// assembler without assuming what sync.Pool.Get returns.
			for range 300 << 10 {
				m.Ret()
			}
			m.Jmp(m.NewLabel())
			if _, err := m.Finish(); err == nil {
				t.Fatal("unbound label accepted")
			}
			giveNativeMasm(m)
			if a, ok := m.(*amd64Masm); ok {
				if len(a.tables) != 0 {
					t.Fatal("tables not reset")
				}
				for _, table := range a.tables[:cap(a.tables)] {
					if table.b != nil {
						t.Fatal("table bytes retained")
					}
				}
			}
			if l := m.NewLabel(); l != 0 {
				t.Fatalf("first label after reset: %d", l)
			}
			m.Bind(0)
			m.Ret()
			m.Tables()
			got, err := m.Finish()
			if err != nil {
				t.Fatal(err)
			}
			fresh := newMasm(arch)
			fresh.Ret()
			want, err := fresh.Finish()
			if err != nil || !bytes.Equal(got, want) {
				t.Fatalf("reset assembler: %x, %v; fresh: %x", got, err, want)
			}
		})
	}
	if borrowNativeMasm("unsupported") != nil {
		t.Fatal("unsupported architecture accepted")
	}
}

func BenchmarkNativeCompile(b *testing.B) {
	var patterns []*Regexp
	for _, tc := range nativeCompileCases {
		re, err := Compile(tc.pattern, tc.flags)
		if err != nil {
			b.Fatal(err)
		}
		patterns = append(patterns, re)
	}
	for _, arch := range []string{"amd64", "arm64"} {
		b.Run(arch, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				for _, re := range patterns {
					for _, wide := range []bool{false, true} {
						genNative(arch, re.prog, 2*(re.groupCount+1), wide, re.flags&FlagUnicode != 0)
					}
				}
			}
		})
	}
}
