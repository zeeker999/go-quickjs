//go:build quickjs_jit && !android && !ios && (linux || windows)

package jit

import (
	"crypto/sha256"
	"fmt"
	"os"

	"math/rand/v2"
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/compiler"
	jitcompile "github.com/go-quickjs/go-quickjs/internal/jit/compile"
	"github.com/go-quickjs/go-quickjs/internal/parser"
)

// TestTmpFingerprint hashes the amd64 code compiled for a fixed set of
// programs, to show a refactoring leaves it byte-identical.
func TestTmpFingerprint(t *testing.T) {
	h := sha256.New()
	var out *os.File
	if path := os.Getenv("FP_OUT"); path != "" {
		out, _ = os.Create(path)
		defer out.Close()
	}
	r := rand.New(rand.NewPCG(21, 22))
	// The flag's address varies from build to build; the code is only
	// compiled here.
	defer func(old uint64) { testEncoding.WriteBarrier = old }(testEncoding.WriteBarrier)
	testEncoding.WriteBarrier = 0x50000000
	n := 0
	for attempt := 0; attempt < 20000 && n < 3000; attempt++ {
		p, l := ssaTestProgram(r)
		l.generic, l.genericEntries = nil, nil
		// Holders' addresses vary from run to run; the code is only
		// compiled here, so fixed ones stand in.
		for pc, s := range l.sites {
			for i := range s.Holders {
				if s.Holders[i].Object != 0 {
					s.Holders[i].Object = uintptr(0x70000 + 0x100*i)
				}
			}
			l.sites[pc] = s
		}
		if p.Validate() != nil {
			continue
		}
		c, err := compileNative(p, l)
		if err != nil || c == nil {
			continue
		}
		h.Write(c.mc.Bytes)
		if out != nil {
			fmt.Fprintf(out, "== %d len %d\n", n, len(c.mc.Bytes))
			_ = sha256.Sum256
		}
		c.code.Close()
		n++
	}
	for _, src := range ssaNativeCorpus {
		prog, _ := parser.Parse(src, parser.Options{})
		top, _ := compiler.Compile(prog, compiler.Options{})
		for _, k := range top.Constants {
			if k.Kind != bytecode.ConstFunction {
				continue
			}
			p, err := jitcompile.LowerSSA(k.Fn)
			if err != nil {
				continue
			}
			c, err := compileNative(p, layout{k.Fn.LocalCount, -1, nil, nil, nil, nil})
			if err == nil && c != nil {
				h.Write(c.mc.Bytes)
				if out != nil {
					fmt.Fprintf(out, "== corpus %d len %d\n", n, len(c.mc.Bytes))
				}
				c.code.Close()
				n++
			}
		}
	}
	t.Logf("fingerprint %x over %d functions", h.Sum(nil), n)

}
