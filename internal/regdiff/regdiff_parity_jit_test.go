//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package regdiff

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/compiler"
	"github.com/go-quickjs/go-quickjs/internal/v8bench"
	"github.com/go-quickjs/go-quickjs/internal/vm"
)

// parityCorpus is a program of the feedback-parity test and the calls that
// warm it.
type parityCorpus struct {
	name, src string
	calls     []string
}

// parityCorpora are the programs TestJITFeedbackParity runs: the
// micro-benchmarks, and the V8 suite, Octane and Are We Fast Yet where
// QUICKJS_V8_DIR, QUICKJS_OCTANE_DIR and QUICKJS_AWFY_DIR name their
// checkouts (v8bench -fetch).
func parityCorpora(t *testing.T) []parityCorpus {
	var out []parityCorpus
	src, calls := regDiffMicro(t)
	out = append(out, parityCorpus{"micro", src, calls})
	suite := func(name, env string, runs int) {
		dir := os.Getenv(env)
		if dir == "" {
			t.Logf("%s: %s names no checkout", name, env)
			return
		}
		files, _, err := v8bench.Corpus(name, dir)
		if err != nil {
			t.Logf("%s: %v", name, err)
			return
		}
		files += fmt.Sprintf(`
function parityBench(name) {
	var out = [];
	BenchmarkSuite.suites.forEach(function (s) {
		s.benchmarks.forEach(function (b) {
			if (b.name !== name) return;
			try {
				b.Setup();
				for (var i = 0; i < %d; i++) b.run();
				b.TearDown();
			} catch (e) { out.push("threw " + e) }
		});
	});
	return out.join(",");
}
function parityNames() {
	var out = [];
	BenchmarkSuite.suites.forEach(function (s) { s.benchmarks.forEach(function (b) { out.push(b.name) }) });
	return out;
}
`, runs)
		out = append(out, parityCorpus{name, files, []string{"__names__"}})
	}
	suite("v8", "QUICKJS_V8_DIR", 12)
	suite("octane", "QUICKJS_OCTANE_DIR", 3)
	if dir := os.Getenv("QUICKJS_AWFY_DIR"); dir != "" {
		var b strings.Builder
		b.WriteString("var __awfyDefs = {}, __awfyCache = {};\n")
		b.WriteString(`function __awfyRequire(path) {
	var name = path.replace(/^\.\//, "").replace(/\.js$/, "");
	var m = __awfyCache[name];
	if (m) return m.exports;
	m = __awfyCache[name] = { exports: {} };
	__awfyDefs[name](m, m.exports, __awfyRequire);
	return m.exports;
}
`)
		var calls []string
		for _, f := range []string{"benchmark", "som", "bounce", "cd", "deltablue", "havlak", "json", "list", "mandelbrot", "nbody", "permute", "queens", "richards", "sieve", "storage", "towers"} {
			text, err := os.ReadFile(dir + "/" + f + ".js")
			if err != nil {
				t.Logf("awfy: %v", err)
				calls = nil
				break
			}
			fmt.Fprintf(&b, "__awfyDefs[%s] = function (module, exports, require) {\n%s\n};\n", strconv.Quote(f), text)
			if f != "benchmark" && f != "som" {
				calls = append(calls, fmt.Sprintf(`(function () { try { var b = __awfyRequire("./%s").newInstance(); for (var i = 0; i < 10; i++) b.innerBenchmarkLoop(1); return "" } catch (e) { return "threw " + e } })()`, f))
			}
		}
		if calls != nil {
			out = append(out, parityCorpus{"awfy", b.String(), calls})
		}
	}
	return out
}

// TestJITFeedbackParity is the register form's JIT's feedback-parity gate
// (docs/register-interpreter.md, Stage 3): each corpus run in two
// runtimes that run the JIT, warmed alike -- one on the stack form's tiers,
// one on the register tier -- and, for every function both compiled, what
// each form's feedback answers compared site by site, and what each
// form's build decides from it (vm.CompareFeedback). It reports every gap;
// QUICKJS_PARITY_STRICT=1 fails on any.
func TestJITFeedbackParity(t *testing.T) {
	if testing.Short() {
		t.Skip("runs the benchmarks' corpora twice")
	}
	regDiffAlone.Lock()
	defer regDiffAlone.Unlock()
	for _, c := range parityCorpora(t) {
		t.Run(c.name, func(t *testing.T) {
			vm.RecordJITCompiles(true)
			defer vm.RecordJITCompiles(false)
			jitReg := vm.SetJITRegTier(true)
			defer func() {
				env := bytecode.RegDefault()
				compiler.SetRegisterForm(env)
				vm.SetRegTier(env)
				vm.SetJITRegTier(jitReg)
			}()
			var rts [2]*vm.Runtime
			for i := range rts {
				// The program compiled with the register form for both, its
				// stack form the same: the first runtime runs it on the old
				// tiers, the second on the register tier.
				compiler.SetRegisterForm(true)
				top := regLowerCompile(c.src)
				if top == nil {
					t.Fatal("not compiled")
				}
				vm.SetRegTier(i == 1)
				r := vm.New(vm.Config{MemoryLimit: 1 << 30, JIT: true})
				defer r.Close()
				rts[i] = r
				ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
				r.SetContext(ctx)
				if _, err := r.Run(top); err != nil {
					cancel()
					// Octane's code-load and run.js want eval, which a
					// bare runtime has none of.
					t.Skipf("run: %v", err)
				}
				calls := c.calls
				if len(calls) == 1 && calls[0] == "__names__" {
					names, err := r.Call(parityGlobal(r, "parityNames"), vm.Undefined, nil)
					if err != nil {
						cancel()
						t.Fatal(err)
					}
					calls = nil
					s, _ := r.Call(parityGlobal(r, "String"), vm.Undefined, []vm.Value{names})
					for _, n := range strings.Split(s.String().Go(), ",") {
						calls = append(calls, fmt.Sprintf("parityBench(%q)", n))
					}
				}
				for _, call := range calls {
					prog := regLowerCompile(call)
					if prog == nil {
						continue
					}
					if v, err := r.Run(prog); err != nil {
						t.Logf("%s: %v", call, err)
					} else if s := v.String(); s != nil && s.Go() != "" && strings.HasPrefix(s.Go(), "threw") {
						t.Logf("%s: %s", call, s.Go())
					}
				}
				cancel()
				r.SetContext(context.Background())
			}
			res := vm.CompareFeedback(rts[0], rts[1])
			t.Log("\n" + res.String())
			if res.Functions == 0 {
				t.Error("no function compiled by both forms")
			}
			if os.Getenv("QUICKJS_PARITY_STRICT") == "1" && (len(res.Gaps) != 0 || len(res.Decisions) != 0) {
				t.Error("the register form's feedback differs from the stack form's")
			}
		})
	}
}

// parityGlobal is the global name of r's.
func parityGlobal(r *vm.Runtime, name string) vm.Value {
	v, _ := r.GetProp(vm.Obj(r.Global()), r.Intern(name))
	return v
}
