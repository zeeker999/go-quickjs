package compile

import (
	"cmp"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/compiler"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
	"github.com/go-quickjs/go-quickjs/internal/parser"
)

// regLoweringOf is what the register lowering makes of fn, as ssaRefusal
// tells the stack form's: "lowered" for Go's entry (LowerReg); for a
// function the entry leaves for doing no loop's or array's work, "lowered
// callee" or "callee: <reason>" (LowerRegInline), or, a generator's or an
// async function's, "lowered hot" or "hot: <reason>" (LowerRegHot); and
// "entry: <reason>" for one the entry refuses outright. It returns the
// programs it made, for their validation.
func regLoweringOf(fn *bytecode.Function) (string, []error) {
	if fn.Reg == nil {
		return "entry: no register code", nil
	}
	reason := func(err error) string {
		var rf *Refusal
		if errors.As(err, &rf) {
			return rf.Reason
		}
		return "error: " + err.Error()
	}
	var invalid []error
	check := func(lower func(*bytecode.RegCode, *bytecode.Function) (*ir.Program, error)) error {
		p, err := lower(fn.Reg, fn)
		if err == nil {
			if verr := p.Validate(); verr != nil {
				invalid = append(invalid, fmt.Errorf("%s: %w", fn.Name, verr))
			}
		}
		return err
	}
	err := check(LowerReg)
	switch {
	case err == nil:
		return "lowered", invalid
	case HostOnly(err) && (fn.Async || fn.Generator):
		if err := check(LowerRegHot); err != nil {
			return "hot: " + reason(err), invalid
		}
		return "lowered hot", invalid
	case HostOnly(err):
		if err := check(LowerRegInline); err != nil {
			return "callee: " + reason(err), invalid
		}
		return "lowered callee", invalid
	}
	return "entry: " + reason(err), invalid
}

// TestLowerRegCorpusRefusals is Stage 3a-2's gate (docs/register-
// interpreter.md, Stage 3, section 6): every function of the .js files
// under each directory QUICKJS_JIT_REG_REFUSALS lists (a path list: the V8
// suite's, Octane's, bundles') compiled with the register backend on, and
// lowered both ways, the stack form's (ssaRefusal) and the register code's
// (regLoweringOf). It prints how many each lowers and refuses, by reason,
// and fails where the register lowering refuses more than the stack
// form's, or makes a program ir.Validate rejects.
func TestLowerRegCorpusRefusals(t *testing.T) {
	dirs := filepath.SplitList(os.Getenv("QUICKJS_JIT_REG_REFUSALS"))
	if len(dirs) == 0 {
		t.Skip("QUICKJS_JIT_REG_REFUSALS names no directories of JavaScript files")
	}
	compiler.SetRegisterForm(true)
	defer compiler.SetRegisterForm(false)
	for _, dir := range dirs {
		stack, regs := map[string]int{}, map[string]int{}
		var invalid []error
		files, total := 0, 0
		var walk func(fn *bytecode.Function, count bool)
		walk = func(fn *bytecode.Function, count bool) {
			if count && !fn.TopLevel {
				total++
				stack[ssaRefusal(fn)]++
				r, bad := regLoweringOf(fn)
				regs[r]++
				invalid = append(invalid, bad...)
			}
			for _, c := range fn.Constants {
				if c.Kind == bytecode.ConstFunction && c.Fn != nil {
					walk(c.Fn, true)
				}
			}
		}
		err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
			if err != nil || d.IsDir() || !strings.HasSuffix(p, ".js") {
				return err
			}
			b, err := os.ReadFile(p)
			if err != nil {
				return err
			}
			// Each file as a script's code is: its functions are what
			// count, the top level's own code not.
			text := string(b)
			prog, err := parser.Parse(text, parser.Options{})
			if err != nil {
				t.Logf("%s: %v", p, err)
				return nil
			}
			fn, err := compiler.Compile(prog, compiler.Options{Source: p, Text: text})
			if err != nil {
				t.Logf("%s: %v", p, err)
				return nil
			}
			files++
			walk(fn, false)
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		refused := func(m map[string]int) int {
			n := 0
			for k, v := range m {
				if !strings.HasPrefix(k, "lowered") {
					n += v
				}
			}
			return n
		}
		var out strings.Builder
		fmt.Fprintf(&out, "%s: %d files, %d functions; refused: stack form %d, register code %d\n", dir, files, total, refused(stack), refused(regs))
		keys := map[string]bool{}
		for k := range stack {
			keys[k] = true
		}
		for k := range regs {
			keys[k] = true
		}
		rows := slices.Collect(func(yield func(string) bool) {
			for k := range keys {
				if !yield(k) {
					return
				}
			}
		})
		slices.SortFunc(rows, func(a, b string) int {
			if c := cmp.Compare(max(stack[b], regs[b]), max(stack[a], regs[a])); c != 0 {
				return c
			}
			return strings.Compare(a, b)
		})
		fmt.Fprintf(&out, "%7s %7s  %s\n", "stack", "reg", "what")
		for _, k := range rows {
			fmt.Fprintf(&out, "%7d %7d  %s\n", stack[k], regs[k], k)
		}
		t.Log("\n" + out.String())
		for i, err := range invalid {
			if i == 20 {
				t.Errorf("and %d more", len(invalid)-20)
				break
			}
			t.Errorf("invalid register lowering: %v", err)
		}
		if refused(regs) > refused(stack) {
			t.Errorf("%s: the register lowering refuses %d functions, the stack form's %d", dir, refused(regs), refused(stack))
		}
	}
}
