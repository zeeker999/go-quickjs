package v8bench

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	quickjs "github.com/go-quickjs/go-quickjs"
	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/compiler"
	"github.com/go-quickjs/go-quickjs/internal/jit/compile"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
	"github.com/go-quickjs/go-quickjs/internal/parser"
)

// regLowerStats is what TestRegLowerEvaluate found over the corpus.
type regLowerStats struct {
	functions, lowered, invalid int
	refused                     map[string]int
	// runs is how many evaluations ran, compared those that returned a
	// value the register interpreter's answer was compared with, by
	// what they ended in otherwise.
	runs, compared, generic, budget, handles int
	exits                                    map[string]int
}

// regLowerArgs are the numbers the corpus's functions are called with: the
// edges of int32 arithmetic, -0, NaN and the infinities among them.
var regLowerArgs = []float64{0, 1, 2, 3, 5, 7, 10, 31, 32, 100, -1, -7, 0.5, -0.5, 2147483647, -2147483648,
	4294967295, math.NaN(), math.Copysign(0, -1), 1e10, math.Inf(1), 12, 64, 255}

// regLowerIdent is a name a call can be written with.
var regLowerIdent = regexp.MustCompile(`^[A-Za-z_$][A-Za-z0-9_$]*$`)

// regLowerLiteral is v as JavaScript writes it.
func regLowerLiteral(v ir.Value) (string, bool) {
	switch v.Kind {
	case ir.Number:
		x := math.Float64frombits(v.Bits)
		switch {
		case math.IsNaN(x):
			return "NaN", true
		case math.IsInf(x, 1):
			return "Infinity", true
		case math.IsInf(x, -1):
			return "-Infinity", true
		case x == 0 && math.Signbit(x):
			return "-0", true
		}
		return strconv.FormatFloat(x, 'g', -1, 64), true
	case ir.Boolean:
		return strconv.FormatBool(v.Bits != 0), true
	case ir.Null:
		return "null", true
	case ir.Undefined:
		return "undefined", true
	}
	return "", false
}

// regLowerCheck lowers every function of src's register code, validates
// each program, and runs those of its top-level functions with numbers as
// their arguments through ir's evaluator: each that returns a value is
// compared with what the register interpreter returns for the same call.
func regLowerCheck(t *testing.T, st *regLowerStats, src string) {
	t.Helper()
	compiler.SetRegisterForm(true)
	defer compiler.SetRegisterForm(os.Getenv("QJS_REG") != "")
	prog, err := parser.Parse(src, parser.Options{})
	if err != nil {
		t.Logf("not a script: %v", err)
		return
	}
	top, err := compiler.Compile(prog, compiler.Options{Text: src})
	if err != nil {
		t.Logf("not compiled: %v", err)
		return
	}
	programs := map[*bytecode.Function]*ir.Program{}
	var walk func(fn *bytecode.Function)
	walk = func(fn *bytecode.Function) {
		for _, c := range fn.Constants {
			if c.Kind != bytecode.ConstFunction || c.Fn == nil {
				continue
			}
			f := c.Fn
			walk(f)
			if f.Reg == nil {
				continue
			}
			st.functions++
			p, err := compile.LowerRegInline(f.Reg, f)
			if err != nil {
				var rf *compile.Refusal
				if errors.As(err, &rf) {
					st.refused[rf.Reason]++
				} else {
					st.refused[err.Error()]++
				}
				continue
			}
			if err := p.Validate(); err != nil {
				st.invalid++
				t.Errorf("%s: invalid lowering: %v\n%s", f.Name, err, f.Reg)
				continue
			}
			st.lowered++
			programs[f] = p
		}
	}
	walk(top)
	// The top level's functions by name, those of one name alone.
	named := map[string]*bytecode.Function{}
	seen := map[string]int{}
	for _, c := range top.Constants {
		if c.Kind == bytecode.ConstFunction && c.Fn != nil && c.Fn.Name != "" {
			seen[c.Fn.Name]++
			named[c.Fn.Name] = c.Fn
		}
	}
	withRegTier(true, func() {
		rt := quickjs.New()
		defer rt.Close()
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		_, err := rt.EvalContext(ctx, src)
		cancel()
		if err != nil {
			t.Logf("eval: %v", err)
			return
		}
		names := slices.Sorted(func(yield func(string) bool) {
			for n := range named {
				if !yield(n) {
					return
				}
			}
		})
		for k, name := range names {
			fn := named[name]
			p := programs[fn]
			if p == nil || seen[name] != 1 || p.Params > 4 || fn.Kind != bytecode.KindNormal || !regLowerIdent.MatchString(name) {
				continue
			}
			// The global of the name is this function: what is called by
			// it is what was evaluated.
			text, _ := json.Marshal(fn.Text)
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			v, err := rt.EvalContext(ctx, fmt.Sprintf(`typeof %s === "function" && %s.toString() === %s`, name, name, text))
			cancel()
			if err != nil || v.String() != "true" {
				continue
			}
			for j := range 8 {
				slots := make([]ir.Value, p.Locals)
				slots[0] = ir.Value{Kind: ir.Opaque}
				slots[1] = ir.Value{Kind: ir.Opaque, Bits: 1}
				args := make([]string, p.Params)
				for i := range p.Params {
					x := regLowerArgs[(j*7+i*3+k)%len(regLowerArgs)]
					if j == 0 {
						x = 3
					}
					slots[p.ParamBase+i] = ir.Float(x)
					args[i], _ = regLowerLiteral(ir.Float(x))
				}
				for i := p.UpvalueBase; i < p.UpvalueBase+p.Upvalues; i++ {
					slots[i] = ir.Value{Kind: ir.Opaque, Bits: 2}
				}
				st.runs++
				exit, err := p.Evaluate(slots, 0, 1<<18)
				switch {
				case errors.Is(err, ir.ErrGeneric):
					st.generic++
					continue
				case err != nil:
					t.Errorf("%s(%s): %v", name, strings.Join(args, ", "), err)
					continue
				case exit.Kind == ir.BudgetExit:
					st.budget++
					continue
				case exit.Kind != ir.Returned:
					st.exits[fn.Reg.Code[exit.State.PC].Op().String()]++
					continue
				}
				want, ok := regLowerLiteral(exit.Value)
				if !ok {
					st.handles++
					continue
				}
				call := fmt.Sprintf("%s(%s)", name, strings.Join(args, ", "))
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				v, err := rt.EvalContext(ctx, fmt.Sprintf(`(function () { var r = %s; return Object.is(r, %s) ? "same" : typeof r + " " + String(r) })()`, call, want))
				cancel()
				st.compared++
				if err != nil || v.String() != "same" {
					got := ""
					if err == nil {
						got = v.String()
					}
					t.Errorf("%s: the evaluator returned %s, the register interpreter %s (%v)\n%s", call, want, got, err, fn.Reg)
				}
			}
		}
	})
}

// TestRegLowerEvaluate is LowerReg's differential test: the register
// interpreter's corpus (the regDiff tests) compiled, every function's
// register code lowered and validated, and its top-level functions run by
// ir's evaluator with numbers as their arguments, each value it returns
// compared with what the register interpreter returns for the same call.
// What the evaluator cannot make -- a global's read, a property's, a call,
// what Go makes -- leaves, and is counted by the register instruction it
// left at.
func TestRegLowerEvaluate(t *testing.T) {
	st := &regLowerStats{refused: map[string]int{}, exits: map[string]int{}}
	regDiffHook = func(t *testing.T, src string, calls []string) { regLowerCheck(t, st, src) }
	defer func() { regDiffHook = nil }()
	for _, tc := range []struct {
		name string
		f    func(*testing.T)
	}{
		{"Appends", TestRegDiffAppends}, {"ArgumentsForms", TestRegDiffArgumentsForms}, {"BlockFunctions", TestRegDiffBlockFunctions},
		{"CallTargets", TestRegDiffCallTargets}, {"ChainsAndClosures", TestRegDiffChainsAndClosures}, {"CharCodeAt", TestRegDiffCharCodeAt},
		{"Classes", TestRegDiffClasses}, {"Constructions", TestRegDiffConstructions}, {"ElemWrites", TestRegDiffElemWrites},
		{"Eval", TestRegDiffEval}, {"EvalCode", TestRegDiffEvalCode}, {"FarJumps", TestRegDiffFarJumps}, {"FarNames", TestRegDiffFarNames},
		{"FarRegisters", TestRegDiffFarRegisters}, {"FinallyCopies", TestRegDiffFinallyCopies}, {"FinallyDeep", TestRegDiffFinallyDeep},
		{"ForOfIn", TestRegDiffForOfIn}, {"GeneratorSteps", TestRegDiffGeneratorSteps}, {"Generators", TestRegDiffGenerators},
		{"ImportCalls", TestRegDiffImportCalls}, {"Int32", TestRegDiffInt32}, {"LabelsSwitchDelete", TestRegDiffLabelsSwitchDelete},
		{"LeafRun", TestRegDiffLeafRun}, {"Length", TestRegDiffLength}, {"Literals", TestRegDiffLiterals}, {"LoopBindings", TestRegDiffLoopBindings},
		{"Microbenchmarks", TestRegDiffMicrobenchmarks}, {"Names", TestRegDiffNames}, {"ObjectLiterals", TestRegDiffObjectLiterals},
		{"Patterns", TestRegDiffPatterns}, {"RegExpLiteralCalls", TestRegDiffRegExpLiteralCalls}, {"RegExpLiterals", TestRegDiffRegExpLiterals},
		{"RegExpWideConstants", TestRegDiffRegExpWideConstants}, {"RestArguments", TestRegDiffRestArguments}, {"Sites", TestRegDiffSites},
		{"StackSitesPast16Bits", TestRegDiffStackSitesPast16Bits}, {"StacksAndNames", TestRegDiffStacksAndNames}, {"Steps", TestRegDiffSteps},
		{"StrictGlobalStores", TestRegDiffStrictGlobalStores}, {"TailCalls", TestRegDiffTailCalls}, {"TopLevel", TestRegDiffTopLevel},
		{"Try", TestRegDiffTry}, {"Typed", TestRegDiffTyped}, {"Using", TestRegDiffUsing}, {"V8Suite", TestRegDiffV8Suite},
		{"WideForms", TestRegDiffWideForms}, {"With", TestRegDiffWith},
	} {
		t.Run(tc.name, tc.f)
	}
	type row struct {
		k string
		n int
	}
	sorted := func(m map[string]int) []row {
		var rows []row
		for k, n := range m {
			rows = append(rows, row{k, n})
		}
		slices.SortFunc(rows, func(a, b row) int {
			if c := cmp.Compare(b.n, a.n); c != 0 {
				return c
			}
			return strings.Compare(a.k, b.k)
		})
		return rows
	}
	var out strings.Builder
	fmt.Fprintf(&out, "%d functions: %d lowered, %d refused, %d invalid\n", st.functions, st.lowered, st.functions-st.lowered-st.invalid, st.invalid)
	for _, r := range sorted(st.refused) {
		fmt.Fprintf(&out, "  %6d refused: %s\n", r.n, r.k)
	}
	fmt.Fprintf(&out, "%d evaluations: %d returned and compared, %d a handle, %d out of budget, %d Go's after a commit\n",
		st.runs, st.compared, st.handles, st.budget, st.generic)
	for _, r := range sorted(st.exits) {
		fmt.Fprintf(&out, "  %6d left at %s\n", r.n, r.k)
	}
	t.Log("\n" + out.String())
	if st.compared == 0 {
		t.Error("no evaluation was compared")
	}
}
