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
	"github.com/go-quickjs/go-quickjs/internal/jit/oracle"
	"github.com/go-quickjs/go-quickjs/internal/parser"
)

// regLowerStats is what TestRegLowerEvaluate found over the corpus.
type regLowerStats struct {
	functions, lowered, invalid int
	refused                     map[string]int
	// runs is how many calls were evaluated: by the evaluator alone, of
	// which returned ran to the end; and with the oracle's machine making
	// what the evaluator does not, of which compared returned a primitive
	// the register interpreter's answer was compared with, handles a value
	// of the machine's heap.
	runs, returned, generic, budget int
	compared, handles, mGeneric     int
	// threw is how many of those compared threw.
	threw int
	// exits are where the evaluator alone left, stops where it stopped
	// with the machine, by register instruction.
	exits, stops map[string]int
}

// regLowerArgs are the numbers the corpus's functions are called with: the
// edges of int32 arithmetic, -0, NaN and the infinities among them.
var regLowerArgs = []float64{0, 1, 2, 3, 5, 7, 10, 31, 32, 100, -1, -7, 0.5, -0.5, 2147483647, -2147483648,
	4294967295, math.NaN(), math.Copysign(0, -1), 1e10, math.Inf(1), 12, 64, 255}

// regLowerIdent is a name a call can be written with.
var regLowerIdent = regexp.MustCompile(`^[A-Za-z_$][A-Za-z0-9_$]*$`)

// regLowerHelper is the runtime's side of the machine's reader of
// globals: a global's value as a graph -- primitives, plain objects and
// their prototypes, dense arrays, functions by their source text -- in
// JSON, or "no" where it has what the machine does not model: an accessor,
// a symbol, a hole, an object of another kind, a frozen one, or more than
// 2,000 objects.
const regLowerHelper = `globalThis.__oracleGraph = function (root) {
	var nodes = [], seen = new Map();
	function ref(v) {
		if (v === undefined) return { u: 1 };
		if (v === null || typeof v === "boolean" || typeof v === "string") return { v: v };
		if (typeof v === "number") return { n: Object.is(v, -0) ? "-0" : String(v) };
		if (typeof v !== "object" && typeof v !== "function") throw 0;
		if (seen.has(v)) return { r: seen.get(v) };
		if (nodes.length >= 2000 || !Object.isExtensible(v) || Object.getOwnPropertySymbols(v).length) throw 0;
		var id = nodes.length, node = { k: [] }, isArray = Array.isArray(v), p = Object.getPrototypeOf(v);
		nodes.push(node);
		seen.set(v, id);
		if (typeof v === "function") {
			if (p !== Function.prototype) throw 0;
			node.f = Function.prototype.toString.call(v);
		} else if (isArray) {
			if (p !== Array.prototype) throw 0;
			node.a = [];
			for (var i = 0; i < v.length; i++) {
				if (!(i in v)) throw 0;
				node.a.push(ref(v[i]));
			}
		} else if (p === null) {
			node.np = 1;
		} else if (p !== Object.prototype) {
			if (typeof p !== "object") throw 0;
			node.p = ref(p);
		}
		var ks = Object.getOwnPropertyNames(v);
		for (var j = 0; j < ks.length; j++) {
			var k = ks[j];
			if (isArray && (k === "length" || /^[0-9]+$/.test(k))) continue;
			if (typeof v === "function" && (k === "length" || k === "name" || k === "arguments" || k === "caller")) continue;
			var d = Object.getOwnPropertyDescriptor(v, k);
			if (!("value" in d) || /^[0-9]/.test(k)) throw 0;
			if ((!d.writable || !d.configurable) && !(typeof v === "function" && k === "prototype")) throw 0;
			node.k.push([k, d.enumerable ? 1 : 0, ref(d.value)]);
		}
		return { r: id };
	}
	try {
		return JSON.stringify({ r: ref(root), nodes: nodes });
	} catch (e) {
		return "no";
	}
};`

// regLowerGraph is the graph regLowerHelper wrote of a value.
type regLowerGraph struct {
	R     regLowerRef
	Nodes []struct {
		F  *string
		A  []regLowerRef
		P  *regLowerRef
		NP int
		K  [][3]json.RawMessage
	}
}

// regLowerRef is a value of the graph: undefined, a primitive, a number's
// text, or an object's index.
type regLowerRef struct {
	U int
	V any
	N *string
	R *int
}

// regLowerBuild is the graph g on m's heap, each function the template of
// its source text in texts where only one template has it, a foreign
// function otherwise.
func regLowerBuild(m *oracle.Machine, g *regLowerGraph, texts map[string][]*bytecode.Function) (ir.Value, bool) {
	objs := make([]ir.Value, len(g.Nodes))
	for i, n := range g.Nodes {
		switch {
		case n.F != nil:
			if fns := texts[*n.F]; len(fns) == 1 {
				objs[i] = m.NewFunction(fns[0])
			} else {
				objs[i] = m.NewForeign()
			}
		case n.A != nil:
			objs[i] = m.NewArray()
		default:
			objs[i] = m.NewObject()
		}
	}
	value := func(r regLowerRef) (ir.Value, bool) {
		switch {
		case r.U != 0:
			return ir.Value{}, true
		case r.R != nil:
			return objs[*r.R], *r.R < len(objs)
		case r.N != nil:
			switch *r.N {
			case "NaN":
				return ir.Float(math.NaN()), true
			case "Infinity":
				return ir.Float(math.Inf(1)), true
			case "-Infinity":
				return ir.Float(math.Inf(-1)), true
			case "-0":
				return ir.Float(math.Copysign(0, -1)), true
			}
			x, err := strconv.ParseFloat(*r.N, 64)
			return ir.Float(x), err == nil
		}
		switch x := r.V.(type) {
		case nil:
			return ir.Value{Kind: ir.Null}, true
		case bool:
			return ir.Bool(x), true
		case string:
			return m.String(x), true
		}
		return ir.Value{}, false
	}
	for i, n := range g.Nodes {
		for _, e := range n.A {
			v, ok := value(e)
			if !ok {
				return ir.Value{}, false
			}
			m.Push(objs[i], v)
		}
		if n.P != nil {
			p, ok := value(*n.P)
			if !ok {
				return ir.Value{}, false
			}
			m.SetProto(objs[i], p, false)
		} else if n.NP != 0 {
			m.SetProto(objs[i], ir.Value{}, true)
		}
		for _, kv := range n.K {
			var k string
			var en int
			var r regLowerRef
			if json.Unmarshal(kv[0], &k) != nil || json.Unmarshal(kv[1], &en) != nil || json.Unmarshal(kv[2], &r) != nil {
				return ir.Value{}, false
			}
			v, ok := value(r)
			if !ok {
				return ir.Value{}, false
			}
			m.Define(objs[i], k, v, en != 0)
		}
	}
	return value(g.R)
}

// regLowerCheck lowers every function of src's register code, validates
// each program, and runs those of its top-level functions with numbers as
// their arguments: through ir's evaluator alone, and with the oracle's
// machine making what the evaluator does not, each value it returns
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
	// texts are the program's functions by their source text, which a
	// global's graph names its functions by.
	texts := map[string][]*bytecode.Function{}
	var walk func(fn *bytecode.Function)
	walk = func(fn *bytecode.Function) {
		for _, c := range fn.Constants {
			if c.Kind != bytecode.ConstFunction || c.Fn == nil {
				continue
			}
			f := c.Fn
			walk(f)
			texts[f.Text] = append(texts[f.Text], f)
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
		eval := func(src string) (string, error) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			v, err := rt.EvalContext(ctx, src)
			if err != nil {
				return "", err
			}
			return v.String(), nil
		}
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		_, err := rt.EvalContext(ctx, src)
		cancel()
		if err != nil {
			t.Logf("eval: %v", err)
			return
		}
		if _, err := rt.Eval(regLowerHelper); err != nil {
			t.Fatal(err)
		}
		// verified reports whether the global of name is the program's one
		// function of the name: what is called by it is what was lowered.
		checked := map[string]bool{}
		verified := func(name string) bool {
			ok, done := checked[name]
			if done {
				return ok
			}
			if fn := named[name]; fn != nil && seen[name] == 1 && regLowerIdent.MatchString(name) {
				text, _ := json.Marshal(fn.Text)
				v, err := eval(fmt.Sprintf(`typeof %s === "function" && %s.toString() === %s`, name, name, text))
				ok = err == nil && v == "true"
			}
			checked[name] = ok
			return ok
		}
		// global is the machine's reader of a global the code reads, from
		// the runtime as it is before the call, its value's graph copied.
		global := func(m *oracle.Machine) func(string) (ir.Value, bool) {
			return func(name string) (ir.Value, bool) {
				if !regLowerIdent.MatchString(name) {
					return ir.Value{}, false
				}
				d, err := eval(fmt.Sprintf(`(function () { try { return __oracleGraph(%s) } catch (e) { return "no" } })()`, name))
				if err != nil || d == "no" {
					return ir.Value{}, false
				}
				var g regLowerGraph
				if json.Unmarshal([]byte(d), &g) != nil {
					return ir.Value{}, false
				}
				return regLowerBuild(m, &g, texts)
			}
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
			if p == nil || p.Params > 4 || fn.Kind != bytecode.KindNormal || !verified(name) {
				continue
			}
			for j := range 8 {
				slots := make([]ir.Value, p.Locals)
				slots[0] = ir.Value{Kind: ir.Opaque}
				slots[1] = ir.Value{Kind: ir.Opaque, Bits: 1}
				args := make([]string, p.Params)
				vals := make([]ir.Value, p.Params)
				for i := range p.Params {
					x := regLowerArgs[(j*7+i*3+k)%len(regLowerArgs)]
					if j == 0 {
						x = 3
					}
					vals[i] = ir.Float(x)
					slots[p.ParamBase+i] = vals[i]
					args[i] = oracle.NumberString(x)
					if x == 0 && math.Signbit(x) {
						args[i] = "-0"
					}
				}
				for i := p.UpvalueBase; i < p.UpvalueBase+p.Upvalues; i++ {
					slots[i] = ir.Value{Kind: ir.Opaque, Bits: 2}
				}
				st.runs++
				// The evaluator alone, which leaves at what it cannot make.
				exit, err := p.Evaluate(slots, 0, 1<<18)
				switch {
				case errors.Is(err, ir.ErrGeneric):
					st.generic++
				case err != nil:
					t.Errorf("%s(%s): %v", name, strings.Join(args, ", "), err)
				case exit.Kind == ir.BudgetExit:
					st.budget++
				case exit.Kind != ir.Returned:
					st.exits[fn.Reg.Code[exit.State.PC].Op().String()]++
				default:
					st.returned++
				}
				// With the machine making what the evaluator does not: the
				// program's functions called by evaluating theirs.
				m := oracle.New()
				m.Budget = 1 << 19
				m.Global = global(m)
				r, ok, err := m.Call(m.Function(fn), ir.Value{}, vals)
				switch {
				case errors.Is(err, ir.ErrGeneric):
					st.mGeneric++
					continue
				case err != nil:
					t.Errorf("%s(%s), with the machine: %v", name, strings.Join(args, ", "), err)
					continue
				case !ok && m.Thrown != "":
					// The machine is sure the call throws: so must the
					// register interpreter's, the same kind.
					call := fmt.Sprintf("%s(%s)", name, strings.Join(args, ", "))
					got, err := eval(fmt.Sprintf(`(function () { try { %s; return "returned" } catch (e) { return (%s) ? "same" : String(e) } })()`, call, m.Thrown))
					st.compared++
					st.threw++
					if err != nil || got != "same" {
						t.Errorf("%s: the evaluator threw an exception of which %s, the register interpreter %s (%v)\n%s", call, m.Thrown, got, err, fn.Reg)
					}
					continue
				case !ok:
					st.stops[m.Stop]++
					continue
				}
				call := fmt.Sprintf("%s(%s)", name, strings.Join(args, ", "))
				want, ok := m.Text(r)
				check := fmt.Sprintf(`(function () { var r = %s; return Object.is(r, %s) ? "same" : typeof r + " " + String(r) })()`, call, want)
				if !ok {
					// An array's or an object's, compared as JSON writes it.
					if want, ok = m.JSON(r); !ok {
						st.handles++
						continue
					}
					q, _ := json.Marshal(want)
					check = fmt.Sprintf(`(function () { var r = JSON.stringify(%s); return r === %s ? "same" : r })()`, call, q)
				}
				got, err := eval(check)
				st.compared++
				if err != nil || got != "same" {
					t.Errorf("%s: the evaluator returned %s, the register interpreter %s (%v)\n%s", call, want, got, err, fn.Reg)
				}
			}
		}
	})
}

// TestRegLowerEvaluate is LowerReg's differential test: the register
// interpreter's corpus (the regDiff tests) compiled, every function's
// register code lowered and validated, and its top-level functions run
// with numbers as their arguments -- by ir's evaluator alone, and with
// the oracle's machine making what it leaves to Go -- each value returned
// compared with what the register interpreter returns for the same call.
// What neither can make leaves, counted by the register instruction it
// left at.
func TestRegLowerEvaluate(t *testing.T) {
	st := &regLowerStats{refused: map[string]int{}, exits: map[string]int{}, stops: map[string]int{}}
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
	fmt.Fprintf(&out, "%d calls. The evaluator alone: %d returned, %d out of budget, %d Go's after a commit; left at\n",
		st.runs, st.returned, st.budget, st.generic)
	for i, r := range sorted(st.exits) {
		if i < 15 {
			fmt.Fprintf(&out, "  %6d %s\n", r.n, r.k)
		}
	}
	stopped := 0
	for _, n := range st.stops {
		stopped += n
	}
	fmt.Fprintf(&out, "With the machine: %d ran to the end and agreed with the register interpreter (%d by the exception both threw); %d returned a value JSON does not write; %d Generic it could not make; %d stopped, at\n",
		st.compared, st.threw, st.handles, st.mGeneric, stopped)
	for i, r := range sorted(st.stops) {
		if i < 60 {
			fmt.Fprintf(&out, "  %6d %s\n", r.n, r.k)
		}
	}
	t.Log("\n" + out.String())
	if st.compared == 0 {
		t.Error("no evaluation was compared")
	}
}
