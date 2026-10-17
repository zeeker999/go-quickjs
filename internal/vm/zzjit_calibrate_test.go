//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
	"weak"

	"github.com/go-quickjs/go-quickjs/internal/jit"
)

// TestJITCalibration measures what each way a native stretch ends costs,
// against what the tree tier pays for the same work, and prints it as
// break-even ratios: an ending of kind K costs as much as N_K instructions
// of work native code saves over the tree tier. jitSSAProfit's constants
// come from it (jitCostHost and the others). Ratios, not times: both
// sides move with the CPU, and they carry across machines far better.
//
//	QUICKJS_JIT_CALIBRATE=1 go test -tags quickjs_jit -run '^TestJITCalibration$' -count=1 -v -timeout 60m ./internal/vm
//
// Each kernel is a loop timed in a runtime without the JIT (the tree tier)
// and in one with it, which judges nothing (jitJudgeOff): code that does
// not pay stays native, so that what it pays can be measured. A kind's cost
// is a difference of differences -- the loop with the ending against the
// same loop without, natively, less the same difference in the tree tier
// -- so that what the tree tier pays for the operation itself (a call of
// indexOf, a typeof) is not counted against native code. The work a native
// instruction saves is measured on loops of arithmetic, property and
// element access and inlined calls, each by the work jitSSAProfit
// estimates for it. Zen 2's op cache moves a single layout of JIT code by
// up to a quarter, so the whole is measured with the JIT's code placed at
// several offsets within 4 KB (jit.FirstOffset), and the median taken.
// QUICKJS_JIT_CALIBRATE=pads=0:1024,ms=20 picks the offsets, and how long
// each kernel is timed for (40 ms by default; each the fastest of five).
func TestJITCalibration(t *testing.T) {
	cfg := os.Getenv("QUICKJS_JIT_CALIBRATE")
	if cfg == "" {
		t.Skip("set QUICKJS_JIT_CALIBRATE=1 to calibrate the JIT's profit check")
	}
	if !jitSSABackend {
		t.Skip("no SSA backend on this architecture")
	}
	pads := []int{0, 512, 1024, 1536, 2048, 2560, 3072, 3584}
	target := 40 * time.Millisecond
	for _, opt := range strings.Split(cfg, ",") {
		switch k, v, _ := strings.Cut(opt, "="); k {
		case "pads":
			pads = nil
			for _, p := range strings.Split(v, ":") {
				n, err := strconv.Atoi(p)
				if err != nil {
					t.Fatalf("pads: %v", err)
				}
				pads = append(pads, n)
			}
		case "ms":
			n, err := strconv.Atoi(v)
			if err != nil {
				t.Fatalf("ms: %v", err)
			}
			target = time.Duration(n) * time.Millisecond
		}
	}
	jitJudgeOff = true
	defer func() { jitJudgeOff = false }()
	firstOffset := jit.FirstOffset
	defer func() { jit.FirstOffset = firstOffset }()

	tree := New(Config{})
	defer func() { tree.Close(); tree.ReleaseClosed() }()
	if _, err := tree.Run(compileForTest(t, calibSetup)); err != nil {
		t.Fatal(err)
	}
	trees := map[string]float64{}
	for _, k := range calibKernels {
		trees[k.name], _ = calibRate(t, tree, k.call, target)
	}
	var runs []calibRun
	for i, pad := range pads {
		jit.FirstOffset = pad
		run := calibMeasure(t, trees, target, i == 0)
		run.pad = pad
		runs = append(runs, run)
		t.Logf("pad %5d: %s", pad, run.line())
	}
	t.Logf("")
	t.Logf("over %d offsets (median [min, max]):", len(runs))
	t.Logf("  tree tier     %s ns an instruction", calibSpread(runs, func(r calibRun) float64 { return r.treeInstr }))
	t.Logf("  native code   %s ns an instruction", calibSpread(runs, func(r calibRun) float64 { return r.nativeInstr }))
	t.Logf("  saved         %s ns an instruction (tree - native)", calibSpread(runs, func(r calibRun) float64 { return r.saved }))
	for _, w := range calibWork {
		t.Logf("    %-10s  %s ns saved an instruction, %4.1f instructions an iteration", w, calibSpread(runs, func(r calibRun) float64 { return r.savedBy[w] }), runs[0].work[w])
	}
	t.Logf("  each kind: ns more than the tree tier; the instructions of work saved that cost as much; the model's (zzjit_ssa.go)")
	for _, kind := range append(calibKinds, calibKind{name: "gobuiltin", what: "a call of Go of a built-in: the median of those above (jitCostGoBuiltin)"}) {
		model := "-"
		if c, ok := calibModel[kind.name]; ok {
			model = strconv.Itoa(c)
		}
		t.Logf("  %-10s %s ns  %s  %4s  %s", kind.name, calibSpread(runs, func(r calibRun) float64 { return r.cost[kind.name] }),
			calibSpread(runs, func(r calibRun) float64 { return r.ratio[kind.name] }), model, kind.what)
	}
}

// calibModel is the profit check's cost of each kind the calibration
// measures, which it prints beside the measured ratio; a native call's is
// what it saves, a negative cost.
var calibModel = map[string]int{
	"gobuiltin": jitCostGoBuiltin, "host": jitCostHost, "host16": jitCostHost, "inlexit": jitCostInlineLeft, "native": -jitCostNativeCall,
	"callgo": jitCostGoScript, "callgonat": jitCostGoEntered, "unwind": jitCostCalleeLeft,
	"unwind2":  jitCostCalleeLeft + jitCostCalleeLevel,
	"gonested": jitCostNestedBuiltin, "goscript": jitCostNestedScript, "return": jitCostReturn, "deopt": jitCostDeopt,
}

// calibSetup is the kernels' script, which both runtimes run. Each kernel's
// loop does its work, or its work and one ending of a kind, an iteration;
// the loops of base's shape differ from it only in what they add to s. The
// callees whose names begin with nop do nothing but return, with a loop
// that runs no iteration, which keeps them from being inlined. tret and
// tdeo call out of the tree tier: delete, which the JIT does not compile,
// keeps them there.
var calibSetup = calibFixed + calibSweep()

const calibFixed = `var str='abcdefgh';
	function O(){this.a=1;this.b=2;this.c=3}
	O.prototype.get=function(){return this.a};
	O.prototype.gt=function(x){const t=typeof x;return this.a+t.length};
	O.prototype.g0=function(x){const t='number';return this.a+t.length};
	var o=new O, arr=new Array(64).fill(1);
	function wi(n){let s=0;for(let i=0;i<n;i++){s=(s*31+i)|0;s^=s>>>7;s=(s+i*i)|0;s^=s<<3}return s}
	function wf(n){let x=0.5,y=1.5;for(let i=0;i<n;i++){x=x*0.999+y*0.001;y=y-x*0.01+i*1e-9}return x+y}
	function wp(n){let s=0;for(let i=0;i<n;i++){o.a=o.b+i;o.b=(o.a^o.c)|0;s=(s+o.b)|0}return s}
	function wa(n){let s=0;for(let i=0;i<n;i++){const k=i&63;arr[k]=(arr[(k+1)&63]+i)|0;s=(s+arr[k])|0}return s}
	function wc(n){let s=0;for(let i=0;i<n;i++){s=(s+o.get()+i)|0}return s}
	function nop(i){let s=i;for(let j=0;j<0;j++)s++;return s}
	function nopgo(i){let s=i;for(let j=0;j<0;j++)s++;return s}
	function nopnat(i){let s=i;for(let j=0;j<0;j++)s++;return s}
	function noph(i,x){let s=i;for(let j=0;j<0;j++)s++;const t=typeof x;return (s+t.length)|0}
	function nopg(i){let s=i;for(let j=0;j<0;j++)s++;return (s+str.indexOf('c'))|0}
	function nopgo2(i){let s=i;for(let j=0;j<0;j++)s++;return s}
	function nops(i){let s=i;for(let j=0;j<0;j++)s++;return (s+nopgo2(i))|0}
	function mid(i,x){let s=i;for(let j=0;j<0;j++)s++;return (s+nop(i))|0}
	function midh(i,x){let s=i;for(let j=0;j<0;j++)s++;return (s+noph(i,x))|0}
	function P(i){this.v=i;for(let j=0;j<0;j++);}
	function Pgo(i){this.v=i;for(let j=0;j<0;j++);}
	function base(n,x){let s=0;for(let i=0;i<n;i++){const t='number';s=(s*31+i+t.length)|0}return s}
	function gocall(n,x){let s=0;for(let i=0;i<n;i++){const t='number';s=(s*31+i+t.length+str.indexOf('c'))|0}return s}
	function host(n,x){let s=0;for(let i=0;i<n;i++){const t=typeof x;s=(s*31+i+t.length)|0}return s}
	function cnop(n,x){let s=0;for(let i=0;i<n;i++){const t='number';s=(s*31+i+t.length+nop(i))|0}return s}
	function cgo(n,x){let s=0;for(let i=0;i<n;i++){const t='number';s=(s*31+i+t.length+nopgo(i))|0}return s}
	function cgonat(n,x){let s=0;for(let i=0;i<n;i++){const t='number';s=(s*31+i+t.length+nopnat(i))|0}return s}
	function chost(n,x){let s=0;for(let i=0;i<n;i++){const t='number';s=(s*31+i+t.length+noph(i,x))|0}return s}
	function cmid(n,x){let s=0;for(let i=0;i<n;i++){const t='number';s=(s*31+i+t.length+mid(i,x))|0}return s}
	function cmidh(n,x){let s=0;for(let i=0;i<n;i++){const t='number';s=(s*31+i+t.length+midh(i,x))|0}return s}
	function cngo(n,x){let s=0;for(let i=0;i<n;i++){const t='number';s=(s*31+i+t.length+nopg(i))|0}return s}
	function cnsc(n,x){let s=0;for(let i=0;i<n;i++){const t='number';s=(s*31+i+t.length+nops(i))|0}return s}
	function cnew(n,x){let s=0;for(let i=0;i<n;i++){const t='number';s=(s*31+i+t.length+new P(i).v)|0}return s}
	function cnewgo(n,x){let s=0;for(let i=0;i<n;i++){const t='number';s=(s*31+i+t.length+new Pgo(i).v)|0}return s}
	function inl(n,x){let s=0;for(let i=0;i<n;i++){const t='number';s=(s*31+i+t.length+o.gt(x))|0}return s}
	function inl0(n,x){let s=0;for(let i=0;i<n;i++){const t='number';s=(s*31+i+t.length+o.g0(x))|0}return s}
	function small(k){let s=0;for(let i=0;i<k;i++)s=(s+i)|0;return s}
	function small0(k){let s=0;for(let i=0;i<k;i++)s=(s+i)|0;return s}
	function tret(n){let s=0;for(let i=0;i<n;i++)s=(s+small(0))|0;if(n<0)delete o.q;return s}
	function tret0(n){let s=0;for(let i=0;i<n;i++)s=(s+small0(0))|0;if(n<0)delete o.q;return s}
	function dsmall(x,k){let s=0;for(let i=0;i<k;i++)s=(s+i)|0;return (s+x*2)|0}
	function tdeo(n,x){let s=0;for(let i=0;i<n;i++)s=(s+dsmall(x,0))|0;if(n<0)delete o.q;return s}
	`

// calibKernel is a loop of the calibration's: call, with N its iterations;
// fn the function whose counts say what an iteration did natively, and
// want, what it must have done an iteration, each within a tenth: its
// entries, exits to Go (a native callee's exit and a call Go makes for a
// native call among them), failed guards and calls of Go, and the calls
// native callers made of callee.
type calibKernel struct {
	name, call, fn, callee string
	want                   calibCounts
	// prep, after the warm-up, sets the JIT's state for what is measured;
	// cold marks a kernel not warmed up.
	prep func(r *Runtime, entry func(string) *jitEntry)
	cold bool
}

type calibCounts struct{ goCalls, hosts, guards, entries, calls float64 }

// calibNotNative has native callers call the function name through Go
// (notNative), and Go run it in the tree tier (slow) or enter its code.
func calibNotNative(name string, slow bool) func(*Runtime, func(string) *jitEntry) {
	return func(r *Runtime, entry func(string) *jitEntry) {
		e := entry(name)
		e.notNative, e.link.entry, e.link.body, e.entrySlow = true, 0, 0, slow
	}
}

var calibKernels = []calibKernel{
	{name: "wi", call: "wi(N)", fn: "wi"},
	{name: "wf", call: "wf(N)", fn: "wf"},
	{name: "wp", call: "wp(N)", fn: "wp"},
	{name: "wa", call: "wa(N)", fn: "wa"},
	{name: "wc", call: "wc(N)", fn: "wc"},
	{name: "base", call: "base(N,1)", fn: "base"},
	{name: "gocall", call: "gocall(N,1)", fn: "gocall", want: calibCounts{goCalls: 1}},
	{name: "host", call: "host(N,1)", fn: "host", want: calibCounts{hosts: 1, entries: 1}},
	{name: "cnop", call: "cnop(N,1)", fn: "cnop", callee: "nop", want: calibCounts{calls: 1}},
	{name: "cgo", call: "cgo(N,1)", fn: "cgo", want: calibCounts{goCalls: 1}, prep: calibNotNative("nopgo", true)},
	{name: "cgonat", call: "cgonat(N,1)", fn: "cgonat", want: calibCounts{goCalls: 1}, prep: calibNotNative("nopnat", false)},
	// A native callee leaving: its caller's stretch ends, a host exit of
	// the caller's, and Go finishes the callee (jitUnwindNative).
	{name: "chost", call: "chost(N,1)", fn: "chost", callee: "noph", want: calibCounts{hosts: 1, entries: 1, calls: 1}},
	{name: "cngo", call: "cngo(N,1)", fn: "nopg", callee: "nopg", want: calibCounts{goCalls: 1, calls: 1}},
	// A native callee's native callee leaving: Go makes two frames, and
	// resumes two levels.
	{name: "cmid", call: "cmid(N,1)", fn: "cmid", callee: "mid", want: calibCounts{calls: 1}},
	{name: "cmidh", call: "cmidh(N,1)", fn: "cmidh", callee: "midh", want: calibCounts{hosts: 1, entries: 1, calls: 1}},
	{name: "cnsc", call: "cnsc(N,1)", fn: "nops", callee: "nops", want: calibCounts{goCalls: 1, calls: 1}, prep: calibNotNative("nopgo2", true)},
	{name: "cnew", call: "cnew(N,1)", fn: "cnew", callee: "P", want: calibCounts{calls: 1}},
	// A construction of a function native callers do not call, which Go
	// makes (jitGoCall).
	{name: "cnewgo", call: "cnewgo(N,1)", fn: "cnewgo", want: calibCounts{goCalls: 1}, prep: calibNotNative("Pgo", true)},
	// An inlined callee's exit: Go makes its frame and finishes it.
	{name: "inl", call: "inl(N,1)", fn: "inl", want: calibCounts{hosts: 1, entries: 1}},
	{name: "inl0", call: "inl0(N,1)", fn: "inl0"},
	// Called from the tree tier: an entry from Go and a return, or, small0
	// given up, a call in the tree tier.
	{name: "tret", call: "tret(N)", fn: "small", want: calibCounts{entries: 1}},
	{name: "tret0", call: "tret0(N)", fn: "small0",
		prep: func(r *Runtime, entry func(string) *jitEntry) { entry("small0").entrySlow = true }},
	// dsmall, compiled for numbers and compiled again no more, deoptimizes
	// at x*2 at each call of tdeo(N,'q') -- not warmed up with it, which
	// would have its code compiled for Go to multiply -- and returns at each
	// of tdeo(N,1).
	{name: "tdeo1", call: "tdeo(N,1)", fn: "dsmall", want: calibCounts{entries: 1}},
	{name: "tdeoq", call: "tdeo(N,'q')", fn: "dsmall", want: calibCounts{guards: 1, entries: 1}, cold: true,
		prep: func(r *Runtime, entry func(string) *jitEntry) { entry("dsmall").reopts = jitReoptimizations }},
}

// calibSites is how many sites of its built-in a sweep kernel's loop has
// (calibSweep): one site alone flatters a call, the processor's
// predictors and caches all its.
const calibSites = 16

// calibBuiltins are the built-ins the sweep calls by a call of Go, each
// written for site k of a loop over i; the model's cost of a call of Go of
// a built-in is their median (jitCostGoBuiltin).
var calibBuiltins = []struct{ name, expr string }{
	{"indexOf", "w%[1]d.indexOf('x')"},
	{"substring", "w%[1]d.substring(1,3).length"},
	{"split", "w%[1]d.split(' ').length"},
	{"replace", "w%[1]d.replace(rg%[1]d,'x').length"},
	{"test", "(rt%[1]d.test(w%[1]d)?1:0)"},
	{"mapget", "bm.get((i+%[1]d)&63)"},
	{"sethas", "(bs.has('k'+((i+%[1]d)&63))?1:0)"},
	{"keys", "Object.keys(bo).length"},
	{"parseInt", "parseInt(w%[1]d,10)|0"},
	{"String", "String(i+%[1]d).length"},
	{"slice", "ba.slice(0,2).length"},
	{"join", "[i,%[1]d].join(',').length"},
}

// calibSweep is the sweep's script: for each built-in, a loop of
// calibSites sites calling it, and two of the same shape, one adding 0 at
// each site, one leaving for typeof at each (calibKernels' b_ kernels).
func calibSweep() string {
	var b strings.Builder
	b.WriteString("var bm=new Map(), bs=new Set(), bo={a:1,b:2}, ba=[1,2,3,4];\n")
	b.WriteString("for(let i=0;i<64;i++){bm.set(i,i);bs.add('k'+i)}\n")
	for k := range calibSites {
		fmt.Fprintf(&b, "var w%d='w%d xyz', rg%d=/x%d?/g, rt%d=/q%d|y/;\n", k, k, k, k, k, k)
	}
	loop := func(name, expr string) {
		fmt.Fprintf(&b, "function b_%s(n,x){let s=0;for(let i=0;i<n;i++){", name)
		for k := range calibSites {
			fmt.Fprintf(&b, "s=(s+%s)|0;", strings.ReplaceAll(expr, "%[1]d", strconv.Itoa(k)))
		}
		b.WriteString("}return s}\n")
	}
	loop("none", "0")
	loop("typeof", "(typeof x).length")
	for _, c := range calibBuiltins {
		loop(c.name, c.expr)
	}
	return b.String()
}

func init() {
	calibKernels = append(calibKernels,
		calibKernel{name: "b_none", call: "b_none(N,1)", fn: "b_none"},
		calibKernel{name: "b_typeof", call: "b_typeof(N,1)", fn: "b_typeof", want: calibCounts{hosts: calibSites, entries: calibSites}})
	calibKinds = append(calibKinds, calibKind{name: "host16", with: "b_typeof", without: "b_none", per: calibSites,
		what: "an exit for one operation, and the entry after it, at each of 16 sites"})
	for _, c := range calibBuiltins {
		k := "b_" + c.name
		calibKernels = append(calibKernels, calibKernel{name: k, call: k + "(N,1)", fn: k, want: calibCounts{goCalls: calibSites}})
		calibKinds = append(calibKinds, calibKind{name: c.name, with: k, without: "b_none", per: calibSites, builtin: true,
			what: "a call of Go of " + c.name + ", at each of 16 sites"})
	}
}

// calibWork are the kernels the work an instruction saves is measured on.
var calibWork = []string{"wi", "wf", "wp", "wa", "wc"}

// calibKind is a way a stretch ends, or a call that costs, measured as the
// difference of a kernel with it and one without, less the tree tier's
// difference, per of them an iteration (1 if 0); plus is a kind whose cost
// this one's difference is on top of, and builtin marks a built-in's call,
// whose median is the cost of one (gobuiltin).
type calibKind struct {
	name, with, without, what string
	plus                      string
	per                       float64
	builtin                   bool
}

// calibKinds are the ways a stretch ends, and the calls that cost.
var calibKinds = []calibKind{
	{name: "gocall1", with: "gocall", without: "base", what: "a call of Go of String.prototype.indexOf, the loop's one site"},
	{name: "host", with: "host", without: "base", what: "an exit to Go for an operation and the entry after it (typeof)"},
	{name: "inlexit", with: "inl", without: "inl0", what: "an exit inside an inlined callee, Go making its frame (jitUnwindNative)"},
	{name: "native", with: "cnop", without: "base", what: "a native call of a native callee that does nothing"},
	{name: "callgo", with: "cgo", without: "base", what: "a call of Go of a script function, run in the tree tier"},
	{name: "callgonat", with: "cgonat", without: "base", what: "a call of Go of a script function, its native code entered"},
	{name: "unwind", with: "chost", without: "cnop", what: "a native callee's exit, Go finishing it (jitUnwindNative), its caller resumed"},
	{name: "unwind2", with: "cmidh", without: "cmid", what: "a native callee's exit two native calls deep"},
	{name: "gonested", with: "cngo", without: "cnop", what: "a native callee's call of Go of a built-in (jitGoCallNested)"},
	{name: "goscript", with: "cnsc", without: "cnop", what: "a native callee's call of Go of a script function"},
	{name: "construct", with: "cnew", without: "base", what: "a native construction of a native callee"},
	{name: "newgo", with: "cnewgo", without: "base", what: "a construction of a script function by a call of Go, run in the tree tier"},
	{name: "return", with: "tret", without: "tret0", what: "an entry from the tree tier's call and the return"},
	{name: "deopt", with: "tdeoq", without: "tdeo1", plus: "return", what: "an entry from the tree tier's call and a failed guard, the interpreter finishing"},
}

// calibRun is what one placement of the JIT's code measured.
type calibRun struct {
	pad                           int
	treeInstr, nativeInstr, saved float64
	savedBy, work                 map[string]float64
	cost, ratio                   map[string]float64
}

func (r calibRun) line() string {
	var b strings.Builder
	fmt.Fprintf(&b, "saved %.2f ns/instr (tree %.2f, native %.2f)", r.saved, r.treeInstr, r.nativeInstr)
	for _, k := range calibKinds {
		if !k.builtin {
			fmt.Fprintf(&b, "  %s %.0f ns=%.0f", k.name, r.cost[k.name], r.ratio[k.name])
		}
	}
	fmt.Fprintf(&b, "  gobuiltin %.0f ns=%.0f", r.cost["gobuiltin"], r.ratio["gobuiltin"])
	return b.String()
}

// calibMeasure times every kernel in a new runtime with the JIT, and works
// out the costs against trees, the tree tier's times. check has it test
// that each kernel did what it is to measure.
func calibMeasure(t *testing.T, trees map[string]float64, target time.Duration, check bool) calibRun {
	r := New(Config{JIT: true})
	defer func() { r.Close(); r.ReleaseClosed() }()
	if _, err := r.Run(compileForTest(t, calibSetup)); err != nil {
		t.Fatal(err)
	}
	entry := func(name string) *jitEntry {
		cl := r.global.getOwn(r.atoms.intern(name)).value().Object().fn().closure
		return r.jit.cache[weak.Make(cl.fn)]
	}
	// Warmed up, every kernel compiled and settled, before any is changed
	// for what it measures (prep): a deoptimization's kernel is warmed with
	// numbers alone.
	for _, k := range calibKernels {
		if !k.cold {
			for range 3 {
				calibRun1(t, r, k.call, 20000)
			}
		}
	}
	for _, k := range calibKernels {
		if k.prep != nil {
			k.prep(r, entry)
		}
	}
	nat := map[string]float64{}
	work := map[string]float64{}
	for _, k := range calibKernels {
		e := entry(k.fn)
		if e == nil || e.ssa == nil {
			t.Fatalf("%s: %s was not compiled: %+v", k.name, k.fn, e)
		}
		// What an iteration does, counted over one run, before the
		// timing's.
		const n = 20000
		var callee *jitEntry
		calls := uint64(0)
		if k.callee != "" {
			if callee = entry(k.callee); callee == nil || callee.link == nil {
				t.Fatalf("%s: %s has no link: %+v", k.name, k.callee, callee)
			}
			calls = callee.link.in
		}
		before, unwound := e.ssaStats, r.jit.unwound
		calibRun1(t, r, k.call, n)
		after, unwinds := e.ssaStats, r.jit.unwound-unwound
		if callee != nil {
			calls = callee.link.in - calls
		}
		per := func(a, b uint64) float64 { return float64(a-b) / n }
		work[k.name] = per(after.work, before.work)
		nat[k.name], _ = calibRate(t, r, k.call, target)
		if check {
			got := calibCounts{goCalls: per(after.goCalls, before.goCalls), hosts: per(after.hosts, before.hosts),
				guards: per(after.guards, before.guards), entries: per(after.entries, before.entries), calls: float64(calls) / n}
			t.Logf("%-7s tree %7.1f ns, native %7.1f ns an iteration; %s an iteration: %.2f entries, %.2f exits, %.2f guards, %.2f calls of Go, %.2f unwinds, %.1f work; %.2f native calls",
				k.name, trees[k.name], nat[k.name], k.fn, got.entries, got.hosts, got.guards, got.goCalls, float64(unwinds)/n, work[k.name], got.calls)
			near := func(got, want float64) bool { return got >= want*0.9-0.02 && got <= want*1.1+0.02 }
			if !near(got.goCalls, k.want.goCalls) || !near(got.hosts, k.want.hosts) || !near(got.guards, k.want.guards) ||
				k.want.entries != 0 && !near(got.entries, k.want.entries) || !near(got.calls, k.want.calls) {
				t.Errorf("%s: %s did %+v an iteration, not %+v", k.name, k.fn, got, k.want)
			}
		}
		if e.entrySlow != (k.name == "tret0") || e.ssaStats.entries < after.entries {
			t.Errorf("%s: %s judged or dropped meanwhile: %+v", k.name, k.fn, e)
		}
	}
	run := calibRun{savedBy: map[string]float64{}, work: map[string]float64{}, cost: map[string]float64{}, ratio: map[string]float64{}}
	var tw, nw, ww float64
	var saved []float64
	for _, w := range calibWork {
		run.work[w] = work[w]
		run.savedBy[w] = (trees[w] - nat[w]) / work[w]
		saved = append(saved, run.savedBy[w])
		tw += trees[w]
		nw += nat[w]
		ww += work[w]
	}
	// The kernels' median: their work is not alike, and none is typical.
	run.saved = calibMedian(saved)
	run.treeInstr, run.nativeInstr = tw/ww, nw/ww
	var builtins []float64
	for _, k := range calibKinds {
		c := (nat[k.with] - nat[k.without]) - (trees[k.with] - trees[k.without])
		if k.per != 0 {
			c /= k.per
		}
		if k.plus != "" {
			c += run.cost[k.plus]
		}
		run.cost[k.name] = c
		run.ratio[k.name] = c / run.saved
		if k.builtin {
			builtins = append(builtins, c)
		}
	}
	run.cost["gobuiltin"] = calibMedian(builtins)
	run.ratio["gobuiltin"] = run.cost["gobuiltin"] / run.saved
	return run
}

// calibRun1 runs call once with N iterations in r.
func calibRun1(t *testing.T, r *Runtime, call string, n int) time.Duration {
	fn := compileForTest(t, fmt.Sprintf("(function(){var N=%d;return %s})()", n, call))
	t0 := time.Now()
	if _, err := r.Run(fn); err != nil {
		t.Fatalf("%s: %v", call, err)
	}
	return time.Since(t0)
}

// calibRate is what an iteration of call takes in r: the fastest of five
// runs, the iterations grown until that takes target -- Windows' clock
// moves in steps of half a millisecond.
func calibRate(t *testing.T, r *Runtime, call string, target time.Duration) (float64, int) {
	n := 10000
	for {
		best := time.Duration(1<<63 - 1)
		for range 5 {
			best = min(best, calibRun1(t, r, call, n))
		}
		if best >= target {
			return float64(best.Nanoseconds()) / float64(n), n
		}
		n = int(float64(n) * min(16, max(1.5, 1.2*float64(target)/float64(max(best, time.Microsecond)))))
	}
}

func calibMedian(xs []float64) float64 {
	s := slices.Clone(xs)
	slices.Sort(s)
	if len(s)%2 == 1 {
		return s[len(s)/2]
	}
	return (s[len(s)/2-1] + s[len(s)/2]) / 2
}

// calibSpread is the median of f over runs, with its least and greatest.
func calibSpread(runs []calibRun, f func(calibRun) float64) string {
	var xs []float64
	for _, r := range runs {
		xs = append(xs, f(r))
	}
	return fmt.Sprintf("%7.2f [%7.2f, %7.2f]", calibMedian(xs), slices.Min(xs), slices.Max(xs))
}
