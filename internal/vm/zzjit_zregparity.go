//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"cmp"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/jit"
	jitcompile "github.com/go-quickjs/go-quickjs/internal/jit/compile"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
	"github.com/go-quickjs/go-quickjs/internal/jit/ssa"
)

// The feedback-parity check of the register form's JIT against the stack
// form's (docs/register-interpreter.md, Stage 3): a program run in two
// runtimes, one on the stack form's tiers and its JIT, one on the register
// tier and the JIT that compiles register code, warmed alike; then, for
// every function both compiled, what each form's feedback answers --
// jitFeedback's by stack pc, jitRegFeedback's by register pc -- is
// compared site by site, a register instruction matched with the stack
// instruction its source position is (regCode's src), and what each
// form's build decides from it, function by function. Only tests run it
// (TestJITFeedbackParity, internal/regdiff).

// jitParityLog, set by RecordJITCompiles, has every runtime note the
// closures it compiles, by form.
var jitParityLog struct {
	sync.Mutex
	on   atomic.Bool
	seen map[*Runtime]map[*bytecode.Function]*closure
}

// RecordJITCompiles has every runtime note the closures it compiles, of
// either form, for CompareFeedback, or stop and forget them.
func RecordJITCompiles(on bool) {
	jitParityLog.Lock()
	defer jitParityLog.Unlock()
	jitParityLog.on.Store(on)
	jitParityLog.seen = nil
}

// jitParityNote notes that r compiles cl's function (RecordJITCompiles).
func (r *Runtime) jitParityNote(cl *closure) {
	if cl == nil {
		return
	}
	jitParityLog.Lock()
	defer jitParityLog.Unlock()
	if !jitParityLog.on.Load() {
		return
	}
	if jitParityLog.seen == nil {
		jitParityLog.seen = map[*Runtime]map[*bytecode.Function]*closure{}
	}
	m := jitParityLog.seen[r]
	if m == nil {
		m = map[*bytecode.Function]*closure{}
		jitParityLog.seen[r] = m
	}
	m[cl.fn] = cl
}

// FeedbackParity is what CompareFeedback found.
type FeedbackParity struct {
	// Functions is how many functions both runtimes compiled and were
	// compared; Answers how many non-empty answers of the stack form's
	// feedback there were, Matched how many the register form's answered
	// alike at a register instruction of the same source position.
	Functions, Answers, Matched int
	// Gaps counts the answers not matched, by method and why: the
	// register form's feedback has no such method ("unanswered"), no
	// register instruction is at the stack instruction's position ("no
	// instruction"), or its answers there differ ("differs").
	Gaps map[string]int
	// Decisions counts the functions whose builds decided differently, by
	// what: refused by one form, inlined, host operations left to Go,
	// loops' entries escape analysis dropped.
	Decisions map[string]int
	// Examples are a few of each gap and decision, as text.
	Examples map[string][]string
}

// String is the report, gaps and decisions most first.
func (p FeedbackParity) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%d functions compiled by both forms; %d answers of the stack form's feedback, %d answered alike by the register form's\n",
		p.Functions, p.Answers, p.Matched)
	for _, part := range []struct {
		title string
		m     map[string]int
	}{{"gaps", p.Gaps}, {"decisions that differ", p.Decisions}} {
		keys := slices.SortedFunc(func(yield func(string) bool) {
			for k := range part.m {
				if !yield(k) {
					return
				}
			}
		}, func(a, b string) int {
			if c := cmp.Compare(part.m[b], part.m[a]); c != 0 {
				return c
			}
			return strings.Compare(a, b)
		})
		fmt.Fprintf(&b, "%s:\n", part.title)
		for _, k := range keys {
			fmt.Fprintf(&b, "  %7d %s\n", part.m[k], k)
			for _, ex := range p.Examples[k] {
				fmt.Fprintf(&b, "            %s\n", ex)
			}
		}
	}
	return b.String()
}

// CompareFeedback compares the feedback of stack's JIT, a runtime that ran
// the stack form, with reg's, a runtime that ran the same program,
// compiled again, on the register tier, for every function both compiled
// since RecordJITCompiles: a function is the same in both where its name
// and its source text are.
func CompareFeedback(stack, reg *Runtime) FeedbackParity {
	jitParityLog.Lock()
	ss, rs := jitParityLog.seen[stack], jitParityLog.seen[reg]
	jitParityLog.Unlock()
	res := FeedbackParity{Gaps: map[string]int{}, Decisions: map[string]int{}, Examples: map[string][]string{}}
	note := func(m map[string]int, k, ex string) {
		m[k]++
		if len(res.Examples[k]) < 3 {
			res.Examples[k] = append(res.Examples[k], ex)
		}
	}
	key := func(fn *bytecode.Function) string { return fn.Name + "\x00" + fn.Text }
	byKey := map[string]*closure{}
	for fn, cl := range rs {
		byKey[key(fn)] = cl
	}
	fns := slices.SortedFunc(func(yield func(*bytecode.Function) bool) {
		for fn := range ss {
			if !yield(fn) {
				return
			}
		}
	}, func(a, b *bytecode.Function) int { return strings.Compare(key(a), key(b)) })
	for _, fn := range fns {
		scl, rcl := ss[fn], byKey[key(fn)]
		if rcl == nil || rcl.reg == nil {
			continue
		}
		se := stack.jit.cache[stack.jitKeyOf(fn)]
		if se == nil || se.ssaProg == nil {
			continue
		}
		sp := se.ssaProg
		rp, err := jitcompile.LowerReg(rcl.reg.rc, rcl.fn)
		if err != nil {
			continue
		}
		res.Functions++
		sfb := &jitFeedback{r: stack, fn: fn, cl: scl, e: se, prog: sp}
		rfb := reg.jitRegFeedbackFor(rcl, rp)
		if rfb == nil {
			continue
		}
		name := fn.Name
		if name == "" {
			name = "(anonymous)"
		}
		// The register program's instructions by the stack pc their
		// register instruction's position is.
		byStack := map[int][]int{}
		for j := range rp.Code {
			rpc := int(rp.Origin[j])
			if rpc < len(rcl.reg.src) {
				s := int(rcl.reg.src[rpc])
				byStack[s] = append(byStack[s], j)
			}
		}
		sv, rv := reflect.ValueOf(sfb), reflect.ValueOf(rfb)
		st := sv.Type()
		stext := func(m reflect.Value, args []reflect.Value) (string, bool) { return jitParityCall(stack, m, args) }
		rtext := func(m reflect.Value, args []reflect.Value) (string, bool) { return jitParityCall(reg, m, args) }
		for m := range st.NumMethod() {
			meth := st.Method(m)
			mt := meth.Type
			pcArg := mt.NumIn() == 2 && mt.In(1).Kind() == reflect.Int
			if !pcArg && mt.NumIn() != 1 || mt.NumOut() == 0 {
				continue
			}
			rm := rv.MethodByName(meth.Name)
			if !pcArg {
				a, ok := stext(sv.Method(m), nil)
				if !ok || a == "" {
					continue
				}
				res.Answers++
				switch b, _ := rtext(rm, nil); {
				case !rm.IsValid():
					note(res.Gaps, meth.Name+": unanswered", fmt.Sprintf("%s: %s", name, a))
				case b != a:
					note(res.Gaps, meth.Name+": differs", fmt.Sprintf("%s: stack %s, register %s", name, a, b))
				default:
					res.Matched++
				}
				continue
			}
			asked, known := jitParityMethods[meth.Name]
			for s := range sp.Code {
				if known && !jitParityAsked(asked, sp.Code[s].Op) {
					// The build never asks the method there (jitSnapOps):
					// its answer, whatever it is, is no feedback.
					continue
				}
				a, ok := stext(sv.Method(m), []reflect.Value{reflect.ValueOf(s)})
				if !ok || a == "" {
					continue
				}
				res.Answers++
				where := fmt.Sprintf("%s pc %d (%v)", name, s, jitParityStackOp(fn, s))
				if !rm.IsValid() {
					note(res.Gaps, meth.Name+": unanswered", fmt.Sprintf("%s: %s", where, a))
					continue
				}
				// The register instructions of the same position and of the
				// same operation; a register instruction made of several
				// stack ones -- a constant folded into its user, a method's
				// read and its call -- has the position of one of them: the
				// nearest of the same operation, a few instructions on; and
				// failing those, those of the position, of other operations.
				var js []int
				for d := 0; d <= 8; d++ {
					for _, q := range []int{s + d, s - d} {
						if d == 0 && q != s+d {
							continue
						}
						for _, j := range byStack[q] {
							if rp.Code[j].Op == sp.Code[s].Op && !slices.Contains(js, j) {
								js = append(js, j)
							}
						}
					}
				}
				if len(js) == 0 {
					js = byStack[s]
				}
				if len(js) == 0 {
					note(res.Gaps, meth.Name+": no instruction", fmt.Sprintf("%s: %s", where, a))
					continue
				}
				matched := false
				var got string
				for _, j := range js {
					arg := j
					if meth.Name == "Resumed" || meth.Name == "EntryGeneric" {
						// Asked by a state's PC, register code's a register
						// pc.
						arg = int(rp.Maps[j].PC)
					}
					b, _ := rtext(rm, []reflect.Value{reflect.ValueOf(arg)})
					if b == a {
						matched = true
						break
					}
					if got == "" || b != "" {
						got = fmt.Sprintf("%s at %d (ir op %d, %v)", b, j, rp.Code[j].Op, rcl.reg.rc.Code[rp.Origin[j]].Op())
					}
				}
				if matched {
					res.Matched++
				} else {
					note(res.Gaps, meth.Name+": differs", fmt.Sprintf("%s: stack %s; register %s", where, a, got))
				}
			}
		}
		jitParityDecisions(&res, name, sp, sfb, rp, rfb, note)
	}
	return res
}

// jitParityStackOp is the stack instruction at pc of fn's code.
func jitParityStackOp(fn *bytecode.Function, pc int) any {
	if pc < len(fn.Code) {
		return fn.Code[pc].Op
	}
	return "past the code"
}

// jitParityMethods are the methods of a PC by name, as the snapshot numbers
// them (jitSnapNames).
var jitParityMethods = func() map[string]jitSnapMethod {
	m := map[string]jitSnapMethod{}
	for i, name := range jitSnapNames {
		if jitSnapAtPC[i] {
			m[name] = jitSnapMethod(i)
		}
	}
	return m
}()

// jitParityAsked reports whether the build asks method m at an instruction
// of op (jitSnapAsks): the sparse methods, Generic, EntryGeneric and
// Resumed, it asks at every one.
func jitParityAsked(m jitSnapMethod, op ir.Op) bool {
	return jitSnapSparse&(1<<m) != 0 || int(op) < len(jitSnapAsks) && jitSnapAsks[op]&(1<<m) != 0
}

// jitParityCall calls method m with args and returns its results as text,
// "" where they are all zero, a property's or a global's key by its name in
// r; ok is false where it panicked.
func jitParityCall(r *Runtime, m reflect.Value, args []reflect.Value) (s string, ok bool) {
	if !m.IsValid() {
		return "", false
	}
	defer func() {
		if recover() != nil {
			s, ok = "", false
		}
	}()
	out := m.Call(args)
	if sites, ok := out[0].Interface().([]ssa.CallSite); ok && len(out) == 1 {
		// Native calls: what is called, and how, not the callee's frame,
		// which each form lays out its own way (ssa.CallSite's Window).
		out[0] = reflect.ValueOf(jitParitySites(sites))
	}
	if last := out[len(out)-1]; last.Kind() == reflect.Bool && !last.Bool() && len(out) > 1 {
		// An answer with its ok false is none.
		return "", true
	}
	zero := true
	parts := make([]string, len(out))
	for i, v := range out {
		if !v.IsZero() {
			zero = false
		}
		parts[i] = jitParityText(r, v, 0)
	}
	if zero {
		return "", true
	}
	return strings.Join(parts, " "), true
}

// jitParitySites are sites, the native calls a call makes, with the fields
// of the callee's frame cleared -- its parameters, locals, operand slots,
// receiver's slot, captured bindings' cells, register window, where its
// function object and its call's arguments go, its tail calls' frame,
// whether its start makes its locals -- each form's own, and a receiver's
// coercion only where the callee reads it (the builder clears it
// otherwise).
func jitParitySites(sites []ssa.CallSite) []ssa.CallSite {
	out := make([]ssa.CallSite, len(sites))
	for i, s := range sites {
		s.Coerce = s.Coerce && (s.ThisSlot >= 0 || s.Window != 0)
		s.Params, s.LocalCount, s.MaxStack, s.ThisSlot, s.StartOnly, s.Upvalues, s.Window = 0, 0, 0, 0, false, 0, 0
		s.Function, s.FrameArgs, s.Tail, s.TailFrame = false, false, false, false
		out[i] = s
	}
	return out
}

// jitParityText is v as text that two runtimes' answers share where they
// mean the same: an address is only whether it is one, a pointer's target
// is shown, to a depth, and a key, an atom of r's, is its name.
func jitParityText(r *Runtime, v reflect.Value, depth int) string {
	switch v.Kind() {
	case reflect.Uintptr, reflect.UnsafePointer:
		if v.IsZero() {
			return "0"
		}
		return "@"
	case reflect.Pointer, reflect.Interface:
		if v.IsNil() {
			return "nil"
		}
		if depth > 2 {
			return "&"
		}
		return "&" + jitParityText(r, v.Elem(), depth+1)
	case reflect.Struct:
		var b strings.Builder
		b.WriteString("{")
		t := v.Type()
		for i := range v.NumField() {
			if v.Field(i).IsZero() {
				continue
			}
			if f := v.Field(i); t.Field(i).Name == "Key" && f.Kind() == reflect.Uint32 {
				fmt.Fprintf(&b, "Key:%q ", r.AtomName(Atom(f.Uint())))
				continue
			}
			fmt.Fprintf(&b, "%s:%s ", t.Field(i).Name, jitParityText(r, v.Field(i), depth+1))
		}
		b.WriteString("}")
		return b.String()
	case reflect.Slice, reflect.Array:
		if v.Kind() == reflect.Slice && v.IsNil() {
			return "[]"
		}
		n := v.Len()
		var b strings.Builder
		fmt.Fprintf(&b, "[%d:", n)
		for i := range min(n, 8) {
			b.WriteString(jitParityText(r, v.Index(i), depth+1))
			b.WriteString(" ")
		}
		b.WriteString("]")
		return b.String()
	case reflect.Map, reflect.Func, reflect.Chan:
		if v.IsNil() {
			return "nil"
		}
		return "#"
	}
	return fmt.Sprint(v.Interface())
}

// jitParityDecisions builds both forms' programs with their feedback and
// notes where what the builds decided differs.
func jitParityDecisions(res *FeedbackParity, name string, sp *ir.Program, sfb ssa.Feedback, rp *ir.Program, rfb ssa.Feedback, note func(map[string]int, string, string)) {
	type built struct {
		refused                   bool
		why                       string
		inlined, hosts, unentered int
	}
	build := func(p *ir.Program, fb ssa.Feedback) (b built) {
		defer func() {
			if x := recover(); x != nil {
				b.refused, b.why = true, fmt.Sprint("panic: ", x)
			}
		}()
		w := jit.GetWorkspace()
		defer jit.PutWorkspace(w)
		f, err := ssa.BuildIn(&w.SSA, p, fb)
		if err != nil {
			return built{refused: true, why: err.Error()}
		}
		return built{inlined: len(f.Inlined), hosts: len(f.HostsLeft), unentered: len(f.Unentered)}
	}
	s, r := build(sp, sfb), build(rp, rfb)
	pair := func(what string, a, b int) {
		if a != b {
			note(res.Decisions, what, fmt.Sprintf("%s: stack %d, register %d", name, a, b))
		}
	}
	switch {
	case s.refused != r.refused:
		note(res.Decisions, "refused", fmt.Sprintf("%s: stack %v, register %v (%s%s)", name, s.refused, r.refused, s.why, r.why))
		return
	case s.refused:
		return
	}
	pair("inlined", s.inlined, r.inlined)
	pair("host operations left to Go", s.hosts, r.hosts)
	pair("loops' entries dropped", s.unentered, r.unentered)
}
