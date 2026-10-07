//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"context"
	"errors"
	"fmt"
	"math"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"
	"weak"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/compiler"
	"github.com/go-quickjs/go-quickjs/internal/parser"
)

const jitSumSource = `function sum(n) { let s=0; for(let i=0;i<n;i++) s+=i; return s } sum(10000)`

func TestJITDenseArraysAndCallbacks(t *testing.T) {
	var value Value
	if unsafe.Sizeof(value) != 16 || unsafe.Offsetof(value.num) != 0 || unsafe.Offsetof(value.ref) != 8 {
		t.Fatal("VM dense cell ABI changed")
	}
	for _, tc := range []struct {
		name, source string
		want         float64
		guard        bool
	}{
		{"captured dimensions", `let n=3; function f(a,b){for(let i=0;i<n;i++)a[i]+=2*b[i];return a[0]+a[1]+a[2]} f([1,2,3],[4,5,6])`, 36, false},
		{"alias", `function f(a,b){for(let i=0;i<a.length;i++)a[i]+=b[i];return a[0]+a[1]} let a=[2,3];f(a,a)`, 10, false},
		{"prefix and postfix", `function f(a){let i=0,s=a[i++]+a[++i]+a[i--]+a[--i];return s*10+i}f([2,3,5])`, 140, false},
		{"assignment result", `function f(a){let x=a[1]=7;return x+a[1]}f([2,3])`, 14, false},
		{"callback snapshots", `let n=4;function step(a){n=2;a.push(9);a[0]=8}function f(a,cb){let s=0;for(let i=0;i<n;i++){s+=a[i];cb(a)}return s*10+a.length}f([1,2,3,4],step)`, 36, false},
		{"method receiver", `function f(a){return Math.sqrt(a[0])+a.length}f([9,2])`, 5, false},
		{"negative zero key", `function f(a){return a[-0]}f([7])`, 7, false},
		{"NaN write", `function f(a){a[0]=0/0;return a[0]}f([1])`, math.NaN(), false},
		{"numeric int32", `function f(a){return (a[0]|0)+(a[1]|0)}f([2.9,-1.9])`, 1, false},
		{"int32 wrap fallback", `function f(a){return a[0]|0}f([4294967297])`, 1, true},
		{"hole inherited read", `Array.prototype[0]=7;function f(a){return a[0]}f([,])`, 7, true},
		{"accessor", `function f(a){let i=0;return a[i++]+i*10}let a=[1];Object.defineProperty(a,'0',{get(){return 7}});f(a)`, 17, true},
		{"negative key", `function f(a){return a[-1]}let a=[1];a[-1]=9;f(a)`, 9, true},
		{"fractional key", `function f(a){return a[.5]}let a=[1];a[.5]=8;f(a)`, 8, true},
		{"proxy", `function f(a){return a[0]}f(new Proxy([1],{get(){return 8}}))`, 8, true},
		{"inherited setter", `let hits=0;Object.defineProperty(Array.prototype,'0',{set(v){hits+=v}});function f(a){a[0]=3;return hits}f([,])`, 3, true},
		{"frozen", `function f(a){a[0]=3;return a[0]}f(Object.freeze([7]))`, 7, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, trees := range []bool{false, true} {
				r := jitRuntimeForTest(t, Config{JIT: true})
				previous := treeTier.Swap(trees)
				t.Cleanup(func() { treeTier.Store(previous) })
				v, err := r.Run(compileForTest(t, tc.source))
				if err != nil || !v.IsNumber() || v.Number() != tc.want && !(math.IsNaN(v.Number()) && math.IsNaN(tc.want)) {
					t.Fatalf("result %v error %v, want %v", v, err, tc.want)
				}
				if r.jit == nil || r.jit.entries == 0 || (r.jit.guards > 0) != tc.guard || r.jit.rootCount != 0 {
					t.Fatalf("native state %+v", r.jit)
				}
				for _, view := range r.jit.arrays {
					if view.Data != nil {
						t.Fatal("retained borrowed array")
					}
				}
			}
		})
	}
}

func TestJITHostReleaseAndGC(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	calls := 0
	r.global.setOwnRaw(r.atoms.intern("host"), r.NewFunction("host", 1,
		func(rt *Runtime, _ Value, args []Value) (Value, error) {
			calls++
			if rt.jit == nil || rt.jit.rootCount != 0 {
				t.Fatal("callback retained native roots")
			}
			for _, view := range rt.jit.arrays {
				if view.Data != nil {
					t.Fatal("callback retained borrowed storage")
				}
			}
			rt.releaseJIT()
			runtime.GC()
			return args[0], nil
		}), propDefault)
	v, err := r.Run(compileForTest(t, `function f(a){let s=0;for(let i=0;i<a.length;i++){s+=host(a[i]);a[i]=s}return s}f([1,2,3])`))
	if err != nil || v.Number() != 6 || calls != 3 || r.jit == nil || r.jit.rootCount != 0 {
		t.Fatalf("result %v err %v calls %d", v, err, calls)
	}
}

func TestJITHostErrors(t *testing.T) {
	for _, source := range []string{
		`function f(a,cb){for(let i=0;i<a.length;i++)a[i]=cb();return 1} f([1],()=>{throw new RangeError('callback')})`,
		`function f(a){return missing+a[0]} f([1])`,
		`function f(a){return Math.noSuchMethod(a[0])} f([1])`,
		`let a=[1]; Object.defineProperty(globalThis,'mathlike',{get(){throw new SyntaxError('getter')}});function f(a){return mathlike+a[0]}f(a)`,
	} {
		var want string
		for _, enabled := range []bool{false, true} {
			r := jitRuntimeForTest(t, Config{JIT: enabled})
			_, err := r.Run(compileForTest(t, source))
			if err == nil {
				t.Fatal("expected error")
			}
			if !enabled {
				want = err.Error()
			} else if err.Error() != want {
				t.Fatalf("native error %q; want %q", err, want)
			}
			if enabled && (r.jit == nil || r.jit.entries == 0 || r.jit.rootCount != 0) {
				t.Fatal("did not exercise native host exit")
			}
		}
	}
}

func jitRuntimeForTest(t *testing.T, cfg Config) *Runtime {
	t.Helper()
	r := New(cfg)
	// Boundary tests force promotion so guard and budget assertions cannot
	// pass by running only the existing tiers.
	r.jitCallThreshold = 1
	t.Cleanup(func() { r.Close(); r.ReleaseClosed() })
	return r
}

func TestJITRuntimeOptIn(t *testing.T) {
	p := compileForTest(t, jitSumSource)
	for _, enabled := range []bool{false, true} {
		r := jitRuntimeForTest(t, Config{JIT: enabled})
		if r.jit != nil {
			t.Fatal("runtime creation allocated JIT state")
		}
		v, err := r.Run(p)
		if err != nil || v.Number() != 49995000 {
			t.Fatalf("sum = %v, %v", v, err)
		}
		if enabled {
			if r.jit == nil || r.jit.entries == 0 || r.jit.budgets == 0 || r.jitCodeBytes() == 0 {
				t.Fatal("opt-in did not execute bounded native code")
			}
			r.Close()
			r.ReleaseClosed()
			if r.jit != nil {
				t.Fatal("close retained native code")
			}
		} else if r.jit != nil {
			t.Fatal("opt-out allocated JIT state")
		}
	}
}

func TestJITCachedTreeCall(t *testing.T) {
	r := jitRuntimeForTest(t, Config{})
	if _, err := r.Run(compileForTest(t, `function sum(n) { var s=0; for(var i=0;i<n;i++) s+=i; return s } sum(10000); sum(10); sum(20)`)); err != nil {
		t.Fatal(err)
	}
	o := r.global.getOwn(r.atoms.intern("sum"))
	if o.value.Object().fn().treeCall == nil {
		t.Fatal("test did not establish cached tree call")
	}
	// Enable on the same runtime after repeated calls have cached their tree.
	r.initJIT(true)
	v, err := r.Run(compileForTest(t, `for (var j=0;j<8;j++) sum(10000); sum(10000)`))
	if err != nil || v.Number() != 49995000 {
		t.Fatalf("cached call = %v, %v", v, err)
	}
	if r.jit == nil || r.jit.entries == 0 {
		t.Fatal("cached tree bypassed native selection")
	}
}

func TestJITRuntimeGuardReentry(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	v, err := r.Run(compileForTest(t, `
		function sum(n) { let s=0; for(let i=0;i<n;i++) s+=i; return s }
		function f(a) { let s=0; for(let i=0;i<3;i++) { s++; s+=a } return s }
		let count=0;
		let result=f({valueOf() { count++; return sum(3) }});
		result*10+count`))
	if err != nil || v.Number() != 123 {
		t.Fatalf("guard reentry = %v, %v", v, err)
	}
	if r.jit == nil || r.jit.guards != 1 || r.jit.entries < 4 || r.jit.rootCount != 0 {
		t.Fatal("guard or nested native call did not execute, or roots survived")
	}
}

func TestJITRuntimeCancellation(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	p := compileForTest(t, `function forever() { let i=0; for(;;) i++ } forever()`)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	r.SetContext(ctx)
	_, err := r.Run(p)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("cancel = %v", err)
	}
	if r.jit == nil || r.jit.entries == 0 || r.jit.budgets == 0 {
		t.Fatal("loop never entered native code")
	}
}

func TestJITRuntimeCloseAfterGuard(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	called := false
	r.global.setOwnRaw(r.atoms.intern("closeNow"), r.NewFunction("closeNow", 0,
		func(rt *Runtime, _ Value, _ []Value) (Value, error) {
			called = true
			rt.Halt(context.Canceled)
			rt.Close()
			rt.ReleaseClosed()
			if rt.jitCodeBytes() == 0 {
				t.Fatal("released code while frame active")
			}
			return Int32(2), nil
		}), propDefault)
	_, err := r.Run(compileForTest(t, `function f(a) { let s=0; for(let i=0;i<10000;i++) s+=a; return s }
		f({valueOf() { return closeNow() }})`))
	if !called || !errors.Is(err, context.Canceled) {
		t.Fatalf("close = %v, called %v", err, called)
	}
	r.ReleaseClosed()
	if r.jit != nil {
		t.Fatal("native owners survived unwind")
	}
}

func TestJITMemoryRefusal(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true, MemoryLimit: 256 << 10})
	v, err := r.Run(compileForTest(t, jitSumSource))
	if err != nil || v.Number() != 49995000 {
		t.Fatalf("optimization refusal stopped script: %v, %v", v, err)
	}
	if r.jitCodeBytes() != 0 {
		t.Fatal("compiled beyond optional memory allowance")
	}
	_, err = r.Run(compileForTest(t, `new ArrayBuffer(1024*1024)`))
	if !errors.Is(err, ErrMemoryLimit) {
		t.Fatalf("script exhaustion = %v", err)
	}

	large := jitRuntimeForTest(t, Config{JIT: true, MemoryLimit: 32 << 20})
	if _, err := large.Run(compileForTest(t, jitSumSource)); err != nil {
		t.Fatal(err)
	}
	if large.jitCodeBytes() == 0 {
		t.Fatal("ample memory prevented native compilation")
	}
	if large.meter.live < large.jitCodeBytes() {
		t.Fatal("new native pages were not immediately charged")
	}
	with := large.meter.walk(large)
	codeBytes := large.jitCodeBytes()
	large.releaseJIT()
	without := large.meter.walk(large)
	if with-without < codeBytes {
		t.Fatal("meter omitted native allocation")
	}
}

//go:noinline
func jitWeakOwnerForTest(t *testing.T, r *Runtime) weak.Pointer[bytecode.Function] {
	fn := jitFunctionForTest(t, `function f(n) { let s=0; for(let i=0;i<n;i++) s+=i; return s }`)
	if e := r.jitFor(fn); e == nil || e.code == nil {
		t.Fatal("native compilation refused")
	}
	return weak.Make(fn)
}

func TestJITWeakCache(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	p := jitWeakOwnerForTest(t, r)
	deadline := time.Now().Add(3 * time.Second)
	for p.Value() != nil && time.Now().Before(deadline) {
		runtime.GC()
		runtime.Gosched()
	}
	if p.Value() != nil {
		t.Fatal("native cache retained function graph")
	}
	fn := jitFunctionForTest(t, `function g(n) { let s=0; for(let i=0;i<n;i++) s+=i; return s }`)
	if e := r.jitFor(fn); e == nil || e.code == nil {
		t.Fatal("compilation refused")
	}
	if len(r.jit.cache) != 1 {
		t.Fatal("dead native owner was not swept")
	}
	runtime.KeepAlive(fn)
}

func TestJITCacheBounds(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	template := jitFunctionForTest(t, `function f(n) { var s=0; for(var i=0;i<n;i++) s+=i; return s }`)
	var functions []*bytecode.Function
	for i := 0; i < 2*jitCacheEntries; i++ {
		fn := new(bytecode.Function)
		*fn = *template
		functions = append(functions, fn)
		if e := r.jitFor(fn); e == nil || e.code == nil {
			t.Fatalf("entry %d refused", i)
		}
		if len(r.jit.cache) > jitCacheEntries || r.jitCodeBytes() > jitCacheBytes {
			t.Fatal("cache exceeded limits")
		}
	}
	if len(r.jit.cache) != jitCacheEntries {
		t.Fatal("test did not reach eviction")
	}
	var metadata int
	for _, e := range r.jit.cache {
		metadata += e.code.MetadataSize()
	}
	if metadata > jitMetadataBytes {
		t.Fatal("metadata exceeded limits")
	}
	runtime.KeepAlive(functions)
}

func TestJITSharedTemplateRuntimes(t *testing.T) {
	p := compileForTest(t, jitSumSource)
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r := New(Config{JIT: true})
			r.jitCallThreshold = 1
			defer func() { r.Close(); r.ReleaseClosed() }()
			for j := 0; j < 3; j++ {
				v, err := r.Run(p)
				if err != nil || v.Number() != 49995000 {
					t.Errorf("shared program = %v, %v", v, err)
					return
				}
			}
			if r.jit == nil || r.jit.entries == 0 {
				t.Error("shared program did not enter native code")
			}
		}()
	}
	wg.Wait()
}

func TestJITBudgetAllocations(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	p := compileForTest(t, jitSumSource)
	if _, err := r.Run(p); err != nil {
		t.Fatal(err)
	}
	if r.jit == nil || r.jit.budgets == 0 {
		t.Fatal("loop did not enter bounded native execution")
	}
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	for i := 0; i < 10; i++ {
		if _, err := r.Run(p); err != nil {
			t.Fatal(err)
		}
	}
	runtime.ReadMemStats(&after)
	// Preparing a new top-level closure allocates; each native budget exit
	// must not box the 4 KiB root array into an interface. The byte bound
	// also allows small ABI temporaries checkptr forces onto the heap.
	if bytes := (after.TotalAlloc - before.TotalAlloc) / 10; bytes > 16<<10 {
		t.Fatalf("native budget loop allocated %d bytes per call", bytes)
	}
}

func TestJITClosureRemembersRefusal(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	v, err := r.Run(compileForTest(t, `function f(n) { var s=0; for(var i=0;i<n;i++) s+=new Number(i).valueOf(); return s } f(3); f`))
	if err != nil {
		t.Fatal(err)
	}
	cl := v.Object().fn().closure
	if !cl.jitRefused {
		t.Fatal("unsupported function did not remember its refusal")
	}
	r.releaseJIT()
	got, err := r.Call(v, Undefined, []Value{Int32(10)})
	if err != nil || got.Number() != 45 {
		t.Fatalf("refused call = %v, %v", got, err)
	}
	if r.jit != nil {
		t.Fatal("permanent refusal repeated the weak-cache lookup")
	}
}

func TestJITCallPromotion(t *testing.T) {
	for _, keyword := range []string{"var", "let"} {
		t.Run(keyword, func(t *testing.T) {
			r := New(Config{JIT: true})
			t.Cleanup(func() { r.Close(); r.ReleaseClosed() })
			v, err := r.Run(compileForTest(t, "function sum(n) { "+keyword+" s=0; for("+keyword+" i=0;i<n;i++) s+=i; return s } sum"))
			if err != nil {
				t.Fatal(err)
			}
			cl := v.Object().fn().closure
			for i := 1; i <= jitHotCalls; i++ {
				got, err := r.Call(v, Undefined, []Value{Int32(100)})
				if err != nil || got.Number() != 4950 {
					t.Fatalf("call %d = %v, %v", i, got, err)
				}
				if int(cl.jitCalls) != i {
					t.Fatalf("call %d: hotness = %d", i, cl.jitCalls)
				}
				if i < jitHotCalls && r.jit != nil {
					t.Fatalf("cold call %d allocated native state", i)
				}
			}
			if r.jit == nil || r.jit.entries != 1 {
				t.Fatal("hot function did not promote")
			}
			for i := 0; i < 300; i++ {
				got, err := r.Call(v, Undefined, []Value{Int32(10)})
				if err != nil || got.Number() != 45 {
					t.Fatalf("hot call = %v, %v", got, err)
				}
			}
			if cl.jitCalls != jitHotCalls || r.jit.entries != 301 {
				t.Fatal("hot counter overflowed or native selection stopped")
			}
		})
	}
}

func TestJITPromotionRetriesMemoryRefusal(t *testing.T) {
	r := New(Config{JIT: true, MemoryLimit: 256 << 10})
	t.Cleanup(func() { r.Close(); r.ReleaseClosed() })
	v, err := r.Run(compileForTest(t, `function sum(n) { let s=0; for(let i=0;i<n;i++) s+=i; return s } sum`))
	if err != nil {
		t.Fatal(err)
	}
	cl := v.Object().fn().closure
	for i := 0; i < jitHotCalls; i++ {
		if _, err := r.Call(v, Undefined, []Value{Int32(10)}); err != nil {
			t.Fatal(err)
		}
	}
	if cl.jitRefused || cl.jitCalls != 0 || r.jit != nil {
		t.Fatal("memory refusal was permanent or retained native state")
	}
	r.meter.limit = 32 << 20
	for i := 1; i <= jitHotCalls; i++ {
		got, err := r.Call(v, Undefined, []Value{Int32(10)})
		if err != nil || got.Number() != 45 {
			t.Fatalf("retry %d = %v, %v", i, got, err)
		}
		if i < jitHotCalls && r.jit != nil {
			t.Fatal("temporary refusal retried before another warmup")
		}
	}
	if r.jit == nil || r.jit.entries != 1 {
		t.Fatal("temporary refusal never retried")
	}
}

func TestJITPromotionRuntimeIsolation(t *testing.T) {
	p := compileForTest(t, `function sum(n) { var s=0; for(var i=0;i<n;i++) s+=i; return s } sum`)
	var cold *Runtime
	for _, calls := range []int{1, jitHotCalls} {
		r := New(Config{JIT: true})
		t.Cleanup(func() { r.Close(); r.ReleaseClosed() })
		v, err := r.Run(p)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < calls; i++ {
			got, err := r.Call(v, Undefined, []Value{Int32(10)})
			if err != nil || got.Number() != 45 {
				t.Fatalf("call = %v, %v", got, err)
			}
		}
		if calls == 1 {
			cold = r
		} else if r.jit == nil || r.jit.entries != 1 {
			t.Fatal("hot runtime did not promote")
		}
	}
	if cold.jit != nil {
		t.Fatal("shared template propagated promotion to a cold runtime")
	}
}

func TestJITPromotionKeepsFramelessCalls(t *testing.T) {
	r := New(Config{JIT: true})
	t.Cleanup(func() { r.Close(); r.ReleaseClosed() })
	v, err := r.Run(compileForTest(t, `function f(a) { return a+1 } var result; for(var i=0;i<1000;i++) result=f(i); result`))
	if err != nil || v.Number() != 1000 {
		t.Fatalf("frameless calls = %v, %v", v, err)
	}
	cl := r.global.getOwn(r.atoms.intern("f")).value.Object().fn().closure
	if cl.jitCalls != 0 || r.jit != nil {
		t.Fatal("frameless calls accumulated framed hotness")
	}
}

func TestJITLoopPromotionFirstCall(t *testing.T) {
	for _, keyword := range []string{"var", "let"} {
		t.Run(keyword, func(t *testing.T) {
			r := New(Config{JIT: true})
			t.Cleanup(func() { r.Close(); r.ReleaseClosed() })
			source := "function sum(n) { " + keyword + " s=0; for(" + keyword + " i=0;i<n;i++) s+=i; return s } sum(10000)"
			v, err := r.Run(compileForTest(t, source))
			if err != nil || v.Number() != 49995000 {
				t.Fatalf("first call = %v, %v", v, err)
			}
			if r.jit == nil || r.jit.osrs != 1 || r.jit.budgets == 0 || r.jit.rootCount != 0 {
				t.Fatal("first long call did not complete bounded OSR")
			}
			cl := r.global.getOwn(r.atoms.intern("sum")).value.Object().fn().closure
			tree := (*tree)(atomic.LoadPointer(&cl.fn.VMCode))
			if keyword == "var" && (tree == nil || tree == noTree) || keyword == "let" && tree != noTree {
				t.Fatal("test did not exercise its expected existing tier")
			}
			if r.jitDeoptDepth != 0 {
				t.Fatal("completed call retained a deoptimized frame")
			}
		})
	}
}

func TestJITLoopPromotionAtEntryPC(t *testing.T) {
	r := New(Config{JIT: true})
	t.Cleanup(func() { r.Close(); r.ReleaseClosed() })
	v, err := r.Run(compileForTest(t, `function f(n) { while(n) n--; return n } f(10000)`))
	if err != nil || v.Number() != 0 {
		t.Fatalf("entry loop = %v, %v", v, err)
	}
	if r.jit == nil || r.jit.osrs != 1 {
		t.Fatal("loop targeting entry did not promote")
	}
}

func TestJITLoopPromotionLiveOperands(t *testing.T) {
	for _, keyword := range []string{"var", "let"} {
		t.Run(keyword, func(t *testing.T) {
			r := New(Config{JIT: true})
			t.Cleanup(func() { r.Close(); r.ReleaseClosed() })
			original := jitFunctionForTest(t, "function f(a,n) { "+keyword+" s=0; for("+keyword+" i=0;i<n;i++) s+=i; return s }")
			fn := *original
			// Carry an opaque operand through the loop, then return it after
			// a swap. Native return must guard and publish that exact operand.
			fn.Code = []bytecode.Instr{{Op: bytecode.OpGetLocal}}
			mapping := make([]uint32, len(original.Code))
			for i, in := range original.Code {
				mapping[i] = uint32(len(fn.Code))
				if in.Op == bytecode.OpReturn {
					fn.Code = append(fn.Code, bytecode.Instr{Op: bytecode.OpSwap})
				}
				fn.Code = append(fn.Code, in)
			}
			for i := 1; i < len(fn.Code); i++ {
				switch fn.Code[i].Op {
				case bytecode.OpJump, bytecode.OpJumpIfTrue, bytecode.OpJumpIfFalse,
					bytecode.OpJumpIfTrueKeep, bytecode.OpJumpIfFalseKeep, bytecode.OpJumpIfCmpFalse:
					fn.Code[i].A = mapping[fn.Code[i].A]
				}
			}
			fn.MaxStack++
			o := r.NewObject()
			v, err := r.run(r.prepare(&fn), Undefined, []Value{Obj(o), Int32(10000)}, Undefined, nil)
			if err != nil || !v.IsObject() || v.Object() != o {
				t.Fatalf("live operand = %v, %v", v, err)
			}
			if r.jit == nil || r.jit.osrs != 1 || r.jit.guards != 1 || r.jit.rootCount != 0 {
				t.Fatal("live operand was lost, unguarded, or retained")
			}
			for _, root := range r.jit.roots {
				if root.IsObject() {
					t.Fatal("OSR retained an opaque root")
				}
			}
		})
	}
}

func TestJITLoopGuardSuppressesReentry(t *testing.T) {
	for _, keyword := range []string{"var", "let"} {
		t.Run(keyword, func(t *testing.T) {
			r := New(Config{JIT: true})
			t.Cleanup(func() { r.Close(); r.ReleaseClosed() })
			count := 0
			var state *jitState
			o := r.NewObject()
			o.setOwnRaw(r.atoms.intern("valueOf"), r.NewFunction("valueOf", 0,
				func(rt *Runtime, _ Value, _ []Value) (Value, error) {
					count++
					if rt.jit != nil {
						if state != nil {
							t.Fatal("deoptimized invocation recompiled after eviction")
						}
						state = rt.jit
						rt.releaseJIT()
					}
					return Int32(1), nil
				}), propDefault)
			r.global.setOwnRaw(r.atoms.intern("operand"), Obj(o), propDefault)
			source := "function f(n,a) { " + keyword + " s=0; for(" + keyword + " i=0;i<n;i++) s+=a; return s } f(10000,operand)"
			v, err := r.Run(compileForTest(t, source))
			if err != nil || v.Number() != 10000 || count != 10000 {
				t.Fatalf("guard = %v, %v; coercions %d", v, err, count)
			}
			if state == nil || state.osrs != 1 || state.guards != 1 || state.rootCount != 0 || r.jitDeoptDepth != 0 {
				t.Fatal("guard did not exit once or restore ownership")
			}
		})
	}
}

func TestJITLoopPromotionCancellation(t *testing.T) {
	for _, keyword := range []string{"var", "let"} {
		t.Run(keyword, func(t *testing.T) {
			r := New(Config{JIT: true})
			t.Cleanup(func() { r.Close(); r.ReleaseClosed() })
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
			defer cancel()
			r.SetContext(ctx)
			_, err := r.Run(compileForTest(t, "function f() { "+keyword+" i=0; for(;;) i++ } f()"))
			if !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("cancel = %v", err)
			}
			if r.jit == nil || r.jit.osrs != 1 || r.jit.budgets == 0 {
				t.Fatal("first infinite call never entered bounded native code")
			}
		})
	}
}

func TestJITLoopGuardErrors(t *testing.T) {
	for _, tc := range []struct{ name, source, message string }{
		{"bigint", `function f(n,a) { var s=0; for(var i=0;i<n;i++) { s+=i; if(i===2000) s+=a } return s } f(10000,1n)`,
			"TypeError: cannot mix BigInt and other types"},
		{"tdz", `function f(n) { let s=0; for(let i=0;i<n;i++) { s+=i; if(i===2000) { let x=x; s+=x } } return s } f(10000)`,
			`ReferenceError: cannot access "x" before initialization`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := compileForTest(t, tc.source)
			var baseline *Thrown
			for _, enabled := range []bool{false, true} {
				r := New(Config{JIT: enabled})
				t.Cleanup(func() { r.Close(); r.ReleaseClosed() })
				_, err := r.Run(p)
				var thrown *Thrown
				if !errors.As(err, &thrown) || thrown.Error() != tc.message {
					t.Fatalf("enabled %v: error = %v", enabled, err)
				}
				if !enabled {
					baseline = thrown
					continue
				}
				if r.jit == nil || r.jit.osrs != 1 || r.jit.guards != 1 || r.jit.rootCount != 0 || r.jitDeoptDepth != 0 {
					t.Fatal("error did not follow OSR and guard fallback")
				}
				if len(thrown.trace.frames) != len(baseline.trace.frames) {
					t.Fatal("OSR changed the thrown trace depth")
				}
				for i, frame := range thrown.trace.frames {
					want := baseline.trace.frames[i]
					if frame.fn != want.fn || frame.pc != want.pc {
						t.Fatalf("trace frame %d = %+v, want %+v", i, frame, want)
					}
				}
			}
		})
	}
}

func BenchmarkJITNumericLoop(b *testing.B) {
	for _, keyword := range []string{"var", "let"} {
		for _, enabled := range []bool{false, true} {
			name := keyword + "/existing"
			if enabled {
				name = keyword + "/native"
			}
			b.Run(name, func(b *testing.B) {
				source := "function sum(n) { " + keyword + " s=0; for(" + keyword + " i=0;i<n;i++) s+=i; return s }"
				ast, err := parser.Parse(source, parser.Options{})
				if err != nil {
					b.Fatal(err)
				}
				p, err := compiler.Compile(ast, compiler.Options{})
				if err != nil {
					b.Fatal(err)
				}
				r := New(Config{JIT: enabled})
				defer func() { r.Close(); r.ReleaseClosed() }()
				if _, err := r.Run(p); err != nil {
					b.Fatal(err)
				}
				ast, err = parser.Parse("sum(10000)", parser.Options{})
				if err != nil {
					b.Fatal(err)
				}
				p, err = compiler.Compile(ast, compiler.Options{})
				if err != nil {
					b.Fatal(err)
				}
				for i := 0; i < jitHotCalls; i++ {
					if _, err := r.Run(p); err != nil {
						b.Fatal(err)
					}
				}
				if enabled && (r.jit == nil || r.jit.entries == 0) {
					b.Fatal("benchmark did not enter native code")
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					v, err := r.Run(p)
					if err != nil || v.Number() != 49995000 {
						b.Fatalf("sum = %v, %v", v, err)
					}
				}
			})
		}
	}
}

func BenchmarkJITCallPromotion(b *testing.B) {
	for _, iterations := range []int{100, 10000} {
		for _, calls := range []int{1, 4, 16, 64} {
			for _, mode := range []string{"existing", "eager", "automatic"} {
				b.Run(fmt.Sprintf("iterations%d/calls%d/%s", iterations, calls, mode), func(b *testing.B) {
					source := fmt.Sprintf(`function sum(n) { let s=0; for(let i=0;i<n;i++) s+=i; return s }
					var result; for(var k=0;k<%d;k++) result=sum(%d); result`, calls, iterations)
					ast, err := parser.Parse(source, parser.Options{})
					if err != nil {
						b.Fatal(err)
					}
					p, err := compiler.Compile(ast, compiler.Options{})
					if err != nil {
						b.Fatal(err)
					}
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						r := New(Config{JIT: mode != "existing"})
						if mode == "eager" {
							r.jitCallThreshold = 1
						}
						v, err := r.Run(p)
						native := r.jit != nil && r.jit.entries != 0
						coldState := r.jit != nil
						r.Close()
						r.ReleaseClosed()
						if err != nil || v.Number() != float64(iterations*(iterations-1)/2) {
							b.Fatalf("sum = %v, %v", v, err)
						}
						if mode != "existing" && (mode == "eager" || calls >= jitHotCalls || iterations > backEdgeCheckInterval+1) && !native {
							b.Fatal("hot benchmark did not enter native code")
						}
						if mode == "automatic" && calls < jitHotCalls && iterations < backEdgeCheckInterval && coldState {
							b.Fatal("cold benchmark allocated native state")
						}
					}
				})
			}
		}
	}
}

func jitNumericKernelCases() []struct{ name, body string } {
	return []struct{ name, body string }{
		{"logistic", `VAR x=0.123; for(VAR i=0;i<n;i++) x=3.99*x*(1-x); return x`},
		{"newton", `VAR x=1,s=0; for(VAR i=0;i<n;i++) { x=(x+2/x)*0.5; s+=x } return s`},
		{"particle", `VAR x=0,y=0,vx=0.2,vy=0.3,s=0; for(VAR i=0;i<n;i++) { vx-=x*0.001; vy-=y*0.001; x+=vx; y+=vy; if(x>10||x< -10) vx=-vx; if(y>10||y< -10) vy=-vy; s+=x*x+y*y } return s`},
	}
}

func jitDenseKernelCases() []struct{ name, body string } {
	return []struct{ name, body string }{
		{"vector", `for(var i=0;i<n;i++)a[i]=2*b[i]+1;return a[n-1]`},
		{"stencil", `for(var i=1;i<n-1;i++)a[i]=(b[i-1]+b[i]+b[i+1])/3;return a[n-2]`},
		{"helper", `for(var k=0;k<4;k++){for(var i=1;i<n-1;i++)a[i]=(b[i-1]+b[i]+b[i+1])/3;boundary(a)}return a[0]`},
	}
}

func BenchmarkJITDenseKernels(b *testing.B) {
	for _, tc := range jitDenseKernelCases() {
		for _, mode := range []string{"interpreter", "existing", "native"} {
			b.Run(tc.name+"/"+mode, func(b *testing.B) {
				previous := treeTier.Swap(mode != "interpreter")
				defer treeTier.Store(previous)
				compile := func(src string) *bytecode.Function {
					ast, err := parser.Parse(src, parser.Options{})
					if err != nil {
						b.Fatal(err)
					}
					p, err := compiler.Compile(ast, compiler.Options{})
					if err != nil {
						b.Fatal(err)
					}
					return p
				}
				setup := compile(`function make(n){function boundary(a){a[0]=a[1];a[n-1]=a[n-2]}return function kernel(a,b){` + tc.body + `}}var kernel=make(8192);var a=new Array(8192),b=new Array(8192);for(var i=0;i<8192;i++){a[i]=1;b[i]=1}`)
				call := compile(`kernel(a,b)`)
				r := New(Config{JIT: mode == "native"})
				defer func() { r.Close(); r.ReleaseClosed() }()
				if _, err := r.Run(setup); err != nil {
					b.Fatal(err)
				}
				var want Value
				for i := 0; i < jitHotCalls; i++ {
					v, err := r.Run(call)
					if err != nil {
						b.Fatal(err)
					}
					want = v
				}
				if mode == "native" && (r.jit == nil || r.jit.entries == 0 || r.jit.guards != 0) {
					b.Fatal("did not stay native")
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					v, err := r.Run(call)
					if err != nil || !jitSameValueForTest(v, want) {
						b.Fatalf("result %v error %v want %v", v, err, want)
					}
				}
				b.StopTimer()
				if r.jit != nil {
					b.ReportMetric(float64(r.jitCodeBytes()), "code+metadata-B")
				}
			})
		}
	}
}

func BenchmarkJITDenseFirstUse(b *testing.B) {
	for _, tc := range jitDenseKernelCases() {
		for _, enabled := range []bool{false, true} {
			mode := "existing"
			if enabled {
				mode = "automatic"
			}
			b.Run(tc.name+"/"+mode, func(b *testing.B) {
				compile := func(src string) *bytecode.Function {
					ast, err := parser.Parse(src, parser.Options{})
					if err != nil {
						b.Fatal(err)
					}
					p, err := compiler.Compile(ast, compiler.Options{})
					if err != nil {
						b.Fatal(err)
					}
					return p
				}
				setup := compile(`function make(n){function boundary(a){a[0]=a[1];a[n-1]=a[n-2]}return function kernel(a,b){` + tc.body + `}}var kernel=make(8192);var a=new Array(8192),b=new Array(8192);for(var i=0;i<8192;i++){a[i]=1;b[i]=1}`)
				call := compile(`kernel(a,b)`)
				b.ReportAllocs()
				b.ResetTimer()
				b.StopTimer()
				for i := 0; i < b.N; i++ {
					r := New(Config{JIT: enabled})
					if _, err := r.Run(setup); err != nil {
						b.Fatal(err)
					}
					b.StartTimer()
					v, err := r.Run(call)
					b.StopTimer()
					native := r.jit != nil && r.jit.entries != 0 && r.jit.guards == 0
					r.Close()
					r.ReleaseClosed()
					want := float64(1)
					if tc.name == "vector" {
						want = 3
					}
					if err != nil || v.Number() != want || enabled && !native {
						b.Fatalf("first call %v error %v native %v", v, err, native)
					}
				}
			})
		}
	}
}

func BenchmarkJITNumericKernels(b *testing.B) {
	for _, tc := range jitNumericKernelCases() {
		for _, keyword := range []string{"var", "let"} {
			source := "function kernel(n) { " + strings.ReplaceAll(tc.body, "VAR", keyword) + " }"
			compile := func(source string) *bytecode.Function {
				ast, err := parser.Parse(source, parser.Options{})
				if err != nil {
					b.Fatal(err)
				}
				p, err := compiler.Compile(ast, compiler.Options{})
				if err != nil {
					b.Fatal(err)
				}
				return p
			}
			definition, call := compile(source), compile("kernel(10000)")
			baseline := New(Config{})
			if _, err := baseline.Run(definition); err != nil {
				b.Fatal(err)
			}
			want, err := baseline.Run(call)
			baseline.Close()
			baseline.ReleaseClosed()
			if err != nil || !want.IsNumber() {
				b.Fatalf("baseline = %v, %v", want, err)
			}
			for _, enabled := range []bool{false, true} {
				mode := "existing"
				if enabled {
					mode = "native"
				}
				b.Run(tc.name+"/"+keyword+"/"+mode, func(b *testing.B) {
					r := New(Config{JIT: enabled})
					defer func() { r.Close(); r.ReleaseClosed() }()
					if _, err := r.Run(definition); err != nil {
						b.Fatal(err)
					}
					for i := 0; i < jitHotCalls; i++ {
						v, err := r.Run(call)
						if err != nil || !jitSameValueForTest(v, want) {
							b.Fatalf("warmup = %v, %v; want %v", v, err, want)
						}
					}
					if enabled && (r.jit == nil || r.jit.entries == 0 || r.jit.guards != 0) {
						b.Fatal("benchmark did not stay in native code")
					}
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						v, err := r.Run(call)
						if err != nil || !jitSameValueForTest(v, want) {
							b.Fatalf("kernel = %v, %v; want %v", v, err, want)
						}
					}
					b.StopTimer()
					if enabled {
						var codeBytes, metadataBytes int
						for _, entry := range r.jit.cache {
							codeBytes += entry.code.Size()
							metadataBytes += entry.code.MetadataSize()
						}
						b.ReportMetric(float64(codeBytes), "native-B")
						b.ReportMetric(float64(metadataBytes), "metadata-B")
					}
				})
			}
		}
	}
}

func BenchmarkJITKernelPromotion(b *testing.B) {
	for _, tc := range jitNumericKernelCases() {
		for _, keyword := range []string{"var", "let"} {
			for _, calls := range []int{1, 2, 4, 8, 16} {
				source := "function kernel(n) { " + strings.ReplaceAll(tc.body, "VAR", keyword) + " }" +
					fmt.Sprintf("var result; for(var k=0;k<%d;k++) result=kernel(10000); result", calls)
				ast, err := parser.Parse(source, parser.Options{})
				if err != nil {
					b.Fatal(err)
				}
				p, err := compiler.Compile(ast, compiler.Options{})
				if err != nil {
					b.Fatal(err)
				}
				baseline := New(Config{})
				want, err := baseline.Run(p)
				baseline.Close()
				baseline.ReleaseClosed()
				if err != nil || !want.IsNumber() {
					b.Fatalf("baseline = %v, %v", want, err)
				}
				for _, enabled := range []bool{false, true} {
					mode := "existing"
					if enabled {
						mode = "automatic"
					}
					b.Run(fmt.Sprintf("%s/%s/calls%d/%s", tc.name, keyword, calls, mode), func(b *testing.B) {
						b.ReportAllocs()
						b.ResetTimer()
						for i := 0; i < b.N; i++ {
							r := New(Config{JIT: enabled})
							v, err := r.Run(p)
							native := r.jit != nil && r.jit.entries != 0 && r.jit.guards == 0
							r.Close()
							r.ReleaseClosed()
							if err != nil || !jitSameValueForTest(v, want) {
								b.Fatalf("kernel = %v, %v; want %v", v, err, want)
							}
							if enabled && !native {
								b.Fatal("benchmark did not stay in native code")
							}
						}
					})
				}
			}
		}
	}
}
