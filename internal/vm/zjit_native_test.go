//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"testing"
	"time"
	"weak"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
	"github.com/go-quickjs/go-quickjs/internal/compiler"
	"github.com/go-quickjs/go-quickjs/internal/parser"
)

const jitSumSource = `function sum(n) { let s=0; for(let i=0;i<n;i++) s+=i; return s } sum(10000)`

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
	v, err := r.Run(compileForTest(t, `function f(n) { var s=0; for(var i=0;i<n;i++) s+=Math.abs(i); return s } f(3); f`))
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
	v, err := r.Run(compileForTest(t, jitSumSource+`; sum`))
	if err != nil {
		t.Fatal(err)
	}
	cl := v.Object().fn().closure
	for i := 1; i < jitHotCalls; i++ {
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
			for _, mode := range []string{"existing", "eager", "delayed"} {
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
						if mode != "existing" && (mode == "eager" || calls >= jitHotCalls) && !native {
							b.Fatal("hot benchmark did not enter native code")
						}
						if mode == "delayed" && calls < jitHotCalls && coldState {
							b.Fatal("cold benchmark allocated native state")
						}
					}
				})
			}
		}
	}
}
