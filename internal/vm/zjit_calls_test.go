//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/go-quickjs/go-quickjs/internal/compiler"
	"github.com/go-quickjs/go-quickjs/internal/parser"
)

func TestJITScalarCallChains(t *testing.T) {
	for _, source := range []string{
		`function h(a,n){let s=0;for(let i=0;i<n;i++){a[0]+=1;s+=a[0]}return s}
         h([0],4);function f(a,h,n){let s=0;for(let i=0;i<n;i++)s+=h(a,4);return s+a[0]}
         let a=[0];f(a,h,4)===152&&a[0]===16`,
		`function h(a,x,y){for(let i=0;i<4;i++)a[0]+=1;return y===undefined?x:a}
         h([0],3);function f(a,h){let s=0;for(let i=0;i<4;i++)s+=h(a,3);return s}
         let a=[0];f(a,h)===12&&a[0]===16`,
		`function h(a,n){'use strict';for(let i=0;i<n;i++)a[0]+=1;return this}
         h([0],4);let p={h};let o=Object.create(p);
         function f(o,a){for(let i=0;i<4;i++)if(o.h(a,4)!==o)return false;return a[0]===16}
         f(o,[0])`,
		`function h(a,n){for(let i=0;i<n;i++)a[0]+=1;return this}
         h([0],4);function f(a,h){let o;for(let i=0;i<4;i++)o=h(a,4);return o===globalThis&&a[0]===16}
         f([0],h)`,
		`function h(a,n){for(let i=0;i<n;i++)a[0]+=1;return a[0]}
         h([0],4);function g(a,h){let s=0;for(let i=0;i<4;i++)s+=h(a,4);return s}
         g([0],h);function f(a,g,h){let s=0;for(let i=0;i<4;i++)s+=g(a,h);return s+a[0]}
         f([0],g,h)===608`,
		`function make(m){return function h(a,n){for(let i=0;i<n;i++)a[0]+=m;return a}}
         let h=make(3);h([0],4);function f(a,h){for(let i=0;i<4;i++)if(h(a,4)!==a)return false;return a[0]===48}
         f([0],h)`,
		`function h(a,x){for(let i=0;i<4;i++)a[0]+=1;return x}
         h([0],3);function f(a,h){let x;for(let i=0;i<4;i++)x=h(a,-0,7);return x}
         let a=[0];Object.is(f(a,h),-0)&&a[0]===16`,
	} {
		t.Run(source[:25], func(t *testing.T) {
			r := jitRuntimeForTest(t, Config{JIT: true})
			v, err := r.Run(compileForTest(t, source))
			if err != nil || !v.IsBool() || !v.Truthy() {
				t.Fatalf("scalar chain: %v, %v", v, err)
			}
			if r.jit == nil || r.jit.transfers == 0 || r.jit.rootCount != 0 || r.frameDepth != 0 {
				t.Fatal("scalar call chain did not execute and unwind")
			}
			if r.jit.callFrames.active {
				t.Fatal("active arena after return")
			}
			for _, q := range r.jit.callFrames.frames {
				if q.f != nil || q.e != nil {
					t.Fatal("arena retained a frame or executable owner")
				}
			}
		})
	}
}

func TestJITCallChainBounds(t *testing.T) {
	var source strings.Builder
	for i := 11; i >= 0; i-- {
		fmt.Fprintf(&source, "function f%d(a){for(let i=0;i<4;i++)a[0]+=1;return ", i)
		if i == 11 {
			source.WriteString("a[0]}")
		} else {
			fmt.Fprintf(&source, "f%d(a)+1}", i+1)
		}
		fmt.Fprintf(&source, "f%d([0]);", i)
	}
	for _, cfg := range []Config{
		{JIT: true}, {JIT: true, MaxCallDepth: 10}, {JIT: true, StackSize: 80},
		{JIT: true, MemoryLimit: 256 << 10}, {JIT: true, MemoryLimit: 32 << 20},
	} {
		r := jitRuntimeForTest(t, cfg)
		if _, err := r.Run(compileForTest(t, source.String())); err != nil {
			if cfg.MaxCallDepth != 0 || cfg.StackSize != 0 {
				if !strings.Contains(err.Error(), "maximum call stack size exceeded") {
					t.Fatalf("wrong depth/stack error: %v", err)
				}
				continue
			}
			t.Fatal(err)
		}
		v, err := r.Run(compileForTest(t, `f0([0])`))
		if err != nil || v.Number() != 59 || r.frameDepth != 0 || r.jit != nil && r.jit.rootCount != 0 {
			t.Fatalf("bounded native arena fallback: %v, %v", v, err)
		}
		if cfg.MemoryLimit == 0 && (r.jit == nil || r.jit.transfers == 0) {
			t.Fatal("deep chain did not use scalar transfers")
		}
	}
}

func TestJITCallChainCommittedFallback(t *testing.T) {
	for _, tail := range []string{
		`let count=0;let bad={valueOf(){count++;return 3}};
         let a=[0,bad];f(a,h)===4&&a[0]===4&&count===4`,
		`let count=0;let bad={valueOf(){count++;throw new TypeError("limb")}};
         let a=[0,bad];let good=false;try{f(a,h)}catch(e){good=e instanceof TypeError&&e.message==="limb"&&e.stack.includes("h")&&e.stack.includes("f")}
         good&&a[0]===1&&count===1`,
	} {
		source := `function h(a,n){for(let i=0;i<n;i++)a[0]+=1;return a[1]*2}
                   h([0,2],4);function f(a,h){let s=0;for(let i=0;i<4;i++){h(a,1);s++}return s}` + tail
		r := jitRuntimeForTest(t, Config{JIT: true})
		v, err := r.Run(compileForTest(t, source))
		if err != nil || !v.IsBool() || !v.Truthy() || r.jit.transfers == 0 || r.jit.rootCount != 0 {
			t.Fatalf("committed chain fallback: %v, %v", v, err)
		}
	}
}

func TestJITCallChainCallbackRelease(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	callbacks := 0
	r.global.setOwnRaw(r.atoms.intern("host"), r.NewFunction("host", 0,
		func(rt *Runtime, _ Value, _ []Value) (Value, error) {
			callbacks++
			if rt.jit.rootCount != 0 || rt.jit.callActive {
				t.Fatal("callback retained borrowed scalar permissions")
			}
			rt.releaseJIT()
			runtime.GC()
			if v, err := rt.Run(compileForTest(t, `reenter([0],4)`)); err != nil || v.Number() != 4 {
				t.Fatalf("reentry: %v, %v", v, err)
			}
			return Undefined, nil
		}), propDefault)
	source := `function reenter(a,n){for(let i=0;i<n;i++)a[0]+=1;return a[0]}
               function h(a,n,cb){for(let i=0;i<n;i++)a[0]+=1;cb();return a[0]}
               h([0],4,()=>{});function f(a,h,cb){let s=0;for(let i=0;i<4;i++)s+=h(a,4,cb);return s}
               let a=[0];f(a,h,()=>host())===40&&a[0]===16`
	v, err := r.Run(compileForTest(t, source))
	if err != nil || !v.IsBool() || !v.Truthy() || callbacks != 4 || r.jit.rootCount != 0 {
		t.Fatalf("callback chain release: %v, %v, callbacks=%d", v, err, callbacks)
	}
}

func TestJITCallChainMutation(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	source := `function h(a,n){for(let i=0;i<n;i++)a[0]+=1;return 1}
               function k(a,n){for(let i=0;i<n;i++)a[0]+=2;return 2}
               h([0],4);k([0],4);let p={h};let o=Object.create(p);let reads=0;
               function f(o,a,cb){let s=0;for(let i=0;i<4;i++){s+=o.h(a,4);cb(i)}return s}
               let a=[0];let sum=f(o,a,i=>{if(i===0)p.h=k;if(i===1)Object.defineProperty(o,"h",{get(){reads++;return h}})});
               sum===5&&a[0]===20&&reads===2`
	v, err := r.Run(compileForTest(t, source))
	if err != nil || !v.IsBool() || !v.Truthy() || r.jit.transfers == 0 || r.jit.rootCount != 0 {
		t.Fatalf("method mutation: %v, %v", v, err)
	}
}

func TestJITCallChainLegacyArguments(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true, NodeQuirks: true})
	source := `function h(a,n,cb){while(--n>=0)a[0]+=1;cb();return a[0]}
               h([0],4,()=>{});let checks=0;let extra={x:7};
               function f(a,h,cb){let s=0;for(let i=0;i<4;i++)s+=h(a,4,cb,extra);return s}
               let a=[0];f(a,h,()=>{let args=h.arguments;if(args.length===4&&args[0]===a&&args[1]===-1&&args[3]===extra&&h.caller===f)checks++})===40&&checks===4`
	v, err := r.Run(compileForTest(t, source))
	if err != nil || !v.IsBool() || !v.Truthy() || r.jit.transfers == 0 || r.jit.rootCount != 0 {
		t.Fatalf("observable scalar call arguments: %v, %v", v, err)
	}
}

func TestJITCallChainCompileEviction(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	var source strings.Builder
	source.WriteString(`function f(a,h){for(let i=0;i<4;i++)a[0]+=h(a);return a[0]}let good=true;`)
	for i := 0; i < 160; i++ {
		fmt.Fprintf(&source, "function h%d(a){let s=0;for(let i=0;i<16;i++)s+=i;return a[0]+1}if(f([0],h%d)!==15)good=false;", i, i)
	}
	source.WriteString("good")
	v, err := r.Run(compileForTest(t, source.String()))
	if err != nil || !v.IsBool() || !v.Truthy() || r.jit.transfers != 640 || r.jit.rootCount != 0 {
		t.Fatalf("compile/evict during scalar chain: %v, %v", v, err)
	}
	if len(r.jit.cache) != jitCacheEntries {
		t.Fatal("callee churn did not fill the bounded code cache")
	}
	f := r.global.getOwn(r.atoms.intern("f")).value.Object().fn().closure
	if r.entryOf(f).code.Size() == 0 {
		t.Fatal("callee compilation evicted its active caller")
	}
}

func TestJITCallChainCancellation(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	source := `function h(a,n){for(let i=0;i<n;i++)a[0]+=1;return a[0]}
               h([0],4);function f(a,h,n){let s=0;for(let i=0;i<n;i++)s+=h(a,4);return s}`
	if _, err := r.Run(compileForTest(t, source)); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	r.ctx = ctx
	_, err := r.Run(compileForTest(t, `f([0],h,Infinity)`))
	if !errors.Is(err, context.DeadlineExceeded) || r.jit.transfers == 0 || r.jit.rootCount != 0 || r.frameDepth != 0 {
		t.Fatalf("chain cancellation: %v", err)
	}
}

func BenchmarkJITScalarCalls(b *testing.B) {
	const setup = `function h(a,n){let s=0;for(let i=0;i<n;i++){a[0]+=1;s+=a[0]}return s}
                   function f(a,h,n){let s=0;for(let i=0;i<n;i++)s+=h(a,4);return s+a[0]}`
	for _, mode := range []string{"interpreter", "existing", "native"} {
		b.Run(mode, func(b *testing.B) {
			previous := treeTier.Swap(mode != "interpreter")
			defer treeTier.Store(previous)
			r := New(Config{JIT: mode == "native"})
			defer func() { r.Close(); r.ReleaseClosed() }()
			for _, source := range []string{setup, `for(let i=0;i<16;i++){h([0],4);f([0],h,16)}`} {
				ast, err := parser.Parse(source, parser.Options{})
				if err != nil {
					b.Fatal(err)
				}
				fn, err := compiler.Compile(ast, compiler.Options{})
				if err != nil {
					b.Fatal(err)
				}
				if _, err := r.Run(fn); err != nil {
					b.Fatal(err)
				}
			}
			ast, err := parser.Parse(`f([0],h,1024)`, parser.Options{})
			if err != nil {
				b.Fatal(err)
			}
			fn, err := compiler.Compile(ast, compiler.Options{})
			if err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				v, err := r.Run(fn)
				if err != nil || v.Number() != 8394752 {
					b.Fatalf("scalar calls: %v, %v", v, err)
				}
			}
			b.StopTimer()
			if mode == "native" && r.jit.transfers == 0 {
				b.Fatal("no scalar transfers")
			}
		})
	}
}

func BenchmarkJITScalarCallsFirstUse(b *testing.B) {
	const source = `function h(a,n){let s=0;for(let i=0;i<n;i++){a[0]+=1;s+=a[0]}return s}
                    function f(a,h,n){let s=0;for(let i=0;i<n;i++)s+=h(a,4);return s+a[0]}`
	for _, enabled := range []bool{false, true} {
		b.Run(fmt.Sprintf("native=%v", enabled), func(b *testing.B) {
			b.ReportAllocs()
			b.StopTimer()
			for range b.N {
				ast, err := parser.Parse(source, parser.Options{})
				if err != nil {
					b.Fatal(err)
				}
				fn, err := compiler.Compile(ast, compiler.Options{})
				if err != nil {
					b.Fatal(err)
				}
				ast, err = parser.Parse(`f([0],h,8192)`, parser.Options{})
				if err != nil {
					b.Fatal(err)
				}
				call, err := compiler.Compile(ast, compiler.Options{})
				if err != nil {
					b.Fatal(err)
				}
				r := New(Config{JIT: enabled})
				if _, err := r.Run(fn); err != nil {
					b.Fatal(err)
				}
				b.StartTimer()
				v, err := r.Run(call)
				b.StopTimer()
				if err != nil || v.Number() != 536920064 {
					b.Fatalf("first call: %v, %v", v, err)
				}
				r.Close()
				r.ReleaseClosed()
			}
		})
	}
}
