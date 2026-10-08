//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
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
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
	"github.com/go-quickjs/go-quickjs/internal/parser"
)

const jitSumSource = `function sum(n) { let s=0; for(let i=0;i<n;i++) s+=i; return s } sum(10000)`

func TestJITPropertyABI(t *testing.T) {
	var p Property
	var cell ir.PropertyCell
	if unsafe.Sizeof(p) != unsafe.Sizeof(cell) || unsafe.Offsetof(p.key) != unsafe.Offsetof(cell.Key) || unsafe.Offsetof(p.flags) != unsafe.Offsetof(cell.Flags) || unsafe.Offsetof(p.value) != unsafe.Offsetof(cell.Bits) || unsafe.Offsetof(p.value)+unsafe.Offsetof(p.value.ref) != unsafe.Offsetof(cell.Reference) {
		t.Fatal("VM property layout differs from borrowed native cells")
	}
	if propWritable != 1 || propDefault != 7 || propAccessor|propDeleted|propPrivate|propNamespaceExport|propUninit != 0xf8 {
		t.Fatal("VM property permissions differ from native attributes")
	}
}

func TestJITNumericProperties(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	v, err := r.Run(compileForTest(t, `function f(o,n){let s=0;for(let i=0;i<n;i++){s=(s+o.x)|0;o.x=o.x+1}return s}let o={a:7,x:3};f(o,1000)===502500&&o.x===1003&&o.a===7`))
	if err != nil || !v.IsBool() || !v.Truthy() || r.jit == nil || r.jit.entries == 0 || r.jit.hosts != 0 || r.jit.guards != 0 || r.jit.rootCount != 0 {
		t.Fatalf("numeric fields did not stay native: %v, %v", v, err)
	}
}

func TestJITThisSnapshot(t *testing.T) {
	for _, source := range []string{
		`function f(a){'use strict';let n=this;for(let i=0;i<4;i++)a[i]=n*2;return a[3]}f.call(3,[0,0,0,0])===6`,
		`function f(a){a[0]+=1;return this}let o={x:3};f.call(o,[0])===o`,
		`function f(a){'use strict';a[0]+=1;return this}f.call(null,[0])===null&&f.call(undefined,[0])===undefined`,
		`function f(a){a[0]+=1;return this}let a=[0];f.call(null,a)===globalThis&&a[0]===1`,
		`function f(a,cb){let x=this.x;cb();a[0]+=1;return this.x+x}let o={x:3};f.call(o,[0],()=>{o.x=7})===10`,
		`let o={x:3,f(a){a[0]+=1;return ()=>{a[0]+=1;return this.x+a[0]}}};o.f([0])()===5`,
		`let a=[0],good=false;class B{}class D extends B{constructor(){let f=()=>{a[0]+=1;return this};try{f()}catch(e){good=e instanceof ReferenceError}super();if(f()!==this)throw Error('receiver')}}new D();good&&a[0]===2`,
	} {
		for _, enabled := range []bool{false, true} {
			r := jitRuntimeForTest(t, Config{JIT: enabled})
			v, err := r.Run(compileForTest(t, source))
			if err != nil || !v.IsBool() || !v.Truthy() || enabled && (r.jit == nil || r.jit.entries == 0 || r.jit.rootCount != 0) {
				t.Fatalf("receiver JIT %v: %v, %v; %s", enabled, v, err, source)
			}
		}
	}
}

func TestJITPropertyFallbacks(t *testing.T) {
	for _, tc := range []struct{ name, source string }{
		{"reference", `function f(o,a){let v=o.x;a[0]+=1;return v}let x={};f({x},[0])===x`},
		{"negative zero", `function f(o,a,v){o.x=v;a[0]+=1;return o.x}let o={x:2};Object.is(f(o,[0],-0),-0)&&Object.is(o.x,-0)`},
		{"NaN", `function f(o,a,v){o.x=v;a[0]+=1;return o.x}let o={x:2};Number.isNaN(f(o,[0],NaN))&&Number.isNaN(o.x)`},
		{"readonly", `function f(o,a){o.x=7;a[0]+=1;return o.x}let o={x:3};Object.defineProperty(o,'x',{writable:false});f(o,[0])===3`},
		{"readonly strict", `function f(o,a){'use strict';a[0]+=1;o.x=7;return o.x}let a=[0],o={x:3};Object.defineProperty(o,'x',{writable:false});let good=false;try{f(o,a)}catch(e){good=e instanceof TypeError}good&&a[0]===1&&o.x===3`},
		{"accessor", `let hits=0;function f(o,a){o.x=7;a[0]+=1;return o.x}let o={get x(){hits++;return 3},set x(v){hits+=v}};f(o,[0])===3&&hits===8`},
		{"inherited", `function f(o,a){o.x=7;a[0]+=1;return o.x}let p={x:3},o=Object.create(p);f(o,[0])===7&&o.x===7&&p.x===3`},
		{"deleted", `function f(o,a){let v=o.x;a[0]+=1;return v}let o={x:3};delete o.x;f(o,[0])===undefined`},
		{"nonextensible", `function f(o,a){o.x=7;a[0]+=1;return o.x}let o={x:3};Object.preventExtensions(o);f(o,[0])===7`},
		{"large table", `function f(o,a){o.x=7;a[0]+=1;return o.x}let o={x:3,a:1,b:2,c:3,d:4,e:5,f:6,g:7,h:8,i:9};f(o,[0])===7&&o.i===9`},
		{"proxy", `let hits=0;function f(o,a){o.x=7;a[0]+=1;return o.x}let target={x:3},o=new Proxy(target,{get(t,k){hits++;return t[k]},set(t,k,v){hits++;t[k]=v;return true}});f(o,[0])===7&&hits===2`},
		{"alias mutation", `function f(o,p,a){o.x=7;a[0]+=1;return p.x}let o={x:3};f(o,o,[0])===7`},
		{"shape mutation", `function f(o,a,cb){o.x=7;cb(o);a[0]+=1;return o.x}let o={x:3};f(o,[0],o=>{for(let i=0;i<30;i++)o['p'+i]=i;Object.defineProperty(o,'x',{get(){return 11}})})===11`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, enabled := range []bool{false, true} {
				r := jitRuntimeForTest(t, Config{JIT: enabled})
				v, err := r.Run(compileForTest(t, tc.source))
				if err != nil || !v.IsBool() || !v.Truthy() || enabled && (r.jit == nil || r.jit.entries == 0 || r.jit.guards != 0 || r.jit.rootCount != 0) {
					t.Fatalf("property JIT %v: %v, %v", enabled, v, err)
				}
			}
		})
	}
}

func TestJITPropertyViewNotArray(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	v, err := r.Run(compileForTest(t, `function f(o,a){return o.length+a[0]}f({length:7},[3])===10`))
	if err != nil || !v.IsBool() || !v.Truthy() || r.jit == nil || r.jit.guards != 1 || r.jit.rootCount != 0 {
		t.Fatalf("ordinary property view granted array length access: %v, %v", v, err)
	}
}

func TestJITPropertyCallbackRelease(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	r.global.setOwnRaw(r.atoms.intern("host"), r.NewFunction("host", 0,
		func(rt *Runtime, _ Value, _ []Value) (Value, error) {
			if rt.jit == nil || rt.jit.rootCount != 0 {
				t.Fatal("callback retained native property views")
			}
			rt.releaseJIT()
			runtime.GC()
			return Undefined, nil
		}), propDefault)
	v, err := r.Run(compileForTest(t, `function f(o,a,cb){o.x=7;cb();a[0]+=1;return o.x+a[0]}let o={x:3};f(o,[0],()=>{host();delete o.x;o.y=9;o.x=11})===12&&o.y===9`))
	if err != nil || !v.IsBool() || !v.Truthy() || r.jit == nil || r.jit.rootCount != 0 || r.jit.guards != 0 {
		t.Fatalf("property view release: %v, %v", v, err)
	}
}

func TestJITRemainderBoundaries(t *testing.T) {
	for _, tc := range []struct{ left, right, check string }{
		{`13`, `5`, `v===3`}, {`-12`, `3`, `Object.is(v,-0)`},
		{`-0`, `2`, `Object.is(v,-0)`}, {`13.5`, `2.5`, `v===1`},
		{`1e100`, `3`, `v===1`}, {`Infinity`, `3`, `Number.isNaN(v)`},
		{`13`, `0`, `Number.isNaN(v)`}, {`13`, `Infinity`, `v===13`},
		{`'13'`, `5`, `v===3`}, {`13n`, `5n`, `v===3n`},
	} {
		r := jitRuntimeForTest(t, Config{JIT: true})
		if _, err := r.Run(compileForTest(t, `function f(a,x,y){a[0]+=1;return x%y}`)); err != nil {
			t.Fatal(err)
		}
		fn := compileForTest(t, `let v=f([1],`+tc.left+`,`+tc.right+`);`+tc.check)
		v, err := r.Run(fn)
		if err != nil || !v.IsBool() || !v.Truthy() || r.jit == nil || r.jit.guards != 0 || r.jit.rootCount != 0 {
			t.Fatalf("%s %% %s: %v, %v", tc.left, tc.right, v, err)
		}
	}
	for _, source := range []string{
		`function f(a,x){a[0]+=1;return x%3}f([1],14)===2`,
		`function f(a,x){a[0]+=1;let y=x%3;return y}f([1],14)===2`,
		`function f(a,x,y){a[0]+=1;return x%y}let good=false;try{f([1],1n,0n)}catch(e){good=e instanceof RangeError}good`,
		`function f(a,x,y){a[0]+=1;return x%y}let good=false;try{f([1],1n,2)}catch(e){good=e instanceof TypeError}good`,
	} {
		r := jitRuntimeForTest(t, Config{JIT: true})
		v, err := r.Run(compileForTest(t, source))
		if err != nil || !v.IsBool() || !v.Truthy() || r.jit == nil || r.jit.entries == 0 || r.jit.guards != 0 || r.jit.rootCount != 0 {
			t.Fatalf("remainder fused/error result: %v, %v", v, err)
		}
	}
}

func TestJITRemainderCoercionRelease(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	r.global.setOwnRaw(r.atoms.intern("host"), r.NewFunction("host", 0,
		func(rt *Runtime, _ Value, _ []Value) (Value, error) {
			if rt.jit == nil || rt.jit.rootCount != 0 {
				t.Fatal("remainder coercion retained native roots")
			}
			rt.releaseJIT()
			runtime.GC()
			return Undefined, nil
		}), propDefault)
	v, err := r.Run(compileForTest(t, `let order='';let a=[1];function f(a,x,y){a[0]+=1;let rem=x%y;return rem+a[0]+a.length}f(a,{valueOf(){order+='x';host();a.push(7);return 13}},{valueOf(){order+='y';return 5}})===7&&order==='xy'`))
	if err != nil || !v.IsBool() || !v.Truthy() || r.jit == nil || r.jit.rootCount != 0 || r.jit.guards != 0 {
		t.Fatalf("remainder callback order/release: %v, %v", v, err)
	}
}

func TestJITGlobalReads(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		r := jitRuntimeForTest(t, Config{JIT: enabled})
		calls := 0
		r.global.setOwnRaw(r.atoms.intern("host"), r.NewFunction("host", 0,
			func(rt *Runtime, _ Value, _ []Value) (Value, error) {
				calls++
				if enabled && rt.jit.rootCount != 0 {
					t.Fatal("global accessor retained native roots")
				}
				return Undefined, nil
			}), propDefault)
		for _, source := range []string{
			`globalThis.g=3;function f(a){return g+a[0]}f([1])===4`,
			`g=7;f([1])===8`,
			`Object.defineProperty(globalThis,'g',{get(){host();return 9},configurable:true});f([1])===10`,
			`delete globalThis.g;let good=false;try{f([1])}catch(e){good=e instanceof ReferenceError}good`,
			`globalThis.g=20;let g=13;f([1])===14`,
		} {
			v, err := r.Run(compileForTest(t, source))
			if err != nil || !v.IsBool() || !v.Truthy() {
				t.Fatalf("global read JIT %v: %v, %v", enabled, v, err)
			}
		}
		if calls != 1 || enabled && (r.jit == nil || r.jit.guards != 0 || r.jit.rootCount != 0) {
			t.Fatal("global resolution did not preserve accessors or lexical shadowing")
		}
	}
}

func TestJITNumericGlobalLoop(t *testing.T) {
	for _, keyword := range []string{"var", "let", "const"} {
		r := jitRuntimeForTest(t, Config{JIT: true})
		v, err := r.Run(compileForTest(t, keyword+` scale=3,offset=7;function f(a,n){for(let i=0;i<n;i++)a[i]=i*scale+offset;return a[n-1]}let a=[];for(let i=0;i<512;i++)a[i]=0;f(a,512)===1540&&a[0]===7`))
		if err != nil || !v.IsBool() || !v.Truthy() || r.jit == nil || r.jit.entries == 0 || r.jit.hosts != 0 || r.jit.guards != 0 || r.jit.rootCount != 0 {
			t.Fatalf("%s numeric bindings did not stay native: %v, %v", keyword, v, err)
		}
	}
}

func TestJITShortCountdown(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	v, err := r.Run(compileForTest(t, `function f(n,a){while(--n>=0)a[n]=(a[n]+1)|0;return a[0]}let a=[0,0,0,0];f(1,a);f(1,a);f(1,a);f(1,a)===4`))
	if err != nil || !v.IsBool() || !v.Truthy() || r.jit == nil || r.jit.entries != 0 {
		t.Fatalf("short countdown entered native execution: %v, %v", v, err)
	}
	cl := r.global.getOwn(r.atoms.intern("f")).value.Object().fn().closure
	if cl.jitEntry == nil || cl.jitEntry.shortCounter != 1 || cl.jitEntry.code == nil {
		t.Fatal("short countdown lost its reusable native program")
	}
	v, err = r.Run(compileForTest(t, `for(let i=0;i<2048;i++)f(1,a);a[0]===2052`))
	if err != nil || !v.IsBool() || !v.Truthy() || r.jit.entries != 0 {
		t.Fatalf("short countdown promoted at a back edge: %v, %v", v, err)
	}
	v, err = r.Run(compileForTest(t, `f(4,a)===2053&&a[1]===1&&a[2]===1&&a[3]===1`))
	if err != nil || !v.IsBool() || !v.Truthy() || r.jit.entries == 0 || r.jit.guards != 0 || r.jit.rootCount != 0 {
		t.Fatalf("longer countdown lost native execution: %v, %v", v, err)
	}
}

func TestJITNumericGlobalMutation(t *testing.T) {
	for _, source := range []string{
		`var g=1;function f(o,a,n){for(let i=0;i<n;i++){a[i]=g;o.g=g+1}return g+a[n-1]}let a=[];for(let i=0;i<128;i++)a[i]=0;f(globalThis,a,128)===257&&g===129&&a[0]===1`,
		`let g=3;function f(a,cb){a[0]+=g;cb();return g+a[0]}f([1],()=>{g=7})===11`,
		`globalThis.g=3;function f(a,cb){a[0]+=g;cb();return g+a[0]}let hits=0;f([1],()=>{for(let i=0;i<100;i++)globalThis['p'+i]=i;Object.defineProperty(globalThis,'g',{get(){hits++;return 9}})})===13&&hits===1`,
		`globalThis.g=3;function f(a,cb){a[0]+=g;cb();return g+a[0]}let a=[1],good=false;try{f(a,()=>{delete globalThis.g})}catch(e){good=e instanceof ReferenceError}good&&a[0]===4`,
		`function f(a){a[0]+=1;return g+a[0]}let a=[1],good=false;try{f(a)}catch(e){good=e instanceof ReferenceError}let g=7;good&&a[0]===2&&f([1])===9`,
		`let scale=3,offset=7;function f(a,n){'use strict';for(let i=0;i<n;i++)a[i]=this*scale+offset;return a[n-1]}let a=[];for(let i=0;i<128;i++)a[i]=0;f.call(2,a,128)===13`,
		`globalThis.g=3;function f(a,cb){a[0]+=g;cb();return g+a[0]}f([1],()=>{g={valueOf(){return 7}}})===11`,
	} {
		for _, enabled := range []bool{false, true} {
			r := jitRuntimeForTest(t, Config{JIT: enabled})
			v, err := r.Run(compileForTest(t, source))
			if err != nil || !v.IsBool() || !v.Truthy() || enabled && (r.jit == nil || r.jit.entries == 0 || r.jit.rootCount != 0) {
				t.Fatalf("binding mutation JIT %v: %v, %v; %s", enabled, v, err, source)
			}
		}
	}
}

func TestJITGlobalCacheOwner(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	if _, err := r.Run(compileForTest(t, `globalThis.g=3;function f(a){return g+a[0]}f([1])`)); err != nil {
		t.Fatal(err)
	}
	cl := r.global.getOwn(r.atoms.intern("f")).value.Object().fn().closure
	for _, in := range cl.fn.Code {
		if in.Op != bytecode.OpGetGlobal {
			continue
		}
		site := &cl.ic[in.B]
		site.p1 = r.NewObject()
		v, err := r.Run(compileForTest(t, `f([1])`))
		if err != nil || v.Number() != 4 || site.p1 != cl.scope() || r.jit.rootCount != 0 {
			t.Fatalf("global cache retained a stale environment: %v, %v", v, err)
		}
		return
	}
	t.Fatal("missing global cache site")
}

func TestJITCallbackTierSelection(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	if _, err := r.Run(compileForTest(t, `function cb(){};function f(a,n,cb){for(let i=0;i<n;i++){a[0]+=1;cb()}return a[0]}f([0],64,cb)`)); err != nil {
		t.Fatal(err)
	}
	for range 7 {
		if _, err := r.Run(compileForTest(t, `f([0],64,cb)`)); err != nil {
			t.Fatal(err)
		}
	}
	cl := r.global.getOwn(r.atoms.intern("f")).value.Object().fn().closure
	if cl.jitEntry == nil || !cl.jitEntry.entrySlow || cl.jitRefused {
		t.Fatal("callback-heavy entry did not stay eligible for loop promotion")
	}
	before := r.jit.osrs
	v, err := r.Run(compileForTest(t, `f([0],10000,cb)`))
	if err != nil || v.Number() != 10000 || r.jit.osrs == before || r.jit.guards != 0 || r.jit.rootCount != 0 {
		t.Fatalf("later long call did not promote its hot loop: %v, %v", v, err)
	}
}

func TestJITBoundaryDensityBudget(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	v, err := r.Run(compileForTest(t, `function f(a,n){let s=0;for(let i=0;i<n;i++){s+=a[0]%3;a[0]+=3}return s}let a=[2];f(a,20000)===40000&&a[0]===60002`))
	cl := r.global.getOwn(r.atoms.intern("f")).value.Object().fn().closure
	if err != nil || !v.IsBool() || !v.Truthy() || cl.jitEntry == nil || !cl.jitEntry.entrySlow || r.jit.budgets == 0 || r.jit.rootCount != 0 || r.jit.guards != 0 {
		t.Fatalf("boundary density changed effects or failed to select Go: %v, %v", v, err)
	}
}

func TestJITClosureCacheRelease(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	if _, err := r.Run(compileForTest(t, `function f(a){return a[0]+1}`)); err != nil {
		t.Fatal(err)
	}
	call := compileForTest(t, `f([2])`)
	cl := r.global.getOwn(r.atoms.intern("f")).value.Object().fn().closure
	for range 3 {
		v, err := r.Run(call)
		if err != nil || v.Number() != 3 || cl.jitEntry == nil || cl.jitEntry.code.Size() == 0 || r.jit.rootCount != 0 {
			t.Fatalf("closure cache entry: %v, %v", v, err)
		}
		e := cl.jitEntry
		r.releaseJIT()
		if r.jit != nil || e.code.Size() != 0 || e.code.MetadataSize() != 0 {
			t.Fatal("cached closure kept released executable storage")
		}
	}
}

func TestJITArrayGrowth(t *testing.T) {
	for _, tc := range []struct{ name, source string }{
		{"descending", `function f(a){for(let i=31;i>=0;i--)a[i]=i;return a[0]+a[31]+a.length}f([])===63`},
		{"aliases", `function f(a,b){let s=0;for(let i=0;i<300;i++){a[i]=i;s+=b[i]+b.length}return s}let a=[];f(a,a)===90000`},
		{"holes", `function f(a){a[4]=7;a[0]=3;a[2]=5;return a.length+a[0]+a[2]+a[4]}let a=[];f(a)===20&&!(1 in a)&&!(3 in a)`},
		{"reference barrier", `function f(a,v){a[0]=v;a[16]=v;return a.length}let v={x:9},a=[];f(a,v)===17&&a[0]===v&&a[16]===v`},
		{"readonly length", `function f(a){a[1]=9;return a.length}let a=[2];Object.defineProperty(a,'length',{writable:false});f(a)===1&&!(1 in a)`},
		{"nonextensible hole", `function f(a){a[0]=9;return a.length}let a=[,];Object.preventExtensions(a);f(a)===1&&!(0 in a)`},
		{"inherited readonly", `Object.defineProperty(Array.prototype,'2',{value:8,writable:false});function f(a){a[2]=9;return a.length}let a=[];f(a)===0&&a[2]===8`},
		{"inherited setter", `let hits=0;Object.defineProperty(Array.prototype,'2',{set(v){hits+=v}});function f(a){a[2]=9;return a.length}let a=[];f(a)===0&&hits===9&&!(Object.hasOwn(a,'2'))`},
		{"sparse gap", `function f(a){a[2000]=9;return a.length}let a=[];f(a)===2001&&a[2000]===9&&!(1999 in a)`},
		{"nonindex", `function f(a,k){a[k]=9;return a.length}let a=[];f(a,4294967295)===0&&a[4294967295]===9&&f(a,-1)===0&&a[-1]===9`},
		{"proxy", `let hits=0;function f(a){a[2]=9;return 7}let a=[];let p=new Proxy(a,{set(t,k,v){hits++;return Reflect.set(t,k,v)}});f(p)===7&&hits===1&&a[2]===9`},
		{"typed array", `function f(a){a[0]=257;return 7}let a=new Uint8Array(1);f(a)===7&&a[0]===1`},
		{"strict error", `function f(a){'use strict';a[1]=9;return 7}let a=[2];Object.freeze(a);let good=false;try{f(a)}catch(e){good=e instanceof TypeError}good&&a.length===1`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, enabled := range []bool{false, true} {
				r := jitRuntimeForTest(t, Config{JIT: enabled})
				v, err := r.Run(compileForTest(t, tc.source))
				if err != nil || !v.IsBool() || !v.Truthy() {
					t.Fatalf("JIT %v: %v, %v", enabled, v, err)
				}
				if enabled && (r.jit == nil || r.jit.entries == 0 || r.jit.rootCount != 0) {
					t.Fatal("array write did not enter native code or retained roots")
				}
				if enabled && (tc.name == "descending" || tc.name == "aliases" || tc.name == "holes") && r.jit.guards != 0 {
					t.Fatal("ordinary array growth permanently guarded")
				}
			}
		})
	}
}

func TestJITArrayHolePermissionCallbacks(t *testing.T) {
	for _, source := range []string{
		`let hits=0;function cb(){Object.defineProperty(Array.prototype,'1',{set(v){hits+=v}})}function f(a,cb){a[0]=2;cb();a[1]=3;return a.length}let a=new Array(2);f(a,cb)===2&&hits===3&&!Object.hasOwn(a,'1')&&a[0]===2`,
		`function cb(a){Object.defineProperty(a,'length',{writable:false});Object.preventExtensions(a)}function f(a,cb){a[0]=2;cb(a);a[1]=3;return a.length}let a=new Array(2);f(a,cb)===2&&!Object.hasOwn(a,'1')&&a[0]===2`,
		`function cb(a){a.length=40;a[39]=9}function f(a,b,cb){a[0]=2;cb(a);b[1]=3;return b[0]+b[1]+b[39]+b.length}let a=new Array(2);f(a,a,cb)===54`,
	} {
		r := jitRuntimeForTest(t, Config{JIT: true})
		v, err := r.Run(compileForTest(t, source))
		if err != nil || !v.IsBool() || !v.Truthy() || r.jit == nil || r.jit.guards != 0 || r.jit.rootCount != 0 {
			t.Fatalf("hole permission after callback: %v, %v", v, err)
		}
	}
	r := jitRuntimeForTest(t, Config{JIT: true})
	v, err := r.Run(compileForTest(t, `function f(a){for(let i=31;i>=0;i--)a[i]=i;return a[31]+a.length}f([])`))
	if err != nil || v.Number() != 63 || r.jit == nil || r.jit.hosts != 1 || r.jit.fastHosts != 1 || r.jit.guards != 0 {
		t.Fatalf("descending initialization did not fill holes natively: %v, %v", v, err)
	}
}

func TestJITArrayWriteCoercionRelease(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	calls := 0
	r.global.setOwnRaw(r.atoms.intern("host"), r.NewFunction("host", 0,
		func(rt *Runtime, _ Value, _ []Value) (Value, error) {
			calls++
			if rt.jit == nil || rt.jit.rootCount != 0 {
				t.Fatal("array coercion retained native roots")
			}
			for _, view := range rt.jit.arrays {
				if view.Data != nil {
					t.Fatal("array coercion retained borrowed storage")
				}
			}
			rt.releaseJIT()
			runtime.GC()
			return Undefined, nil
		}), propDefault)
	v, err := r.Run(compileForTest(t, `let a=[1];function f(a,k){a[k]=9;return a[0]+a[1]+a.length}f(a,{toString(){host();a.push(3);a[0]=5;return '1'}})`))
	if err != nil || v.Number() != 16 || calls != 1 || r.jit.rootCount != 0 {
		t.Fatalf("array coercion: %v, %v, calls %d", v, err, calls)
	}
}

func TestJITArrayGrowthLimits(t *testing.T) {
	for _, memory := range []bool{false, true} {
		cfg := Config{JIT: true}
		if memory {
			cfg.MemoryLimit = 8 << 20
		}
		r := jitRuntimeForTest(t, cfg)
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		if !memory {
			cancel()
			ctx, cancel = context.WithTimeout(context.Background(), 30*time.Millisecond)
		}
		defer cancel()
		r.SetContext(ctx)
		_, err := r.Run(compileForTest(t, `function f(a){let i=0;for(;;){a[i]=i;i++}}f([])`))
		want := error(context.DeadlineExceeded)
		if memory {
			want = ErrMemoryLimit
		}
		if !errors.Is(err, want) || r.jit == nil || r.jit.fastHosts == 0 || r.jit.budgets == 0 || r.jit.rootCount != 0 {
			t.Fatalf("array growth memory %v: %v", memory, err)
		}
	}
}

func TestJITReferenceReturns(t *testing.T) {
	for _, tc := range []struct{ expression, check string }{
		{`{x:9}`, `v.x===9`},
		{`[7,8]`, `v.length===2&&v[0]===7&&v[1]===8`},
		{`'kept'`, `v==='kept'`},
		{`12345678901234567890n`, `v===12345678901234567890n`},
		{`Symbol('kept')`, `typeof v==='symbol'&&v.description==='kept'`},
		{`function(){return 11}`, `v()===11`},
	} {
		r := jitRuntimeForTest(t, Config{JIT: true})
		if _, err := r.Run(compileForTest(t, `function f(a,out){a[0]+=1;return out}`)); err != nil {
			t.Fatal(err)
		}
		fn := compileForTest(t, `f([1],`+tc.expression+`)`)
		for range 12 {
			v, err := r.Run(fn)
			if err != nil {
				t.Fatal(err)
			}
			runtime.GC()
			r.global.setOwnRaw(r.atoms.intern("v"), v, propDefault)
			check, err := r.Run(compileForTest(t, tc.check))
			if err != nil || !check.IsBool() || !check.Truthy() {
				t.Fatalf("returned %s did not survive root clearing: %v, %v", tc.expression, check, err)
			}
		}
		if r.jit == nil || r.jit.guards != 0 || r.jit.rootCount != 0 {
			t.Fatal("reference return guarded or retained scratch roots")
		}
	}
}

func TestJITEqualityBoundaries(t *testing.T) {
	for _, tc := range []struct {
		left, op, right string
		want            bool
	}{
		{`null`, `==`, `undefined`, true}, {`null`, `===`, `undefined`, false},
		{`dda`, `==`, `null`, true}, {`undefined`, `==`, `dda`, true},
		{`dda`, `===`, `undefined`, false}, {`dda`, `!==`, `null`, true},
		{`{}`, `!=`, `null`, true}, {`false`, `==`, `null`, false},
		{`true`, `===`, `true`, true}, {`true`, `==`, `1`, true},
		{`'3'`, `==`, `3`, true}, {`3n`, `==`, `'3'`, true},
		{`3n`, `===`, `3`, false}, {`Symbol()`, `==`, `3`, false},
		{`NaN`, `===`, `NaN`, false}, {`-0`, `===`, `0`, true},
	} {
		for _, branch := range []bool{false, true} {
			r := jitRuntimeForTest(t, Config{JIT: true})
			dda := r.NewObject()
			dda.MarkHTMLDDA()
			r.global.setOwnRaw(r.atoms.intern("dda"), Obj(dda), propDefault)
			body := `return x` + tc.op + `y`
			if branch {
				body = `if(x` + tc.op + `y)return true;return false`
			}
			fn := compileForTest(t, `function f(a,x,y){a[0]+=1;`+body+`};f([1],`+tc.left+`,`+tc.right+`)`)
			for range 12 {
				v, err := r.Run(fn)
				if err != nil || !v.IsBool() || v.Truthy() != tc.want {
					t.Fatalf("%s %s %s branch %v: %v, %v; want %v", tc.left, tc.op, tc.right, branch, v, err, tc.want)
				}
			}
			if r.jit == nil || r.jit.entries == 0 || r.jit.guards != 0 || r.jit.rootCount != 0 {
				t.Fatal("equality did not resume native execution or retained roots")
			}
		}
	}
}

func TestJITEqualityBranchBudget(t *testing.T) {
	for _, value := range []string{`null`, `{}`} {
		r := jitRuntimeForTest(t, Config{JIT: true})
		v, err := r.Run(compileForTest(t, `function f(a,x){let n=0;for(let i=0;i<20000;i++){if(x!=null)n+=a[0]}return n}f([2],`+value+`)`))
		want := float64(40000)
		if value == `null` {
			want = 0
		}
		if err != nil || v.Number() != want || r.jit == nil || r.jit.budgets == 0 || r.jit.guards != 0 {
			t.Fatalf("branch budget %s: %v, %v", value, v, err)
		}
	}
}

func TestJITEqualityCoercionRelease(t *testing.T) {
	for _, branch := range []bool{false, true} {
		r := jitRuntimeForTest(t, Config{JIT: true})
		calls := 0
		r.global.setOwnRaw(r.atoms.intern("host"), r.NewFunction("host", 0,
			func(rt *Runtime, _ Value, _ []Value) (Value, error) {
				calls++
				if rt.jit == nil || rt.jit.rootCount != 0 {
					t.Fatal("equality coercion retained native roots")
				}
				for _, view := range rt.jit.arrays {
					if view.Data != nil {
						t.Fatal("equality coercion retained borrowed storage")
					}
				}
				rt.releaseJIT()
				runtime.GC()
				return Undefined, nil
			}), propDefault)
		body := `let same=o==7;return same?a[0]+a.length:0`
		if branch {
			body = `if(o!=7)return 0;return a[0]+a.length`
		}
		v, err := r.Run(compileForTest(t, `let a=[1];let o={valueOf(){host();a.push(9);a[0]=5;return 7}};function f(a,o){a[0]+=1;`+body+`}f(a,o)`))
		if err != nil || v.Number() != 7 || calls != 1 || r.jit == nil || r.jit.rootCount != 0 {
			t.Fatalf("equality callback branch %v: %v, %v, calls %d", branch, v, err, calls)
		}
	}
	for _, body := range []string{`return o==7`, `if(o!=7)return 0;return 1`} {
		var baseline string
		for _, enabled := range []bool{false, true} {
			r := jitRuntimeForTest(t, Config{JIT: enabled})
			_, err := r.Run(compileForTest(t, `function f(a,o){a[0]+=1;`+body+`}f([1],{valueOf(){throw new RangeError('compare')}})`))
			if err == nil {
				t.Fatal("equality coercion did not throw")
			}
			if !enabled {
				baseline = err.Error()
			} else if err.Error() != baseline {
				t.Fatalf("equality error %v; want %s", err, baseline)
			}
		}
	}
}

func TestJITBitwiseAndReceiver(t *testing.T) {
	for _, tc := range []struct{ left, op, right, want string }{
		{"4294967297", "|", "2", "3"},
		{"1e20", "|", "0", "1661992960"},
		{"9223372036854777856", "|", "0", "2048"},
		{"-9223372036854777856", "|", "0", "-2048"},
		{"NaN", "^", "Infinity", "0"},
		{"-1", ">>>", "0", "4294967295"},
		{"-2147483648", ">>>", "-1", "1"},
		{"4294967295", "<<", "33", "-2"},
		{"-8", ">>", "4294967297", "-4"},
		{"3.9", "&", "-1.9", "3"},
		{"2147483648", "|", "1", "-2147483647"},
		{"1", "<<", "-1", "-2147483648"},
		{"1", "<<", "NaN", "1"},
	} {
		for _, trees := range []bool{false, true} {
			previous := treeTier.Swap(trees)
			r := jitRuntimeForTest(t, Config{JIT: true})
			source := `let obj={array:[` + tc.left + `],f:function(a){return this.array[0]` + tc.op + `a[0]}};obj.f([` + tc.right + `])`
			v, err := r.Run(compileForTest(t, source))
			treeTier.Store(previous)
			if err != nil {
				t.Fatal(err)
			}
			s, err := r.toString(v)
			if err != nil || s.Go() != tc.want {
				t.Fatalf("%s %s %s: %v, %v; want %s", tc.left, tc.op, tc.right, v, err, tc.want)
			}
			if r.jit == nil || r.jit.entries == 0 || r.jit.guards != 0 || r.jit.hosts != 1 || r.jit.rootCount != 0 {
				t.Fatalf("did not exercise receiver/property and native bitwise execution")
			}
		}
	}
}

func TestJITReceiverCallbacks(t *testing.T) {
	for _, tc := range []struct {
		source, want string
		guard        bool
	}{
		{`let hits=0,obj={get array(){hits++;return [9]}};function f(a){return this.array[0]+a[0]}let v=f.call(obj,[2]);v*10+hits`, "111", false},
		{`let obj=new Proxy({array:[9]},{get(t,k){return [7]}});function f(a){return this.array[0]+a[0]}f.call(obj,[2])`, "9", false},
		{`let events='';function f(a,b){let s=0;for(let i=0;i<2;i++)s+=a|b;return s}let v=f({valueOf(){events+='a';return 3}},{valueOf(){events+='b';return 4}});v+':'+events`, "14:abab", true},
		{`function f(a,b){let s=a[0]&b[0];return s}f([3n],[2n])`, "2", true},
	} {
		r := jitRuntimeForTest(t, Config{JIT: true})
		v, err := r.Run(compileForTest(t, tc.source))
		if err != nil {
			t.Fatal(err)
		}
		s, err := r.toString(v)
		if err != nil || s.Go() != tc.want {
			t.Fatalf("result %v error %v, want %s", v, err, tc.want)
		}
		if r.jit == nil || r.jit.entries == 0 || (r.jit.guards > 0) != tc.guard || r.jit.rootCount != 0 {
			t.Fatal("did not exercise native callbacks or expected guard")
		}
	}
	for _, source := range []string{
		`function f(a){return this.array[0]+a[0]}f.call(null,[1])`,
		`let obj={get array(){throw new RangeError('getter')}};function f(a){return this.array[0]+a[0]}f.call(obj,[1])`,
		`function f(a,b){let s=0;for(let i=0;i<2;i++)s+=a>>>b;return s}f(3n,1n)`,
	} {
		var want string
		for _, enabled := range []bool{false, true} {
			r := jitRuntimeForTest(t, Config{JIT: enabled})
			_, err := r.Run(compileForTest(t, source))
			if err == nil {
				t.Fatal("expected exception")
			}
			if !enabled {
				want = err.Error()
			} else if err.Error() != want || r.jit == nil || r.jit.entries == 0 || r.jit.rootCount != 0 {
				t.Fatalf("native exception %v, want %s", err, want)
			}
		}
	}
}

func TestJITDataProperties(t *testing.T) {
	for _, tc := range []struct {
		source string
		want   float64
	}{
		{`let o={x:0,f:function(a){let v=this.x=a[0];return v+this.x}};o.f([7])`, 14},
		{`let o={f:function(a){let v=this.x=a[0];return v+this.x}};o.f([7])`, 14},
		{`let o=Object.freeze({x:0});function f(a){let v=this.x=a[0];return v+this.x}f.call(o,[7])`, 7},
		{`let a=[2],o={set x(v){a[0]=v+1;a.push(9)}};function f(a){let v=this.x=a[0];return v+a[0]+a.length}f.call(o,a)`, 7},
		{`let o=new Proxy({x:0},{set(t,k,v){t[k]=v+1;return true}});function f(a){let v=this.x=a[0];return v+this.x}f.call(o,[7])`, 15},
		{`let o={x:[1]};function f(a){this.x=a;return this.x[0]+a[0]}f.call(o,[7])`, 14},
	} {
		for _, trees := range []bool{false, true} {
			previous := treeTier.Swap(trees)
			r := jitRuntimeForTest(t, Config{JIT: true})
			v, err := r.Run(compileForTest(t, tc.source))
			treeTier.Store(previous)
			if err != nil || !v.IsNumber() || v.Number() != tc.want {
				t.Fatalf("data properties: %v, %v; want %v", v, err, tc.want)
			}
			if r.jit == nil || r.jit.entries == 0 || r.jit.rootCount != 0 {
				t.Fatal("data property test never entered native code or retained roots")
			}
		}
	}
	for _, source := range []string{
		`let o=Object.freeze({x:0});function f(a){'use strict';this.x=a[0];return this.x}f.call(o,[7])`,
		`let o={set x(v){throw new TypeError('setter')}};function f(a){this.x=a[0];return 1}f.call(o,[7])`,
	} {
		var want string
		for _, enabled := range []bool{false, true} {
			r := jitRuntimeForTest(t, Config{JIT: enabled})
			_, err := r.Run(compileForTest(t, source))
			if err == nil {
				t.Fatal("expected property write exception")
			}
			if !enabled {
				want = err.Error()
			} else if err.Error() != want || r.jit == nil || r.jit.entries == 0 || r.jit.rootCount != 0 {
				t.Fatalf("property write error %v, want %s", err, want)
			}
		}
	}
}

func TestJITDataPropertyRootLimit(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	// Repeated reads reuse a rooted handle rather than filling the root table.
	v, err := r.Run(compileForTest(t, `function f(o,n){let s=0;for(let i=0;i<n;i++){let a=o.array;for(let j=0;j<16;j++)s+=a[0]}return s}f({array:[1]},2000)`))
	if err != nil || v.Number() != 32000 {
		t.Fatalf("root limit: %v, %v", v, err)
	}
	if r.jit == nil || r.jit.hosts != 0 || r.jit.budgets == 0 || r.jit.rootCount != 0 {
		t.Fatal("property loop did not reuse its bounded root table")
	}
}

func TestJITReferenceFields(t *testing.T) {
	for _, source := range []string{
		`function f(o,n){let s=0;for(let i=0;i<n;i++)s+=o.array[0];return s}f({array:[3]},1000)===3000`,
		`function f(o,n){let s=0;for(let i=0;i<n;i++)s+=o.m.array[0];return s}f({m:{array:[3]}},1000)===3000`,
		`function f(o,n){let s=0;for(let i=0;i<n;i++){s+=o.array[0];o.array[0]++}return s}let a=[3];f({array:a},1000)===502500&&a[0]===1003`,
		`function f(o,n){for(let i=0;i<n;i++)o.x=(o.x+o.array[0])|0;return o.x+0}let o={array:[3],x:7};f(o,1000)===3007&&o.x===3007`,
		`function f(o,n){let s=0;for(let i=0;i<n;i++)s+=o.next.array[0];return s}let o={array:[3]};o.next=o;f(o,1000)===3000`,
		`function f(n){let s=0;for(let i=0;i<n;i++)s+=this.array[0];return s}f.call({array:[3]},1000)===3000`,
		`var scale=7;function f(n){let s=0;for(let i=0;i<n;i++)s+=this.array[0]*scale;return s}f.call({array:[3]},1000)===21000`,
	} {
		for _, enabled := range []bool{false, true} {
			r := jitRuntimeForTest(t, Config{JIT: enabled})
			v, err := r.Run(compileForTest(t, source))
			if err != nil || !v.IsBool() || !v.Truthy() {
				t.Fatalf("JIT %v: %v, %v; %s", enabled, v, err, source)
			}
			if enabled && (r.jit == nil || r.jit.entries == 0 || r.jit.hosts != 0 || r.jit.guards != 0 || r.jit.rootCount != 0) {
				t.Fatalf("reference field left native execution: %s", source)
			}
			if enabled {
				for _, row := range r.jit.references.cells {
					for _, ref := range row {
						if ref.Cell != nil || ref.Reference != nil {
							t.Fatal("borrowed reference retained after return")
						}
					}
				}
			}
		}
	}
}

func TestJITReferenceFieldFallbacks(t *testing.T) {
	for _, source := range []string{
		`let hits=0;function f(o,n){let s=0;for(let i=0;i<n;i++)s+=o.array[0];return s}let o={get array(){hits++;return [3]}};f(o,10)===30&&hits===10`,
		`let hits=0;function f(o,n){let s=0;for(let i=0;i<n;i++)s+=o.array[0];return s}let o=new Proxy({array:[3]},{get(t,k){hits++;return t[k]}});f(o,10)===30&&hits===10`,
		`function f(o,n){let s=0;for(let i=0;i<n;i++)s+=o.array[0];return s}f(Object.create({array:[3]}),100)===300`,
		`function f(o,n){let s=0;for(let i=0;i<n;i++)s+=o.array[0];return s}let o={array:[3]};Object.defineProperty(o,'array',{writable:false});f(o,100)===300`,
		`function f(o,n){let s=0;for(let i=0;i<n;i++)s+=o.array[0];return s}let o={array:[3]};delete o.array;let good=false;try{f(o,10)}catch(e){good=e instanceof TypeError}good`,
		`function f(o,n,cb){cb();let s=0;for(let i=0;i<n;i++)s+=o.array[0];return s}let o={array:[3]};f(o,100,()=>{o.array=[7]})===700`,
		`function f(o,n,b){let a=o.array,s=0;for(let i=0;i<n;i++)s+=a[i];o.array=b;for(let i=0;i<n;i++)s+=o.array[0];return s}let o={array:[3,3,3]};f(o,3,[7])===30&&o.array[0]===7`,
		`function f(o,n){let s=0;for(let i=0;i<n;i++){s+=o.array[0];o=o.next}return s}let o={array:[1]},head=o;for(let i=0;i<300;i++){o.next={array:[1]};o=o.next}f(head,300)===300`,
	} {
		for _, enabled := range []bool{false, true} {
			r := jitRuntimeForTest(t, Config{JIT: enabled})
			v, err := r.Run(compileForTest(t, source))
			if err != nil || !v.IsBool() || !v.Truthy() {
				t.Fatalf("fallback JIT %v: %v, %v; %s", enabled, v, err, source)
			}
			if enabled && (r.jit == nil || r.jit.entries == 0 || r.jit.rootCount != 0) {
				t.Fatalf("missing native fallback: %s", source)
			}
		}
	}
}

func TestJITReferenceCallbackRelease(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	calls := 0
	r.global.setOwnRaw(r.atoms.intern("host"), r.NewFunction("host", 0, func(rt *Runtime, _ Value, _ []Value) (Value, error) {
		calls++
		if rt.jit != nil {
			if rt.jit.rootCount != 0 {
				t.Fatal("callback retained native roots")
			}
			if rt.jit.references != nil {
				for _, view := range rt.jit.arrays {
					if view.Data != nil {
						t.Fatal("callback retained reference permissions")
					}
				}
			}
		}
		rt.releaseJIT()
		runtime.GC()
		return Undefined, nil
	}), propDefault)
	v, err := r.Run(compileForTest(t, `function f(o,n,cb){let s=0;for(let i=0;i<n;i++)s+=o.array[0];cb();for(let i=0;i<n;i++)s+=o.array[0];return s}let o={array:[3]};f(o,100,()=>{host();o.array=[7]})===1000`))
	if err != nil || !v.IsBool() || !v.Truthy() || calls != 1 || r.jit == nil || r.jit.rootCount != 0 {
		t.Fatalf("reference callback: %v, %v, calls %d", v, err, calls)
	}
}

func TestJITReferenceCancellation(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	p := compileForTest(t, `function forever(o){let s=0;for(;;)s+=o.array[0]}forever({array:[1]})`)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	r.SetContext(ctx)
	_, err := r.Run(p)
	if !errors.Is(err, context.DeadlineExceeded) || r.jit == nil || r.jit.entries == 0 || r.jit.budgets == 0 || r.jit.hosts != 0 || r.jit.rootCount != 0 {
		t.Fatalf("reference loop cancellation: %v", err)
	}
	for _, view := range r.jit.arrays {
		if view.Data != nil {
			t.Fatal("cancel retained reference permissions")
		}
	}
}

func TestJITReferenceMemoryRefusal(t *testing.T) {
	for _, limit := range []int64{256 << 10, 32 << 20} {
		r := jitRuntimeForTest(t, Config{JIT: true, MemoryLimit: limit})
		v, err := r.Run(compileForTest(t, `function f(o,n){let s=0;for(let i=0;i<n;i++)s+=o.array[0];return s}f({array:[3]},1000)`))
		if err != nil || v.Number() != 3000 {
			t.Fatalf("reference memory refusal: %v, %v", v, err)
		}
		if limit == 256<<10 {
			if r.jitCodeBytes() != 0 || r.jit != nil && r.jit.references != nil {
				t.Fatal("prepared native references beyond optional allowance")
			}
		} else {
			if r.jit == nil || r.jit.references == nil || r.jit.entries == 0 || r.jit.hosts != 0 {
				t.Fatal("reference loop did not enter native code")
			}
			if r.jitCodeBytes() < int64(unsafe.Sizeof(jitReferences{})) {
				t.Fatal("reference arena not charged to the JIT's budget")
			}
		}
	}
}

func TestJITCryptoCorpus(t *testing.T) {
	dir := os.Getenv("QUICKJS_JIT_V8_DIR")
	if dir == "" {
		t.Skip("set QUICKJS_JIT_V8_DIR to the external V8 v7 suite")
	}
	var source strings.Builder
	for _, name := range []string{"base.js", "crypto.js"} {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		source.Write(data)
		source.WriteByte('\n')
	}
	source.WriteString(`for(let i=0;i<8;i++){encrypt();decrypt()}true`)
	r := New(Config{JIT: true})
	t.Cleanup(func() { r.Close(); r.ReleaseClosed() })
	v, err := r.Run(compileForTest(t, source.String()))
	if err != nil || !v.IsBool() || !v.Truthy() {
		t.Fatalf("Crypto corpus: %v, %v", v, err)
	}
	prop := r.global.getOwn(r.atoms.intern("am3"))
	if !prop.value.IsObject() || prop.value.Object().fn() == nil {
		t.Fatal("missing am3 benchmark function")
	}
	fn := prop.value.Object().fn().closure.fn
	entry := r.jit.cache[weak.Make(fn)]
	if entry == nil || entry.code == nil || entry.misses != 0 || r.jit.entries == 0 || r.jit.fastHosts == 0 || r.jit.rootCount != 0 {
		t.Fatal("am3 did not execute native code without guards")
	}
	t.Logf("native entries=%d fast hosts=%d callback hosts=%d guards=%d budgets=%d transfers=%d am3 code=%d metadata=%d",
		r.jit.entries, r.jit.fastHosts, r.jit.hosts-r.jit.fastHosts, r.jit.guards, r.jit.budgets, r.jit.transfers, entry.code.Size(), entry.code.MetadataSize())
	for key, e := range r.jit.cache {
		if fn := key.Value(); fn != nil && e.calls && e.probeHosts != 0 {
			t.Logf("caller %s steps/host=%d slow=%v", fn.Name, e.probeSteps/e.probeHosts, e.entrySlow)
		}
	}
}

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
		{"int32 wrap", `function f(a){return a[0]|0}f([4294967297])`, 1, false},
		{"hole inherited read", `Array.prototype[0]=7;function f(a){return a[0]}f([,])`, 7, true},
		{"accessor", `function f(a){let i=0;return a[i++]+i*10}let a=[1];Object.defineProperty(a,'0',{get(){return 7}});f(a)`, 17, true},
		{"negative key", `function f(a){return a[-1]}let a=[1];a[-1]=9;f(a)`, 9, true},
		{"fractional key", `function f(a){return a[.5]}let a=[1];a[.5]=8;f(a)`, 8, true},
		{"proxy", `function f(a){return a[0]}f(new Proxy([1],{get(){return 8}}))`, 8, true},
		{"inherited setter", `let hits=0;Object.defineProperty(Array.prototype,'0',{set(v){hits+=v}});function f(a){a[0]=3;return hits}f([,])`, 3, false},
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

func TestJITPropertyReleaseAndGC(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	calls := 0
	r.global.setOwnRaw(r.atoms.intern("host"), r.NewFunction("host", 0,
		func(rt *Runtime, _ Value, _ []Value) (Value, error) {
			calls++
			if rt.jit == nil || rt.jit.rootCount != 0 {
				t.Fatal("property callback retained native roots")
			}
			for _, view := range rt.jit.arrays {
				if view.Data != nil {
					t.Fatal("property callback retained borrowed storage")
				}
			}
			rt.releaseJIT()
			runtime.GC()
			return Undefined, nil
		}), propDefault)
	v, err := r.Run(compileForTest(t, `
		function f(a,o){let s=0;for(let i=0;i<3;i++){s+=o.plain+a[0];o.x=s;s+=o.x}return s+a.length}
		let a=[1],o={plain:0,get x(){host();a.push(8);return a[0]},set x(v){host();a[0]=v+1}};
		f(a,o)`))
	if err != nil || !v.IsNumber() || v.Number() != 39 || calls != 6 || r.jit == nil || r.jit.rootCount != 0 {
		t.Fatalf("result %v err %v property callbacks %d", v, err, calls)
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

func TestJITNestedTreeOSR(t *testing.T) {
	for _, tail := range []bool{false, true} {
		r := jitRuntimeForTest(t, Config{JIT: true})
		r.jitCallThreshold = 100
		body := `let value=sum(10000); return value+7`
		want := float64(49995007)
		if tail {
			body = `return sum(10000)`
			want = 49995000
		}
		source := `function sum(n){var s=0;for(var i=0;i<n;i++)s+=i;return s}
			sum(2);sum(2);function outer(){` + body + `}outer()`
		v, err := r.Run(compileForTest(t, source))
		if err != nil || v.Number() != want {
			t.Fatalf("tail=%v: result %v error %v, want %v", tail, v, err, want)
		}
		if r.jit == nil || r.jit.osrs == 0 {
			t.Fatal("nested call never promoted its loop")
		}
		if r.frameDepth != 0 || r.stackTop != 0 {
			t.Fatalf("nested promotion leaked frames: depth %d stack %d", r.frameDepth, r.stackTop)
		}
		v, err = r.Run(compileForTest(t, `sum(3)+11`))
		if err != nil || v.Number() != 14 {
			t.Fatalf("subsequent call: %v, %v", v, err)
		}
	}
}

func TestJITDebuggerFallback(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true, Debug: true})
	// Hidden host scripts need not contain debugger instructions. The runtime
	// still has debugger callbacks at interrupt checks, so it cannot borrow views.
	v, err := r.Run(compileForTest(t, jitSumSource))
	if err != nil || v.Number() != 49995000 {
		t.Fatalf("debugger fallback: %v, %v", v, err)
	}
	if r.jitEnabled || r.jit != nil {
		t.Fatal("debugger runtime allocated native state")
	}
}

func TestJITNestedTreeOSRThrow(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	r.jitCallThreshold = 100
	v, err := r.Run(compileForTest(t, `
		function sum(n,a){var s=0;for(var i=0;i<n;i++)s+=i;return s+a}
		sum(2,0);sum(2,0);
		function outer(){return sum(10000,{valueOf(){throw new RangeError('nested')}})+7}
		let result='';try{outer()}catch(e){result=e.name+':'+e.message}result`))
	if err != nil || !v.IsString() || v.String().Go() != "RangeError:nested" {
		t.Fatalf("nested exception: %v, %v", v, err)
	}
	if r.jit == nil || r.jit.osrs == 0 || r.jit.guards == 0 {
		t.Fatal("nested call did not promote and guard to coercion")
	}
	if r.frameDepth != 0 || r.stackTop != 0 {
		t.Fatalf("exception leaked frames: depth %d stack %d", r.frameDepth, r.stackTop)
	}
	v, err = r.Run(compileForTest(t, `sum(3,11)`))
	if err != nil || v.Number() != 14 {
		t.Fatalf("call after exception: %v, %v", v, err)
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

func TestJITPropertyLoopCancellation(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	p := compileForTest(t, `function forever(o){for(;;){o.x=(o.x+1)|0}}forever({x:0})`)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	r.SetContext(ctx)
	_, err := r.Run(p)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("property loop cancellation: %v", err)
	}
	if r.jit == nil || r.jit.entries == 0 || r.jit.hosts != 0 || r.jit.budgets == 0 || r.jit.rootCount != 0 {
		t.Fatal("native property loop did not check its budget")
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

// Halt stops a script at its next call or backward jump: the straight-line
// code after the host call that halted it still runs, and nothing after. A
// native loop must stop where the interpreter does, whether the host exit is
// an ordinary call (the call coordinator) or an accessor (the plain loop).
func TestJITHaltAfterHostExit(t *testing.T) {
	for _, tc := range []struct{ name, source string }{
		{"call", `var arr = [0,0,0,0,0,0,0,0,0,0];
			function f(a) { for (let i = 0; i < a.length; i++) { a[i] = 1; stop(i) } return 0 }
			f(arr)`},
		{"accessor", `var arr = [0,0,0,0,0,0,0,0,0,0], n = 0;
			var o = { get x() { stop(n++); return 1 } };
			function f(o, a) { let s = 0; for (let i = 0; i < a.length; i++) { a[i] = 1; s += o.x } return s }
			f(o, arr)`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			run := func(jit bool) (string, *Runtime) {
				r := jitRuntimeForTest(t, Config{JIT: jit})
				r.global.setOwnRaw(r.atoms.intern("stop"), r.NewFunction("stop", 1,
					func(rt *Runtime, _ Value, args []Value) (Value, error) {
						if len(args) > 0 && args[0].Number() == 3 {
							rt.Halt(context.Canceled)
						}
						return Undefined, nil
					}), propDefault)
				if _, err := r.Run(compileForTest(t, tc.source)); !errors.Is(err, context.Canceled) {
					t.Fatalf("jit=%v: run = %v, want context.Canceled", jit, err)
				}
				r.ClearStop()
				v, err := r.Run(compileForTest(t, `arr.join("")`))
				if err != nil {
					t.Fatal(err)
				}
				return v.String().Go(), r
			}
			want, _ := run(false)
			got, r := run(true)
			if r.jit == nil || r.jit.entries == 0 || r.jit.hosts == 0 {
				t.Fatal("loop never reached a native host exit")
			}
			if got != want || want != "1111000000" {
				t.Fatalf("after Halt: native %q, interpreter %q, want 1111000000", got, want)
			}
		})
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
	// Native code is not the script's memory: the script's measure, and so
	// whether it fails with ErrMemoryLimit, is the same with the JIT on and off.
	with := large.meter.walk(large)
	large.releaseJIT()
	if without := large.meter.walk(large); with != without {
		t.Fatalf("meter counted native code: %d bytes with it, %d without", with, without)
	}
	if int64(large.jitBudget()) != large.meter.limit/8 {
		t.Fatalf("JIT budget %d under a %d-byte limit", large.jitBudget(), large.meter.limit)
	}
}

// Compiling must not measure the heap: before, every attempt walked it, and
// a refusal retried every warmup, so a memory-limited runtime walked its whole
// heap every few calls of each hot function.
func TestJITNoHeapWalks(t *testing.T) {
	walks := func(jit bool, limit int64) uint16 {
		r := jitRuntimeForTest(t, Config{JIT: jit, MemoryLimit: limit})
		v, err := r.Run(compileForTest(t, `function sum(n) { let s=0; for(let i=0;i<n;i++) s+=i; return s } sum`))
		if err != nil {
			t.Fatal(err)
		}
		before := r.meter.epoch
		for i := 0; i < 200; i++ {
			if got, err := r.Call(v, Undefined, []Value{Int32(10)}); err != nil || got.Number() != 45 {
				t.Fatalf("sum = %v, %v", got, err)
			}
		}
		if jit && limit > 1<<20 && (r.jit == nil || r.jit.entries == 0) {
			t.Fatal("the function never ran natively")
		}
		return r.meter.epoch - before
	}
	// A small limit refuses every attempt; a large one compiles.
	for _, limit := range []int64{256 << 10, 32 << 20} {
		if off, on := walks(false, limit), walks(true, limit); on != off {
			t.Fatalf("limit %d: %d heap walks with the JIT, %d without", limit, on, off)
		}
	}
}

// A program refused for want of code budget is not recompiled at every
// warmup: the refusal is cached until the cache has released code.
func TestJITBudgetRefusalWaitsForRelease(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true, MemoryLimit: 1 << 20})
	var refused *bytecode.Function
	var deferred *jitEntry
	for i := 0; i < 64 && deferred == nil; i++ {
		fn := jitFunctionForTest(t, fmt.Sprintf(`function f(n) { let s=0; for(let i=0;i<n;i++) s+=i*%d; return s }`, i+1))
		if e := r.jitFor(fn); e != nil && e.deferred {
			refused, deferred = fn, e
		}
	}
	if deferred == nil || deferred.code != nil {
		t.Fatalf("no refusal for want of budget within %d bytes", r.jitBudget())
	}
	if e := r.jitFor(refused); e != deferred {
		t.Fatal("a budget refusal was retried before any code was released")
	}
	for key, e := range r.jit.cache {
		if e.code != nil && key.Value() != refused {
			if !r.jit.dropEntry(key, e) {
				t.Fatal("could not release code")
			}
			break
		}
	}
	if e := r.jitFor(refused); e == nil || e == deferred || e.code == nil {
		t.Fatal("a budget refusal was not retried once code was released")
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
			// a swap. Native return must decode that exact rooted operand.
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
			if r.jit == nil || r.jit.osrs != 1 || r.jit.guards != 0 || r.jit.rootCount != 0 {
				t.Fatal("live operand was lost, guarded, or retained")
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

// The external corpus keeps Tom Wu's license with its implementation. This
// benchmark executes its actual am3 method, and checks every output limb using
// an independent integer multiply/add model after stopping the timer.
func BenchmarkJITCryptoWorkload(b *testing.B) {
	dir := os.Getenv("QUICKJS_JIT_V8_DIR")
	if dir == "" {
		b.Skip("set QUICKJS_JIT_V8_DIR to the external V8 v7 suite")
	}
	var source strings.Builder
	for _, name := range []string{"base.js", "crypto.js"} {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			b.Fatal(err)
		}
		source.Write(data)
		source.WriteByte('\n')
	}
	compile := func(text string) *bytecode.Function {
		ast, err := parser.Parse(text, parser.Options{})
		if err != nil {
			b.Fatal(err)
		}
		fn, err := compiler.Compile(ast, compiler.Options{})
		if err != nil {
			b.Fatal(err)
		}
		return fn
	}
	// The external decrypt function asserts the recovered plaintext, so every
	// timed pair validates the whole RSA path, not only its limb kernel.
	for _, mode := range []string{"interpreter", "existing", "native"} {
		b.Run(mode, func(b *testing.B) {
			previous := treeTier.Swap(mode != "interpreter")
			defer treeTier.Store(previous)
			// Bytecode caches its first tree decision; each tier needs fresh code.
			setup := compile(source.String())
			call := compile(`encrypt();decrypt();true`)
			r := New(Config{JIT: mode == "native"})
			defer func() { r.Close(); r.ReleaseClosed() }()
			if _, err := r.Run(setup); err != nil {
				b.Fatal(err)
			}
			run := func() {
				if v, err := r.Run(call); err != nil || !v.IsBool() || !v.Truthy() {
					b.Fatalf("Crypto pair: %v, %v", v, err)
				}
			}
			for range jitHotCalls {
				run()
			}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				run()
			}
			b.StopTimer()
			if mode == "native" {
				cl := r.global.getOwn(r.atoms.intern("am3")).value.Object().fn().closure
				e := r.jit.cache[weak.Make(cl.fn)]
				if e == nil || e.code == nil || e.misses != 0 || r.jit.rootCount != 0 {
					b.Fatal("Crypto limb method did not stay native")
				}
				b.ReportMetric(float64(r.jitCodeBytes()), "code+metadata-B")
			}
		})
	}
}

func BenchmarkJITCryptoLimb(b *testing.B) {
	dir := os.Getenv("QUICKJS_JIT_V8_DIR")
	if dir == "" {
		b.Skip("set QUICKJS_JIT_V8_DIR to the external V8 v7 suite")
	}
	var source strings.Builder
	for _, name := range []string{"base.js", "crypto.js"} {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			b.Fatal(err)
		}
		source.Write(data)
		source.WriteByte('\n')
	}
	for _, n := range []int{1, 4, 16, 32, 8192} {
		for _, mode := range []string{"interpreter", "existing", "native"} {
			b.Run(fmt.Sprintf("n%d/%s", n, mode), func(b *testing.B) {
				previous := treeTier.Swap(mode != "interpreter")
				defer treeTier.Store(previous)
				compile := func(text string) *bytecode.Function {
					ast, err := parser.Parse(text, parser.Options{})
					if err != nil {
						b.Fatal(err)
					}
					fn, err := compiler.Compile(ast, compiler.Options{})
					if err != nil {
						b.Fatal(err)
					}
					return fn
				}
				setup := compile(source.String() + fmt.Sprintf(`var lhs={array:new Array(%d)},rhs={array:new Array(%d)};
					for(var i=0;i<%d;i++){lhs.array[i]=(i*1664525+1013904223)&268435455;rhs.array[i]=0}`, n, n, n))
				call := compile(fmt.Sprintf(`am3.call(lhs,0,67123451,rhs,0,0,%d)`, n))
				r := New(Config{JIT: mode == "native"})
				defer func() { r.Close(); r.ReleaseClosed() }()
				if _, err := r.Run(setup); err != nil {
					b.Fatal(err)
				}
				input, output := make([]uint32, n), make([]uint32, n)
				for i := range input {
					input[i] = uint32(uint64(i)*1664525+1013904223) & 268435455
				}
				advance := func() uint64 {
					carry := uint64(0)
					for i, x := range input {
						total := uint64(x)*67123451 + uint64(output[i]) + carry
						output[i] = uint32(total & 268435455)
						carry = total >> 28
					}
					return carry
				}
				for range jitHotCalls {
					v, err := r.Run(call)
					if err != nil || v.Number() != float64(advance()) {
						b.Fatalf("warm carry: %v, %v", v, err)
					}
				}
				var last Value
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					var err error
					last, err = r.Run(call)
					if err != nil || !last.IsNumber() {
						b.Fatalf("carry: %v, %v", last, err)
					}
				}
				b.StopTimer()
				var want uint64
				for i := 0; i < b.N; i++ {
					want = advance()
				}
				if last.Number() != float64(want) {
					b.Fatalf("carry %v, want %v", last.Number(), want)
				}
				obj := r.global.getOwn(r.atoms.intern("rhs")).value.Object()
				array, ok := plainOwn(obj, r.atoms.intern("array"))
				if !ok || len(array.Object().elems) != n {
					b.Fatal("missing output limbs")
				}
				for i, want := range output {
					if got := array.Object().elems[i]; !got.IsNumber() || got.Number() != float64(want) {
						b.Fatalf("limb %d: %v, want %v", i, got, want)
					}
				}
				if mode == "native" {
					if r.jit == nil || n > 1 && r.jit.entries == 0 || r.jit.guards != 0 || r.jit.rootCount != 0 {
						b.Fatal("limb loop did not stay native")
					}
					b.ReportMetric(float64(r.jitCodeBytes()), "code+metadata-B")
				}
			})
		}
	}
}

func BenchmarkJITNumericFields(b *testing.B) {
	for _, mode := range []string{"interpreter", "existing", "native"} {
		b.Run(mode, func(b *testing.B) {
			previous := treeTier.Swap(mode != "interpreter")
			defer treeTier.Store(previous)
			compile := func(source string) *bytecode.Function {
				ast, err := parser.Parse(source, parser.Options{})
				if err != nil {
					b.Fatal(err)
				}
				fn, err := compiler.Compile(ast, compiler.Options{})
				if err != nil {
					b.Fatal(err)
				}
				return fn
			}
			setup := compile(`function fields(o,n){let s=0;for(let i=0;i<n;i++){s=(s+o.x)|0;o.x=o.x+1}return s}var fieldObject={x:3}`)
			call := compile(`fields(fieldObject,4096)`)
			r := New(Config{JIT: mode == "native"})
			defer func() { r.Close(); r.ReleaseClosed() }()
			if _, err := r.Run(setup); err != nil {
				b.Fatal(err)
			}
			x := uint64(3)
			run := func() {
				want := float64(int32(uint32(x*4096 + 4096*4095/2)))
				v, err := r.Run(call)
				if err != nil || !v.IsNumber() || v.Number() != want {
					b.Fatalf("fields: %v, %v; want %v", v, err, want)
				}
				x += 4096
			}
			for range jitHotCalls {
				run()
			}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				run()
			}
			b.StopTimer()
			if mode == "native" {
				if r.jit == nil || r.jit.hosts != 0 || r.jit.guards != 0 || r.jit.rootCount != 0 {
					b.Fatal("numeric fields left native execution")
				}
				b.ReportMetric(float64(r.jitCodeBytes()), "code+metadata-B")
			}
		})
	}
}

func BenchmarkJITNumericGlobals(b *testing.B) {
	for _, mode := range []string{"interpreter", "existing", "native"} {
		b.Run(mode, func(b *testing.B) {
			previous := treeTier.Swap(mode != "interpreter")
			defer treeTier.Store(previous)
			compile := func(source string) *bytecode.Function {
				ast, err := parser.Parse(source, parser.Options{})
				if err != nil {
					b.Fatal(err)
				}
				fn, err := compiler.Compile(ast, compiler.Options{})
				if err != nil {
					b.Fatal(err)
				}
				return fn
			}
			setup := compile(`var scale=3,offset=7;function globals(n){let s=0;for(let i=0;i<n;i++)s+=i*scale+offset;return s}`)
			call := compile(`globals(4096)`)
			r := New(Config{JIT: mode == "native"})
			defer func() { r.Close(); r.ReleaseClosed() }()
			if _, err := r.Run(setup); err != nil {
				b.Fatal(err)
			}
			run := func() {
				v, err := r.Run(call)
				const want = 3*4096*4095/2 + 7*4096
				if err != nil || !v.IsNumber() || v.Number() != want {
					b.Fatalf("globals: %v, %v; want %d", v, err, want)
				}
			}
			for range jitHotCalls {
				run()
			}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				run()
			}
			b.StopTimer()
			if mode == "native" {
				if r.jit == nil || r.jit.entries == 0 || r.jit.hosts != 0 || r.jit.guards != 0 || r.jit.rootCount != 0 {
					b.Fatal("numeric globals left native execution")
				}
				b.ReportMetric(float64(r.jitCodeBytes()), "code+metadata-B")
			}
		})
	}
}

func BenchmarkJITReferenceFields(b *testing.B) {
	for _, mode := range []string{"interpreter", "existing", "native"} {
		b.Run(mode, func(b *testing.B) {
			previous := treeTier.Swap(mode != "interpreter")
			defer treeTier.Store(previous)
			compile := func(source string) *bytecode.Function {
				ast, err := parser.Parse(source, parser.Options{})
				if err != nil {
					b.Fatal(err)
				}
				fn, err := compiler.Compile(ast, compiler.Options{})
				if err != nil {
					b.Fatal(err)
				}
				return fn
			}
			setup := compile(`function references(o,n){let s=0;for(let i=0;i<n;i++)s+=o.m.array[0];return s}var referenceObject={m:{array:[3]}}`)
			call := compile(`references(referenceObject,4096)`)
			r := New(Config{JIT: mode == "native"})
			defer func() { r.Close(); r.ReleaseClosed() }()
			if _, err := r.Run(setup); err != nil {
				b.Fatal(err)
			}
			run := func() {
				v, err := r.Run(call)
				const want = 3 * 4096
				if err != nil || !v.IsNumber() || v.Number() != want {
					b.Fatalf("references: %v, %v; want %d", v, err, want)
				}
			}
			for range jitHotCalls {
				run()
			}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				run()
			}
			b.StopTimer()
			if mode == "native" {
				if r.jit == nil || r.jit.entries == 0 || r.jit.hosts != 0 || r.jit.guards != 0 || r.jit.rootCount != 0 {
					b.Fatal("reference fields left native execution")
				}
				b.ReportMetric(float64(r.jitCodeBytes()), "code+metadata-B")
			}
		})
	}
}

func BenchmarkJITReferenceFirstUse(b *testing.B) {
	compile := func(source string) *bytecode.Function {
		ast, err := parser.Parse(source, parser.Options{})
		if err != nil {
			b.Fatal(err)
		}
		fn, err := compiler.Compile(ast, compiler.Options{})
		if err != nil {
			b.Fatal(err)
		}
		return fn
	}
	setup := compile(`function references(o,n){let s=0;for(let i=0;i<n;i++)s+=o.m.array[0];return s}var referenceObject={m:{array:[3]}}`)
	call := compile(`references(referenceObject,4096)`)
	for _, enabled := range []bool{false, true} {
		mode := "existing"
		if enabled {
			mode = "automatic"
		}
		b.Run(mode, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			b.StopTimer()
			for range b.N {
				r := New(Config{JIT: enabled})
				if _, err := r.Run(setup); err != nil {
					b.Fatal(err)
				}
				b.StartTimer()
				v, err := r.Run(call)
				b.StopTimer()
				native := r.jit != nil && r.jit.entries != 0 && r.jit.hosts == 0 && r.jit.guards == 0
				r.Close()
				r.ReleaseClosed()
				if err != nil || v.Number() != 12288 || enabled && !native {
					b.Fatalf("first reference call: %v, %v, native %v", v, err, native)
				}
			}
		})
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
