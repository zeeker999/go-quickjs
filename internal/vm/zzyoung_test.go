//go:build quickjs_young

package vm

import (
	"fmt"
	"runtime"
	"testing"
	"unsafe"
	"weak"
)

func TestYoungGenerationReuse(t *testing.T) {
	h := newYoungHeap(2)
	a, b := h.allocate(nil, nil), h.allocate(nil, nil)
	child := newObject(nil, ClassObject)
	ref := weak.Make(child)
	a.props = append(a.props, Property{value: Obj(child)})
	h.markWord(uintptr(unsafe.Pointer(b)))
	h.collect()
	c := h.allocate(nil, nil)
	if c != a || len(c.props) != 0 || c.elems != nil || c.data != nil || c.weakMapRefs != nil {
		t.Fatal("dead object was not cleared and reused")
	}
	runtime.GC()
	if ref.Value() != nil {
		t.Fatal("recycled object retains a dead child")
	}
	if h.stats.Reused != 1 || h.stats.Reclaimed != 1 {
		t.Fatalf("stats: %+v", h.stats)
	}
	runtime.KeepAlive(h)
}

func TestYoungGenerationCyclesAndTenure(t *testing.T) {
	h := newYoungHeap(4)
	a, b := h.allocate(nil, nil), h.allocate(nil, nil)
	a.props = append(a.props, Property{value: Obj(b)})
	b.props = append(b.props, Property{value: Obj(a)})
	h.allocate(nil, nil)
	h.markWord(uintptr(unsafe.Pointer(&a.props[0].value)))
	h.collect()
	if h.stats.Young != 2 || h.stats.Reclaimed != 1 {
		t.Fatalf("cycle lost: %+v", h.stats)
	}
	c := h.allocate(nil, nil)
	b.props = append(b.props, Property{value: Obj(c)})
	h.markWord(uintptr(unsafe.Pointer(a)))
	h.collect()
	if h.stats.Young != 0 || h.stats.Promoted != 3 || a.props[0].value.Object() != b || b.props[1].value.Object() != c {
		t.Fatalf("tenure moved objects or left a young descendant: %+v", h.stats)
	}
	h.allocate(nil, nil)
	if a.props[0].value.Object() != b {
		t.Fatal("tenured allocation reused")
	}
}

func TestYoungGenerationDeadCycle(t *testing.T) {
	h := newYoungHeap(2)
	a, b := h.allocate(nil, nil), h.allocate(nil, nil)
	a.props = append(a.props, Property{value: Obj(b)})
	b.props = append(b.props, Property{value: Obj(a)})
	h.collect()
	if h.stats.Young != 0 || h.stats.Reclaimed != 2 {
		t.Fatalf("dead cycle retained: %+v", h.stats)
	}
}

func TestYoungGenerationPromotionReleasesOwnership(t *testing.T) {
	h := newYoungHeap(2)
	a, b := h.allocate(nil, nil), h.allocate(nil, nil)
	ref := weak.Make(b)
	h.promoteAll()
	runtime.GC()
	if ref.Value() != nil {
		t.Fatal("nursery retains promoted object")
	}
	if h.stats.Young != 0 || h.stats.Promoted != 2 || len(h.owners) != 0 {
		t.Fatalf("stats: %+v", h.stats)
	}
	runtime.KeepAlive(a)
	runtime.KeepAlive(h)
}

func youngInterpreterForTest(t *testing.T, capacity int) *Runtime {
	old := treeTier.Load()
	treeTier.Store(false)
	t.Cleanup(func() { treeTier.Store(old) })
	r := New(Config{YoungGeneration: capacity})
	t.Cleanup(func() { r.Close(); r.ReleaseClosed() })
	return r
}

func TestYoungGenerationInterpreter(t *testing.T) {
	for _, capacity := range []int{1, 2, 8, 64} {
		r := youngInterpreterForTest(t, capacity)
		v, err := r.Run(compileForTest(t, `function f(n){let s=0;for(let i=0;i<n;i++){let o={a:i,b:i+1};s+=o.a+o.b}return s}f(500)===250000`))
		st := r.YoungGenerationStats()
		t.Logf("capacity %d: %+v", capacity, st)
		if err != nil || !v.Truthy() || st.Reused == 0 || st.Collections == 0 || st.Young != 0 || st.Allocated > uint64(capacity+10) {
			t.Fatalf("capacity %d: %v, %v, %+v", capacity, v, err, st)
		}
	}
}

func TestYoungGenerationInterpreterEscapes(t *testing.T) {
	for _, source := range []string{
		`function churn(){for(let i=0;i<100;i++){let o={a:i,b:i+1}}}function f(){let o={a:17,b:19};churn();return o}let o=f();churn();o.a===17&&o.b===19`,
		`let saved;function f(){for(let i=0;i<100;i++){let o={a:i,b:i+1};if(i===3)saved=o}}f();saved.a===3&&saved.b===4`,
		`let old={};function f(){for(let i=0;i<100;i++){let o={a:i,b:i+1};if(i===3)old.saved=o}}f();old.saved.a===3&&old.saved.b===4`,
		`function f(){let list=null;for(let i=0;i<100;i++)list={a:i,b:list};return list}let p=f(),s=0;while(p){s+=p.a;p=p.b}s===4950`,
		`let a=[];function f(){for(let i=0;i<100;i++){let o={a:i,b:i+1};if(i===3)a.push(o)}}f();a[0].a===3&&a[0].b===4`,
		`let m=new Map;function f(){for(let i=0;i<100;i++){let o={a:i,b:i+1};if(i===3)m.set(0,o)}}f();m.get(0).a===3`,
		`let m=new WeakMap;let k;function f(){for(let i=0;i<100;i++){let o={a:i,b:i+1};if(i===3){k=o;m.set(o,42)}}}f();m.get(k)===42&&k.a===3`,
		`let w;function f(){for(let i=0;i<100;i++){let o={a:i,b:i+1};if(i===3)w=new WeakRef(o)}}f();w.deref().a===3`,
		`let fn;function f(){let o={a:7,b:9};fn=()=>o;for(let i=0;i<100;i++)o={a:i,b:i+1}}f();fn().a===99`,
		`function f(){let o={a:7,b:9};try{throw o}catch(e){for(let i=0;i<100;i++){let x={a:i,b:i+1}}return e}}f().a===7`,
		`function* f(){let o={a:7,b:9};yield o;for(let i=0;i<100;i++){let x={a:i,b:i+1}}return o}let g=f(),o=g.next().value;g.next().value===o&&o.a===7`,
		`let saved;function f(){let o={a:7,b:9};let x={valueOf(){saved=o;return 1}};let n=x+1;for(let i=0;i<100;i++){let p={a:i,b:i+1}}return n}f()===2&&saved.a===7`,
		`let saved;function f(){let o={a:7,b:9};let x={get p(){saved=o;return 1}};let n=x.p;for(let i=0;i<100;i++){let p={a:i,b:i+1}}return n}f()===1&&saved.a===7`,
		`function f(){let o={a:7,b:9};Object.defineProperty(o,'a',{get(){return 11}});for(let i=0;i<100;i++){let p={a:i,b:i+1}}return o.a}f()===11`,
		`function f(){let a={a:1,b:2};a.b=a;for(let i=0;i<100;i++){let o={a:i,b:i+1}}return a}let a=f();a.b===a&&a.a===1`,
	} {
		for _, capacity := range []int{1, 2, 8} {
			r := youngInterpreterForTest(t, capacity)
			v, err := r.Run(compileForTest(t, source))
			if err != nil || !v.Truthy() || r.YoungGenerationStats().Young != 0 {
				t.Fatalf("capacity %d: %v, %v; %s", capacity, v, err, source)
			}
		}
	}
}

func TestYoungGenerationHostRetention(t *testing.T) {
	r := youngInterpreterForTest(t, 2)
	var held Value
	r.Global().setOwnRaw(r.Intern("save"), r.NewFunction("save", 1, func(_ *Runtime, _ Value, args []Value) (Value, error) {
		held = args[0]
		runtime.GC()
		_, err := r.Run(compileForTest(t, `for(let j=0;j<100;j++){let p={a:j,b:j+1}}`))
		return Undefined, err
	}), propDefault)
	v, err := r.Run(compileForTest(t, `function f(){for(let i=0;i<100;i++){let o={a:i,b:i+1};if(i===3)save(o)}return true}f()`))
	if err != nil || !v.Truthy() || held.Object().props[0].value != Int(3) || r.YoungGenerationStats().Young != 0 {
		t.Fatalf("host-held value changed: %v, %v, %v", held, v, err)
	}
}

func TestYoungGenerationLazyAndDisabled(t *testing.T) {
	for _, cfg := range []Config{{YoungGeneration: 8}, {}, {YoungGeneration: 8, MemoryLimit: 1 << 20}, {YoungGeneration: 8, Debug: true}} {
		r := New(cfg)
		if r.young != nil {
			t.Fatal("runtime construction allocated nursery")
		}
		if cfg.MemoryLimit != 0 || cfg.Debug || cfg.YoungGeneration == 0 {
			if r.youngCapacity != 0 {
				t.Fatal("nursery enabled in unsupported mode")
			}
		}
		r.Close()
		r.ReleaseClosed()
	}
}

func TestYoungGenerationWithGoGC(t *testing.T) {
	r := youngInterpreterForTest(t, 64)
	done := make(chan struct{})
	go func() {
		for range 10 {
			runtime.GC()
		}
		close(done)
	}()
	v, err := r.Run(compileForTest(t, `function f(n){let s=0;for(let i=0;i<n;i++){let o={a:i,b:i+1};s+=o.a+o.b}return s}f(10000)===100000000`))
	<-done
	if err != nil || !v.Truthy() || r.YoungGenerationStats().Reused == 0 {
		t.Fatalf("Go GC: %v, %v", v, err)
	}
}

func BenchmarkYoungGenerationObjects(b *testing.B) {
	old := treeTier.Load()
	treeTier.Store(false)
	b.Cleanup(func() { treeTier.Store(old) })
	for _, capacity := range []int{0, 64, 4096} {
		b.Run(fmt.Sprintf("slots-%d", capacity), func(b *testing.B) {
			r := New(Config{YoungGeneration: capacity})
			defer func() { r.Close(); r.ReleaseClosed() }()
			fn := compileForTest(b, `function f(n){let s=0;for(let i=0;i<n;i++){let o={a:i,b:i+1};s+=o.a+o.b}return s}f(10000)`)
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				v, err := r.Run(fn)
				if err != nil || v.Number() != 100000000 {
					b.Fatalf("%v, %v", v, err)
				}
			}
			b.StopTimer()
			st := r.YoungGenerationStats()
			b.ReportMetric(float64(st.Reused)/float64(b.N*10000), "reuse/object")
		})
	}
}
