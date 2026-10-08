//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || (darwin && arm64))

package vm

import (
	"os"
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/compiler"
	"github.com/go-quickjs/go-quickjs/internal/jit/ir"
	"github.com/go-quickjs/go-quickjs/internal/parser"
)

func TestJITStringPacking(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	v, err := r.Run(compileForTest(t, `function pack(s,a){for(let i=0;i<64;i+=4)a[i>>2]=s.charCodeAt(i)+(s.charCodeAt(i+1)<<8)+(s.charCodeAt(i+2)<<16)+(s.charCodeAt(i+3)<<24);return a[0]}let a=new Array(16).fill(0);pack('a'.repeat(64),a)`))
	if err != nil || !v.IsNumber() || v.Number() != 1633771873 || r.jit == nil || r.jit.hosts != 0 {
		if r.jit != nil {
			t.Logf("hosts=%d guards=%d fast=%d", r.jit.hosts, r.jit.guards, r.jit.fastHosts)
		}
		t.Fatalf("packing: %v, %v", v, err)
	}
}

func TestJITStringBoundaries(t *testing.T) {
	for _, tc := range []struct{ name, setup, body, check string }{
		{"utf16", `let s='A\u1234\ud83d\ude00\ud800\udc00';s.charCodeAt(0);`, `a[i]=s.charCodeAt(i);`, `a.join(',')==='65,4660,55357,56832,55296,56320'`},
		{"rope", `let s='A'+'\u1234'+'\ud83d'+'\ude00'+'\ud800'+'\udc00';`, `a[i]=s.charCodeAt(i);`, `a.join(',')==='65,4660,55357,56832,55296,56320'`},
		{"indices", `let s='ABCDEF';`, `a[i]=s.charCodeAt(i-0.5);`, `a.join(',')==='65,65,66,67,68,69'`},
		{"out of bounds", `let s='ABCDEF';`, `a[i]=s.charCodeAt(i-1);`, `Number.isNaN(a[0])&&a.slice(1).join(',')==='65,66,67,68,69'`},
		{"replace", `let s='ABCDEF';String.prototype.charCodeAt=function(i){return 100+i};`, `a[i]=s.charCodeAt(i);`, `a.join(',')==='100,101,102,103,104,105'`},
		{"getter", `let s='ABCDEF',hits=0;let original=String.prototype.charCodeAt;Object.defineProperty(String.prototype,'charCodeAt',{get(){hits++;return original}});`, `a[i]=s.charCodeAt(i);`, `hits===6&&a.join(',')==='65,66,67,68,69,70'`},
		{"index coercion", `let s='ABCDEF',hits=0,index={valueOf(){return hits++}};`, `a[i]=s.charCodeAt(index);`, `hits===6&&a.join(',')==='65,66,67,68,69,70'`},
		{"mutate during coercion", `let s='ABCDEF',hits=0,index={valueOf(){hits++;String.prototype.charCodeAt=function(){return 99};return 0}};`, `a[i]=s.charCodeAt(index);`, `hits===1&&a.join(',')==='65,99,99,99,99,99'`},
		{"callback mutation", `let s='ABCDEF';function cb(i){if(i===2)String.prototype.charCodeAt=function(){return 88}}`, `a[i]=s.charCodeAt(i);cb(i);`, `a.join(',')==='65,66,67,88,88,88'`},
		{"custom method", `let s={charCodeAt(i){return 10+i}};`, `a[i]=s.charCodeAt(i);`, `a.join(',')==='10,11,12,13,14,15'`},
		{"throw", `let s='ABCDEF',hits=0,index={valueOf(){if(++hits===3)throw new TypeError('stop');return hits-1}};`, `a[i]=s.charCodeAt(index);`, `good&&hits===3&&a.join(',')==='65,66,0,0,0,0'`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, enabled := range []bool{false, true} {
				r := jitRuntimeForTest(t, Config{JIT: enabled})
				source := tc.setup + `let a=new Array(6).fill(0),good=false;function f(s,a){for(let i=0;i<6;i++){` + tc.body + `}return a[0]}try{f(s,a)}catch(e){good=e instanceof TypeError&&e.message==='stop'};` + tc.check
				v, err := r.Run(compileForTest(t, source))
				if err != nil || !v.IsBool() || !v.Truthy() || enabled && (r.jit == nil || r.jit.entries == 0 || r.jit.rootCount != 0) {
					t.Fatalf("JIT %v: %v, %v", enabled, v, err)
				}
			}
		})
	}
}

func TestJITArrayLiteralBoundary(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		r := jitRuntimeForTest(t, Config{JIT: enabled})
		v, err := r.Run(compileForTest(t, `let original={x:7};function f(o){let a=[o,1,2];for(let i=1;i<3;i++)a[i]+=1;return a}let a=f(original);a[0]===original&&a[1]===2&&a[2]===3&&a.length===3`))
		if err != nil || !v.IsBool() || !v.Truthy() || enabled && (r.jit == nil || r.jit.hosts == 0 || r.jit.rootCount != 0) {
			t.Fatalf("array literal JIT %v: %v, %v", enabled, v, err)
		}
	}
}

func TestJITStringFullRootTable(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	s := &jitState{charCodeAt: r.jitCharCodeAt}
	for range ir.MaxSlots {
		v := s.encode(Str(NewString("abc")))
		if v.Kind != ir.String {
			t.Fatal(v)
		}
	}
	s.prepareStringMethods(r)
	if s.rootCount != ir.MaxSlots {
		t.Fatal("intrinsic preparation exceeded the root bound")
	}
	for _, view := range s.arrays {
		if view.WritableHole != 0 {
			t.Fatal("granted a method without a rooted intrinsic")
		}
	}
	s.clearRoots()
	if s.rootCount != 0 || s.roots[0].ref != nil || s.arrays[0].Data != nil {
		t.Fatal("string roots or borrowed data survived release")
	}
}

func TestJITStringCalleePermissions(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	v, err := r.Run(compileForTest(t, `function pack(s,a,method){for(let i=0;i<64;i++)a[i]=s.charCodeAt(i);return a[0]}function outer(s,a,method){let sum=0;for(let j=0;j<4;j++)sum+=pack(s,a,method);return sum}let a=new Array(64).fill(0);outer('a'.repeat(64),a,String.prototype.charCodeAt)===388`))
	// Only the four global reads and four call transfers leave machine code.
	if err != nil || !v.IsBool() || !v.Truthy() || r.jit == nil || r.jit.transfers != 4 || r.jit.hosts != 8 || r.jit.rootCount != 0 {
		if r.jit != nil {
			t.Logf("transfers=%d hosts=%d guards=%d", r.jit.transfers, r.jit.hosts, r.jit.guards)
		}
		t.Fatalf("string callee permissions: %v, %v", v, err)
	}
}

func TestJITStringRealms(t *testing.T) {
	r := jitRuntimeForTest(t, Config{JIT: true})
	other := r.NewRealm()
	setup := compileForTest(t, `function pack(s,a){for(let i=0;i<6;i++)a[i]=s.charCodeAt(i);return a[0]}var a=new Array(6).fill(0);true`)
	call := compileForTest(t, `pack('ABCDEF',a)===65&&a[5]===70`)
	for _, realm := range []*Realm{r.Realm, other, r.Realm, other} {
		if _, err := r.RunIn(realm, setup); err != nil {
			t.Fatal(err)
		}
		v, err := r.RunIn(realm, call)
		if err != nil || !v.IsBool() || !v.Truthy() || r.jit == nil || r.jit.hosts != 0 || r.jit.rootCount != 0 {
			t.Fatalf("realm character read: %v, %v", v, err)
		}
	}
}

// Use the external library unchanged and validate every complete hash.
func BenchmarkJITMD5(b *testing.B) {
	path := os.Getenv("QUICKJS_JIT_MD5_SOURCE")
	if path == "" {
		b.Skip("set QUICKJS_JIT_MD5_SOURCE to SparkMD5 3.0.2")
	}
	source, err := os.ReadFile(path)
	if err != nil {
		b.Fatal(err)
	}
	for _, mode := range []string{"interpreter", "existing", "native"} {
		b.Run(mode, func(b *testing.B) {
			previous := treeTier.Swap(mode != "interpreter")
			defer treeTier.Store(previous)
			ast, err := parser.Parse("var module={exports:{}},exports=module.exports;"+string(source)+`var spark=module.exports, input='a'.repeat(1024);function hashes(){return spark.hash(input)}`, parser.Options{})
			if err != nil {
				b.Fatal(err)
			}
			setup, err := compiler.Compile(ast, compiler.Options{})
			if err != nil {
				b.Fatal(err)
			}
			ast, err = parser.Parse(`hashes()`, parser.Options{})
			if err != nil {
				b.Fatal(err)
			}
			call, err := compiler.Compile(ast, compiler.Options{})
			if err != nil {
				b.Fatal(err)
			}
			r := New(Config{JIT: mode == "native"})
			defer func() { r.Close(); r.ReleaseClosed() }()
			if _, err := r.Run(setup); err != nil {
				b.Fatal(err)
			}
			run := func() {
				v, err := r.Run(call)
				if err != nil || !v.IsString() || v.String().Go() != "c9a34cfc85d982698c6ac89f76071abd" {
					b.Fatalf("MD5: %v, %v", v, err)
				}
			}
			for range 100 {
				run()
			}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				run()
			}
			b.StopTimer()
			if mode == "native" {
				packed := false
				if r.jit != nil {
					for key, entry := range r.jit.cache {
						if fn := key.Value(); fn != nil && fn.Name == "md5blk" {
							packed = entry.code != nil && !entry.entrySlow && entry.misses == 0
						}
					}
				}
				if !packed {
					b.Fatal("block packing did not retain native execution")
				}
			}
			if r.jit != nil {
				b.ReportMetric(float64(r.jitCodeBytes()), "code+metadata-B")
				b.ReportMetric(float64(r.jit.hosts)/float64(b.N+100), "hosts/hash")
				b.ReportMetric(float64(r.jit.guards)/float64(b.N+100), "guards/hash")
				for key, entry := range r.jit.cache {
					fn := key.Value()
					if fn != nil {
						b.Logf("%s: code=%v slow=%v misses=%d probes=%d hosts=%d steps=%d", fn.Name, entry.code != nil, entry.entrySlow, entry.misses, entry.probes, entry.probeHosts, entry.probeSteps)
					}
				}
			}
		})
	}
}
