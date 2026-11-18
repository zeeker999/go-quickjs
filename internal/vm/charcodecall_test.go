package vm

import (
	"testing"

	"github.com/go-quickjs/go-quickjs/internal/bytecode"
)

const charCodeCallLegacySource = `
var o={charCodeAt:function method(x){return method.arguments.length+":"+method.arguments[0]}};
function legacy(n){let result;for(let i=0;i<20;i++)result=o.charCodeAt(n+0);return ''+result}`

func TestCharCodeCallLegacyArguments(t *testing.T) {
	r := New(Config{NodeQuirks: true})
	defer r.Close()
	program := compileForTest(t, charCodeCallLegacySource+`;legacy(42)`)
	for range 3 {
		v, err := r.Run(program)
		if err != nil || v.String().Go() != "1:42" {
			t.Fatalf("got %v, %v; want 1:42", v, err)
		}
	}
}

func TestCharCodeCallDeepOverride(t *testing.T) {
	r := New(Config{})
	defer r.Close()
	program := compileForTest(t, `
var o={charCodeAt:function recurse(n){if(n===0)return 0;let result=o.charCodeAt(n-1);return result+1}};
function deep(n){let result=o.charCodeAt(n+0);return result}
''+deep(200)`)
	for range 3 {
		v, err := r.Run(program)
		if err != nil || v.String().Go() != "200" {
			t.Fatalf("got %v, %v; want 200", v, err)
		}
	}
}

func TestCharCodeCallArguments(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct{ name, source, want string }{
		{"updates", `function f(s) { let i=0, a=s.charCodeAt(++i), b=s.charCodeAt(i++), c=s.charCodeAt(i+1); return [a,b,c,i].join() } f("abcde")`, "98,98,100,2"},
		{"getter before increment", `function f() { let i=0, s={get charCodeAt(){i=2;return function(x){return x}}}; let a=s.charCodeAt(++i); return [a,i].join() } f()`, "3,3"},
		{"getter preserves receiver", `function f() { let i=0, s, original; original=s={get charCodeAt(){s="changed";return function(x){return this===original ? x : -1}}}; let a=s.charCodeAt(++i); return [a,i,s].join() } f()`, "1,1,changed"},
		{"getter throws before argument", `function f() { let i=0,s={get charCodeAt(){throw new RangeError("get")}};try{s.charCodeAt(++i)}catch(e){return e.name+":"+i} } f()`, "RangeError:0"},
		{"non callable after argument", `function f() { let i=0,s={charCodeAt:1};try{s.charCodeAt(++i)}catch(e){return e.name+":"+i} } f()`, "TypeError:1"},
		{"null before argument", `function f() { let i=0,s=null;try{s.charCodeAt(++i)}catch(e){return e.name+":"+i} } f()`, "TypeError:0"},
		{"argument replaces method", `function f() { let s="abc",saved=String.prototype.charCodeAt,i={valueOf(){String.prototype.charCodeAt=function(){return 7};return 0}},a=s.charCodeAt(++i);String.prototype.charCodeAt=saved;return [a,i].join() } f()`, "98,1"},
		{"argument reads getter once", `function f() { let n=0,s={get charCodeAt(){n++;return function(x){return x}}},i={valueOf(){return 2}};let a=s.charCodeAt(i+1);return [a,n].join() } f()`, "3,1"},
		{"argument conversion once", `function f() { let n=0,s="abc",i={valueOf(){n++;return 0}};let a=s.charCodeAt(i+1);return [a,n].join() } f()`, "98,1"},
		{"bigint increment", `function f() { let i=0n,s="abc";try{s.charCodeAt(++i)}catch(e){return e.name+":"+i} } f()`, "TypeError:1"},
		{"bound builtin", `function f() { let i=0,s={charCodeAt:String.prototype.charCodeAt.bind("xyz")};let a=s.charCodeAt(++i);return String(a) } f()`, "121"},
		{"script method frame", `function f(){let i=0,s={n:7,charCodeAt:function(x){let h=0;for(let j=0;j<3;j++)h+=j;return this.n+x+h}};let a=s.charCodeAt(++i);return [a,i].join()} f()`, "11,1"},
		{"script method templates", `function f(){let i=0,h=0,s={};for(let k=0;k<10;k++){s.charCodeAt=(function(n){return function(x){let h=0;for(let j=0;j<3;j++)h+=j;return n+x+h}})(k);h+=s.charCodeAt(++i)}return h} ''+f()`, "130"},
		{"numeric edges", `function f() { let s="a\ud800c",i=-1.5,out=[];out.push(s.charCodeAt(i+1));i=NaN;out.push(s.charCodeAt(i+1));i=Infinity;out.push(s.charCodeAt(i+1));i=0;out.push(s.charCodeAt(++i));return out.join() } f()`, "97,97,NaN,55296"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := regI32Eval(t, tc.source); got != tc.want {
				t.Fatalf("got %q, want %q", got, tc.want)
			}
		})
	}
}

func TestCharCodeCallLowering(t *testing.T) {
	fn := compileForTest(t, `function f(s){let i=0,h=0;for(let j=0;j<4;j++){h+=s.charCodeAt(++i);h+=s.charCodeAt(i+1)}return h} f("abcdefg")`)
	var code *bytecode.RegCode
	for _, k := range fn.Constants {
		if k.Kind == bytecode.ConstFunction && k.Fn.Name == "f" {
			code = k.Fn.RegCode()
		}
	}
	if code == nil {
		t.Fatal("no register code")
	}
	count := 0
	for _, w := range code.Code {
		if w.Op().String() == "char_code_call" {
			count++
		}
	}
	if count != 2 {
		t.Fatalf("got %d char_code_call instructions, want 2:\n%s", count, code)
	}
}
