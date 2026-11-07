package vm

import (
	"testing"
	"unsafe"
)

// sharedElemsForTest reports whether v is an array whose elements an array
// literal's arrays share, and their storage.
func sharedElemsForTest(v Value) (bool, unsafe.Pointer) {
	if !v.IsObject() || v.Object().class != ClassArray {
		return false, nil
	}
	o := v.Object()
	return o.flags&objSharedElems != 0, unsafe.Pointer(unsafe.SliceData(o.elemsOf()))
}

// TestSharedArrayLiteral pins that the tree tier makes the arrays of a
// literal of constants over one storage of the constants, marked shared,
// and that the interpreter, whose case of new_array is unchanged, makes each
// its own. A literal with anything but constants, or none, has no site.
func TestSharedArrayLiteral(t *testing.T) {
	defer SetTreeTier(true)
	for _, tree := range []bool{true, false} {
		SetTreeTier(tree)
		r := New(Config{})
		v, err := r.Run(compileForTest(t, `
		function lit() { return [1, 2.5, 'x', true, null, undefined, '', -3]; }
		function notConst(x) { return [1, x]; }
		function empty() { return []; }
		[lit(), lit(), notConst(1), empty(), JSON.stringify(lit())]`))
		if err != nil {
			t.Fatal(err)
		}
		get := func(i int) Value {
			e, err := r.getIndexed(v, Int(i))
			if err != nil {
				t.Fatal(err)
			}
			return e
		}
		s1, p1 := sharedElemsForTest(get(0))
		s2, p2 := sharedElemsForTest(get(1))
		if s1 != tree || s2 != tree || (p1 == p2) != tree {
			t.Errorf("tree %v: literal arrays shared %v %v, one storage %v", tree, s1, s2, p1 == p2)
		}
		if s, _ := sharedElemsForTest(get(2)); s {
			t.Errorf("tree %v: a literal of a variable shares its elements", tree)
		}
		if s, _ := sharedElemsForTest(get(3)); s {
			t.Errorf("tree %v: an empty literal shares its elements", tree)
		}
		if got := get(4).String().Go(); got != `[1,2.5,"x",true,null,null,"",-3]` {
			t.Errorf("tree %v: got %s", tree, got)
		}
		r.Close()
	}
}

// TestSharedArrayLiteralWrites pins that every way a script can write an
// array leaves a shared literal's other arrays as they were, and does to the
// array written what it does to an array of its own: each write is run on
// a shared array, with the tree tier, and on one of its own, in the
// interpreter, and the two must agree. Each has a literal of its own, which
// gives sharing up after a few copies.
func TestSharedArrayLiteralWrites(t *testing.T) {
	defer SetTreeTier(true)
	writes := []string{
		`a[0] = 9`,
		`a[1] += 1`,
		`a[0]++`,
		`var b = a; for (var i = 0; i < 2; i++) b[i] += 1`,
		`a.push(7)`,
		`a.push(7); a[0] = 8`,
		`a.pop(); a.push(42)`,
		`a.pop(); a[a.length] = 5`,
		`a.shift()`,
		`a.unshift(0)`,
		`a.splice(1, 2, 'p', 'q', 'r')`,
		`a.splice(1)`,
		`a.splice('1', 1, 'q')`,
		`a.splice(0, 0, 'p')`,
		`a.splice(0, 0, 'p'); a[1] = 'w'`,
		`a.splice(1, 1, 'q')`,
		`a.splice(1, 2, 'p')`,
		`a.splice(0, 2)`,
		`a.reverse()`,
		`a.sort()`,
		`a.sort(function (x, y) { return String(y) < String(x) ? -1 : 1; })`,
		`a.fill(0)`,
		`a.fill(0, 1, 3)`,
		`a.copyWithin(0, 3)`,
		`a.length = 2; a.push('z', 'z', 'z', 'z')`,
		`a.length = 10; a[7] = 1`,
		`a.length = 0; a[0] = 'w'`,
		`delete a[0]`,
		`Object.defineProperty(a, 1, { get: function () { return 0; } })`,
		`Object.defineProperty(a, 1, { value: 'd', writable: false })`,
		`Object.freeze(a)`,
		`Object.seal(a); a[0] = 's'`,
		`Object.preventExtensions(a); a[1] = 'p'`,
		`a[100000] = 1`,
		`a[8] = 1`,
		`Reflect.set(a, 2, 'r')`,
		`Object.assign(a, ['a', 'b'])`,
		`(function () { arguments[0] = 'arg'; }).apply(null, a)`,
		`(function (x) { x = 3; arguments[1] = 4; }).apply(null, a)`,
		`(function (x, y) { x = 1; y = 2; arguments[0] = 9; })(...a)`,
		`var i = 0; for (var v of a) { a[i++] = 'f'; }`,
		`a.forEach(function (v, i, arr) { arr[i] = 'e'; })`,
		`a.map(function (v) { return v; })[0] = 'm'`,
		`a.concat([1])[0] = 'c'`,
		`a.slice()[0] = 's'`,
		`[...a][0] = 's'`,
		`Array.from(a)[0] = 'f'`,
		`Object.setPrototypeOf(a, null); a[0] = 'n'`,
		`a.with(0, 'w')[0] = 'q'`,
		`a.toSorted()[0] = 't'`,
		`a.toSpliced(0, 1)[0] = 't'`,
		`a.toReversed()[0] = 't'`,
		`for (var i = 0; i < a.length; i++) a[i] = i * 2`,
		`for (var i = 0; i < 20; i++) a.push(i)`,
		`while (a.length) a.pop()`,
		`for (var i = 0; i < 3; i++) a.length--`,
		`a[0] = a; a[1] = [a]`,
		`Array.prototype.push.call(a, 7)`,
		`Array.prototype.pop.call(a)`,
		`Object.preventExtensions(a); a.length = 1`,
		`Object.seal(a); a.length = 10`,
		`a[a.length] = 5`,
		`for (var i = 0; i < 3; i++) a[a.length] = i`,
		`a.__defineGetter__('1', function () { return 0; })`,
		`var b = a; b[1] += 2`,
		`var b = a, i = 1, y = 2; b[i] += y`,
		`var b = a, y = 2; b[1] -= y`,
		`var b = a, i = 2; b[i] *= 3`,
		`var b = a; b[0] |= 4`,
		`var b = a, i = 0; b[i]++`,
		`var b = a; for (var i = 0; i < 2; i++) { var y = i + 1; b[i] += y; }`,
		`var b = a; b[1] += Math.max(1, 2)`,
		`var b = a, i = 1; b[i] += Math.max(1, 2)`,
		`var b = a; for (var i = 0; i < 2; i++) b[i] -= i`,
		`var b = a; for (var i = 0; i < 2; i++) b[0] += i`,
		`var b = a, i = 1; var r = (b[i] += 2); b[0] = r`,
		`var b = a, i = 0; var r = b[i]++; b[2] = r`,
		`a.push(...[7, 8])`,
		`a.pop(...[])`,
		`Array.prototype.push.apply(a, [7])`,
		`Reflect.apply(Array.prototype.pop, a, [])`,
		`Object.defineProperty(a, 'p', { get: Array.prototype.pop }); a.p`,
	}
	for _, form := range []string{"function", "script", "interpreter"} {
		for _, w := range writes {
			var got [2]string
			// Inside a function, which the tree tier builds, or as the script's
			// own code, which runs in the interpreter, where its store tests
			// the flag in executeAt's case.
			run := `(function () { ` + w + ` })();`
			switch form {
			case "script":
				run = w + `;`
			case "interpreter":
				// A function with a direct eval, which the tree tier does not
				// build: its stores are executeAt's OpSetIndex.
				run = `(function () { ` + w + `; if (a.length < 0) eval(''); })();`
			}
			for k, tree := range []bool{true, false} {
				SetTreeTier(tree)
				r := New(Config{})
				v, err := r.Run(compileForTest(t, `
			var lit = function () { return [1, 2.5, 'x', true, null, undefined, 'z']; };
			var a = lit();
			a`))
				if err != nil {
					t.Fatal(err)
				}
				if s, _ := sharedElemsForTest(v); s != tree {
					t.Fatalf("tree %v: the array written is shared: %v", tree, s)
				}
				v, err = r.Run(compileForTest(t, `
			var thrown = '';
			try { `+run+` } catch (e) { thrown = e.name; }
			// A write after it, which finds the array's flag as the write left
			// it: shared still where the storage changed is caught here.
			try { a[0] = 'after'; } catch (e) { thrown += '+' + e.name; }
			var keys = Object.getOwnPropertyNames(a).join();
			var vals = [];
			for (var i = 0; i < a.length; i++) vals.push(a[i] === a ? 'self' : typeof a[i] === 'object' && a[i] !== null ? 'obj' : String(a[i]));
			[thrown, a.length, keys, vals.join(), Object.isFrozen(a), JSON.stringify(lit()), JSON.stringify(lit())].join(' | ')`))
				if err != nil {
					t.Fatalf("%s: %v", w, err)
				}
				got[k] = v.String().Go()
				r.Close()
			}
			if got[0] != got[1] {
				t.Errorf("%s, %s:\n shared: %s\n    own: %s", form, w, got[0], got[1])
			}
			const pristine = `[1,2.5,"x",true,null,null,"z"] | [1,2.5,"x",true,null,null,"z"]`
			if len(got[0]) < len(pristine) || got[0][len(got[0])-len(pristine):] != pristine {
				t.Errorf("%s, %s: the literal's next arrays changed: %s", form, w, got[0])
			}
		}
	}
}

// TestSharedArrayLiteralOtherRealm pins that another realm's push and pop,
// which a call does not apply as this realm's leaves (the built-in's own
// fast path then), copy a shared literal's elements before writing them.
func TestSharedArrayLiteralOtherRealm(t *testing.T) {
	r := New(Config{})
	defer r.Close()
	other := r.NewRealm()
	for _, method := range []string{"pop", "push"} {
		f, err := r.RunIn(other, compileForTest(t, `Array.prototype.`+method))
		if err != nil {
			t.Fatal(err)
		}
		r.global.setOwnRaw(r.atoms.intern("otherMethod"), f, propDefault)
		a, err := r.Run(compileForTest(t, `var lit = function () { return [1, 2, 3]; }; var a = lit(); a`))
		if err != nil {
			t.Fatal(err)
		}
		if s, _ := sharedElemsForTest(a); !s {
			t.Fatal("the literal's array is not shared")
		}
		// Called by this realm's code, where the method is not one of its
		// leaves; then a store, which finds the array as the call left it.
		v, err := r.Run(compileForTest(t, `otherMethod.call(a, 9); a[0] = 7; JSON.stringify(lit())`))
		if err != nil {
			t.Fatal(err)
		}
		if got := v.String().Go(); got != `[1,2,3]` {
			t.Errorf("another realm's %s wrote the literal's elements: %s", method, got)
		}
	}
}

// TestSharedArrayLiteralGivesUp pins that a literal whose arrays keep being
// written stops sharing, and makes each array of its own, as before.
func TestSharedArrayLiteralGivesUp(t *testing.T) {
	r := New(Config{})
	defer r.Close()
	v, err := r.Run(compileForTest(t, `
	function mk() { return [0, 0, 0, 0]; }
	var first = mk();
	function acc() { var s = mk(); for (var i = 0; i < 16; i++) s[i & 3] += i; return s; }
	var out = [];
	for (var k = 0; k < 100; k++) out.push(acc().join());
	[out.every(function (s) { return s === '24,28,32,36'; }), mk(), first]`))
	if err != nil {
		t.Fatal(err)
	}
	ok, err := r.getIndexed(v, Int(0))
	if err != nil {
		t.Fatal(err)
	}
	if !ok.Truthy() {
		t.Error("a written literal's arrays came out wrong")
	}
	last, err := r.getIndexed(v, Int(1))
	if err != nil {
		t.Fatal(err)
	}
	if s, _ := sharedElemsForTest(last); s {
		t.Error("a literal written each time still shares its elements")
	}
	first, err := r.getIndexed(v, Int(2))
	if err != nil {
		t.Fatal(err)
	}
	if s, _ := sharedElemsForTest(first); !s {
		t.Error("a literal's first array, never written, does not share its elements")
	}
}
