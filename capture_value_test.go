package quickjs_test

import "testing"

// TestCaptureByValueEval pins that a direct eval's assignment of a strict
// function's parameter is seen by a closure over it: the eval, in the
// function or in a closure of it, keeps the closure sharing the binding
// rather than taking its value (bytecode.UpvalueDesc's ByValue).
func TestCaptureByValueEval(t *testing.T) {
	checkEval(t, `'use strict'; function f(x) { const g = () => x; eval("x = 6"); return g() } f(1)`, "6")
	checkEval(t, `'use strict'; function f(x) { const g = () => x; (() => eval("x = 7"))(); return g() } f(1)`, "7")
	checkEval(t, `'use strict'; function f(x) { const g = () => x; (function () { eval("x = 8") })(); return g() } f(1)`, "8")
}
