//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

// regMappedArgsSrc is earley-boyer's numeric comparisons, sloppy
// functions whose mapped arguments object aliases the parameter x they
// assign while they read the rest by index, and one whose assigned
// parameter is dead by name past the assignment, read only through
// arguments[0] after host exits; then mappedArgsSetup.
const regMappedArgsSrc = `
function sc_greater(x, y) {
	for (var i = 1; i < arguments.length; i++) { if (x <= arguments[i]) return false; x = arguments[i]; }
	return true;
}
function sc_lessEqual(x, y) {
	for (var i = 1; i < arguments.length; i++) { if (x > arguments[i]) return false; x = arguments[i]; }
	return true;
}
function sc_less(x, y) {
	for (var i = 1; i < arguments.length; i++) { if (x >= arguments[i]) return false; x = arguments[i]; }
	return true;
}
function sc_greaterEqual(x, y) {
	for (var i = 1; i < arguments.length; i++) { if (x < arguments[i]) return false; x = arguments[i]; }
	return true;
}
function deadParam(a, b) { a = b * 2; var s = 0; for (var i = 0; i < 3; i++) s += b.length | 0; return arguments[0] + "/" + s }
` + mappedArgsSetup
