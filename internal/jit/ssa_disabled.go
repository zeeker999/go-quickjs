//go:build !quickjs_jit || android || ios || !((linux && amd64) || (windows && amd64) || (darwin && arm64))

package jit

// supported is false where the JavaScript tier is not built: without the
// tag, or on a platform whose only native code is RawCode (linux/arm64).
const supported = false
