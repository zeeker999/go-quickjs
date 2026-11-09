//go:build quickjs_jit && !android && !ios && arm64

package jit

// executablePolicy allows code on Linux: an anonymous mapping made
// executable with mprotect runs.
func executablePolicy() error { return nil }
