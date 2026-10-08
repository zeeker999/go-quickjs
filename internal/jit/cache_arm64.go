//go:build quickjs_jit && !android && !ios && darwin

package jit

import "runtime"

func flushCode(code []byte) {
	flushInstructionCache(&code[0], uintptr(len(code)))
	runtime.KeepAlive(code)
}

//go:noescape
func flushInstructionCache(start *byte, size uintptr)
