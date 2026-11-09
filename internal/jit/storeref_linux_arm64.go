//go:build quickjs_jit && !android

package jit

import "unsafe"

// On linux/arm64 the new pipeline is not built, but RawCode is, and code
// another compiler made -- the WebAssembly JIT's -- stores pointer words
// as the pipeline's code does: storeRef is darwin/arm64's (entry_arm64.s),
// in storeref_linux_arm64.s.

// storeRef is where native code stores a pointer word while the collector
// marks, and storeRefAddr its address.
func storeRef()
func storeRefAddr() uintptr

// StoreRef is where native code jumps to store a pointer word with Go's
// write barrier.
func StoreRef() uint64 { return uint64(storeRefAddr()) }

// storeRefClobber is storeRef, every register native code may keep given
// junk first, and storeRefClobberAddr its address.
func storeRefClobber()
func storeRefClobberAddr() uintptr

// StoreRefClobber is StoreRef for tests: what native code jumps to there
// fills every register it may keep with junk before it stores, so that code
// that did not save one it keeps across the store goes wrong.
func StoreRefClobber() uint64 { return uint64(storeRefClobberAddr()) }

// storeRefGo is what storeRef calls: the store, which compiled Go makes
// with its write barrier while the collector marks. It never grows the
// stack, which storeRef's caller, native code, did not check, nor
// preempts. Neither the race detector nor checkptr is told of the store.
//
//go:nosplit
//go:norace
//go:nocheckptr
func storeRefGo(slot *unsafe.Pointer, p unsafe.Pointer) { *slot = p }
