// Package nativegen generates the register interpreter's native handlers
// (docs/register-interpreter.md, "Native handlers"): machine code made by
// the JIT's assembler, offline, and written into internal/vm as an
// assembled block of bytes and a native table of the handlers'
// addresses, which the linker resolves. Package vm holds no assembler:
// what its binary holds is the block.
//
// go generate ./internal/vm runs it (cmd), after the package's other
// generators and before layoutpad, which measures what it writes; vm's
// TestRegNativeGenerated fails while a file is not what it makes now.
//
// # vm's layout
//
// The handlers read and write vm's private types: the native context,
// the frame records, regState, frames, the Runtime's fields, objects,
// shapes and caches. The generator never imports vm. It reads the
// layout as plain numbers (Layout): offsets by "Type.field", sizes by
// type and constants by name, which vm's NativeLayout gives in a build
// with the quickjs_nativelayout tag, which nothing else builds; the
// helper cmd/layout, built with that tag and the build's others for the
// target architecture, prints them as JSON (ReadLayout).
//
// # Architectures
//
// Each architecture has its emitter, in files of its own (amd64*.go), and
// its handlers' file. What they share is the layout and the offsets read
// from it (offsets.go, consts.go), the list of the operations the
// handlers answer, in the order every emitter lays them out (handlerOps),
// and the writing of the block and its table (source.go).
package nativegen
