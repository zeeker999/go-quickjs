//go:build quickjs_nativelayout

// Command layout prints package vm's layout as the native handlers'
// generator reads it (nativegen.Layout, as JSON): vm's NativeLayout,
// which only the quickjs_nativelayout tag builds. nativegen.ReadLayout
// runs it, built for the architecture and the tags whose layout it reads.
package main

import (
	"encoding/json"
	"log"
	"os"

	"github.com/go-quickjs/go-quickjs/internal/vm"
	"github.com/go-quickjs/go-quickjs/internal/vm/internal/nativegen"
)

func main() {
	offsets, sizes, consts := vm.NativeLayout()
	l := nativegen.Layout{Offsets: map[string]int64{}, Sizes: map[string]int64{}, Consts: consts}
	for k, v := range offsets {
		l.Offsets[k] = int64(v)
	}
	for k, v := range sizes {
		l.Sizes[k] = int64(v)
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "\t")
	if err := enc.Encode(l); err != nil {
		log.Fatal(err)
	}
}
