//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"reflect"
	"slices"
	"strings"

	"github.com/go-quickjs/go-quickjs/internal/jit/mir"
)

// jitCodeHash hashes what the backend made: its code, every word in it
// that may be an address taken by the order it first appears in, and what
// describes it, its addresses so.
func jitCodeHash(mc *mir.Code) string {
	h := sha256.New()
	ids := map[uint64]int{}
	id := func(v uint64) uint64 {
		if v < 1<<32 || v >= 1<<47 {
			return v
		}
		n, ok := ids[v]
		if !ok {
			n = len(ids)
			ids[v] = n
		}
		return uint64(n) | 0xadd<<48
	}
	// The instructions, a 64-bit immediate's address (amd64's movabs) by id;
	// past them the constants they load, of eight bytes each (movImm), the
	// addresses among them so.
	b := mc.Bytes
	for i := 0; i < mc.Text; {
		if i+10 <= mc.Text && b[i]&0xfe == 0x48 && b[i+1] >= 0xb8 && b[i+1] <= 0xbf {
			if v := binary.LittleEndian.Uint64(b[i+2:]); v >= 1<<32 && v < 1<<47 {
				h.Write(b[i : i+2])
				binary.Write(h, binary.LittleEndian, id(v))
				i += 10
				continue
			}
		}
		h.Write(b[i : i+1])
		i++
	}
	for i := mc.Text; i < len(b); i += 8 {
		if i+8 > len(b) {
			h.Write(b[i:])
			break
		}
		binary.Write(h, binary.LittleEndian, id(binary.LittleEndian.Uint64(b[i:])))
	}
	entries := make([]int, 0, len(mc.Entries))
	for pc := range mc.Entries {
		entries = append(entries, pc)
	}
	slices.Sort(entries)
	for _, pc := range entries {
		fmt.Fprintf(h, "e%d:%d,", pc, mc.Entries[pc])
	}
	fmt.Fprintf(h, "c%d r%d t%d refs%d;", mc.CallEntry, mc.Resume, mc.Text, mc.CallRefs)
	seen := map[uintptr]bool{}
	jitHashValue(h, reflect.ValueOf(mc.Calls), id, seen)
	jitHashValue(h, reflect.ValueOf(mc.GoCalls), id, seen)
	jitHashValue(h, reflect.ValueOf(mc.Exits), id, seen)
	return hex.EncodeToString(h.Sum(nil)[:12])
}

// jitHashValue hashes v, its pointers by what they point to, once, and
// its words that may be addresses by id.
func jitHashValue(h hash.Hash, v reflect.Value, id func(uint64) uint64, seen map[uintptr]bool) {
	switch v.Kind() {
	case reflect.Pointer:
		if v.IsNil() {
			h.Write([]byte("nil"))
			return
		}
		p := v.Pointer()
		fmt.Fprintf(h, "p%d", id(uint64(p)))
		if seen[p] {
			return
		}
		seen[p] = true
		jitHashValue(h, v.Elem(), id, seen)
	case reflect.Slice, reflect.Array:
		fmt.Fprintf(h, "[%d", v.Len())
		for i := range v.Len() {
			jitHashValue(h, v.Index(i), id, seen)
		}
		h.Write([]byte("]"))
	case reflect.Struct:
		h.Write([]byte("{"))
		for i := range v.NumField() {
			jitHashValue(h, v.Field(i), id, seen)
		}
		h.Write([]byte("}"))
	case reflect.Map:
		keys := v.MapKeys()
		slices.SortFunc(keys, func(a, b reflect.Value) int { return strings.Compare(fmt.Sprint(a), fmt.Sprint(b)) })
		for _, k := range keys {
			jitHashValue(h, k, id, seen)
			jitHashValue(h, v.MapIndex(k), id, seen)
		}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		fmt.Fprintf(h, "%d,", v.Int())
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		fmt.Fprintf(h, "%d,", id(v.Uint()))
	case reflect.Bool:
		fmt.Fprintf(h, "%t,", v.Bool())
	case reflect.Float32, reflect.Float64:
		fmt.Fprintf(h, "%v,", v.Float())
	case reflect.String:
		fmt.Fprintf(h, "%q,", v.String())
	case reflect.Interface:
		if v.IsNil() {
			h.Write([]byte("nil"))
			return
		}
		jitHashValue(h, v.Elem(), id, seen)
	case reflect.Invalid:
		h.Write([]byte("-"))
	default:
		panic(errors.New("jitHashValue: " + v.Kind().String()))
	}
}
