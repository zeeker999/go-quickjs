package vm

import (
	"encoding/binary"
	"math/bits"
	"strconv"
	"strings"
	"unicode/utf8"
	"unsafe"

	"github.com/go-quickjs/go-quickjs/internal/jsonscan"
	"github.com/go-quickjs/go-quickjs/internal/wtf8"
)

// JSON.stringify's strings: a string quoted, and the escapes JSON needs.

// appendJSONString appends a quoted string, escaping the characters JSON
// requires plus any lone surrogate, which must be written as an escape because
// it has no valid UTF-8 form. It says how many it escaped so. ascii says the
// string is all ASCII.
//
// Text that needs no escape is found a word at a time, and written as it is
// found while the string is ASCII, as most are: most strings are short, for
// which a call to copy them costs more than the copying.
func appendJSONString(buf []byte, s string, ascii bool) ([]byte, int) {
	if cap(buf)-len(buf) < len(s)+2 {
		buf = growJSON(buf, len(s)+2)
	}
	buf = append(buf, '"')
	i := 0
	if ascii && len(s) < jsonscan.MinLength {
		to := len(buf)
		out := buf[:cap(buf)]
		for ; i+8 <= len(s); i += 8 {
			w := wtf8.Load64(s, i)
			if jsonStops(w) != 0 {
				break
			}
			binary.LittleEndian.PutUint64(out[to:], w)
			to += 8
		}
		for ; i < len(s); i++ {
			c := s[i]
			if c < 0x20 || c == '"' || c == '\\' {
				break
			}
			out[to] = c
			to++
		}
		buf = out[:to]
		if i == len(s) {
			return append(buf, '"'), 0
		}
	}
	plain, surrogates := i, 0
	for {
		var at int
		if ascii {
			at = jsonEscapeASCII(s, i)
		} else {
			at = jsonEscapeWide(s, i)
		}
		if at == len(s) {
			break
		}
		c := s[at]
		if c >= utf8.RuneSelf {
			if r, ok := wtf8.DecodeSurrogateAt(s, at); ok {
				// A lone surrogate is emitted as an escape, which is what the
				// specification's well-formed stringify requires.
				buf = append(buf, s[plain:at]...)
				buf = append(buf, `\u`...)
				buf = strconv.AppendUint(buf, uint64(r), 16)
				surrogates++
				i = at + 3
				plain = i
				continue
			}
			i = at + 1
			continue
		}
		buf = append(buf, s[plain:at]...)
		switch c {
		case '"':
			buf = append(buf, `\"`...)
		case '\\':
			buf = append(buf, `\\`...)
		case '\n':
			buf = append(buf, `\n`...)
		case '\r':
			buf = append(buf, `\r`...)
		case '\t':
			buf = append(buf, `\t`...)
		case '\b':
			buf = append(buf, `\b`...)
		case '\f':
			buf = append(buf, `\f`...)
		default:
			const hex = "0123456789abcdef"
			buf = append(buf, `\u00`...)
			buf = append(buf, hex[c>>4], hex[c&0xF])
		}
		i = at + 1
		plain = i
	}
	buf = append(buf, s[plain:]...)
	return append(buf, '"'), surrogates
}

// growJSON is buf with room for n more bytes. A document that is mostly long
// strings grows its buffer here, and append takes a large one up by only a
// quarter, copying it about four times over: this doubles it. One string
// longer than all there is, though, is most of the text, and the buffer can
// be the text itself if it is about that size.
func growJSON(buf []byte, n int) []byte {
	size := 2 * cap(buf)
	if n >= cap(buf) {
		size = len(buf) + n + n/8
	}
	// It is written before it is read, so it need not be cleared first:
	// only a Builder gets memory that is not.
	var b strings.Builder
	b.Grow(size)
	b.Write(buf)
	b.WriteByte(0)
	out := unsafe.Slice(unsafe.StringData(b.String()), b.Cap())
	return out[:len(buf)]
}

// jsonEscapeASCII is the index from i of the first byte of s, all ASCII, that
// needs an escape, or its length.
func jsonEscapeASCII(s string, i int) int {
	n := len(s)
	i += jsonscan.ASCII(s[i:])
	for ; i+8 <= n; i += 8 {
		if m := jsonStops(wtf8.Load64(s, i)); m != 0 {
			return i + bits.TrailingZeros64(m)>>3
		}
	}
	for ; i < n; i++ {
		if c := s[i]; c < 0x20 || c == '"' || c == '\\' {
			return i
		}
	}
	return n
}

// jsonEscapeWide is jsonEscapeASCII for text wider than ASCII: the index of
// the first byte that needs an escape or may begin a surrogate.
func jsonEscapeWide(s string, i int) int {
	n := len(s)
	i += jsonscan.Wide(s[i:], 0xED)
	for ; i+8 <= n; i += 8 {
		w := wtf8.Load64(s, i)
		q, b, d := w^jsonQuotes, w^jsonSlashs, w^jsonLeadED
		if m := ((q-jsonOnes)&^q | (b-jsonOnes)&^b | (d-jsonOnes)&^d | (w-jsonSpaces)&^w) & jsonHigh; m != 0 {
			return i + bits.TrailingZeros64(m)>>3
		}
	}
	for ; i < n; i++ {
		if c := s[i]; c < 0x20 || c == '"' || c == '\\' || c == 0xED {
			return i
		}
	}
	return n
}
