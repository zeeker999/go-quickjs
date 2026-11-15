//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"strings"
)

// jitListSource is AWFY List's walk, a loop whose phis, read at its back
// edge's poll, are each written there by the edge's moves.
const jitListSource = `
	class Element { constructor(v) { this.val = v; this.next = null } }
	function makeList(n) { if (n === 0) return null; const e = new Element(n); e.next = makeList(n - 1); return e }
	function isShorterThan(x, y) {
		let xTail = x, yTail = y
		while (yTail !== null) {
			if (xTail === null) return true
			xTail = xTail.next
			yTail = yTail.next
		}
		return false
	}
	const lists = []
	for (let i = 0; i <= 12; i++) lists.push(makeList(i))
	function run(rounds) {
		let s = ""
		for (let k = 0; k < rounds; k++)
			for (let i = 0; i <= 12; i++)
				for (let j = 0; j <= 12; j++) s += isShorterThan(lists[i], lists[j]) ? "1" : "0"
		return s
	}`

// jitListWant is run(rounds)'s result, as the interpreter makes it.
func jitListWant(rounds int) string {
	var b strings.Builder
	for range rounds {
		for i := 0; i <= 12; i++ {
			for j := 0; j <= 12; j++ {
				if i < j {
					b.WriteByte('1')
				} else {
					b.WriteByte('0')
				}
			}
		}
	}
	return b.String()
}
