//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

// staleInlineSetup's run inlines get and calls step natively, whose
// forEach -- a call of Go -- calls replaceRun at the run's 30th object and
// its 50th, which acts on the second, native code going on after it; the
// 102nd object, the last, whose k is a getter, leaves get's inlined code.
const staleInlineSetup = `
	function get(o) { return o.k }
	function step(k) { if (k) [0].forEach(replaceRun); return 1 }
	var plain = [], mixed = []
	for (let i = 0; i < 102; i++) {
		plain.push({ k: i })
		mixed.push(i < 101 ? { k: i } : { get k() { return 7 } })
	}
	function run(os, at) {
		let s = 0
		for (let i = 0; i < os.length; i++) {
			s += step(i === at || i === at - 20)
			s += get(os[i])
		}
		return s
	}
`

