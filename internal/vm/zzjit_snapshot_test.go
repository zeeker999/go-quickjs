//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"fmt"
	"io"
	"regexp"
	"slices"
	"sort"
	"strings"
	"sync"
)

// jitSnapPending is a build the oracle is to compare: its snapshot, and
// its feedback's lists before it.
type jitSnapPending struct {
	snap          *jitSnapshot
	asked, before *jitFeedback
}

// jitSnapClone is a feedback with f's lists, copied.
func jitSnapClone(f *jitFeedback) *jitFeedback {
	return &jitFeedback{shapes: slices.Clone(f.shapes), holders: slices.Clone(f.holders), cells: slices.Clone(f.cells),
		links: slices.Clone(f.links), inlined: slices.Clone(f.inlined), made: slices.Clone(f.made),
		wanted: slices.Clone(f.wanted), linked: slices.Clone(f.linked), pools: slices.Clone(f.pools)}
}

// jitSnapListsDiffer says how the lists of a, the build's, and b, the
// replay's, differ, or "": by identity, but cells, which the feedback makes
// anew at each question, by their values.
func jitSnapListsDiffer(a, b *jitFeedback) string {
	same := func(name string, x, y int, eq func(i int) bool) string {
		if x != y {
			return fmt.Sprintf("%s: %d against %d", name, x, y)
		}
		for i := range x {
			if !eq(i) {
				return fmt.Sprintf("%s[%d] differs", name, i)
			}
		}
		return ""
	}
	for _, d := range []string{
		same("shapes", len(a.shapes), len(b.shapes), func(i int) bool { return a.shapes[i] == b.shapes[i] }),
		same("holders", len(a.holders), len(b.holders), func(i int) bool { return a.holders[i] == b.holders[i] }),
		same("cells", len(a.cells), len(b.cells), func(i int) bool { return a.cells[i].SameValue(*b.cells[i]) }),
		same("links", len(a.links), len(b.links), func(i int) bool { return a.links[i] == b.links[i] }),
		same("inlined", len(a.inlined), len(b.inlined), func(i int) bool { return a.inlined[i] == b.inlined[i] }),
		same("made", len(a.made), len(b.made), func(i int) bool { return a.made[i] == b.made[i] }),
		same("wanted", len(a.wanted), len(b.wanted), func(i int) bool { return a.wanted[i] == b.wanted[i] }),
		same("linked", len(a.linked), len(b.linked), func(i int) bool { return a.linked[i] == b.linked[i] }),
		same("pools", len(a.pools), len(b.pools), func(i int) bool { return a.pools[i] == b.pools[i] }),
	} {
		if d != "" {
			return d
		}
	}
	return ""
}

var jitSnapAddresses = regexp.MustCompile(`0x[0-9a-f]+`)

// jitSnapChecks is what the oracle found.
var jitSnapChecks = struct {
	mu      sync.Mutex
	builds  int
	misses  map[string]int
	found   []string
	note    func(string)
	pending map[*jitFeedback]jitSnapPending
}{misses: map[string]int{}, pending: map[*jitFeedback]jitSnapPending{}}

func init() {
	jitSnapChecks.note = func(s string) { jitSnapChecks.found = append(jitSnapChecks.found, s) }
}

// jitSnapReport writes what the oracle found and reports whether it found
// anything.
func jitSnapReport(w io.Writer) bool {
	fmt.Fprintf(w, "snapshot oracle: %d builds, %d differing or missing\n", jitSnapChecks.builds, len(jitSnapChecks.found))
	var ms []string
	for m, n := range jitSnapChecks.misses {
		ms = append(ms, fmt.Sprintf("%s:%d", m, n))
	}
	sort.Strings(ms)
	if len(ms) != 0 {
		fmt.Fprintf(w, "  misses: %s\n", strings.Join(ms, " "))
	}
	for i, s := range jitSnapChecks.found {
		if i == 40 {
			fmt.Fprintf(w, "  ...\n")
			break
		}
		fmt.Fprintf(w, "  %s\n", s)
	}
	return len(jitSnapChecks.found) != 0
}
