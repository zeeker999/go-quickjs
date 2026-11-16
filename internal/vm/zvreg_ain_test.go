package vm

import (
	"fmt"
	"testing"
)

// hasSet spreads the keys a program asks of one layout over the sets of
// hasCache: names that differ only in their middle or in a digit at their
// end, and single letters, each family no more than a few to a set, where
// two to a set is what an entry's set holds. It reads what the string and
// the layout hold, so the spread is the same in every run.
func TestHasSetSpread(t *testing.T) {
	r := New(Config{})
	defer r.Close()
	v, err := r.Run(compileForTest(t, `({ a: 1, b: 2, c: 3 })`))
	if err != nil {
		t.Fatal(err)
	}
	sh := v.Object().shape
	if sh == nil {
		t.Fatal("no layout")
	}
	families := map[string][]string{}
	for i := range 100 {
		families["item%d"] = append(families["item%d"], fmt.Sprintf("item%d", i))
		families["k%d"] = append(families["k%d"], fmt.Sprintf("k%d", i))
		families["prop_%d_x"] = append(families["prop_%d_x"], fmt.Sprintf("prop_%d_x", i))
	}
	for c := 'a'; c <= 'z'; c++ {
		families["letters"] = append(families["letters"], string(c))
	}
	families["keyed"] = []string{"a", "z", "c", "y"}
	for name, keys := range families {
		sets := map[uint64]int{}
		most := 0
		for _, k := range keys {
			set := hasSet(NewString(k), sh)
			sets[set]++
			most = max(most, sets[set])
		}
		// Keys spread at random over the 128 sets would put, of 100, at most
		// four or five in one; of four, two at most.
		limit := 5
		if len(keys) <= 4 {
			limit = 2
		}
		if most > limit {
			t.Errorf("%s: %d keys in %d sets, %d in one, want at most %d", name, len(keys), len(sets), most, limit)
		}
	}
}
