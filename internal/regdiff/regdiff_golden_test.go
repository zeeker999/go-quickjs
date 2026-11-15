package regdiff

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"testing"
)

// The differential tests' recorded answers. Each comparison's program and
// calls were run on the stack interpreter and the tree tier, the tiers the
// register interpreter replaced, and every answer recorded in
// testdata/golden.json, keyed by the test's name and the comparison's
// place among its test's. A call whose two runs there answered
// differently -- a clock, Math.random -- is recorded as null and not
// compared. QJS_REGDIFF_RECORD=1 records the file anew.

// goldenPath is the recorded answers' file.
const goldenPath = "testdata/golden.json"

// goldenCase is one comparison's recorded answers: a hash of its program
// and calls, which says the case is still the one recorded, and each
// call's answer, an answer longer than goldenLong by its hash.
type goldenCase struct {
	Program string    `json:"program"`
	Answers []*string `json:"answers"`
}

// goldenLong is the length past which an answer is recorded by its hash.
const goldenLong = 160

var (
	goldenMu      sync.Mutex
	goldenLoaded  bool
	goldenCases   map[string]goldenCase
	goldenSeq     = map[string]int{}
	goldenRecords = map[string]goldenCase{}
)

// goldenRecording reports whether this run records the answers anew.
func goldenRecording() bool { return os.Getenv("QJS_REGDIFF_RECORD") != "" }

// goldenKey is the key of t's next comparison.
func goldenKey(t *testing.T) string {
	goldenMu.Lock()
	defer goldenMu.Unlock()
	n := goldenSeq[t.Name()]
	goldenSeq[t.Name()] = n + 1
	return fmt.Sprintf("%s#%d", t.Name(), n)
}

// goldenProgram is the hash of a comparison's program and calls.
func goldenProgram(src string, calls []string) string {
	h := sha256.New()
	h.Write([]byte(src))
	for _, c := range calls {
		h.Write([]byte{0})
		h.Write([]byte(c))
	}
	return hex.EncodeToString(h.Sum(nil))[:24]
}

// goldenAnswer is answer as it is recorded.
func goldenAnswer(answer string) string {
	if len(answer) <= goldenLong {
		return answer
	}
	sum := sha256.Sum256([]byte(answer))
	return fmt.Sprintf("sha256:%s len %d", hex.EncodeToString(sum[:16]), len(answer))
}

// goldenRecord records a comparison's answers, nil where they vary.
func goldenRecord(key, src string, calls []string, answers []*string) {
	goldenMu.Lock()
	defer goldenMu.Unlock()
	goldenRecords[key] = goldenCase{Program: goldenProgram(src, calls), Answers: answers}
}

// goldenWrite writes the answers recorded, where this run records them.
func goldenWrite() error {
	if !goldenRecording() {
		return nil
	}
	goldenMu.Lock()
	defer goldenMu.Unlock()
	keys := make([]string, 0, len(goldenRecords))
	for k := range goldenRecords {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	ordered := make([]json.RawMessage, 0, len(keys))
	for _, k := range keys {
		b, err := json.Marshal(map[string]goldenCase{k: goldenRecords[k]})
		if err != nil {
			return err
		}
		ordered = append(ordered, b)
	}
	if err := os.MkdirAll(filepath.Dir(goldenPath), 0o755); err != nil {
		return err
	}
	// One case a line, so that a change to a case is a change to its line.
	var out []byte
	out = append(out, "{\n"...)
	for i, b := range ordered {
		out = append(out, b[1:len(b)-1]...)
		if i+1 < len(ordered) {
			out = append(out, ',')
		}
		out = append(out, '\n')
	}
	out = append(out, "}\n"...)
	return os.WriteFile(goldenPath, out, 0o644)
}

// goldenCheck compares the answers on with those recorded for key, and
// fails where they differ. It returns how many calls it skipped, whose
// recorded answers varied.
func goldenCheck(t *testing.T, key, src string, calls, on []string) (skipped int) {
	t.Helper()
	goldenMu.Lock()
	if !goldenLoaded {
		goldenLoaded = true
		if b, err := os.ReadFile(goldenPath); err == nil {
			if err := json.Unmarshal(b, &goldenCases); err != nil {
				goldenMu.Unlock()
				t.Fatalf("%s: %v", goldenPath, err)
			}
		}
	}
	c, ok := goldenCases[key]
	goldenMu.Unlock()
	if !ok {
		t.Fatalf("%s has no answers for %s: record them (QJS_REGDIFF_RECORD=1)", goldenPath, key)
	}
	if c.Program != goldenProgram(src, calls) || len(c.Answers) != len(calls) {
		t.Fatalf("%s: the case is not the one recorded in %s; its recorded answers are to be updated with it", key, goldenPath)
	}
	bad := 0
	for i, want := range c.Answers {
		if want == nil {
			skipped++
			continue
		}
		if got := goldenAnswer(on[i]); got != *want {
			if bad++; bad <= 20 {
				t.Errorf("%s:\n  recorded: %.300s\n  register: %.300s", calls[i], *want, got)
			}
		}
	}
	if bad > 20 {
		t.Errorf("and %d more", bad-20)
	}
	return skipped
}
