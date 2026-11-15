//go:build quickjs_jit && !android && !ios && ((linux && amd64) || (windows && amd64) || ((darwin || linux) && arm64))

package vm

import (
	"os"
	"strings"
	"testing"
)

// jitBackgroundWork is what TestJITBackgroundRuntimes runs: micro-benchmarks
// of every kind of code -- properties, arrays and typed arrays, calls and
// closures, strings, regular expressions, BigInts, generators, iterators,
// exceptions, private fields, spreads, the callbacks of built-ins, sorts --
// each with a count small enough for a run under -race, large enough for
// its loops to be compiled. None reads the clock or Math.random.
var jitBackgroundWork = []struct {
	name string
	n    int
}{
	{"prop_read", 8192}, {"prop_write", 8192}, {"prop_update", 4096}, {"prop_create", 512},
	{"prop_delete", 128}, {"array_read", 4096}, {"array_write", 2048}, {"array_push", 32},
	{"array_pop", 16}, {"array_slice", 32}, {"typed_array_read", 2048}, {"arguments_read", 512},
	{"global_read", 8192}, {"local_destruct", 1024}, {"func_call", 4096}, {"func_closure_call", 4096},
	{"int_arith", 32}, {"float_arith", 16}, {"map_set_string", 2}, {"map_delete", 4},
	{"weak_map_set", 8}, {"array_for_in", 64}, {"array_for_of", 64}, {"regexp_ascii", 4},
	{"regexp_replace", 2}, {"string_build1", 8}, {"string_build2", 8}, {"int_to_string", 2048},
	{"float_toFixed", 1024}, {"string_to_float", 2048}, {"bigint64_arith", 16}, {"bigint256_arith", 8},
	{"generator_for_of", 64}, {"generator_next", 32}, {"generator_delegate", 16}, {"try_throw", 8192},
	{"try_finally", 8192}, {"iterator_for_of", 32}, {"object_for_in", 512}, {"closure_make", 1024},
	{"closure_callback", 512}, {"tail_mutual", 8192}, {"method_call", 4096}, {"string_switch_mixed", 4096},
	{"has_own_call_for_in", 512}, {"array_push_args5", 4}, {"private_field", 8192}, {"prop_in", 2048},
	{"object_spread", 512}, {"object_rest", 512}, {"spread_call", 512}, {"array_map", 64},
	{"array_filter", 64}, {"array_reduce", 64}, {"typed_forEach", 64}, {"typed_map", 64},
}

// jitBackgroundRuntimes is how many runtimes TestJITBackgroundRuntimes runs
// at once, and jitBackgroundRounds how many times each runs all the work.
const (
	jitBackgroundRuntimes = 4
	jitBackgroundRounds   = 4
)


// jitMicrobenchForTest is QuickJS's micro-benchmarks and ours, as the
// v8bench runner's micro mode loads them: microbench.js without the call
// of its main that ends it, then microbench_extra.js.
func jitMicrobenchForTest(t *testing.T) string {
	t.Helper()
	base, err := os.ReadFile("../v8bench/microbench.js")
	if err != nil {
		t.Fatal(err)
	}
	extra, err := os.ReadFile("../v8bench/microbench_extra.js")
	if err != nil {
		t.Fatal(err)
	}
	s := string(base)
	end := strings.LastIndex(s, "\nif (typeof scriptArgs === \"undefined\")")
	if end < 0 {
		t.Fatal("microbench.js: no call of main to leave out")
	}
	return s[:end+1] + string(extra)
}
