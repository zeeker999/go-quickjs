# JIT measurements

Measurements of the native tier, newest first. Each entry says:
- the host and Go version;
- the commit measured;
- how the measurement was run.

Compare like with like: fresh processes, the same host, and `placements` for
anything under 3%. The baseline is the tree tier, not the bytecode
interpreter (see [the production plan](jit-production-plan.md)).

The branch's original measurement diary (Apple M5 Max, October 7-8, 2026)
covers each stage it built, with Node comparisons and first-use costs. It is
kept in git history, on branch `jit-wip-backup`, as `internal/jit/README.md`
and `docs/jit-plan.md`. Its evidence directories lived outside the repository
and are not reproducible from it.

## 2026-10-08: Phase 0 (Ryzen 5 3600, Windows 10, amd64, Go 1.27.1)

**Binary size** of `cmd/qjs` at 43efbe2:

| Build | Bytes | Against main 6c3dd16 |
|---|---|---|
| main | 41,909,248 | |
| this branch, untagged | 41,912,320 | +3,072 (the `--jit` flag and glue outside the interpreter) |
| this branch, `quickjs_jit` | 42,417,152 | +507,904 (+1.2%) |

**Interpreter identity:** `hotdiff` finds `executeAt`, `runTree`,
`runTreeNested`, `callObject`, `callTree`, `runFD` and `backEdgeCheck` the
same as main in the untagged build.

**JIT off against main** (`placements`, eight placements, three rounds,
fixed work; main 6c3dd16):

| Suite | main | tagged, JIT off | change | untagged |
|---|---|---|---|---|
| Richards | 13.7 | 13.9 | +1.6% | +0.7% |
| DeltaBlue | 23.7 | 23.9 | +0.7% | -0.1% |
| Crypto | 274.2 | 263.7 | -3.8% | -1.3% |
| RayTrace | 137.1 | 137.6 | +0.3% | -0.2% |
| EarleyBoyer | 375.0 | 377.0 | +0.5% | -0.4% |
| RegExp | 274.0 | 272.2 | -0.6% | +1.6% |
| Splay | 279.8 | 281.7 | +0.7% | +0.1% |
| NavierStokes | 185.2 | 181.5 | -2.0% | -0.5% |
| TOTAL | 1600.6 | 1585.7 | -0.9% | -0.2% |

The untagged figures are from the run before the last fix, which touched only
the tagged build's call path and file order (`hotdiff` shows the untagged hot
code unchanged). Before that fix the tagged build was +2.1% overall, with
Crypto +7.0% and NavierStokes +6.6%.

**Removing the selectors fitted to Crypto and MD5** (70d2c86) costs nothing
measurable. V8 fixed work, 20 iterations, JIT on, three alternating fresh
processes on one layout:

| Suite | Before | After |
|---|---|---|
| Crypto | 1009.4 ms | 1020.0 ms (+1.0%) |
| NavierStokes | 301.5 ms | 300.5 ms |

**Kernels** at c55bc9e (`BenchmarkJITNumericKernels`, `BenchmarkJITDenseKernels`;
three processes, medians). Native code against the tree tier ("existing"):

| Kernel | Tree tier | Native | Gain |
|---|---|---|---|
| vector, 8192 elements | 187 µs | 50.6 µs | 3.7x |
| stencil, 8192 elements | 428 µs | 87 µs | 4.9x |
| stencil with helper calls | 1720 µs | 339 µs | 5.1x |
| logistic, `var` | 154 µs | 27.4 µs | 5.6x |
| Newton, `var` | 192 µs | 63.2 µs | 3.0x |
| particle, `var` | 659 µs | 110.6 µs | 6.0x |

With `let` locals the tree tier itself is 2.4-2.8x slower than with `var`
(logistic: 431 µs). That gap is the default tiers' to close.

**test262 with `-conformance.jit`:** 99,599 passed, 0 failed, 342 skipped, as
without the JIT. But native code ran in only 452 of the 99,941 runs, or 4,484
with a call threshold of 1, so this is weak evidence for the compiled code.

## Calibration (from the branch's diary, Apple M5 Max)

V8 v7 fixed work, 50 iterations, ms. Node `--jitless` comes from an earlier
table on the same host and driver. The last two columns compare each JIT's
gain over its own interpreter.

| Suite | Tree tier | Branch JIT | Node `--jitless` | Node | V8 JIT ÷ jitless | Branch JIT ÷ tree |
|---|---|---|---|---|---|---|
| Richards | 96.1 | 96.3 | 62.7 | 4.1 | 15.3x | 1.0x |
| DeltaBlue | 141.2 | 147.2 | 105.8 | 5.9 | 17.9x | 0.96x |
| Crypto | 2066 | 1017 | 2175 | 69.2 | 31.4x | 2.0x |
| RayTrace | 791 | 794 | 421 | 26.7 | 15.8x | 1.0x |
| EarleyBoyer | 2369 | 2422 | 1202 | 109.8 | 10.9x | 0.98x |
| RegExp | 1128 | 1118 | 707 | 194.4 | 3.6x | 1.0x |
| Splay | 153 | 152 | 66.7 | 23.3 | 2.9x | 1.0x |
| NavierStokes | 1299 | 370 | 1793 | 98.1 | 18.3x | 3.5x |
| Total | 8052 | 6127 | 6570 | 564 | 11.7x | 1.31x |
