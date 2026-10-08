# JIT progress tracker

This tracks the work in [jit-production-plan.md](jit-production-plan.md)
across sessions. Update it **in the same commit** as the work it records.

## Resume here

- **Worktree:** `D:\Data\go-quickjs-jit`, branch `jit-wip`, based on main
  6c3dd16. Pushed to `zk/jit-wip` when CI is wanted on Linux or macOS (the
  user allows that); `zk/codex/jit-wip` keeps the old history, never
  force-pushed.
- **Original history:** branch `jit-wip-backup` (82aa960). Everything up to
  that point is squashed into c55bc9e.
- **Test commands** (Windows/amd64 here; `QUICKJS_REQUIRE_JIT=1` fails on
  silent fallback):

  ```sh
  go test -tags quickjs_jit -count=1 ./internal/jit/... ./internal/vm
  go test -tags quickjs_jit -count=1 ./...        # full tagged
  go test -count=1 ./...                          # full untagged
  TEST262_DIR=d:/Data/test262 go test -tags quickjs_jit ./conformance \
    -run TestConformance -count=1 -timeout 60m -v -args -conformance.jit
  ```

- **Next item:** P4b (arrays through views), while the 24-hour fuzz, the
  last Phase 1 gate, runs.
- **Stress test262:** `QJS_JIT_STRESS=threshold,budget=1 TEST262_DIR=d:/Data/test262 go test -tags quickjs_jit ./conformance -run TestConformance -v -args -conformance.jit`;
  `QJS_JIT_PIPELINE=ssa` runs the new pipeline wherever it compiles.

## Phase 0: stabilize and cut

| ID | Item | Status | Commit |
|---|---|---|---|
| R1 | Native loops honor `Halt`/`Close` after a host exit | done | jit: stop native loops at Halt after a host exit |
| R2 | JIT code outside the script's memory budget, evictable, never the cause of `ErrMemoryLimit` | done | jit: keep native code out of the script's memory |
| R3 | No full heap walk per compile attempt; budget refusals cached until code is released (instead of exponential back-off) | done | jit: keep native code out of the script's memory |
| R4a | `recover` in compilation becomes a permanent refusal | done | jit: refuse instead of crashing when compilation panics |
| R4b | Size `inferIntegerResults`'s origin table from the final origin count, with an invariant test | done | jit: refuse instead of crashing when compilation panics |
| R7 | No silent miscompile paths: `lower` has no default `Nop` (an untranslated opcode panics and is refused), kind inference names every operation (else refused), `Validate` rejects handle literals; `TestLowerCoversDescribedOpcodes` walks every opcode. A single shared per-opcode table waits for Phase 2's lowering. | done | jit: refuse what lowering and kind inference do not know |
| Z1 | `closure` stays 128 B in tagged builds (index, not pointer); size and field-order checks in both builds (`TestJITFieldLayout`) | done | jit: cost nothing while off |
| Z2 | `jitTreeRecovery` only for frames whose function can enter a native loop | done | jit: cost nothing while off |
| C1 | Delete `dispatch*.go`/`.s` and `loop*.go` with their tests | done | jit: delete the unused dispatch layer and loop prototype |
| C2 | Delete the shape-matched selectors (`selectShortCountdown`, `selectArrayGrowth` with native growth, the 16-cell preallocation, the string-packing host discount); re-measure. `charCodeAt` stays: its native call is guarded by the realm's intrinsic, so it is general; only its selection (the name anywhere in the function) is crude, and moves to per-site selection in Phase 2's SSA. | done | jit: drop selectors fitted to Crypto and MD5 |
| C3 | Untagged hot functions identical to main: `internal/cmd/hotdiff` compares them instruction by instruction (ignoring NOP inline marks, padding and addresses); struct sizes by `TestJITFieldLayout`. A local gate like `placements`, not CI: an intended interpreter change would fail it. Run it before merging JIT work. | done | jit: compile the interpreter's hooks out of builds without the JIT |
| D1 | Rewrite `internal/jit/README.md` as contracts; move measurements to `docs/jit-results.md`; `qjs --jit` warns when the build lacks the JIT | done | jit: document contracts, not a diary |

**Gate (met 2026-10-08):**
- [x] Full suites (tagged and untagged) and stress-mode test262 pass: 99,599
  passed, 0 failed, with native code in 452 runs (4,484 under stress), as
  before Phase 0.
- [x] Untagged hot functions identical to main (`hotdiff`, all seven).
- [x] Tagged JIT-off `placements` level against main: -0.9% total. Richards
  is +1.6% (13.7 to 13.9 ms, inside both sides' 13-14 ms spread); every other
  suite is within -3.8% to +0.7%. The untagged build is -0.2% total.

  The first tagged run was +2.1% (Crypto +7.0%, NavierStokes +6.6%), for two
  reasons, both fixed in "jit: keep the JIT's code out of the interpreter's
  way":
  - the hooks did not inline, so every call paid a function call;
  - zjit_*.go sorted before ztree_*.go and moved the tree tier's code.

## Phase 1: verification infrastructure

| ID | Item | Status | Commit |
|---|---|---|---|
| V1 | Stress knobs, internal only: `QJS_JIT_STRESS=threshold,budget=N,deopt=N` (compile on the first call; return to Go every N native instructions; finish in the interpreter at every Nth return, from wherever native code got to). `TestJITStressDifferential` runs a corpus under five settings against the JIT off. | done | jit: stress mode and native coverage in test262 |
| V2 | `-conformance.jit` reports per area how many tests ran natively, and fails a JIT build in which none did (per-area failure was too strict: Intl areas rarely compile). Counters come from `vm.Runtime.JITStats` through `hostaccess`. | done | jit: stress mode and native coverage in test262 |
| V3 | `FuzzJITDifferential` generates programs in the JIT's subset (numbers and edge values, arrays with holes and out-of-range indices, bitwise and comparison operators, properties and accessors, helper calls that throw, break/continue, BigInt and `valueOf` operands) and compares the interpreter, the tree tier and the JIT under three stress settings, including an effect log. `TestJITDifferentialRandom` runs 300 fixed programs in every test run. 10 minutes: 635,214 programs, no divergence. | done | jit: fuzz the JIT against the interpreter |
| V4 | `internal/jit/verify`, a module of its own, disassembles every program both emitters make from a JavaScript corpus and 3,000 random IR programs with `x/arch` (v0.22.0, Go 1.24): each must decode in full, with no call, push, pop, system call or trap, and no use of Go's reserved registers. Both emitters now build on every supported target (`emit_x86.go`, `emit_a64.go`), so arm64 is checked off macOS. Per-instruction golden tests remain for Phase 2's encoders. | done | jit: check both emitters against a disassembler on every target |
| V5 | CI: the native job runs all `internal/jit` tests, the VM's JIT tests, checkptr and the encoder verifier on the three targets with Go 1.24 and 1.27; `jit-conformance` runs test262 under `QJS_JIT_STRESS=threshold,budget=3,deopt=7` on the three targets (test262 pinned at 7ab7faf); `jit-fuzz` fuzzes for 5 minutes on Linux. Every command was run locally on windows/amd64; Green on GitHub from run 37801457816. | done | ci: run the JIT's verification on every target |
| V6 | Round trips measured on amd64 (`BenchmarkNativeRoundTrip`, `BenchmarkJITHostRoundTrip`): a bare entry and exit costs 23.7 ns (D7's target is 15 ns or less); a host exit as the VM takes one today costs 43 ns for `%` and 114 ns for a Go call, against 7 and 30 ns in the tree tier. arm64 is still to measure on a Mac. | done | jit: measure what a return to Go costs |

**Gate:**
- [x] Stress test262 passes on all three platforms: CI run 37801457816
  (2b0d682), every job green, linux/amd64, windows/amd64 and darwin/arm64.
  The first run (958ac22) failed only off the JIT: the runners' zone,
  Etc/UTC, was named "Etc/UTC" instead of "UTC" (fixed on main, 24c5fa5),
  Windows checked test262 out with CRLFs (the job sets `core.autocrlf
  false`), and the differential test wanted SSA entries on arm64, which has
  no SSA backend yet (`jitSSABackend`).
- [ ] Fuzzer runs 24 hours with no divergence. The first start (2026-10-08,
  10:16 EDT) stopped after 18 minutes on a program over the memory limit
  that the interpreter finished and the JIT stopped: not a divergence, since
  the meter measures on process-wide allocation, but the harness compared
  it. Such runs are now inconclusive (`outOfMemory`), and the input is a
  seed. The meter itself lets a rope that shares itself be written out
  unmetered, past the limit, on main too: a main-branch follow-up, not the
  JIT's. Restarted 2026-10-08 as a detached process (6 workers,
  coverage-instrumented binary). Progress goes to
  `D:/Data/quickjs-jit-results/2026-10-08-review/fuzz-24h.err` (stderr); a failing
  input lands in `fuzzwork/testdata/fuzz/FuzzJITDifferential/` beside it.
  Before that: 15 minutes in two runs, 700,000 programs, no divergence.
- [x] Round-trip costs published (amd64; arm64 waits for a Mac or the macOS
  runner).

## Phase 2: the new pipeline at parity

Design: [jit-phase2-design.md](jit-phase2-design.md). P2 gates the rest.

| ID | Item | Status | Commit |
|---|---|---|---|
| P1 | `internal/jit/ssa`: types, builder (Braun et al.) from the slot IR, evaluator, Phase 2 passes, for the numeric subset (copies, stack shuffles, every arithmetic, bitwise and comparison operator, updates, branches, TDZ checks, returns; host operations as exits). Arrays, properties, strings and calls are refused until P4. 3,000 random programs and a JavaScript corpus match the slot IR on 172,809 comparisons, built and optimized, with and without polls. | done (numeric subset) | jit: build typed SSA from the slot IR |
| P2 | Walking skeleton on amd64: `asm/amd64` (golden tests), `mir` (SSA to amd64), `jit.SSACode`, and the VM running functions through it (`QJS_JIT_PIPELINE=ssa`, or `Runtime.jitSSA`). Gate: bare round trip **4.0 ns** (target 15 ns or less, met); numeric kernels faster than the old pipeline (met: logistic 1.47x, Newton 1.31x, particle 2.05x; 4-22x the tree tier); helper round trip **43 ns for `%`** (target 25 ns or less, not met). The profile puts that cost in Go re-decoding the bytecode instruction at every exit (`jitHost`, `jitBinaryAt`), which P5's helper table replaces, so it moves to P5's gate. A Go call through the new pipeline (66 ns per iteration) is level with the tree tier (72 ns). | done (helper cost moves to P5) | jit: run functions through the new pipeline |
| P3 | The skeleton on arm64, measured on a Mac or the macOS runner | todo | |
| P4 | Coverage: every slot-IR operation built; stress corpus and fuzzer pass on the new pipeline | in progress | |
| P4a | References carried natively: no entry check; exits leave references to Go through records (copy, store over, "maybe" for a phi of two slots, which carries a run-time *shadow* of its origin), `RetFrom` for returns; the evaluator checks every origin. `unboxPhis` unboxes a phi only for an unboxed use, so a phi that merely carries a number no longer guards an entry load that may hold a reference (kernels' code unchanged). Harness: 281 copies, 1,549 stores over references, 26,066 maybes, 273 reference returns over 2,000 programs; `TestJITSSAReferences`. | done | jit: carry references through native code |
| P4b | Arrays through read-only views (D8): element reads and numeric writes | next | |
| P4c | Properties, globals, captured bindings, strings; calls as helper exits | todo | |
| P5 | Helpers: contained and reentrant, re-validation, generation counter; `%` and Go calls no slower than the tree tier | todo | |
| P6 | Code arena (R8) | todo | |
| P7 | Parity on both architectures, then delete the old pipeline | todo | |

**Gate:**
- [ ] Every kernel and suite at least as fast as the old pipeline on both
  architectures.
- [ ] No divergence under the Phase 1 tools.
- [ ] Compile budget met.

## Phases 3-6

These are tracked here once Phase 2's gate is met; see the plan for their
contents.

## Decisions

| Date | Decision |
|---|---|
| 2026-10-08 | Baseline is the tree tier. Targets per category (plan section 2), calibrated by V8 JIT ÷ jitless. |
| 2026-10-08 | JIT off must cost nothing, in both untagged and tagged builds. |
| 2026-10-08 | Go-side calls use wazero's exit/resume on a native stack. No pointer stores or allocation from native code. |

**Open:** D8's reference-reassignment question (Phase 4 spike); macOS
hardened-runtime support (Phase 6).

## Baseline measurements (2026-10-08, Ryzen, Windows/amd64, Go 1.27.1)

- **Kernels, native vs tree tier** (`BenchmarkJITNumericKernels`/
  `DenseKernels`):
  - vector 3.7x, stencil 4.9x, stencil with helper calls 5.1x;
  - logistic 5.6x, Newton 3.0x, particle 6.0x (`var` locals).
- **test262 with `-conformance.jit`:** 99,599 passed, 0 failed, 342 skipped.
  Native entries in 452 of 99,941 runs; at threshold 1, in 4,484.
- **Struct sizes in a tagged build:** closure 144 B (main 128), Runtime 7952 B
  (main 7936), Realm 1264 B (main 1248). After Z1: closure 128 B, and
  Runtime and Realm grow only at their ends.
- **Untagged hot functions:** symbol sizes alone hid a difference.
  `executeAt` and `runTree` had main's sizes but different instruction
  order and registers, perturbed by the hooks' stubs. Since C3, `hotdiff`
  finds all seven hot functions identical to main. `callTree` still carries
  three `NOPL` inline marks, which take it from 1760 to 1792 B.

  ```sh
  go build -o main.exe ./cmd/qjs   # in a main checkout (or: git checkout --detach main here)
  go build -o head.exe ./cmd/qjs   # here
  go run ./internal/cmd/hotdiff main.exe head.exe
  ```
