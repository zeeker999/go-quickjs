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

- **Next item:** P7. The V8 suite is level with the old pipeline (P7f);
  left before retiring it: deoptimize-and-reoptimize with a failed site
  made generic and a give-up limit (deopt loops now only run in the
  interpreter); native prototype method lookups and cheaper exits, so
  object-oriented code is worth running natively rather than demoted;
  same-length string contents; make the new pipeline the default and run
  the 24-hour fuzz; then delete the old pipeline. Both architectures are
  supported (the user's requirement): test every change on amd64 here and
  on arm64 through CI's macOS jobs, whose JIT kernels job tabulates both
  pipelines. The user asked (2026-10-09) to optimize as a V8 engineer
  would, not by copying the old JIT, and to fix the allocation counts.
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
  JIT's. The second run (from 10:46 EDT, built at 57dd1e7) stopped after
  2 h 39 min, 8 million programs, on a hang in the interpreter's own run:
  a BigInt squared in a loop under the harness's 16 MB limit, which main
  now reserves before computing (6bf6058, cherry-picked); its input is a
  seed. A third run (from 17:35, at 72f22fd) was stopped clean after 20
  minutes and 689,000 programs: by the user's rule, the 24-hour run is made
  when a milestone is reached -- here, with Phase 2's other gate items --
  not left running during development. Meanwhile the seeds run in `go
  test` and CI fuzzes for 5 minutes on every push. Progress goes to
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
| P3 | The new pipeline on arm64. Done: mir split into a shared core and per-architecture code generators (amd64's output byte-identical); an A64 encoder (`asm/arm64`, golden-tested against arm64asm); arm64's code generator, every op amd64's has; `SSACode`, the harness and the VM glue for darwin/arm64. CI run 37856570182 (ae18d4a), macos-15: the harness and VM JIT tests pass on arm64, and test262 under stress with `QJS_JIT_PIPELINE=ssa` passes 99,599 of 99,599 with 1,403,956 entries into arm64 code; the verify module decodes the arm64 output of 2,000 functions on any host. Still to do: performance on arm64 (no Mac here; the runner is shared and noisy), and linux/arm64. | done (performance not yet measured) | jit: the new pipeline on arm64 (P3, first cut) |
| P4 | Coverage: every slot-IR operation built (`TestBuildEveryOperation`); stress test262 and the fuzzer pass on the new pipeline | done (P4a-P4g) | |
| P4a | References carried natively: no entry check; exits leave references to Go through records (copy, store over, "maybe" for a phi of two slots, which carries a run-time *shadow* of its origin), `RetFrom` for returns; the evaluator checks every origin. `unboxPhis` unboxes a phi only for an unboxed use, so a phi that merely carries a number no longer guards an entry load that may hold a reference (kernels' code unchanged). Harness: 281 copies, 1,549 stores over references, 26,066 maybes, 273 reference returns over 2,000 programs; `TestJITSSAReferences`. | done | jit: carry references through native code |
| P4b | Arrays in place (D8): `ArrayOf` finds the array through its value's origin or shadow and checks the class; element reads of numbers, number stores into number elements, length (sparse included), index keys; holes, other values, other objects and other keys exit as the slot IR's views do (hole writes go to Go). Evaluators share the slot IR's view semantics; the harness lays out fake objects through `abi.Encoding`; sabotaged reads, writes, lengths and shadow lookups each fail it. `BenchmarkJITArrayKernels` (fuzzer running): vector 8.8 ns/element against 12.5 old and 40.5 tree; stencil 18.9 / 22.7 / 103.9; dot 7.7 / 11.3 / 45.7. | done | jit: read and write arrays in place |
| P4b+ | Array loops at native speed: hoist `ArrayOf` and the elements' header out of loops (nothing changes them natively), int32 induction variables, bounds-check elimination. 7.7 ns per element of a dot product is ~50 instructions. | todo (Phase 3) | |
| P4c | Captured bindings read in place through their cells (`Context.Upvalues`, `Func.FrameLocals`), the shadow lookup included; the builder records the slots instructions write (`Func.Written`, apart from the phis reads record), and mir refuses a function that writes a captured binding. Harness: captured values in cells, the frame's entries poisoned; `TestSSANativeCapturedShadow`; `TestJITSSACaptured`. `BenchmarkJITDenseKernels` (closures over n): vector 41 us against 65 old and 297 tree; stencil 89 / 164 / 542; helper 504 / 548 / 1,911. | done | jit: read captured bindings in place |
| P4d | The receiver, read from the context (`Context.This`, `Func.ThisSlot`), and properties in place (D8): a shape guard against the site's cache (`ssa.Feedback`, kept alive by the entry) gives the index; any other ordinary object with at most 8 properties is searched for the key, unrolled, as the VM's small objects are (they have no shape until a cache asks); numbers only, the rest exits to Go. `jitcompile.LowerSSA` drops the old pipeline's profitability rules (properties only in loops with arrays or bitwise work, the receiver only in property loops) and leaves global and reference reads to Go; the VM lowers for the new pipeline first. Harness: shapes, keys and flags; unpolled runs bounded at 2**20 back-edges, so a diverging loop fails rather than hangs. `BenchmarkJITNumericFields`: 55 us against 69 old and 651 tree. | done | jit: read the receiver and properties in place |
| P4e | References read from objects (`ReferenceRead`), per D8's decision A: `PropCell` finds the property's cell, `LoadCell` its value, whose shadow is the cell's address; shadows are 64-bit `Source` values (slot, -1, or a cell at or above 256), and `ObjectOf` finds an object through a cell too, so `o.a.b`, `n = n.next` and `this.items[i]` stay native; Go copies from the cell at exit (`vm.jitSource`). The evaluator checks every reference against its cell. `TestJITSSACellsUnderGC` builds, walks and drops lists while the collector runs without pause. | done | jit: carry references read from objects by their cells |
| P4f | Globals read in place, as cells (`GlobalCell`): the binding where the interpreter last found it in the closure's scope (`Context.Global`), the name's and plain, initialized data, unless a script-level lexical binding of the name shadows it (its bit in `Context.LexNames`, as `lexShadows` asks). `LowerSSA` keeps global reads, with no binding-view slots. `TestJITSSAGlobals`: numbers, a function a call is given, an object's field, Math, and a `let` that comes to shadow a global property. | done | jit: read globals in place |
| P4g | Strings: `x.length` of a string as of an array (`OpLength`, rope or not), and charCodeAt (`StringMethod`, `StringCode`): Go puts the intrinsic in a context cell before an entry of code that calls it, while String.prototype still has it as made; a call of it reads a flat string's byte (ASCII) or cached code unit natively, and leaves the rest to Go. `TestSSANativeStrings` covers every form, index and callee; `TestJITSSAStrings` a hash loop, a rope, and a replaced charCodeAt. | done | jit: read strings in place |
| P5 | Helpers: contained and reentrant, re-validation, generation counter; `%` and Go calls no slower than the tree tier. The gate is met (P5a, P5b). The rest moves to the phase that hoists guards and adds native calls (see Decisions, 2026-10-09): re-validation re-checks hoisted guards and borrowed views, and the new pipeline has neither -- it reads the heap in place after every exit. | done (gate met) | |
| P5a | `%` natively, without a helper: integers below 2**63 by `IDIV` (a zero remainder takes the dividend's sign; a divisor of -1 does not divide), the rest -- fractions, NaN, infinities, a zero divisor -- exits to Go, which computes `math.Mod`. A slot IR `Mod` operator, made only by `LowerSSA`. x87's `FPREM` was tried and dropped: it is exact, but microcoded, absent on arm64, and Go itself computes `math.Mod` in software on every architecture (the user asked). `TestSSANativeRemainder`, including a remainder stored through the operand stack after `IDIV` takes RDX. `BenchmarkJITHostRoundTrip/remainder`: 5.0 ns per iteration, against 48.6 in the tree tier and 53.5 in the old JIT (0 host exits). | done | jit: compute integer remainders natively |
| P5b | A call of a Go function leaves native code once, as fast as the tree tier. The callee and arguments only Go uses stay native: `LowerSSA` keeps a global read whose value no native operation takes (the old pipeline's numeric-use rule made it an exit) and makes such a property read a reference read, both carried by their cells. A function compiled before the interpreter ran a global read finds the global where the global object has it now; the cell's key check catches a move. An exit's value with a shadow needs no record when it turns out to be a number stored where no reference is, or its own slot's reference: native code checks both (amd64 and arm64); Go applies one record without a buffer. `TestJITSSACallExits` (one exit per call; first-call compilation; a global deleted and defined again), `TestJITSSAReferences/overwrite` rewritten so the slot really holds the reference a number replaces. `BenchmarkJITHostRoundTrip/go-call`: 59.6 ns per iteration, from 77 (2 exits and 2 records per call), against 70.3 in the tree tier and 119 in the old JIT. Stress test262 on the new pipeline: 99,599 passed, 0 failed. | done | jit: leave native code once per call to Go |
| P6 | Code arena (R8): a runtime's code in its `jit.Arena`, chunks of 64 KiB (a larger function gets its own), functions packed at 64-byte alignment; placing one opens the pages it touches, copies and seals them again (never writable and executable at once; sound because a runtime's native code always returns before Go compiles); a chunk is unmapped with its last function; the arena locks for finalizers. Both pipelines (`CompileIn`, `NewSSACode(arena, ...)`); a nil arena gives a code its own. Darwin's code-signing check is asked once. `TestArenaPacksCode` (200 functions, one spanning pages, run again after half are released and 100 more placed among them), `TestArenaHoldsBothPipelines`, `TestWindowsArenaProtectionAndRelease` and `TestLinuxArenaProtectionAndRelease` (read/execute page by page, committed while any code is owned, freed with the last), `TestJITArenaMappings`: 101 functions in 2 mappings (new pipeline) or 6 (old), where each was one -- on Windows 64 KiB of address space each. The harness places its thousands of programs in one shared arena. First use level (0.96-1.04x, inside the noise of the JIT-off rows). Stress test262: 99,599 passed, 0 failed, on both pipelines. | done | jit: keep a runtime's native code in an arena |
| P7 | Parity on both architectures, then delete the old pipeline. Note: the old pipeline's call-chain tests (`zzjit_calls_test.go`) assert its own counters (`transfers`); under the new pipeline their programs answer correctly but those assertions fail, and they go with the old pipeline or move to Phase 5's inlining. | in progress | |
| P7a | Measured on both architectures: CI's `JIT kernels` job (macos-15 and ubuntu-latest) runs every kernel with a new-pipeline mode, `ScalarCalls`, and the promotion benchmarks once with each pipeline, and tabulates the new pipeline's time over the old one's and the tree tier's in its summary. arm64's first run put logistic at 1.13x the old pipeline: the arm64 generator wrote every result to scratch and copied it, built 1.0 from a general register, and jumped to the next instruction after a branch. Results now go to their registers on both architectures (amd64's SSE arithmetic in two instructions, not three), arm64 encodes FMOV immediates (`FloatImm`, all 256 checked against x/arch) and zero from XZR. CI run 37868061808 (46e6f4b), arm64: every kernel at or below the old pipeline, 0.46x (particle) to 0.92x (ScalarCalls), logistic 0.77x. amd64 (idle Ryzen): particle 0.77x and logistic 0.87x of the new pipeline's own previous time. A compiled function calling a compiled function (`ScalarCalls`, a new ssa mode) is already faster through Go than the old coordinator's transfers: 172 us against 180, 385 in the tree tier; so native calls are not what retiring the old pipeline waits on. | done | ci: time the JIT's kernels on arm64 and amd64; jit: write arithmetic results to their registers |
| P7b | Compile time (`BenchmarkJITCompile`, the VM's own path, both pipelines): the new pipeline took twice the old one's. Its maps keyed by SSA values are slices by ID (mir's core, liveness as bit sets, `ssa.OriginMap`, the optimizer), mir's location table is formatted on demand, instructions are chosen by switch, values and slices come from slabs; the machine code is byte-identical throughout (an amd64 fingerprint of 3,011 functions). Now (idle Ryzen): sum 45 us (old 40), particle 128 (old 145), a 158-instruction region 205 (old 340), 246 KB. The budget is 50 us and 64 KB for 200 instructions: still about 4x off. What is left is spread: the SSA builder (38%), mir (32%), the optimizer (24%); a Value is 112 bytes; mir's use list takes a use per slot of every guard's state. | in progress | jit: time compiling, in both pipelines; jit: compile with tables indexed by ID, not maps; jit: optimize SSA with tables indexed by ID; jit: choose arithmetic instructions by switch, not a map literal; jit: take SSA values and slices from slabs |
| P7c | The V8 suite through each pipeline (2026-10-09, `-mode fixed -n 3`, idle Ryzen): the new pipeline 1,622 ms against the old one's 1,417 and the tree tier's 1,603; Crypto 2.0x the old pipeline, Richards 1.58x, DeltaBlue 1.65x. Per-function counters (`jitSSAStats`) and exact guard sites (`abi.Context.ExitSite`, the IR PC of the failing operation, -1 for an entry's speculation; `FrameState.Site`) found three causes. (1) Callees the old coordinator compiled for itself only, which ran in the tree tier when new-pipeline code called them -- Crypto's am3 among them: fixed, Crypto 327 to 154.5 ms (old 165.5), the suite 1,451. (2) Comparisons with null or undefined taken for numbers, whose unbox then counted as evidence that a phi carries numbers, so the entry speculated it -- Richards' list walks failed a guard on every call: `OpStrictNullish` and `OpLooseNullish` (Annex B's [[IsHTMLDDA]] read through the object's origin) compare natively; only guard unboxes count as evidence; a global that can never change (undefined, NaN, Infinity: neither writable nor configurable) is a constant behind a deopt-only check. (3) Array elements that are objects fail `ElemRead`, and code exits to Go at every call (DeltaBlue): next, with deoptimize-and-reoptimize and a cost model. | done | jit: leave callees to the new pipeline when it runs; jit: compare with null natively, and report exit sites |
| P7d | Element reads by cell (`ElemCell`, an IR `Reference` flag lowering sets for a result no native operation takes as a number) and equality reading its operands as they are; then a profitability policy instead of the old pipeline's probes: each native stretch's work is estimated (IR instructions from entry to exit, a mean loop length per native back-edge) and a function whose stretches mostly end leaving for Go after fewer than ten instructions runs in the tree tier (`jitSSAProfit`, `entrySlow`), mid-invocation too; `jitHost` uses the site caches. DeltaBlue 40 to 25 ms. | done | jit: read elements carried as references by their cells; jit: let equality take its operands as they are; jit: run in the tree tier what costs more native than it saves |
| P7e | Compile allocations, V8's way: `ssa.Workspace` and `mir.Workspace` (arenas a runtime rewinds after each compile, `arena.KeepBig` for big tables), tables by ID instead of maps, reused assemblers. `BenchmarkJITCompile` against the old pipeline: sum 52 allocations, 6.5 KB, 26 us (61, 62 KB, 40 us); region 82, 41 KB, 115 us (90, 762 KB, 340 us). The V8 suite's allocations within 15-30% of the old pipeline's (Crypto 63,881 to 11,170; old 9,673). `TestWorkspaceCompilesTheSame` checks the bytes are the same. | done | jit: allocate less to compile; jit: compile in workspaces a runtime rewinds; jit: keep a compile's tables, lists and assembler from one to the next |
| P7f | Spill slots reused once their values die (a slot only where its last occupant ended before the spilled interval began: taking any freed one miscompiled, `TestJITSSAManySpills`, `TestSSANativeHighPressure`), so NavierStokes' project and lin_solve2 compile; native equality of any two values (`OpEqTagged`: words, objects by pointer through their origins, numbers as numbers, strings by identity and length) so EarleyBoyer's sc_assq stays native. V8 suite, medians of five: new 1,413.9 ms against old 1,425.6; Crypto ahead (153 against 164), Richards and DeltaBlue about level, EarleyBoyer about 7 ms behind (same-length strings still go to Go). | done | jit: reuse spill slots once their values are dead; jit: compare any two values natively where their words tell; jit: compare strings natively by identity and length |

**Gate:**
- [ ] Every kernel and suite at least as fast as the old pipeline on both
  architectures. amd64, 2026-10-08 (a1e3973, idle Ryzen, medians of 3 at
  300 ms): every `BenchmarkJIT*` kernel is, from 0.09x the old pipeline's
  time (remainder) through 0.44x (particle), 0.62-0.67x (logistic), 0.65x
  (dot), 0.74-0.81x (vector, stencil) and 0.88x (numeric globals) to 1.02x
  (numeric fields, level). Against the tree tier one was behind: a Go call,
  86 us per 1,000 iterations against 71.5; since P5b it is ahead, 59.6
  against 70.3 (2026-10-08, after P5b every kernel is again at or below
  the old pipeline's time, numeric fields level at 34.1 against 34.0 us).
  arm64, 2026-10-09 (CI run 37868061808, macos-15, medians of 5): every
  kernel is too, from 0.22x (remainder) and 0.46x (particle) to 0.92x
  (ScalarCalls); a Go call 0.52x the old pipeline and 0.83x the tree tier.
  Kernels are met on both; suites (the V8 suite through each pipeline) are
  not yet measured.
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
| 2026-10-09 | P5's gate is met; its re-validation blocks, generation counter and helper classes wait for guard hoisting and native calls, which give them something to protect. Until then every exit re-enters at an entry that reloads and re-checks what it reads. |

**Open:** macOS hardened-runtime support (Phase 6).

**Decided 2026-10-08 (the user chose it):** D8's reference question. A
reference loaded from
the heap (`o.a.b`, `this.items[i]`, a global function) carries its cell's
address the way an ambiguous phi carries its slot (a shadow); native code
reads through it and never stores it, and an exit record has Go copy from
that cell. Sound while Go's heap does not move and the object graph does
not change while native code runs -- it changes only between entries -- but
outside `unsafe.Pointer`'s documented rules, so it needs a stress test and
a check with each new Go version. (Rejected: keeping such values out of
native code, every read of one exiting to Go.)

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
