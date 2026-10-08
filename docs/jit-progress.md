# JIT progress tracker

This tracks the work in [jit-production-plan.md](jit-production-plan.md)
across sessions. Update it **in the same commit** as the work it records.

## Resume here

- **Worktree:** `D:\Data\go-quickjs-jit`, branch `jit-wip`, based on main
  6c3dd16. It is not pushed; the remote branch `zk/codex/jit-wip` still has
  the old history.
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

- **Next item:** the first unchecked one below, in order. Phase 0 items are
  independent unless noted.

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

**Gate:**
- [ ] Full suites (tagged and untagged) and stress-mode test262 pass.
- [ ] Untagged hot functions byte-identical to main.
- [ ] Tagged JIT-off `placements` level (≤0.5% total, ≤1% for any suite).

## Phase 1: verification infrastructure

| ID | Item | Status | Commit |
|---|---|---|---|
| V1 | Stress knobs: threshold 1, deoptimize every Nth guard, poll exit at every back-edge, force publish. Internal only. | todo | |
| V2 | Conformance runner reports native entries, compiles, guards and host exits per area; an area with loops and no entries fails. Start from `D:/Data/quickjs-jit-results/2026-10-08-review/jit-stress-counters.patch` or rewrite it. | todo | |
| V3 | JavaScript differential fuzzer: interpreter vs tree vs stress JIT, comparing result, error and an effect log | todo | |
| V4 | Encoder golden tests and a register-discipline check (`x/arch`, nested test module) | todo | |
| V5 | CI: stress test262 on linux/amd64, windows/amd64 and macos/arm64; the fuzzer; Go 1.24 and the newest Go | todo | |
| V6 | Measure the helper round trip and region-entry cost per architecture | todo | |

**Gate:**
- [ ] Stress test262 passes on all three platforms.
- [ ] Fuzzer runs 24 hours with no divergence.
- [ ] Round-trip costs published.

## Phase 2: the new pipeline at parity

| ID | Item | Status | Commit |
|---|---|---|---|
| P1 | Typed SSA from the slot IR (CFG, loops, phis, guards with frame state) | todo | |
| P2 | SSA evaluator and differential check against the slot IR | todo | |
| P3 | Machine IR, linear-scan register allocation, amd64 and arm64 encoders | todo | |
| P4 | Native stack and wazero-style exit/resume (D7 mechanism) | todo | |
| P5 | Direct frame entry and exit (D5); back-edge polling through `r.backEdges` (D6) | todo | |
| P6 | Deoptimization from SSA frame state (D4) | todo | |
| P7 | Chunked per-runtime code arena (R8) | todo | |
| P8 | Parity, then delete the old emitters | todo | |

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
  go build -o main.exe ./cmd/qjs   # in a main checkout
  go build -o head.exe ./cmd/qjs   # here
  go run ./internal/cmd/hotdiff main.exe head.exe
  ```
