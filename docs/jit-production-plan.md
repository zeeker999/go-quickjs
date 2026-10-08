# JIT: from `codex/jit-wip` to production

Review of `zk/codex/jit-wip` at 82aa960 (20 commits on main 6c3dd16, +16.5k
lines: about 7.7k production, 6.7k tests, 2.1k notes): the target
architecture, and the plan to get there. Written October 8, 2026.

## 1. Goal

A JIT that is **fast compared with our own tiers** and **robust**.

1. **Baseline is the tree tier**, which is what users run today. Measuring
   against bytecode (`QJS_NOTREE=1`) inflates every result by the tree tier's
   own 1.3-1.5x.
2. **JIT off costs nothing.** A runtime without `WithJIT()` runs exactly as
   fast, and in exactly as much memory, as main, in an ordinary build and in a
   `quickjs_jit` build.
3. **JIT on never loses.** Tiering up must not make any workload measurably
   slower than the tree tier.
4. **Correctness is verified, not inferred.** Native code is checked
   differentially against the interpreter on inputs that actually reach it.

## 2. Calibration: what a good interpreter and a good JIT achieve

Node `--jitless` (Ignition) shows what a good interpreter achieves. Node with
its JIT shows how much a production JIT gains over that interpreter. These are
fixed-work V8 v7 runs from the branch's notes, on an Apple M5 Max, 50
iterations, in ms. Node `--jitless` comes from the same host and driver, in an
earlier table.

| Suite | Tree tier | Branch JIT | Node `--jitless` | Node JIT | Tree ÷ jitless | **V8: JIT gain** | **Ours: JIT gain** |
|---|---|---|---|---|---|---|---|
| Richards | 96.1 | 96.3 | 62.7 | 4.1 | 1.53 | 15.3x | 1.0x |
| DeltaBlue | 141.2 | 147.2 | 105.8 | 5.9 | 1.33 | 17.9x | 0.96x |
| Crypto | 2066 | 1017 | 2175 | 69.2 | 0.95 | 31.4x | 2.0x |
| RayTrace | 791 | 794 | 421 | 26.7 | 1.88 | 15.8x | 1.0x |
| EarleyBoyer | 2369 | 2422 | 1202 | 109.8 | 1.97 | 10.9x | 0.98x |
| RegExp | 1128 | 1118 | 707 | 194.4 | 1.59 | 3.6x | 1.0x |
| Splay | 153 | 152 | 66.7 | 23.3 | 2.29 | 2.9x | 1.0x |
| NavierStokes | 1299 | 370 | 1793 | 98.1 | 0.72 | 18.3x | 3.5x |
| **Total** | 8052 | 6127 | 6570 | 564 | 1.23 | 11.7x | 1.31x |

What this says:

- **The tree tier is a good interpreter on numeric code** (Crypto 0.95x and
  NavierStokes 0.72x of Ignition's time). **It is not on object code:**
  1.3-2.3x behind Ignition. That gap is in calls (about 21 ns of frame
  machinery per call), allocation and the object model. A JIT that refuses
  that code can't close it. It is a separate track for the default tiers, and
  it also helps every JIT-off user.
- **V8's JIT gains 11-18x even on object code.** Part of that leverage is
  out of reach for a JIT inside a Go process. Go has no supported way for
  generated code to
  - call Go directly: it must exit and be resumed, as wazero does (D7);
  - allocate on the Go heap;
  - store a pointer (the write barrier is runtime-private).

  So inline allocation and inline reference stores, a large part of V8's gain
  on Richards, DeltaBlue, RayTrace, EarleyBoyer and Splay, are unavailable.
  What is available: hidden-class (shape) guarded reads, inlining, unboxed
  numbers, guard hoisting, and avoiding allocation of temporaries that never
  escape.
- **On numeric code the ceiling is high.** On `lin_solve`'s inner cell
  (Ryzen, measured during the tree tier's design), the costs in ns per cell
  are:
  - bytecode: 140;
  - tree tier: 82;
  - plain Go over `Value`s with the same checks: 14;
  - plain Go over `float64`: 4.4.

  That is up to 18x over the tree tier for code the JIT keeps unboxed. The
  branch reaches 3.5x on NavierStokes.

Targets follow from this, by category, against the tree tier:

| Category | Examples | V8 JIT gain | **Our target** | Branch today |
|---|---|---|---|---|
| Numeric and array loops | NavierStokes, kernels, stencils | 18x | **≥5x** (stretch 10x) | 3-5x |
| Bitwise and integer kernels | Crypto, hashing, base64 | 31x | **≥4x** | 2.0-2.2x |
| Loops reading object fields | particles over objects, matrix rows | 11-18x | **≥2x** | not compiled |
| Call- and allocation-heavy | Richards, DeltaBlue, Splay | 3-18x | **≥0.98x** (never slower) | 0.96-1.0x |
| RegExp-bound | RegExp | 3.6x | unaffected (regexp engine) | 1.0x |

For every suite, report what fraction of V8's JIT gain we capture:
our JIT ÷ tree, set against V8 JIT ÷ jitless. The **tree ÷ jitless** column is
the default tiers' own target (≤1.0), tracked separately.

## 3. Where the branch stands

### Measured here (Windows 10, Ryzen, amd64, Go 1.27.1)

**Correctness:**
- Its notes say Windows execution was never verified; it now has been.
- Tagged vet passes, the native `internal/jit/...` tests pass (also on
  Go 1.24.0), and the full tagged suite passes.
- test262 with `-conformance.jit`: 99,599 passed, 0 failed, 342 skipped,
  the same as without the JIT.

**How much of test262 actually reached native code** (temporary counters, in
the scratchpad as `jit-stress-counters.patch`):

| Mode | Runs that entered native code | Functions compiled | Lowering refusals |
|---|---|---|---|
| Normal thresholds | **452 of 99,941 (0.45%)** | 2,776 | 49,139 |
| Stress (call threshold 1) | **4,484 (4.5%)** | 11,762 | 1,842,203 |

Both modes pass. But even in stress mode, 95% of test262 never runs a native
instruction, because the accepted subset is narrow. **test262 cannot be the
main correctness evidence for this JIT**; Phase 1 below supplies it.

**Fuzzing:**
- 60,000 random scalar IR programs, entered at random PCs with random budgets,
  match the IR oracle exactly.
- `FuzzLower` ran 2M inputs with no failure.

**Kernels** (3 fresh processes): native code is 3.0-6.0x the tree tier with
`var` locals (vector 3.7x, stencil 4.9x, stencil with helper calls 5.1x,
logistic 5.6x, Newton 3.0x, particle 6.0x). These ratios match the Mac.

**Side finding for the default tiers:** with `let` locals the tree tier is
2.4-2.8x slower than with `var` (logistic: 431 vs 154 µs). That should be
fixed regardless of the JIT.

**Struct sizes:**

| | closure | Runtime | Realm |
|---|---|---|---|
| main | 128 | 7936 | 1248 |
| branch, untagged | 128 | 7936 | 1248 |
| branch, tagged, JIT off | **144** | 7952 | 1264 |

### Why it stops at 2-5x

Each of these has a measured or code-level cause, and the architecture in
section 4 removes each one:

| Cost | Branch design | Fix |
|---|---|---|
| Per-instruction bookkeeping | 4096-*instruction* budget, precharged per region, refunded at guards | Poll the interpreter's back-edge counter (D6) |
| Kind checks every iteration | Slot IR with 16-byte `{bits, kind}` scalars; facts reset at every join and loop header | Typed SSA, unboxed int32/float64, guards hoisted out of loops (D2, D3) |
| Entry and exit copying | Go encodes the whole frame into scratch on entry and publishes it on exit | Native code reads live-ins from the VM frame and writes back only what changed (D5) |
| Host operations | Each host exit clears roots, then Go re-encodes the whole frame | Helper exits keep native state; a full frame is built only when JavaScript can run (D7) |
| Object code | Refused, or handled through Go-prepared views | Native shape-guarded reads (D8) |
| Calls | Go coordinator; native call dispatch built but unused (return chaining measured +9%) | Inlining only (D9) |
| One bad guard | 8 misses refuse the whole function | Per-site deopt feedback and recompile (D4) |
| Selection | Shape-matched selectors (`am3` countdown, MD5 array growth) plus sampled heuristics | One cost model (D10) |

### What is sound and kept

- **The boundary.** Leaf code is reached by a tail jump and never calls; it
  is W^X throughout, uses no cgo and no runtime internals, and works from
  Go 1.24 to 1.27. Preemption and SIGPROF treat the native PC as "not Go"
  and retry.
- **GC discipline.** Native code stores only number bits, and only into cells
  whose pointer word is nil.
- **Untagged builds** are byte-identical to main in `executeAt`, `runTree`,
  `callObject` and `backEdgeCheck`.
- **Numeric semantics** are correct: NaN comparisons, ToInt32/ToUint32,
  shift masking, signed zero, and guards before effects.
- **Exact frame state at every bytecode PC.** This is the branch's most
  valuable asset. It becomes the deoptimization contract.
- **The slot-IR evaluator**, kept as the oracle.

### What must be fixed whatever the architecture

| # | Finding | Where |
|---|---|---|
| R1 | `Halt`/`Close` from a host call is ignored: native code re-enters for up to 4096 steps, including backward jumps and writes. `Halt` (runtime.go:1001) promises the script stops within its current straight-line code. | zzjit_native.go:564-590, zzjit_calls.go:258 |
| R2 | JIT code counts as live script memory and is never evicted. A script can fail with `ErrMemoryLimit` only because the JIT is on. | memory.go:178 |
| R3 | Each compile attempt walks the whole heap. A refusal restarts warmup, so the walk repeats every 8 calls. | zzjit_native.go:140, 455 |
| R4 | A panic in the compiler crashes the host. One is confirmed: an out-of-range index in `inferIntegerResults`. | program_native.go:13, program_ranges.go:136 |
| R5 | A fault in native code kills the process on every OS. Go cannot recover a fault at a non-Go PC. | runtime behaviour |
| R6 | About 310 hand-encoded instruction sites, never checked against a disassembler. Nothing verifies that reserved registers are untouched. `memory()` mis-encodes base registers 4 and 12. | program_amd64.go, program_arm64.go |
| R7 | Silent miscompile paths: an opcode `lower` doesn't handle becomes a `Nop`; kind inference has no default case. | lower.go:815, program_assembler.go:307 |
| R8 | One OS mapping per compiled function. Many runtimes exhaust `vm.max_map_count`. | program_native.go:48 |
| R9 | Call copying of up to 512 cells is charged 1 budget unit. | dispatch_*.s:177 |
| Z1 | In a tagged build, `closure` is 144 bytes instead of 128, even with the JIT off. | runtime.go:554 |
| Z2 | With the JIT on, every nested tree call goes through defer/recover. | zzjit_native.go:110 |

## 4. Target architecture

### D1. Scope: a loop-region accelerator beside the tree tier

The native tier compiles **hot loop nests**, with small callees inlined.
- Entry is by on-stack replacement at the loop header, from the tree tier or
  the interpreter.
- Exit is at loop exits and side exits back to the bytecode tiers, at an
  exact PC.
- Function-entry compilation is only for small hot functions that themselves
  contain a loop.

Everything else stays in the tree tier.

**Why:** in a Go process each runtime service is an exit to Go and a
re-entry, costing about 10-20 ns, or 3-5 tree nodes. Native code only pays
where it keeps running, and loops are both where the time goes and where that
is achievable. Compiling only the region also keeps cold code (and its
exits) out of native code, and keeps compile time small.

The object-allocating, call-heavy code in Richards, DeltaBlue and Splay is
**not** this tier's job. Improving it is the default-tier track in section 2,
which shares this tier's type feedback.

### D2. Pipeline

```text
bytecode + feedback (tree-tier caches, deopt counters)
  -> region selection (loop nest + inlinable callees)
  -> typed SSA: CFG, loops, phis, guard nodes carrying frame state
  -> optimizations: GVN/CSE, LICM with guard hoisting, range analysis and
     bounds-check elimination, representation selection, DCE,
     (later) escape analysis / scalar replacement
  -> machine IR with virtual registers
  -> linear-scan register allocation over the whole region
  -> per-architecture encoder (amd64, arm64)
  -> chunked per-runtime code arena, sealed W^X
```

- Every pass is linear or near-linear, and the compiler allocates from
  `internal/arena`.
- **Compile budget:** below 50 µs and 64 KB transient for a 200-instruction
  region. The branch's MD5 first use took 2 ms and allocated 9 MB.
- The branch's slot IR and its evaluator stay. They describe frame state and
  serve as the oracle. They are not the optimization IR.

### D3. Representation

- **SSA types:** int32, float64, boolean, read-only object reference,
  string reference, and tagged `Value`. The last appears only at boundaries
  and in generic paths.
- **Unboxing a VM `Value`** is one compare of `num` against the tag base, and
  numbers need no unpacking.
- **int32 is speculated** for induction variables, indices, bitwise results
  and integer-valued feedback:
  - overflow or a result of -0 is a guard;
  - after repeated failures at a site, it recompiles as float64.
- **Doubles are materialized exactly at exits,** with NaN canonicalized in one
  per-architecture `storeNumber` helper.
- **No reassociation, no fused multiply-add, and no integer arithmetic
  wherever double rounding could differ.** Integer multiply is used only when
  range analysis proves the product exact.

### D4. Guards, frame state and deoptimization

- **Every guard node carries the frame state of its bytecode PC.** That state
  maps each local and each operand-stack slot to an SSA value, a constant, or
  a rematerializable expression. This is the branch's exact-state invariant,
  attached to SSA.
- **Deoptimization writes that state into the VM frame and resumes
  `executeAt` at the PC.** Effects already committed are never replayed.
- **A failure counter per guard site.** Past a threshold, the region is
  recompiled without that speculation: the operation becomes a generic helper
  call. After at most three recompiles the region is blocklisted.
- **No whole-function refusal for one unstable site.**

### D5. Entry and exit without copying the frame

- **A per-runtime context block**, owned by a typed Go value, holds:
  - the frame's slot base, recomputed by Go at every entry because the VM
    stack can be reallocated;
  - the address of `r.backEdges`;
  - the constants table;
  - the exit record.
- **On entry, native code reads its live-in slots straight from the VM frame**
  (a `Value`'s `num`, with `ref` tested for nil) and unboxes them. Go copies
  nothing.
- **On exit, native code writes numbers back into frame slots whose `ref` is
  nil.** That is a scalar store, with no pointer word touched. The exit
  record lists the few slots Go must write itself: reference results, or a
  slot that held a reference. Go writes those with the write barrier.
- **Exit kinds:** deoptimize; helper; poll; return.

### D6. Polling: exactly the interpreter's contract

- **Native code decrements `r.backEdges` at each back-edge and exits when it
  reaches zero or below.** That is the same check as the interpreter's
  `if r.backEdges--; r.backEdges <= 0` (vm.go:1463). The exit runs
  `backEdgeCheck`, so `Halt`, cancellation, the memory limit and GC safe
  points behave exactly as they do in the interpreter. R1 disappears by
  construction.
- **There is no per-instruction budget.** Straight-line code between
  back-edges is bounded by region size.
- **Calls and helper exits** check `r.stopped` before re-entering native code.

### D7. Helpers: runtime services without leaving the region

Helpers use wazero's mechanism (wazevo, `call_engine.go`):

- **Native code runs on its own stack,** a pointer-free buffer that Go
  allocates and grows through an exit, not on the goroutine stack.
- **To call Go, native code exits:**
  1. it saves its SP, FP and return address in the context block;
  2. it sets an exit code and the helper's arguments;
  3. it returns to the Go loop that entered it.
- **Go runs the helper, then resumes native code** at the saved return
  address with the saved SP and FP (wazero's
  `afterGoFunctionCallEntrypoint`). Native frames, inlined or called, survive
  the Go call intact.
- **This is the only supported way for generated code to reach Go.** Go's
  stack growth and GC stack scanning need metadata for every frame on a
  goroutine stack, so native frames can never sit on one, and generated code
  can't call a Go function directly. wazero calls no Go function from
  generated code either.

Helpers fall in two classes:

- **Contained:** no JavaScript can run. Examples: an element store that
  grows an array, a reference store, string concatenation, allocation of
  values that escape, a `Math` function not inlined. Native state stays in
  place and nothing is published.
- **Reentrant:** JavaScript can run. Examples: getters, setters, proxies,
  `valueOf`, non-inlined calls. Go first builds the full frame from the
  guard's frame state, so exceptions, stack traces, GC and the debugger see a
  normal frame. It then bumps a **heap generation counter** and re-enters at a
  re-validation block, which re-checks every hoisted guard and reloads
  borrowed views.

**The helper round trip is a first-class benchmark per architecture.** Its
target is ≤15 ns. The cost model (D10) uses it.

### D8. Heap access from native code: read only

- **Native code may load from VM objects reachable from the frame:**
  - an object's `shape`, compared with a constant the region's metadata keeps
    alive;
  - `props[idx].value`;
  - `elems[i]`;
  - flat string bytes and UTF-16 units.
- **The tree tier's `propCache` supplies the shapes.** A shape the JIT relies
  on is marked `seen`, so a layout change replaces the shape instead of
  mutating it (vm_shape.go). That makes a shape guard sound.
- **Number stores** go into existing number cells (`ref == nil`): array
  elements, data properties, frame slots.
- **Reference stores and allocation** are contained helpers.
- **Native code never stores a pointer anywhere.** wazero never has to:
  - Wasm linear memory is a pointer-free `[]byte`;
  - a table reference is a `uintptr` (`wasm.Reference`) into memory its
    module keeps alive;
  - memory and table growth are exits.

  So wazero never needs a write barrier; it avoids the problem rather than
  solving it.
- **Workarounds considered and rejected:**
  - **Linkname the runtime's barrier or `mallocgc`.** Generated code still
    can't call them, Go 1.23+ restricts runtime linknames, and the repository
    rules forbid runtime internals.
  - **Unbarriered stores reasoned safe for one GC.** They are unsound: a
    goroutine stack that has already been scanned can still hold the
    overwritten value.
  - **Owning the heap** (handles plus our own GC). This is the only route to
    V8-style inline allocation, but it means replacing the VM's memory model.
- **What's left:**
  - helpers;
  - deferred store buffers and Go-pre-rooted allocation pools, only if
    profiles justify them;
  - scalar replacement (Phase 5).
- **Open question, settled by the Phase 4 spike:** a reference-typed SSA value
  that can't be recomputed from the frame at exit, such as `n = n.next` in a
  loop. Two choices:
  - **Recommended:** restrict regions so they never reassign reference-typed
    locals except through a helper.
  - **Alternative:** keep the raw pointer in scratch and let Go re-root it at
    exit. This relies on Go's heap not moving and on the object graph not
    changing while native code runs. It is outside `unsafe.Pointer`'s
    documented rules, so it needs an explicit decision, a stress test, and
    a check on every new Go version.

### D9. Calls: inline, or exit

- **Inline small monomorphic callees** inside the region, guarded on the
  callee's identity. The tree tier's call caches supply the callee.
- **Every other call is a reentrant helper.**
- **No native-to-native call ABI at first.** The branch measured native
  return chaining at +9% on Crypto and its Go coordinator at +1%. Both were
  built on stackless leaf code that tail-jumps, so the unused dispatch layer
  is deleted.
- **With D7's native stack, real native calls** (plain CALL/RET on the native
  stack) **become cheap.** They are a Phase 5 candidate for callees too large
  to inline, if profiles show the need.

### D10. Selection: one cost model

- **A region is compiled when both hold:**
  - its loop's back-edge count passes a threshold;
  - the estimated helper exits per iteration, priced at D7's measured round
    trip, leave a projected gain above the compile cost (D2's budget).
- **Feedback updates the estimate.** A region whose observed exit density
  stays high is backed off exponentially, then blocklisted.
- **No selector may match a particular source shape.** Each policy is
  written down with the measurement behind it.

### D11. Ownership, memory and platforms

- **Native code is an evictable per-runtime cache, outside the script's
  memory budget.** It is capped, counted, and dropped first under pressure.
  It never causes `ErrMemoryLimit` (fixes R2).
- **A chunked code arena per runtime** (fixes R8).
- **Release** happens only at Go-side points with no native frame live, as
  today.
- **One machine IR, two encoders,** so amd64 and arm64 stay at parity. Today
  only arm64 keeps integer results; amd64 is the main server architecture.
- **Platforms:** add linux/arm64. It needs a cache flush driven by CTR_EL0,
  plus CI on real hardware.

### D12. Zero cost when off

- **Feedback comes from existing state only:**
  - the tree tier's `propCache`;
  - its call caches;
  - the interpreter's back-edge counter;
  - per-site deopt counters, which live in JIT metadata.

  There is no new per-operation profiling in the interpreter or tree tier.
- **Closure fields fit the existing padding:** 3 counter bytes plus a
  `uint32` index into a runtime-side table, not a pointer. A compile-time
  assertion fixes `closure` at 128 bytes in every build.
- **Hooks only at the back-edge slow path and the function-entry slow path.**
  Recovery for tree-tier on-stack replacement only in frames that have a
  region to enter (fixes Z2).
- **Gate on every change:**
  - untagged builds keep `go tool nm -size` identical for the hot functions;
  - tagged JIT-off builds are level in `placements`.

### D13. Correctness by construction, then by checking

The checks are layered, so each kind of bug shows up in exactly one place:

1. **Frame state:** the slot IR evaluator, already in the branch.
2. **Optimizer:** a Go evaluator for SSA, compared with the slot IR on the
   same region and entry state. It catches optimizer bugs without any machine
   code.
3. **Encoders:** golden tests through `golang.org/x/arch` (`x86asm`,
   `arm64asm`), in a nested test module so the main module gains no
   dependency. Plus a whole-program check that every region is free of calls,
   pushes, SP changes and writes to SP, BP, R18, R28, R29 and R30.
4. **Native code against the SSA evaluator:** random regions, entry states
   and helper results.
5. **JavaScript against the interpreter:** a generator of small programs
   aimed at the region subset (numbers, arrays, holes, shapes, getters and
   proxies at the edges, `valueOf`, BigInt, TDZ, exceptions). Each runs in the
   interpreter, the tree tier and the stress-mode JIT, comparing the result,
   the error type and message, and a log of side effects.
6. **Stress knobs**, internal only:
   - threshold 1;
   - deoptimize on every Nth guard;
   - poll exit at every back-edge;
   - force each reentrant helper to publish.

   test262 and the corpus run under each knob. The runner reports native
   entries per area and fails any area with loops that reached none.
7. **Fault policy:** document that an encoder bug can kill the process (R5).
   Keep a code-range table so a crash names the JavaScript function.

### Keep, rewrite, delete

| Branch code | Fate |
|---|---|
| `memory_*.go`, `policy_darwin.go`, `entry_*.s`, cache flush | **Keep**, with fixes: `dc cvau`, `CS_KILL`, a cached csops result, a chunked arena |
| `ir` (slot IR, evaluator, validator), `compile/lower.go` eligibility and state maps | **Keep** as the frame-state builder and oracle. Add an exhaustive per-opcode table (R7). |
| VM hooks: back-edge OSR from the interpreter and the tree tier, tree recovery | **Keep and narrow** (Z1, Z2) |
| Boundary, alias, callback and cancellation tests | **Keep**, retargeted to the new pipeline |
| `program_assembler.go`, `program_amd64.go`, `program_arm64.go`, `program_ranges.go` | **Replace** with SSA, machine IR and verified encoders. They serve as reference until parity, then go. |
| `zzjit_native.go` `jitHost`/`jitHostFast`, `zzjit_calls.go` coordinator | **Replace** with the helper table (D7) and inlining (D9) |
| `dispatch*.go`/`.s`, `loop*.go` | **Delete** |
| Countdown, array-growth and `charCodeAt`-name selectors | **Delete** (D10) |

## 5. Plan

Each phase ends with a gate. A phase that misses its performance gate stops
and is re-planned; robustness is never traded to make a number.

Measurement rules:
- fresh processes;
- `placements build/compare` for anything that touches `internal/vm`;
- the corpus and the Node calibration table from section 2, reported each
  time.

### Phase 0: stabilize and cut (about 2 weeks)

1. Fix R1-R4 in place. Old code that survives until Phase 2 must be safe.
2. Fix Z1 and Z2. Add the closure size assertion and the
   `go tool nm -size` check to CI.
3. Delete the dispatch layer, the `Loop` prototype and the shape-matched
   selectors. Re-measure honestly.
4. Make R7's per-opcode table exhaustive. Kind inference's default case
   clears all facts.
5. Rewrite `internal/jit/README.md` as contracts only. Move measurements to
   `docs/jit-results.md`. Drop evidence that lives outside the repository.
   `qjs --jit` warns in a build without the JIT.
6. Squash the branch into a reviewable series with descriptive commit
   bodies.

**Gate:**
- Tagged and untagged suites pass, and stress-mode test262 passes.
- Untagged builds are byte-identical to main in the hot functions.
- Tagged JIT-off builds are level in `placements` (≤0.5% total, ≤1% for any
  suite).

### Phase 1: verification infrastructure (about 2 weeks)

D13, items 1, 3, 5, 6 and 7, against the current pipeline:
- Stress knobs and native-entry reporting in the conformance runner
  (start from `jit-stress-counters.patch`).
- The JavaScript differential fuzzer.
- Encoder golden tests and the register-discipline check.
- CI: stress-mode test262 on linux/amd64, windows/amd64 and macos/arm64; the
  fuzzer for a fixed time per run; Go 1.24 and the newest Go.
- **Measure** the helper round trip and the region-entry cost on each
  architecture. They are D10's inputs, and they decide whether D7's ≤15 ns is
  realistic.

**Gate:**
- Stress-mode test262 passes on all three platforms.
- The fuzzer runs 24 hours with no divergence.
- Round-trip costs are published.

### Phase 2: the new pipeline at parity (about 6-8 weeks)

Build D2-D6 and D11's encoders and arena alongside the old pipeline. The
selection between them is internal.
- Typed SSA from the slot IR's eligibility. Linear-scan register allocation.
- Machine IR with amd64 and arm64 encoders, verified from day one.
- Direct frame entry and exit (D5) and back-edge polling (D6).
- Deoptimization with frame state (D4).
- SSA evaluator and differential checks (D13, items 2 and 4).

**Gate:**
- Every kernel and suite is at least as fast as the old pipeline on both
  architectures.
- No divergence under the Phase 1 tools.
- Compile budget met.

Then delete the old emitters.

### Phase 3: loop performance (about 4 weeks)

- int32 speculation with deopt feedback.
- LICM and guard hoisting, with re-validation blocks after reentrant helpers.
- Range analysis and bounds-check elimination for `i < a.length` loops.
- Shape guards on arrays and typed arrays (the latter are pointer-free
  backing stores and ideal for native code).

**Gate:**
- Numeric and array category ≥5x the tree tier.
- Bitwise category ≥4x.
- No corpus workload below 0.98x.
- NavierStokes and Crypto capture-of-V8 ratios published.

### Phase 4: object reads and contained helpers (about 4-6 weeks)

- D7's helper table, with the contained and reentrant classes and the heap
  generation counter.
- D8's shape-guarded reads, fed by the tree tier's `propCache`.
- The spike on reference reassignment in loops: decide between restricting
  regions and re-rooting raw pointers at exit.
- D10's cost model, replacing the branch's sampled heuristics.

**Gate:**
- Field-reading loop category ≥2x the tree tier.
- Call- and allocation-heavy suites ≥0.98x.
- Helper round trip ≤15 ns, or D10 re-tuned to what it is.

### Phase 5: inlining and non-escaping objects (about 6 weeks; spike first)

- D9: inline monomorphic callees using the tree tier's call caches.
- **Spike:** escape analysis and scalar replacement within a region. A
  temporary object that never escapes the region, such as a small vector in
  a RayTrace-style loop, is never allocated. On deoptimization, Go
  materializes it from the frame state. This is the one lever on object code
  that doesn't need native allocation or pointer stores.

**Gate:**
- Corpus geomean against the tree tier improves by a measured margin.
- Every suite at least level.
- If scalar replacement doesn't pay for its complexity, it is dropped and the
  result recorded.

### Phase 6: release (experimental, then supported)

- **Platforms:** linux/arm64. darwin/amd64 and windows/arm64 if they are
  wanted. A decision on macOS hardened-runtime apps: `MAP_JIT` without cgo,
  or documented as unavailable.
- **Diagnostics:**
  - `disasm -jit` shows regions, SSA and machine code;
  - an internal stats report (regions, entries, exits by kind and reason,
    code bytes, compile time).
- **Experimental release,** behind `quickjs_jit` and `WithJIT()`: a minor
  version, published per-platform break-even and first-use figures, and the
  calibration table.
- **Supported release criteria** (drop the build tag; keep per-runtime
  opt-in):
  - two releases with no JIT-only bug;
  - stress test262 and the fuzzer in CI on every supported platform;
  - JIT-off cost still zero;
  - category targets met.
- Default-on is a later, separate decision.

### Parallel track: the default tiers (not part of this plan's gates)

The **tree ÷ jitless** column puts our default tiers 1.3-2.3x behind Ignition
on object code. Those gains reach every user, JIT or not:
- `let` loops in the tree tier (2.4-2.8x slower than `var`);
- call frame cost (about 21 ns);
- allocation volume: 4.4 GB over a 50-iteration V8 run.

## 6. Benchmark corpus

Replace "Crypto 10x" and "MD5" with a fixed corpus, split into a tuning set
and a **held-out set** that isn't looked at while optimizing. The held-out set
is what catches overfitting, like the countdown and string-growth selectors.

- **Tuning set:**
  - the V8 v7 suite (fixed work);
  - QuickJS `microbench.js` (with `qjs -s`);
  - numeric kernels (sum, Newton, particles, matrix multiply, FFT);
  - array and typed-array kernels (vector, stencil, sort, copy);
  - hashing (MD5, SHA-256, CRC32), base64, string scanning;
  - field-reading loops over objects.
- **Held-out set:** hot paths of real libraries, for example a JSON
  pretty-printer, a markdown tokenizer, a small interpreter, a priority queue
  and a Levenshtein distance. Frozen before Phase 2.
- **Reference engines,** on the same host, driver and work:
  - Node `--jitless` and Node, through `internal/cmd/v8bench/external`;
  - C QuickJS.
- **Reported each time:**
  - per category, geomean against the tree tier (and against bytecode, for
    continuity), and the worst workload;
  - capture of V8's JIT gain, per suite;
  - first-use latency and allocations, code and metadata bytes, peak RSS;
  - tagged and untagged binary size.

## 7. Risks

- **Process death on an encoder bug (R5) can't be removed,** only made
  unlikely. D13 stays in CI permanently.
- **D8's read-only heap access depends on two things:** that the runtime is
  single-goroutine and that Go's heap doesn't move. Both hold today, but the
  dependency must be documented and stress-tested on every Go release. The
  raw-pointer alternative in D8 adds a dependency on rules Go doesn't promise;
  it needs an explicit decision.
- **Object-code expectations.** By section 2, most of V8's object-code gain
  needs inline allocation and pointer stores, which a Go-hosted JIT doesn't
  have. Promising more than "never slower, faster in field-reading loops" for
  Richards-like code would be overpromising; that gain comes from the default
  tiers.
- **Scope.** Phases 2-5 are 5-6 months. Each phase gate is a point where the
  work can stop with something that delivers.
