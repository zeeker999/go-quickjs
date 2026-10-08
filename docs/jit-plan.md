# Native JIT implementation plan

Status: experimental numeric and dense-array executor implemented, October 7,
2026. Both amd64 and arm64 emitters execute the pointer-free slot IR. A `quickjs_jit` build plus
`WithJIT()` enables native execution of eligible framed functions, with guards
that resume the interpreter and periodic exits for cancellation and limits.
Runtime-owned caches, executable memory accounting, and deferred cleanup are
implemented. Native budget/PC bookkeeping, scalar register allocation, direct
numeric comparisons, and permanent closure refusal hints reduce execution and
selection overhead. Framed calls promote hot
closures after a bounded warmup without allocating state for short cold calls.
Long calls can enter native code at existing interpreter and tree back-edge
checks using the completed branch's target and spilled state. `disasm -jit`
reports IR eligibility and exit maps. More precise hotness feedback, wider
object coverage, and qualification on mixed object workloads remain future work.
Read-only captured bindings, guarded dense numeric array access and length,
fused index updates, and ordinary calls through resumable Go exits are now
implemented. Straight-line region budgets, local kind facts, and checked-view
reuse reduce native bookkeeping while retaining exact arbitrary-PC fallback.
Numeric bitwise operators now implement full ToInt32/ToUint32 conversion,
including large doubles and nonfinite values. Receiver and property operations
resume in Go; ordinary own data properties use scalar slots directly, while
accessors and exotic objects publish the frame and clear borrowed views.
Bounded host batches share the native instruction budget so property-heavy
loops still check cancellation and limits.
Constant bitwise operands use immediate instructions; rare large-double
conversions sit outside the hot instruction stream. External host entries
return directly in Go, and primitive native returns avoid publishing locals
that cannot remain observable. Reference returns now decode their rooted handle
before clearing native scratch. Nonnumeric equality and indexed writes resume
through Go exits; ordinary array growth refreshes every borrowed alias. Proven
ordinary writable holes can be filled with numbers directly in native code.
Remainder resumes through the existing numeric/coercion implementation, including
fused operands. Ordinary globals avoid frame publication, and closure-local
executable hints avoid repeated weak-key lookups. Sampled native work per host
boundary keeps unsuitable short entries in Go while allowing long-call loop
promotion; budget checks also bound an unsuitable first long invocation.
A full external Crypto workload benchmark
compares fresh bytecode for the interpreter, tree, and native tiers.
See [the implementation notes](../internal/jit/README.md)
for contracts, validation, and current limits.

The active performance target is a measured 5-10x improvement over the bytecode
interpreter on representative hot workloads. Compare the existing tree tier
separately, and report the mixed V8 suite as its own acceptance measure; a hot
kernel improvement does not establish the same gain for the whole engine.
Five times is the minimum usefulness threshold; the target remains incomplete.
For complete Crypto encryption/decryption, the active acceptance target is 10x;
a 5x isolated kernel or a 5x complete result does not complete that target.

## Crypto path to 10x

Whole encryption/decryption, including plaintext validation, is the acceptance
workload. At about 60 ms in bytecode and 20 ms with the current JIT, 10x means
about 6 ms per pair on the same machine and work: another roughly 3.3x over the
current JIT. Node remains a separate reference. The existing Go/tree tier must
also be measured; reducing bytecode dispatch alone overstates progress.

Only about 29% of the pre-field Crypto profile executes generated code. Improving
that code alone cannot close the gap. Conversely, the long limb kernel's roughly
7x bytecode speedup shows that coverage alone does not establish a 10x result.
Both the surrounding execution and native integer throughput need improvement.

Implement and measure these stages separately:

1. Native own numeric fields. Borrow bounded ordinary property tables, guard
   attributes and numeric cells, and preserve Go references. Keep reference
   receivers on direct host exits. Refresh views after callbacks and mutation.
   Measure complete RSA, short limb calls, and balanced V8 placements before
   widening object-loop selection.
2. Native numeric globals and stable reference fields. Read resolved ordinary
   binding cells directly, preserving lexical shadowing, TDZ, missing bindings,
   and accessor ordering. Keep references in Go-owned roots and refresh borrowed
   cells and handles after callbacks. This covers constants such as limb masks
   and receivers such as backing arrays without repeated Go bridges.
3. Short entry and conversion costs. Measure limb counts 1, 4, 16, 32, and 8192.
   Reduce repeated frame encoding, property preparation, and root/view work;
   select Go when a generic trip-count or work estimate proves native entry
   unprofitable. Do not select by benchmark or function name.
4. Native calls and enclosing arithmetic loops. Introduce bounded native frame
   storage and a shared instruction budget. Transfer scalar arguments and
   rooted handles directly between eligible callers and callees. A callee host
   exit must reconstruct its complete call chain without replaying effects;
   cache eviction, exceptions, GC, cancellation, and recursion must retain their
   existing ownership and error behavior. Then evaluate selective inlining.
5. Integer representation and ranges. Keep bitwise results and proven bounded
   arithmetic in integer registers across operations instead of repeatedly
   converting between doubles and integers. Guard speculative input ranges.
   Preserve JavaScript rounding, signed zero, NaN, overflow and shift masking;
   multiplication cannot use integer modulo arithmetic when double rounding
   could change the result. Compare both short and long limb throughput.
6. Profile the remaining complete workload. Extend lowering and optimize the
   remaining division/reduction, allocation and formatting paths where measured
   time justifies it. Accept changes by complete RSA and mixed-suite results,
   with code size, first use, allocations, live heap and peak RSS reported.

These stages are hypotheses with measurable gates, not promised speedups. A
faster isolated kernel or a compiled function does not count as completing a
stage if whole RSA regresses or the work still spends most of its time in Go.

The current implementation covers own numeric fields in selected host-free
loops, live numeric global bindings, and a small-countdown entry hint. The hint
retains Go execution for zero/one remaining iterations of a small single
`while (--parameter >= 0)` loop, while preserving native entry and OSR for later
larger calls. Selected loops now read ordinary own reference fields using a
bounded graph of rooted handles, including backing arrays and chained receivers.
Only live data cells matching their preparation-time identity grant a read;
callbacks discard and rebuild these permissions. A read performed only before
a loop retains its Go bridge because reference preparation regressed complete
RSA on short calls. A bounded Go coordinator now transfers scalar arguments and
rooted handles between compiled callers and callees, retaining guarded numeric
fields across calls. It resolves ordinary inherited data at the original read
and materializes every suspended frame before callbacks or deoptimization.
Small own-field callees can compile for this path while retaining the existing
standalone policy. Reference globals, direct machine-code call transfers and
full integer register representation remain unimplemented; whole RSA remains
about 3x bytecode.

The call coordinator is a foundation, not a completed performance milestone.
Final eight-placement measurements against the rebased pre-call implementation
show Crypto 61.2 -> 61.8 ms (+1.0%), mixed total 503.7 -> 505.4 ms (+0.3%), and
opt-out total 614.3 -> 613.4 ms (-0.1%). Fresh full RSA averages 19.04 ms versus
18.87 ms in the prior JIT; bytecode/tree are 57.50/39.96 ms. The 10x complete
Crypto target remains unmet. Do not count added compiled functions or scalar
transfer counters as an end-to-end speedup.

The next coverage milestone must eliminate the coordinator's repeated assembly
entries and retain more enclosing execution in native code. Compiling
the limb callee alone still pays for the caller's tree execution, argument/frame
conversion and repeated native entry. A bounded native frame arena should hold
scalar slots and return PCs, while typed Go owners root closures, references and
code mappings. Native transfers must preserve the Go SP/FP/g registers and use a
shared instruction budget. Begin with eligible non-recursive callees and exact
callee-exit reconstruction; only then extend to recursive or polymorphic calls.
Host exits must materialize all active frames at their committed PCs, so a
numeric array write is never replayed. Guard exits, exact exceptions, callback
GC/reentry, code eviction, depth limits and cancellation are acceptance tests,
along with complete RSA and balanced V8 comparisons.

### Integer conversion retention and rejected return chaining

A native return-chain prototype resumed suspended callers directly through a
bounded assembly trampoline. It passed exact exit/budget comparisons and the
call-chain semantic corpus, and transferred all 581,954 returns in the Crypto
corpus. Nevertheless, eight-placement comparisons measured Crypto 62.4 ->
68.2 ms (+9.3%). Preparing and reconciling the chain outweighed the removed
transition. The prototype is preserved with its measurements outside the
checkout, and is not part of production. Future native call work must transfer
both directions and retain the enclosing execution, rather than add return
bookkeeping to every existing entry.

The arithmetic step now tracks conservative finite magnitude bounds within a
precharged region. Masks, shifts, copies and bounded arithmetic can prove that
integer conversion cannot overflow. Both emitters omit the general overflow
path for those operands and retain the last converted scalar through copies
and chained bitwise operations. Arm64 additionally retains seven conversion
results in Go's permanent scratch registers R19-R25, keyed by immutable scalar
origins. Copies preserve identity; writes acquire new identities; branches,
external entries and exact small-budget paths discard these facts. Floating
point results are still stored normally: this does not reassociate arithmetic,
replace multiplication with modular integer multiplication, or remove state
needed by a guard exit.

Eight-placement comparisons against e0b2b35 measured Crypto 64.2 -> 63.1 ms
(-1.8%) and mixed total 534.0 -> 533.0 ms (-0.2%). Fresh alternating processes
measured full RSA 19.28 -> 19.17 ms (-0.5%), versus current bytecode/tree
59.91/40.77 ms: approximately 3.1x bytecode, far below the 10x gate. Every pair
checks the decrypted plaintext. Crypto's retained code/metadata shrank from
698,120 to 665,352 bytes; warm allocations stayed about 239.6 KB/918 per pair.
Full-RSA process RSS was 30.6-34.0 MiB for the baseline and 30.9-31.5 MiB for
the change. These small throughput changes do not establish a broad engine win.

The existing first-use call corpus (8192 calls, 100 fresh runtimes per process,
three fresh processes) measured median native latency 717 -> 724 us. Its timed
region includes native compilation/promotion, but excludes runtime construction,
JavaScript compilation and declarations. Allocations rose from about 248.8 to
261.5 KB and 132 to 134 allocations. The range table is compilation-only and
bounded by 4096 instructions times 256 scalar slots (1 MiB); runtime scalar
arenas, rooting, instruction budgets and the opt-in API do not grow.

The balanced opt-out total was 619.7 -> 618.2 ms (-0.2%). Default/tagged qjs
binaries are 40,575,074/40,984,210 bytes: the ordinary build is unchanged and
the tagged binary grew 16,912 bytes. Ordinary runtime construction still does
not compile or allocate executable memory. Full default/tagged Go suites,
vet, race/checkptr, Go 1.24, Linux/amd64 native execution under emulation,
Windows amd64/arm64 and Linux 386 cross-builds, and the 386 length regression
pass. Native language/built-ins Test262 reports 91,492 passed, zero failed,
342 skips, with peak RSS 2,736,275,456 bytes. Native Windows execution remains
a CI/target-machine check. Evidence and the rejected return prototype are in
`../quickjs-jit-results/2026-10-08-native-returns`.

The remaining architectural priorities are direct guarded native calls and
method dispatch, live reference/binding coverage in enclosing reduction and
squaring loops, and integer register allocation across larger arithmetic
regions. The current profile still spends substantial time on both generated
instructions and Go coordination, so improving only one side cannot meet 10x.
Measure each next step with complete RSA and balanced placements; do not use
native-entry counts or an isolated limb loop as the acceptance result.

## Objective and scope

Add an optional native executor for `linux/amd64`, `windows/amd64`, and
`darwin/arm64`. Compile the engine's own bytecode directly. Keep the existing
interpreter, tree executor, and frameless fast paths, and preserve JavaScript
behavior, runtime isolation, resource limits, and portability.

Optional execution is a requirement: the default build excludes native
backends, and a build including them still leaves the JIT disabled unless the
host opts in. Unsupported targets and unavailable executable memory retain
normal execution. Runtime construction must not allocate executable memory.
Debugger-enabled runtimes also retain Go execution: native budget exits cannot
yet refresh borrowed views after debugger evaluation changes frames or arrays.

The implementation uses Go code generation, Go assembly entry/exit bridges,
and OS memory APIs. It requires no cgo, WebAssembly, C compiler, or external
runtime library. Architecture-specific assembly is part of the implementation,
as it is in wazero; ordinary builds continue to work anywhere Go does.

Deliver a numeric baseline first, then expand only where measurements justify
it. Full JavaScript coverage is provided by fallback from the beginning.
Native coverage of every opcode, concurrent compilation, native-to-native
calls, inlining, object allocation in native code, and a general optimizing
SSA compiler are outside the first release.

## Existing integration points

- `internal/vm/vm.go`: `executeAt` resumes a frame at a bytecode position and
  explicit operand-stack depth. It is the initial destination for a bailout.
- `internal/vm/vm_call.go`: `runFD` prepares frames and chooses a tree or the
  interpreter. `callDirect` also has faster routes that bypass this choice.
- `internal/vm/zcall_tree.go`: cached `funcData.treeCall` plans execute trees
  directly. Promotion must reach these calls as well as `runFD`. A nested tree
  that can promote its loop needs its own return recovery; otherwise the native
  return would unwind past the callee and skip its caller's remaining work.
- `internal/vm/vm_leaf_pure.go`: some functions already execute without a
  frame. Keep those paths unless a measured native replacement is better.
- `internal/bytecode/function.go`: `Function.VMCode` currently holds a shared
  tree pointer. Leave that contract intact for the initial implementation.
- `internal/vm/runtime.go`: frames and the fixed-size value stack provide
  canonical JavaScript state. `ReleaseClosed`, rather than the concurrent-safe
  VM `Close`, is the appropriate starting point for deferred code release.
- `internal/vm/value.go`: `Value` contains a tagged number and a Go pointer.
  Derive layouts from the actual type; comments describing historical sizes
  are not a native ABI. Numeric NaNs must obey `Float`'s normalization rule.

## Architecture

Keep machine-code infrastructure in `internal/jit`, with no
dependency on `internal/vm`. Its responsibilities are a small intermediate
representation, instruction emission, relocations, executable allocations,
bridges, and code ownership. `internal/jit/ir` defines the slot IR;
`internal/jit/compile` owns the bytecode adapter and eligibility analysis without
importing the VM or touching its hot-code layout. VM value conversion, hotness,
frame publication, and fallback remain in `internal/vm`.

The initial flow is:

```text
Framed function entries or full back-edge check budgets accumulate hotness
    -> VM checks function eligibility
    -> bytecode becomes a small control-flow IR
    -> amd64 or arm64 emitter produces native code
    -> OS backend seals and publishes executable memory

Prepared VM frame
    -> Go copies primitive state into pointer-free scratch storage
    -> assembly bridge enters native code
    -> native code returns a result, bailout, or budget exit
    -> Go publishes changed state into the frame
    -> return, re-enter native code, or resume executeAt
```

### Execution boundary

Define a documented internal ABI independently of the platform C ABI.
Specify entry registers, preserved registers, stack alignment, permitted
scratch space, floating-point state, and exit-record layout for both CPUs.
Native code must restore Go's stack and reserved registers before returning.
Do not convert an executable address into a Go function value or call arbitrary
Go functions directly from generated code.

The first backend operates only on pointer-free scratch storage. All
JavaScript reference values remain in Go-visible roots. Publish primitive
results and clear overwritten references through Go code. Do not write Go
pointers or `Value.ref` from native code. Keep the typed owners of scratch
storage and code alive across execution; a `uintptr` alone is not a root.

This simplifies GC correctness but does not establish bridge correctness.
Stack scanning, async preemption, CPU profiling, signals, and stack growth
must be investigated and tested explicitly before integrating the executor.
Avoid private Go runtime APIs unless the feasibility work demonstrates an
unavoidable need and documents the supported Go versions and maintenance cost.

### Native compilation and fallback

Use basic blocks with explicit local/operand slots, primitive operations,
guards, branches, returns, and exits. Track bytecode position and stack depth
at every exit. The initial slot emitter now allocates frequently accessed scalar
bits and kinds to registers, initializing them on external entry and spilling
them at every exit. Unallocated slots retain the original scratch layout.
Allocation is bounded and independent of JavaScript type; guard checks still
precede any write by the failing instruction.

The first eligible functions have simple parameters, local primitive
operations, branches, loops, and returns. Exclude exception handlers,
generators, async execution, direct eval, `with`, closure creation, upvalues,
mapped arguments, and captured-local aliasing. Analyze eligibility rather than
assuming locals are private because a function has no explicit calls.

Initially implement numeric constants and locals, addition, subtraction,
multiplication, division, negation, numeric comparisons, boolean branches,
and primitive returns. Guard dynamic operands before consuming them. Leave
remainder, bitwise conversions, BigInt, strings, properties, calls, and other
operations to the existing executor until each has exact semantic tests.
The current executor also supports numeric bitwise operations, read-only
captured bindings, dense numeric arrays, and resumable calls and properties.
Remainder and nonnumeric equality have resumable Go boundaries, and reference
values can return through rooted handles. Other BigInt and string operations
still use interpreter fallback.

A failed guard exits before the failing operation. Commit all earlier work,
including completed loop iterations, exactly once, then resume that operation
in `executeAt`. Never restart the function after native execution has committed
work. Match the interpreter's next-instruction PC convention and separately
record the source location of an operation that can throw.

No floating-point reassociation, implicit fused multiply-add, or fast-math
assumptions. Preserve signed zero, infinities, subnormals, unordered NaN
comparisons, and canonical NaN boxing on both architectures.

### Hotness and tier selection

Function-entry promotion currently uses a saturating per-closure counter,
with compilation allowed on the eighth framed call. Cold selection requires
no map lookup or native state allocation. Temporary compilation refusals
restart that warmup. Compile synchronously between executions; put bounds
on function size, compiler work, and code memory. Internal test controls force
compilation so boundary correctness tests do not depend on threshold tuning.
Long calls also become hot at existing back-edge interrupt checks. These use
the runtime's shared 1024-back-edge budget as coarse work feedback; the initial
interrupt check does not count as work. The call threshold alone does not
estimate the work of a long first invocation.

Instrument `runFD` and cached `callTree` execution without defeating the
existing frameless shortcuts. A cached tree plan must observe promotion to
native code; changing only `runFD` would leave many hot calls on their trees.
Keep per-runtime tier selection separate from `Function.VMCode` and avoid
adding a map lookup to every JIT-disabled call.

On-stack replacement now enters the completed backward branch's target from
the interpreter or tree tier, after interrupt and memory checks. Tree branch
nodes spill live operands before checking the budget. Native selection verifies
the entry map's operand depth and copies both locals and live operands. It
never restarts function setup or re-evaluates a completed branch. A guard
suppresses further OSR for that invocation, including after cache eviction
during coercion. Temporary refusals defer another loop attempt for eight check
budgets. More frequent or exact per-function work profiling remains future work.

Use guard-miss feedback to suspend native attempts for unsuitable functions.
Cache permanent compilation refusals within bounded bookkeeping. Tune
thresholds from measured compilation cost and savings, rather than promising
a particular call count in the public API.

### Ownership, limits, and close

Own native code, hotness, scratch buffers, and miss feedback per runtime in
the first release. A shared `Program` can run concurrently in separate
runtimes without shared mutable JIT state. Do not embed runtime addresses in
the shared bytecode or tree cache.

Use a finite code budget and a finite metadata budget, including refusal and
hotness entries that can retain functions. Account for page-rounded executable
memory, scratch storage, and persistent metadata in the runtime memory limit;
bound transient compilation memory as well. When an optional compilation
cannot fit, abandon it and continue in the existing tier. Actual script heap
exhaustion retains the existing `ErrMemoryLimit` behavior.

Reclaim evicted code only at Go-side points where no native execution or
resume record can refer to it. Release all allocations after execution has
unwound on runtime close. Closing a runtime from a host callback must not unmap
active code. Prefer deterministic ownership over finalizers as the primary
cleanup mechanism. Shared native code and persistent disk caches are deferred.

### Platform implementation

| Target | Emitter | Platform work |
|---|---|---|
| Linux/amd64 | Shared amd64 emitter | Allocate writable pages, seal executable, unmap |
| Windows/amd64 | Shared amd64 emitter | Virtual allocation, protection changes, instruction-cache flush, release |
| macOS/arm64 | arm64 emitter | Executable-memory policy, instruction-cache synchronization, release |

Never execute partially emitted code. Prefer write-then-execute protection;
keep published code immutable instead of patching it in place. Reuse the
existing `golang.org/x/sys` dependency where appropriate. Handle denied
executable allocation as a reported JIT refusal followed by normal execution.

For macOS, test ordinary Go executables and signed hardened executables
separately. Verify whether ordinary writable-to-executable mappings suffice
for each configuration, and what `MAP_JIT`, entitlements, thread write
protection, and CPU cache maintenance require when they do not. An embedding
library cannot grant an entitlement to its host. Prove a no-cgo implementation
for supported configurations and document fallback for the others.

## Delivery milestones

| Milestone | Deliverable | Completion criterion |
|---|---|---|
| 0. Native boundary feasibility | Minimal numeric kernel and bridge experiments on amd64 and arm64; OS allocation probes on all targets | Go 1.24 and newest supported Go survive GC, profiling, preemption, stack-growth, and denied-allocation tests; macOS support conditions documented |
| 1. State and compiler foundation | Small IR, eligibility report, bytecode/exit maps, runtime ownership, memory budgets, internal force/off controls | Go-side IR evaluator agrees with exact regression results; exits reconstruct locals, PC, and stack without replaying effects |
| 2. Linux/amd64 numeric executor | amd64 emission, runtime entry, guards, budget exits, interpreter bailout | Native execution is asserted in differential tests; exact numeric behavior, cancellation, GC stress, and close/reclaim tests pass |
| 3. Automatic promotion | Hotness in normal and cached tree call paths, miss suppression, compiler work limits, diagnostics | Repeated hot calls promote; cold and unstable functions stay cheap; JIT-disabled placement comparisons meet the performance gate |
| 4. Remaining targets | Windows/amd64 allocation and bridge validation; full arm64 backend on macOS | The same corpus actually executes native code on all three targets; OS-specific failure and cache-coherency tests pass |
| 5. Experimental public release | Minimal opt-in API, optional build, CLI/benchmark/conformance controls, documentation, CI | Full validation passes, memory is bounded, startup and break-even measurements published for each target |
| 6. Measured expansion | Loop-header OSR, register allocation, typed-array kernels, guarded data-property reads | Each addition has exact semantic tests and a measured benefit over the current best tier |

Milestone 0 is deliberately a feasibility gate. A numeric kernel that returns
the right answer is insufficient if Go cannot safely scan or interrupt the
execution boundary. Probe both architectures early; Linux is the first full
VM implementation, not the only platform studied before committing to it.

Split each milestone into small PRs: memory/bridge infrastructure, state
mapping, emitter operations, VM dispatch, and validation. Keep machine-code
machinery outside `internal/vm` where possible, and do not rename or move its
existing files. Compare placements whenever integration changes hot layout.

## Configuration and diagnostics

Use a `quickjs_jit` build tag plus an explicit per-runtime `WithJIT()`
option for the experimental release. The option remains callable in ordinary
or unsupported builds and falls back to the existing tiers. Default runtimes
allocate no JIT memory. Tuning knobs remain internal initially.

Test all four combinations of native support included/excluded and runtime
opt-in enabled/disabled. Only a supported build with opt-in may execute native
code. Native test jobs must fail if they unexpectedly fall back; ordinary
application execution continues to work when the backend is unavailable.

Preserve `WithoutCodeGeneration()`'s existing meaning: disabling `eval` and
the Function constructor. Native execution is controlled independently.
Keep standards mode and `WithNodeQuirks` behavior identical across tiers.
An exported option requires a minor release under the repository's versioning
rules.

Add internal reports for eligibility, actual native entries, compile time,
code/metadata bytes, budget exits, guard failures, and fallback reasons. Extend
`internal/cmd/disasm` with a proposed `-jit` report showing eligibility, exit
maps, and native instructions. Reporting must distinguish eligible code from
code that was compiled and executed.

## Validation and performance gates

Run exact tests and differential tests in interpreter-only, current default,
and forced-native configurations. Reuse the tree semantic corpus where
applicable, and add focused cases for every emitted operation and bailout.
Assert native entry counters for eligible tests; silent fallback is not a
passing native-execution test.

Cover negative zero, NaNs, infinities, subnormals, missing arguments, type
changes between invocations, guard failures after loop iterations, side-effect
order after fallback, exact error types/messages/stack locations, memory
limits, interrupted infinite loops, close during callbacks, eviction, repeated
compile/run/close cycles, and concurrent runtimes sharing one Program.

Use GC stress, `GOMAXPROCS=1` and multiple processors, CPU profiling, and
checkptr builds for boundary tests. The race detector does not instrument
generated machine code; supplement race tests with explicit ownership and
concurrent-runtime stress tests. Test Go 1.24 and the newest supported Go.

Run the smallest affected packages first, then `go test ./...`, `go vet ./...`,
and `git diff --check`, in ordinary and JIT-enabled builds. Preserve the
repository's Windows/arm64 and 386 portability checks with native execution
unavailable there. Execute native tests on actual target runners and assert
their GOOS/GOARCH; cross-compiling does not validate a bridge.

If test262 is available, start with numeric expressions and loop statements,
then broaden to language and built-ins with forced JIT eligibility. Add
conformance-runner plumbing so these runs actually enable the JIT. Compare
reported failures with the baseline; do not add skips or weaken expectations.

Benchmark interpreter-only, the current default including trees and frameless
paths, and native execution. Use fresh processes and the existing eight-build
placement comparisons for integration changes. Measure numeric kernels and
the full fixed-work V8 suite, including compilation and scratch-copy overhead.
Measure compilation amortization separately from steady-state execution.

Report binary size, runtime construction cost, first compile/use latency,
compiler allocations, native code bytes, cold-script time, warm throughput,
peak RSS, and retained memory after GC and close. Native mappings are not
reported in Go's live heap, so collect OS memory information explicitly.

Initial engineering gates, to revisit after milestone 0's measurements:

- At least 1.5x steady-state speedup over the existing best tier on a
  representative numeric-loop corpus, including entry/exit and budget costs.
- JIT-disabled V8 aggregate overhead no greater than 1%, with no repeatable
  individual regression above 3%, measured across placements.
- A published compilation break-even point for each kernel; no default-on
  proposal until end-to-end workloads benefit after paying compilation costs.
- Bounded code and metadata under churn, complete executable-memory release
  after close, and no new semantic or conformance failures.

These are acceptance targets, not predicted results. Stop expansion and
revise the design if bridge costs or workload coverage cannot meet them.

The scalar register-allocation snapshot meets the local numeric throughput and
runtime opt-out overhead targets: all four measured kernels beat the best
existing tier by at least 2x when warm, and their long invocations repay
compilation on the first call. The mixed V8 suite remains effectively level;
broader object and array coverage is still needed for an aggregate engine win.
See the implementation notes for fresh-process results, placement comparisons,
memory costs, validation, and the workloads where compilation still loses.

## Reference

Study wazero's execution context, assembly entry/exit, code ownership, and
OS allocation designs. Its statically typed Wasm semantics are not a substitute
for JavaScript guards and fallback. Do not add wazero as a production dependency
or route JavaScript execution through Wasm. Any adapted source needs its
license and attribution preserved.

Source inspected for this plan: wazero commit
[`e234f6fe6ecd4589dd0c643b641bd38f6d826ee5`](https://github.com/tetratelabs/wazero/tree/e234f6fe6ecd4589dd0c643b641bd38f6d826ee5),
particularly `internal/engine/wazevo/entrypoint_{amd64,arm64}.go`,
`internal/engine/wazevo/backend/isa/arm64/abi_entry_arm64.s`, and
`internal/platform/mmap_*.go`.
