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
See [the implementation notes](../internal/jit/README.md)
for contracts, validation, and current limits.

The active performance target is a measured 5-10x improvement over the bytecode
interpreter on representative hot workloads. Compare the existing tree tier
separately, and report the mixed V8 suite as its own acceptance measure; a hot
kernel improvement does not establish the same gain for the whole engine.

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
Remainder, BigInt, and string operations still use interpreter fallback.

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
