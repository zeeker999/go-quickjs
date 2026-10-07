# Optional native executor

This package implements the experimental numeric executor in [the JIT plan](../../docs/jit-plan.md).
It emits numeric JavaScript functions and dense numeric array loops directly
as amd64 or arm64 machine code.
Its `compile` subpackage lowers bytecode into a slot IR, with a Go evaluator in
`ir` for verifying native exits. `WithJIT()` selects the optional VM tier.
The original bounded leaf kernel remains a bridge and allocation probe.

Ordinary builds exclude the machine-code emitters, OS allocation backends,
and assembly bridges. With `-tags quickjs_jit`, a backend is included only on
`linux/amd64`, `windows/amd64`, or `darwin/arm64`. `Supported()` only describes
the build; `NewLoop()` allocates lazily and reports `ErrUnavailable` when
executable-memory policy prevents construction. Native JavaScript execution
additionally requires explicit runtime opt-in.

Runtimes made with `WithDebugger()` use the existing Go execution tiers, even
when `WithJIT()` is also requested. Debugger interrupt callbacks can evaluate
code and mutate paused frames or arrays; native budget exits currently retain
borrowed array views and cannot resume safely across those callbacks.

## Boundary contract

`Loop.Run` performs these operations in order, at most 4096 times per entry:

```go
sum += next
next += step
```

The bridge tail-jumps to a leaf kernel without changing the Go stack or its
return address. Native code cannot call functions, adjust the stack pointer,
retain pointers, or enter another kernel. Its only input is a pointer to four
eight-byte slots: `next`, `step`, `sum`, and the iteration count. The output is
written into the first and third slots. It uses SSE2 on amd64 and scalar
floating-point instructions on arm64, with no reassociation or fused operations.

The amd64 bridge passes the state in DI and uses AX for entry; generated code
clobbers CX, X0-X2, and condition flags. The arm64 bridge passes the state in
R0 and uses R16 for entry; generated code clobbers R1, F0-F2, and condition
flags. Stack/frame pointers, return addresses, and Go's goroutine register
are preserved. The typed owners of state and code remain live until return.

Generated code is not a Go safe point. Bounded batches return to Go so stack
scanning and scheduling can proceed there. This is the contract of this small
leaf kernel, not a general solution for native calls or arbitrary unbounded
JavaScript loops. The CPU profiler can encounter unsymbolized native addresses;
this package does not register native stack or symbol metadata with Go.

Code is allocated writable, copied, then sealed read/execute before publication.
It is never published writable/executable or patched while executing. Darwin
uses instruction-cache invalidation with 64-byte maintenance granules and
barriers, avoiding the cache-description register that traps on Apple Silicon.
Windows calls `FlushInstructionCache`. Allocation and release use the existing
`golang.org/x/sys` dependency. Production builds need no cgo or C compiler.

A `Loop` belongs to one caller, which closes it explicitly after use. Running
and closing the same loop concurrently is unsupported. `Size()` includes the
whole allocated page, which will be needed for VM code-memory accounting.
There is no process-wide code cache and no finalizer-based cleanup.

## Validation

```sh
go test ./internal/jit
CGO_ENABLED=0 QUICKJS_REQUIRE_JIT=1 go test -tags quickjs_jit ./internal/jit
go test -tags quickjs_jit -race -count=1 ./internal/jit
go test -tags quickjs_jit -gcflags=all=-d=checkptr=2 ./internal/jit
go vet -tags quickjs_jit ./internal/jit
```

`QUICKJS_REQUIRE_JIT=1` prevents a native test job from accidentally passing
with an unsupported backend. Native tests compare exact primitive results
with Go, including signed zero, subnormals, infinities, and NaNs. They exercise
batch limits, closed allocations, changing instructions across allocation
reuse, stack growth, GC and stack snapshots with one and two processors, CPU
profiling, and independent owners compiling/running/closing concurrently.
Linux and Windows tests also inspect OS mapping state to verify read/execute
protection and actual release. Generated instructions are not instrumented by
Go's race detector.

On macOS a regression test signs a temporary test executable with hardened
runtime and runs only `TestExecutableMemoryPolicy` in it. The prototype checks
code-signing flags using `csops` before allocation: protection changes alone
do not establish that executing unsigned pages is allowed. Hardened or enforced
hosts receive `ErrUnavailable`, including hosts with JIT entitlements. This
prototype does not yet implement the entitled `MAP_JIT` path.

The CI matrix runs native tests on the three target OS/architecture pairs
with Go 1.24 and Go 1.27, with cgo disabled. Tests under emulation validate
instruction behavior but cannot establish native performance or all hardware
and OS boundary properties.

## Measurements and next milestone

Use fresh test processes for each benchmark sample:

```sh
go test -c -tags quickjs_jit -o /tmp/quickjs-jit.test ./internal/jit
/tmp/quickjs-jit.test -test.run '^$' -test.bench 'Loop$' -test.benchmem
```

`BenchmarkNativeLoop` includes the Go/native boundary and a complete maximum
batch. `BenchmarkGoLoop` does equivalent work in Go; `BenchmarkNewLoop` includes
emission, allocation, sealing, cache publication, and release. These kernels
are feasibility measurements, not JavaScript or VM speedup claims. OS code
pages are not included in Go allocation figures.

Boundary-prototype snapshot before VM integration, October 7, 2026, with Go
1.27 on an Apple M5 Max (the binary sizes are historical):

| Measurement | Result |
|---|---|
| First construction and release, three fresh processes | 45-88 us; 288-304 Go bytes, 1-2 allocations |
| Native maximum batch, including the bridge | 2.31 us; zero Go allocations |
| Equivalent Go maximum batch | 2.52 us; zero Go allocations |
| Emission/allocation/sealing/release after warmup | 21.7 us; zero Go allocations |
| Executable allocation per live kernel | 16 KiB for 48 instruction bytes |
| Benchmark test process peak RSS | 7.45 MiB, including the test harness |
| qjs binary, ordinary and JIT-tagged builds | Both 40,169,442 bytes; the CLI does not yet import this package |

These are exploratory measurements from one host, not acceptance measurements
for the VM. Ordinary and JIT-tagged full suites passed locally, as did the
native race/checkptr tests and the existing concurrent-Intl regression.
The kernel executed on macOS/arm64 with Go 1.24.4 and 1.27, and on Linux/amd64
under emulation. Windows/amd64 was cross-compiled and vetted; execution on
Windows and native Linux hardware remains for CI. Tagged fallback builds for
Windows/arm64 and Linux/386 also compiled successfully.

## Compiler and state-map foundation

`compile.Lower` accepts a conservative numeric subset of the VM's bytecode:
locals, primitive constants, stack copies, arithmetic, numeric comparisons,
branches, increments/decrements, and primitive returns. It handles fused local,
immediate, and comparison instructions without exposing half-completed bytecode
operations. It also accepts read-only captured bindings, dense numeric array
access, length reads, fused index updates, assignment-result insertion, and
numeric bitwise operations over the full double domain. Ordinary calls, method
calls, receiver loads, global reads, property reads and writes, and method
lookups exit to Go and resume native execution.
Host-backed functions need a loop or indexed work to qualify; small wrappers
without that work retain the existing tiers. The new receiver/property coverage
also requires indexed or bitwise work: tree execution is faster for the
object loops measured in DeltaBlue. Captured own locals, writes to
upvalues, arguments objects, direct eval, non-simple parameters, handlers,
`with`, and other unsupported opcodes are refused,
including in unreachable code. Work is bounded to 4096 bytecode instructions and
256 total local/captured-binding/operand slots. It never changes `Function.VMCode`.

The analysis checks local/constant operands, branch targets, stack underflow,
stack capacity, and agreement at control-flow joins. Each reachable instruction
has a map of the next bytecode PC and live operand depth. Short-circuit branches
preserve their operand on the taken edge and pop it on the fallthrough edge.
The IR keeps locals first, read-only captured-binding snapshots next, and
operands last, in the interpreter's order.

IR scratch contains only 16-byte pointer-free scalars. Reference values are
opaque indices into Go-owned roots, and can be copied but cannot participate in
numeric operations. Guards return before changing any operand or local of the
failing instruction. The Go evaluator bounds executed instructions even in an
infinite loop; budget exits can resume at any reachable map. VM publication must
use `vm.Float` to normalize arithmetic NaNs, rather than copying raw IEEE bits
into boxed values.

`internal/vm/zjit_native.go` publishes those maps into real frames and resumes
`executeAt`. Native compilation occurs after a closure becomes hot through
framed calls; existing frameless paths still run first. `internal/vm/jit_state_test.go`
compares native execution with the Go IR oracle and every executed boundary
with uninterrupted interpretation, including signed zero, NaNs, subnormals,
infinities, loops, and short-circuit branches. Guard tests pin earlier committed
writes, exactly-once coercion, TDZ and BigInt error types/messages, and the saved
bytecode location. Integration tests assert native entries, cancellation,
reentrant coercion, memory refusal, cached tree calls, and deferred cleanup.

## Optional runtime execution

Build with `-tags quickjs_jit`, then create a runtime with `quickjs.WithJIT()`.
Both are required. The CLI accepts `--jit`; the conformance runner accepts
`-conformance.jit`, and `internal/cmd/v8bench` accepts `-jit`. Ordinary builds
and unsupported targets keep the existing tiers, even with the option set.
`WithoutCodeGeneration()` independently controls eval and Function.

Each runtime owns a weak-key cache of at most 128 entries, 8 MiB of executable
code and its metadata, and 512 KiB of native metadata. Programs are limited to
4096 instructions and 256 scalar slots. Dead weak keys are swept on cache
misses; a full cache evicts one owner between entries. Eight guard misses
suspend native selection for that cached function, and closures that reach
that cutoff remember it to avoid repeated weak-cache lookups. Shared bytecode templates
carry no native address or mutable JIT state.

Scratch values contain no pointers. Opaque indices refer to typed Go roots,
which are cleared before fallback can call JavaScript or the host. Native
execution returns to Go after at most 4096 instructions, publishes frame
state, and checks cancellation, closure, and memory limits before reentry.

Code allocations are writable during construction and sealed executable
before entry. `WithMemoryLimit` counts executable pages and native metadata.
Optional compilation also checks a conservative transient work allowance;
when it cannot fit, existing execution continues without stopping the script.
Runtime construction allocates no JIT state or executable memory. Closing
releases code after frames and host calls unwind; release failures retain
ownership for retry. A Code finalizer provides backup reclamation for owners
the host abandoned. Native execution is experimental and has no guaranteed
speedup; compilation and guard fallback can cost more than the
existing tree tier.

Native instruction bodies keep the remaining budget in a register and set the
exit PC only when exiting. Straight-line regions precharge their instruction
count once. A guard refunds the uncommitted suffix; a small remaining budget
uses the generic path with per-instruction checks. Every entry and exit retains
exact state and committed-step counts, with at most 4096 committed instructions
per entry. Region sizes fit arm64's immediate encoding, including a maximum-size
4096-instruction program split across regions.

The emitters allocate scalar bits to fourteen FP registers on arm64 and thirteen
on amd64. Six kinds on arm64 and four on amd64 use integer registers; other
kinds and unallocated slots remain in scratch. Copies of opaque handles and NaN
payloads preserve exact bits. Within a region, kind facts remove redundant
numeric guards and kind stores; copied scalar origins allow repeated accesses
to reuse one checked array view. Every arbitrary external entry into a region's
middle uses the generic path until reaching a region boundary. No unchecked
facts or cached view can leak across entry, branch joins, or callbacks.
Every exit spills the registers before returning to Go.

An array handle indexes a separate, typed table of 32-byte borrowed views. Each
view roots dense Go storage and records its dense length, JavaScript length,
and numeric tag boundary. Only ordinary arrays qualify. Native reads and stores
guard exact uint32 indices, bounds, and existing numeric cells. Stores change
only the numeric eight-byte field and canonicalize NaNs; they never change a
Go pointer, grow a slice, fill a hole, or bypass an accessor. The VM keeps owners
alive until return, and typed view pointers also keep backing storage live.
Native code does not retain any address between entries. Holes, proxies,
reconfigured properties, nonnumeric elements, unsupported keys, and coercions
resume the interpreter before committing the failing bytecode instruction.
Fused index updates commit only after all read guards succeed.

Before a host operation that can invoke callbacks the VM publishes locals and operands, then clears all
scalar roots and borrowed views. It runs the existing call/property semantics
in Go with the fetched PC, propagates exact errors, reacquires code after any
cache eviction or release, and rebuilds snapshots and views before reentry.
Captured bindings and array storage can change during callbacks. Host operations
use an iterative native/Go loop, so repeated helper calls do not recursively
nest interpreter resumptions in one JavaScript frame.

Receiver loads and ordinary own data property reads and overwrites can operate
on scalar slots in Go without publishing the frame. The existing plain-property
guards exclude proxies, accessors, private or deleted properties, indexed
storage, and exotic objects. Writes use Go assignments with its write barrier;
new or nonwritable properties use the normal VM operation. Root-table capacity
is bounded; a full table takes the ordinary publication and refresh boundary.
Up to sixteen adjacent host operations and fused local copies share each
boundary. Native work and both host paths share one 4096-instruction budget,
including across reentry, so a loop made of property operations cannot evade
cancellation or memory checks.

Numeric bitwise operations implement JavaScript's ToInt32/ToUint32 semantics
for every double. Native conversion truncates the common range to int64 and
uses the IEEE exponent and significand for large values, with NaNs and
infinities becoming zero. Shift counts are masked to five bits, and unsigned
right shifts produce positive numbers up to 4294967295. Nonnumeric operands
still guard before coercion, preserving BigInt and callback semantics through
the interpreter.

Arithmetic reads allocated registers directly and guards both operands before
writing the result. SSE2's destructive destination needs a temporary when it
aliases the right input. Numeric comparisons avoid internal control-flow
branches: arm64 selects conditions with correct unordered behavior, while
amd64 masks UCOMISD's unordered flags where needed. Arm64 comparison branches
use those flags directly. These changes preserve operation order, NaN behavior,
signed zero, TDZ checks, exact committed-step counts, and arbitrary-PC entry.

Each runtime's closure remembers permanent bytecode refusals, avoiding repeated
weak-cache registration and lookup. Memory, executable-policy, emission, and
dynamic guard refusals are not permanent hints. Shared bytecode stays immutable.

Framed calls accumulate a saturating counter on the closure. Currently the
eighth call permits compilation; short earlier calls allocate no JIT state and
keep using the existing tier. These counts are per closure within a runtime, even
when other closures or runtimes share the bytecode template. The counter fits
existing closure padding. Cached tree calls pass through the same selection
point, while frameless paths remain ahead of native selection. Temporary
resource refusals reset the counter so compilation is retried after another
warmup. The threshold is an internal policy, not an embedding API guarantee.

Long invocations also promote at existing back-edge interrupt checks, using
their runtime-wide 1024-back-edge budget as coarse work feedback. The initial
interrupt check is not a completed work budget. Both interpreter and tree
execution enter at the completed branch's target, after cancellation, memory,
and stale-slot checks. Tree branch nodes already spill live operands there.
The native entry map must match the live operand depth before locals and
operands are encoded. Completed setup and branch effects are never replayed.
Tree execution unwinds its branch nodes to `runTree` only after native execution
or guard fallback finishes; ordinary tree exceptions retain their existing path.

A failed guard selects interpretation for the rest of that invocation, even if
coercion evicts its native cache entry. Nested calls can still enter native code.
The suppression tracks bounded frame depth and restores the parent's value on
return; it retains no frame pointer. Temporary refusals delay further loop
attempts for eight check budgets as well as restarting call warmup. The extra
closure/runtime fields fit existing padding. Ordinary builds inline empty
hooks out of both executors; no new per-iteration counter is introduced.

Boundary tests force early promotion internally and assert native entries.
Separate tests cover call warmup, first-call OSR in both existing tiers, live
opaque operands, cancellation, guard effects with eviction, exact BigInt/TDZ
messages and trace PCs, counter saturation, runtime isolation, and refusals.

### Initial integration validation snapshot (8a96be4)

On October 7, 2026, Go 1.27 on an Apple M5 Max:

- Default and JIT-tagged full Go suites pass. Native VM tests also pass with
  Go 1.24, race detection, and checkptr; Linux/amd64 executes under Docker
  emulation. Windows/amd64 cross-builds and assembly vet pass; actual Windows
  execution is covered by the native CI matrix.
- JIT-enabled test262 runs pass 92,869 language, Annex B, and built-in variants,
  with zero failures and the same 342 unsupported-feature skips. A separate
  expression/statement comparison gives identical default and JIT results.
  Fresh parallel conformance processes sample peak RSS at 657 MiB default and
  402 MiB JIT-enabled; these peaks depend on worker and GC scheduling.
- Eight placements per side, three rounds per placement, compare default builds
  before and after dispatch hooks: total +0.5%, suites -0.4% to +0.9%.
- qjs binaries are 40,169,570 bytes without the tag and 40,291,890 bytes with
  it: 122,320 extra bytes (0.3%). Runtime opt-out allocates no native pages.
- Three alternating fresh V8 processes per setting, using one tagged binary
  and fixed work (`-n 5`), average 992 ms disabled and 1060 ms enabled: about
  7% slower. Go allocations rise from 672.4 to 672.7 MB; peak RSS ranges overlap
  (230-235 MiB). Live Go heap after GC is 9.4 MB for both.

A separate precompiled 10,000-iteration sum, repeated in three fresh processes,
shows the costs that aggregate V8 times hide:

| Measurement | Default | JIT enabled |
|---|---|---|
| First run, var locals | 90-196 us; 24 KB, 37-39 Go allocations | 186-287 us; 41 KB, 47 Go allocations |
| Warm run, var locals | 54-61 us | 124-125 us |
| Warm run, lexical locals | 108-110 us | 128-133 us |
| Warm allocations per whole script run | 852 Go bytes, 8 allocations | 852 Go bytes, 8 allocations |
| Peak RSS of sum measurement process | 13.4-13.6 MiB | 13.8-14.1 MiB |

Runtime construction allocates no JIT state; compilation cost occurs on first
function entry. Warm script allocations are closure preparation, not per-budget
native work. Go allocation figures omit executable pages, which are reported by
`Code.Size()` and counted by the VM meter. Go heap after close in this small
probe is about 4.5 MiB on either path because the stack pool retains the stack;
integration tests independently assert that close releases native ownership.
These results establish correctness and bounded ownership, not a performance
release.

### Instruction and refusal overhead tuning

Follow-up measurements on the same host and toolchain use three alternating
fresh processes per setting. Keeping budget/PC in registers and removing
fallthrough jumps reduces the native 10,000-iteration sum from about 124-125 us
to 49-57 us, about 2.3 times faster. It now roughly matches the existing tree path
for `var` locals and takes about half the interpreter time for `let` locals.
This benchmark verifies that the enabled runtime actually enters native code:

```sh
go test -c -tags quickjs_jit -o /tmp/quickjs-jit-vm.test ./internal/vm
/tmp/quickjs-jit-vm.test -test.run '^$' \
  -test.bench '^BenchmarkJITNumericLoop$' -test.benchtime 300ms -test.benchmem
```

The whole-script VM benchmark retains 804-805 Go bytes and seven allocations;
the public API probe retains 852 bytes and eight allocations. First native
entry still takes 40,704 Go bytes and 47 allocations, with sampled latency
138-356 us. Native pages are accounted separately. The probe's peak RSS is
13.5-13.7 MiB; compilation remains a first-use cost.

Remembering permanent closure refusals also reduces the mixed V8 suite's
repeated lookup cost. Fresh fixed-work runs (`-n 5`) average 1,073 ms for the
initial native tier, 978 ms for the tuned tier, and 983 ms with JIT disabled
in the tuned binary. Enabled and disabled totals are effectively level in this
sample; this is not an overall engine speedup claim. Enabled Go allocations
remain 672.6-672.7 MB versus 672.4 MB disabled; both retain 9.4 MB after GC.
Peak RSS overlaps at 232-237 MiB enabled and 230-238 MiB disabled.

Eight default-build placements per side average -0.7% overall, with individual
suites between -2.4% and +0.7%, showing no default regression in this sample.
The final qjs binaries are 40,169,618 bytes without the tag and 40,291,858 bytes
with it, a 122,240-byte (0.3%) difference.
Default and tagged full suites, Go 1.24 native tests, race/checkptr tests, and
all 92,869 executed JIT-enabled test262 variants pass after tuning.
More precise hotness feedback, broader numeric coverage, and fewer scalar
loads/guards remain future work.

### Call-promotion measurements (8c5769e)

The entry policy's eight-call warmup reduces eager compilation for cold
closures. Three fresh benchmark processes on the same host compare existing
execution, eager native compilation, and delayed promotion. Each operation
constructs and closes a runtime, defines a numeric function, then invokes it
the given number of times. Source compilation is outside the measurement;
native compilation, runtime construction, and release are included.

| Loop iterations per call | Calls per closure | Existing | Eager native | Delayed promotion |
|---|---|---|---|---|
| 100 | 1 | 67.6 us | 97.8 us | 69.8 us |
| 100 | 4 | 72.5 us | 99.8 us | 69.2 us |
| 100 | 16 | 85.1 us | 108.4 us | 115.5 us |
| 100 | 64 | 141.5 us | 145.4 us | 146.2 us |
| 10,000 | 1 | 178.6 us | 150.7 us | 176.1 us |
| 10,000 | 4 | 514.2 us | 315.3 us | 516.5 us |
| 10,000 | 16 | 1832.8 us | 962.6 us | 1352.1 us |
| 10,000 | 64 | 6926.5 us | 3675.8 us | 3947.5 us |

Cold delayed operations retain the existing path's 1,210 Go allocations,
versus 1,245 eager allocations, and allocate no native pages. Go byte totals
are noisier (roughly 290 KB existing/delayed and 307 KB eager for one short
call) because stack-pool reuse varies. These results show why a call threshold
is only a conservative starting policy: cold short calls avoid compilation,
but hot short calls can still fail to repay it, and a long first call can repay
eager compilation immediately. Work-based feedback and OSR are needed to
distinguish these cases. Eight calls is not a universal break-even point.

The warm-loop benchmark now defines its function once, warms past promotion,
and measures only the calling script. It averages about 53 us native versus
107 us interpreted for lexical locals, retaining 244-245 Go bytes and three
allocations per script run. The earlier benchmark redefined the function on
each operation, so its allocation counts are not directly comparable.

Three alternating fresh V8 processes average 1,043 ms eager, 1,023 ms delayed,
and 1,025 ms disabled in the delayed binary. Eight placements per side, three
rounds each, show effectively flat aggregate times: +0.7% with JIT enabled
(individual suites -1.1% to +1.7%) and +0.1% with tagged runtime opt-out
(individual suites -0.3% to +0.3%). These results do not establish an engine
speedup. Default-build execution code is unchanged. Delayed Go
allocations are 672.5-672.6 MB versus 672.4 MB disabled, with 9.4 MB live heap
after GC for both. Peak RSS ranges overlap at 230-239 MiB delayed and
232-238 MiB disabled.

The final qjs binaries are 40,169,618 bytes without the tag and 40,291,874
bytes with it: a 122,256-byte (0.3%) difference. Default and tagged full suites,
both vet configurations, Go 1.24 VM tests, race/checkptr tests, Linux/amd64
execution under emulation, and 92,869 JIT-enabled test262 variants pass.
Windows/amd64 and fallback builds for Windows/arm64 and Linux/386 compile;
actual Windows execution remains for the native CI runners.

```sh
go test -c -tags quickjs_jit -o /tmp/quickjs-jit-vm.test ./internal/vm
/tmp/quickjs-jit-vm.test -test.run '^$' \
  -test.bench '^BenchmarkJITCallPromotion$' -test.benchtime 200ms -test.benchmem
GOFLAGS=-tags=quickjs_jit go run ./internal/cmd/v8bench/placements build -label change
go run ./internal/cmd/v8bench/placements compare -jit -dir ../v8-v7 base change
```

Build both placement labels with the tag in the same checkout. `compare -jit`
enables native execution in both sets; omit it to measure tagged runtime opt-out.

### Loop-promotion measurements

The same fresh-runtime benchmark now selects native code automatically on
call warmup or a full back-edge check budget. Three fresh processes on the
same host include runtime construction, native compilation, and release:

| Loop iterations per call | Calls per closure | Existing | Eager native | Automatic promotion |
|---|---|---|---|---|
| 100 | 1 | 68.4 us | 100.6 us | 70.8 us |
| 100 | 4 | 70.0 us | 104.5 us | 73.6 us |
| 100 | 16 | 86.7 us | 114.5 us | 115.5 us |
| 100 | 64 | 147.6 us | 141.9 us | 146.2 us |
| 10,000 | 1 | 180.0 us | 154.7 us | 160.9 us |
| 10,000 | 4 | 508.5 us | 323.7 us | 327.4 us |
| 10,000 | 16 | 1922.5 us | 967.9 us | 976.2 us |
| 10,000 | 64 | 7003.2 us | 3512.7 us | 3494.7 us |

A single long call can now repay compilation without waiting for call warmup.
Short cold calls still take 1,210 Go allocations and no native pages; eager
compilation takes 1,245 allocations. Automatically promoted runs take 1,243
allocations. A one-call long run uses about 307 KB of Go allocations versus
290 KB existing. Warm lexical loops retain about 53.5 us native versus 108 us
interpreted, with 244-245 Go bytes and three allocations per calling script.
Call-based promotion of frequently called short loops still need not repay
compilation; this policy has no universal performance guarantee.

Three alternating fresh fixed-work V8 processes average 1,057 ms before OSR,
1,045 ms with OSR, and 1,043 ms with JIT disabled in the OSR binary. Treat these
totals as effectively level, not an aggregate speedup claim. Eight placements
per side, three rounds each, confirm flat totals: -0.1% ordinary, effectively
0.0% JIT enabled, and 0.0% tagged runtime opt-out. Individual suites range from
-1.0% to +1.1% ordinary, -1.0% to +0.9% enabled, and -1.2% to +1.3% opt-out.
Go allocations remain about 672.6 MB enabled and 672.4 MB
disabled, with 9.4 MB live after GC for both. Peak RSS overlaps at 234-239 MiB
enabled and 231-241 MiB disabled.

The final qjs binaries are 40,169,618 bytes without the tag and 40,292,418
bytes with it: a 122,800-byte (0.3%) difference. Default and tagged full suites,
both vet configurations, Go 1.24 tests, race/checkptr tests, and Linux/amd64
native and VM tests under emulation pass. JIT-enabled test262 reports 92,869
passed, zero failed, and 342 existing skips. Windows/amd64 and fallback builds
for Windows/arm64 and Linux/386 compile; actual Windows execution remains
unverified locally.

### Scalar register-allocation measurements

On the same Go 1.27 / Apple M5 Max host, three alternating fresh processes
compare the OSR emitter with register allocation and direct comparisons. Each
kernel runs 10,000 iterations. The function is defined and warmed once; timed
calls include the calling script, native entry/exit, and budget checks. Native
selection and the absence of guard exits are asserted, and results must match
existing execution exactly. Source and native compilation are outside these
warm measurements; all times are means in microseconds.

| Kernel / locals | Existing tier | Previous JIT | Register JIT |
|---|---|---|---|
| Sum / var | 55.8 | 52.7 | 26.1 |
| Sum / let | 107.0 | 53.4 | 26.9 |
| Logistic recurrence / var | 89.5 | 55.5 | 25.0 |
| Logistic recurrence / let | 156.2 | 57.8 | 25.9 |
| Newton iteration / var | 108.0 | 88.3 | 53.2 |
| Newton iteration / let | 207.5 | 87.0 | 53.2 |
| Particle integration / var | 327.0 | 155.5 | 93.2 |
| Particle integration / let | 595.1 | 161.9 | 93.1 |

These kernels are about 2.0-3.6 times faster than the tree tier and 3.9-6.4
times faster than the interpreter. Warm calling scripts still take three Go
allocations and about 244-250 bytes. Register allocation introduces no
per-budget Go allocations.

Fresh-runtime measurements include construction, native compilation, and
release. Automatic promotion takes 135.7 us for one long lexical call versus
176.1 us existing, 531.2 us for sixteen long calls versus 1858.6 us, and
1802.9 us for sixty-four versus 6973.0 us. Sixty-four short calls now take
127.2 us versus 140.6 us existing, but sixteen short calls still lose
(117.6 us versus 84.3 us). Cold calls keep 1,210 Go allocations and no native
pages; promoted runs take 1,246 allocations, three more than before register
allocation. A single long call takes about 314 KB versus 291 KB existing,
with stack-pool reuse affecting the Go byte totals.

The fresh-runtime kernel benchmark also includes construction, compilation,
and release, with automatic OSR and 10,000 iterations per call. These long
kernels repay native compilation on their first invocation in this sample:

| Kernel / locals | First call existing | First call automatic | 16 calls existing | 16 calls automatic |
|---|---|---|---|---|
| Logistic recurrence / var | 155.1 us | 121.4 us | 1486.6 us | 483.2 us |
| Logistic recurrence / let | 210.9 us | 126.7 us | 2430.1 us | 472.9 us |
| Newton iteration / var | 161.3 us | 142.9 us | 1770.4 us | 934.7 us |
| Newton iteration / let | 267.0 us | 158.0 us | 3532.9 us | 907.3 us |
| Particle integration / var | 358.9 us | 201.8 us | 4951.8 us | 1545.4 us |
| Particle integration / let | 620.7 us | 252.9 us | 8899.9 us | 1579.7 us |

This break-even point applies to these workloads and this host; shorter calls
still need enough repetitions to cover compilation. The kernel benchmark
checks exact results and actual native execution. Process peak RSS for all its
cases ranges from 32.1 to 34.7 MiB. Each live kernel owns one 16 KiB executable
page on this host plus 488-1544 bytes of native metadata, reported by the warm
benchmark's `native-B` and `metadata-B` metrics.

The mixed fixed-work V8 suite remains effectively level. Three alternating
fresh processes average 1027.3 ms before register allocation, 1012.2 ms after,
and 1033.9 ms with JIT disabled in the new binary. Eight placements per side,
three rounds each, show -0.6% with native execution enabled and -0.7% with
tagged runtime opt-out. Individual suites range from -0.2% to +1.2% enabled
and -1.0% to +0.0% opt-out. The small aggregate differences do not establish a
broad JIT speedup. Most hot functions in this suite still require objects,
arrays, calls, or upvalues that this numeric tier refuses.

V8 Go allocations remain 672.6 MB enabled and 672.4 MB disabled, with 9.4 MB
live after GC for both. Peak RSS is 233.7-238.9 MiB enabled and 230.5-235.0 MiB
disabled. The qjs binary is 40,169,618 bytes ordinarily (unchanged) and
40,309,506 bytes with the tag: a 139,888-byte (0.35%) difference.

Default and tagged full suites, both vet configurations, Go 1.24 tests,
race/checkptr tests, and Linux/amd64 native/VM tests under emulation pass.
Boundary tests now exercise the complete register pool during stack growth,
GC, and stack snapshots. Differential tests cover every entry and budget,
cached and spilled values, opaque handles, NaN payloads, arithmetic aliases,
and every numeric comparison with both branch directions. JIT-enabled test262
reports 92,869 passed, zero failed, and 342 existing skips. Windows/amd64 and
fallback builds for Windows/arm64 and Linux/386 compile; Windows execution
remains unverified locally.

```sh
go test -c -tags quickjs_jit -o /tmp/quickjs-jit-vm.test ./internal/vm
/tmp/quickjs-jit-vm.test -test.run '^$' \
  -test.bench '^BenchmarkJITNumeric' -test.benchtime 200ms -test.benchmem
/tmp/quickjs-jit-vm.test -test.run '^$' \
  -test.bench '^BenchmarkJITKernelPromotion$' -test.benchtime 100ms -test.benchmem
```

Inspect eligibility without native support or executable-memory allocation:

```sh
go run ./internal/cmd/disasm -jit -func sum -e \
  'function sum(n) { let s=0; for(let i=0;i<n;i++) s+=i; return s }'
go test ./internal/jit/... ./internal/cmd/disasm
go test ./internal/vm -run '^TestJIT' -count=1
go test -race ./internal/jit/compile ./internal/jit/ir
go test ./internal/jit/compile -run '^$' -fuzz '^FuzzLower$' -fuzztime=15s
```

Future work includes more precise per-function work feedback, broader numeric
coverage, and reducing scalar loads and guards. Runtime ownership, memory
accounting, native emission, VM dispatch, and limited on-stack replacement are
implemented; the existing `Function.VMCode` tree cache remains independent.

### Dense-array coverage and region measurements

This snapshot precedes the rebase onto upstream `5946c1a`; the refreshed
measurements are recorded after the Node comparison.

The active target is 5-10x over the bytecode interpreter on representative hot
workloads. The following results reach that range for several workloads, while
the mixed suite remains well below that target. All measurements use Go 1.27,
macOS/arm64, and Apple M5 Max. Warm results are means of three fresh processes;
functions and arrays are initialized before timing. Each dense kernel processes
8192 elements, and the helper case performs four stencil passes with ordinary
JavaScript boundary calls. Benchmarks assert actual native entry, zero guard
exits, and exact results.

| Dense kernel | Interpreter | Existing tree tier | Native | Speedup over interpreter |
|---|---|---|---|---|
| Vector arithmetic | 149.4 us | 101.4 us | 25.0 us | 6.0x |
| Three-point stencil | 273.8 us | 182.6 us | 36.7 us | 7.5x |
| Stencil with helper calls | 1078.2 us | 715.8 us | 148.6 us | 7.3x |

Warm dense calls retain three Go allocations and 368 bytes per calling script.
The vector and stencil each own one 16 KiB code allocation plus 536 and 728 bytes
of metadata; the helper owns two such allocations plus 1408 bytes of metadata.
Warm numeric particle integration improves to 54.9 us from 315.0 us in the tree
tier, and 54.5 us from 562.4 us in the interpreter: 5.7x and 10.3x respectively.
Newton iteration remains about 4.0x over the interpreter, below the target.
Process peak RSS for all warm cases is 26.6-28.1 MiB.

First-use measurements create fresh runtimes with precompiled bytecode and
initialized arrays. Timed first calls include automatic loop promotion and
native compilation, but exclude runtime construction, source compilation, and
array initialization. This separates the first native feature use from startup:

| Kernel | First call existing | First call automatic | Go bytes / allocations existing | Go bytes / allocations automatic |
|---|---|---|---|---|
| Vector | 104.2 us | 72.1 us | 370 / 3 | 100585 / 66 |
| Stencil | 181.3 us | 90.2 us | 373 / 3 | 127635 / 69 |
| Helper | 704.5 us | 217.6 us | 401 / 3 | 186331 / 72 |

These long calls repay native compilation in this sample; the warm speedup
is not a first-call speedup or a guarantee for shorter loops. Runtime construction
still allocates no JIT state or executable pages.

The fixed-work V8 comparison uses eight balanced placements per build and three
rounds per placement, at three iterations. NavierStokes falls from 91.7 ms to
30.3 ms (-66.9%); total falls from 655.4 ms to 594.7 ms (-9.3%). The tagged
runtime opt-out comparison is effectively level (-0.9%, with NavierStokes flat).
Most other suites remain level. This is a mixed-workload improvement, not a
5-10x aggregate improvement.

Three alternating fresh processes of the final binary at five iterations give
916.0 ms native versus 1008.9 ms opt-out, with NavierStokes at 45.8 versus
153.3 ms. Native Go allocations rise from 672.4 to 677.5 MB for compilation and
metadata; live heap after collection stays at 9.4 MB for both. Peak RSS overlaps:
234.1-242.1 MiB native, 236.0-239.4 MiB opt-out. At fifty iterations the live heap
still reports 9.4 MB, rather than growing with iteration count.

The qjs binary is 40,169,618 bytes without the tag and 40,360,210 bytes with it:
a 190,592-byte (0.47%) difference. Default/tagged full suites, vet, Go 1.24,
race/checkptr, and Linux/amd64 native and VM tests under emulation pass. Native
fuzzing completes 1.89 million executions without a failure. JIT-enabled test262
reports 92,869 passed, zero failed, and the same 342 skips. Windows/amd64 and
fallback Windows/arm64 and Linux/386 builds pass; actual Windows execution is
not verified on this host.

Reproduce the warm and first-use measurements with:

```sh
go test -tags quickjs_jit ./internal/vm -run '^$' \
  -bench '^BenchmarkJIT(DenseKernels|DenseFirstUse|NumericKernels)$' -count=3
```

Use separate fresh processes for each sample when comparing changes.

### Node comparison

The external runner also measures Node v26.8.1 (V8 14.6.202.34-node.28).
Its fixed-work suite timer now uses Node's monotonic `performance.now()`;
QuickJS's external runner retains `Date.now()`. The wall clock produced suite
timings that exceeded the measured process total on this host, so those samples
were replaced. Node's per-suite output now has 0.1 ms precision.

The following figures are refreshed after rebasing onto upstream `5946c1a`.
They are means of three alternating fresh processes per configuration, with
fifty fixed iterations and identical suite sources, on the same host.
Times are milliseconds. The existing go-quickjs column includes its tree tier;
Node `--jitless` is a separate engine with different interpreter and regexp
implementations, not a control for go-quickjs's own native compiler.

| Suite | Existing go-quickjs | Native go-quickjs | Node | Node --jitless |
|---|---|---|---|---|
| Richards | 94.9 | 94.4 | 4.3 | 63.4 |
| DeltaBlue | 147.2 | 144.6 | 6.4 | 109.2 |
| Crypto | 2103.5 | 2124.3 | 68.8 | 2201.6 |
| RayTrace | 821.7 | 828.8 | 27.7 | 432.1 |
| EarleyBoyer | 2467.1 | 2487.7 | 117.0 | 1213.7 |
| RegExp | 1191.7 | 1186.2 | 203.3 | 720.8 |
| Splay | 160.3 | 169.4 | 23.1 | 69.2 |
| NavierStokes | 1343.3 | 380.5 | 105.7 | 1775.6 |
| TOTAL | 8341.2 | 7426.5 | 593.7 | 6623.9 |

The external TOTAL includes process startup, file loading, and source compilation;
the Go TOTAL includes runtime construction, loading, compilation, and execution,
but excludes launching the Go process. Per-suite timings include setup and
teardown. Larger iteration counts reduce the relative contribution of startup;
these are fixed-work timings, not V8's score mode. Before the rebase, at five
iterations, Node's total is 167.3 ms and Node `--jitless` 794.6 ms, versus 916.0 ms native and
1008.9 ms existing go-quickjs.

At fifty iterations the native tier saves 11.0% of go-quickjs's total elapsed
time. NavierStokes is 3.5x faster than the existing tiers, but Node remains 12.5x
faster overall and 30.9x faster on Crypto. Crypto's bitwise arithmetic and its
property/receiver setup, followed by object-heavy EarleyBoyer, remain major
coverage targets. The 5-10x target has been reached by selected hot kernels,
not by the complete V8 suite.

```sh
go run ./internal/cmd/v8bench/external -engine node -cmd node \
  -dir ../v8-v7 -mode fixed -n 50
go run ./internal/cmd/v8bench/external -engine node -cmd node -arg --jitless \
  -dir ../v8-v7 -mode fixed -n 50
go run -tags quickjs_jit ./internal/cmd/v8bench -jit \
  -dir ../v8-v7 -mode fixed -n 50
```

### Upstream optimization integration

The nine local JIT commits have been rebased onto upstream `5946c1a`, including
the performance changes through `363c64c` and the debugger/source-map support.
The nested tree-call optimization now retains the callee's recovery boundary
when its loop can promote: a native return must leave the callee, then run the
remaining caller operations. Regression tests pin both ordinary and tail
results, exceptions after guard fallback, frame cleanup, and subsequent calls.
With debugger support enabled, native selection is disabled at runtime
construction, including for hidden scripts without debugger instructions.
Paused-frame evaluation and mutation remain covered with both options enabled.

Eight balanced placements per build, three rounds per placement, compare the
previous dense-array JIT build with the rebased build. With JIT enabled, total
falls from 617.0 to 594.7 ms (-3.6%); EarleyBoyer improves 6.3%, while Crypto
and NavierStokes stay essentially level. With JIT opted out, total falls from
671.5 to 644.7 ms (-4.0%), including a 7.4% NavierStokes improvement. These
comparisons measure the upstream integration, not additional native coverage.

Warm dense benchmarks retain exact-result and native-entry assertions. Three
fresh processes per sample set, at 8192 elements, now give:

| Kernel | Interpreter | Existing tree | Native | Speedup over interpreter |
|---|---|---|---|---|
| Vector | 148.1 us | 100.4 us | 24.4 us | 6.1x |
| Stencil | 264.3 us | 178.5 us | 36.5 us | 7.2x |
| Stencil with helper calls | 1054.1 us | 707.5 us | 146.9 us | 7.2x |

Warm calls still allocate 368 bytes in three allocations and own the same
16920, 17112, and 34176 bytes of native code plus metadata. Peak RSS is
25.5-28.3 MiB. First-use samples use three fresh processes, each measuring
100 independently created runtimes; timing excludes runtime construction,
source compilation, and array setup, but includes native compilation and OSR:

| Kernel | First call existing | First call automatic | Go bytes / allocations existing | Go bytes / allocations automatic |
|---|---|---|---|---|
| Vector | 102.6 us | 84.1 us | 389 / 3 | 100605 / 66 |
| Stencil | 181.2 us | 100.0 us | 404 / 3 | 127668 / 69 |
| Helper | 717.9 us | 230.1 us | 424 / 3 | 186378 / 72 |

First-use process RSS is 27.9-31.9 MiB. The fifty-iteration Go V8 run allocates
4407.7 MB with existing tiers and 4413.5 MB with native selection, and both
retain 9.4 MB after collection. Process RSS ranges overlap: 290.2-306.5 MiB
existing and 298.3-324.5 MiB native. Node's process RSS is 184.4-202.9 MiB,
and Node `--jitless` is 190.5-192.5 MiB. The qjs binaries are 40,503,538 bytes
default and 40,694,130 bytes tagged, still a 190,592-byte (0.47%) difference.

Default and tagged full Go suites and vet pass. Focused race/checkptr, Go 1.24,
and Linux/amd64 native VM tests under emulation pass, as do Windows/amd64,
Windows/arm64, and Linux/386 builds and the 386 length regression. Actual
Windows execution remains unverified on this host. Both JIT-enabled test262
and JIT plus debugger compilation report 92,869 passed, zero failures, and
342 existing skips; peak process RSS is 2948.4 and 2807.5 MiB respectively.
The 5-10x goal remains active: selected dense kernels meet it, while the
mixed suite and Crypto still require broader native coverage.

### Bitwise arithmetic and property boundaries

Both native backends now execute numeric AND, OR, XOR, complement, and all
three shifts, including fused bytecode forms and full large-double conversion.
Receiver loads and property reads/writes can resume in Go. Adjacent host
operations share a bounded batch; ordinary own data reads and overwrites use
scalar slots directly, while callbacks publish the frame, clear borrowed
storage, and reacquire code before reentry. Assignment-result insertion keeps
the value in its original stack position. The shared instruction budget covers
both native work and host batches, including property-only numeric loops.

Qualification matters as much as coverage. The new property operations need
indexed or bitwise work to qualify; object loops alone remain in the tree
tier. Closures also remember the eight-guard cutoff instead of looking up a
program that can no longer run. Eight balanced placements, three fresh rounds
per placement, compare this final build with the rebased JIT:

| Suite | Rebased native | Bitwise/property native | Change |
|---|---|---|---|
| Richards | 5.7 ms | 5.7 ms | +0.4% |
| DeltaBlue | 8.7 ms | 8.8 ms | +0.6% |
| Crypto | 126.0 ms | 78.8 ms | -37.5% |
| RayTrace | 49.6 ms | 49.8 ms | +0.5% |
| EarleyBoyer | 148.8 ms | 149.6 ms | +0.6% |
| RegExp | 104.4 ms | 103.8 ms | -0.5% |
| Splay | 107.3 ms | 107.2 ms | -0.1% |
| NavierStokes | 31.1 ms | 31.0 ms | -0.2% |
| TOTAL | 599.2 ms | 548.0 ms | -8.5% |

The external V8 fixture lives at `../v8-v7`, outside the repository and the
system temporary directory. `TestJITCryptoCorpus` executes its encryption and
decryption with the suite's plaintext assertion and checks that the actual
`am3` method runs native code without guards. `BenchmarkJITCryptoLimb` uses
that same external method, keeping Tom Wu's license with the source, and
checks every output limb and the final carry against an independent uint64
multiply/add model. Enable both by setting `QUICKJS_JIT_V8_DIR` to the fixture's
absolute path; Go tests run from their package directory.

Three fresh processes per measurement, Go 1.27 on macOS/arm64 (Apple M5 Max),
give these warm-call means:

| Kernel | Interpreter | Existing tree | Native | Speedup over interpreter |
|---|---|---|---|---|
| Vector, 8192 elements | 147.4 us | 100.3 us | 24.9 us | 5.9x |
| Stencil, 8192 elements | 266.6 us | 180.3 us | 37.2 us | 7.2x |
| Stencil with helper calls | 1112.8 us | 707.5 us | 146.7 us | 7.6x |
| Crypto am3, 32 limbs | 2391.7 ns | 1669.7 ns | 675.0 ns | 3.5x |
| Crypto am3, 8192 limbs | 542.0 us | 352.7 us | 82.4 us | 6.6x |

Dense kernels allocate 368 bytes in three allocations per call; am3 allocates
432 bytes and three allocations in all tiers. Native am3 owns 16384 bytes of
code and 1184 bytes of metadata. Dense code/metadata sizes are unchanged.
Kernel-process RSS is 31.2-31.9 MiB. First-use measurements use 100 independent
runtimes per process, excluding construction, source compilation, and setup
but including native compilation and OSR:

| Kernel | First call existing | First call automatic | Go bytes / allocations existing | Go bytes / allocations automatic |
|---|---|---|---|---|
| Vector | 103.3 us | 88.8 us | 390 / 3 | 100605 / 66 |
| Stencil | 183.8 us | 100.3 us | 404 / 3 | 127668 / 69 |
| Helper | 728.8 us | 228.0 us | 424 / 3 | 186378 / 72 |

First-use RSS is 27.9-29.4 MiB. Runtime construction still allocates no native
code. The qjs binaries are 40,503,538 bytes default and 40,726,690 bytes tagged,
an increase of 223,152 bytes (0.55%).

Fifty-iteration fixed-work measurements alternate fresh processes of the
current existing tiers, the rebased native build, the current native build,
Node v26.8.1, and Node with `--jitless`. Three-process means are:

| Suite | Existing tiers | Rebased native | Current native | Node | Node --jitless |
|---|---|---|---|---|---|
| Richards | 96.4 ms | 94.5 ms | 95.4 ms | 4.4 ms | 62.7 ms |
| DeltaBlue | 145.4 ms | 147.2 ms | 153.4 ms | 6.4 ms | 105.8 ms |
| Crypto | 2114.2 ms | 2112.6 ms | 1251.9 ms | 66.0 ms | 2174.7 ms |
| RayTrace | 828.5 ms | 819.6 ms | 826.5 ms | 27.4 ms | 420.7 ms |
| EarleyBoyer | 2490.6 ms | 2451.7 ms | 2455.4 ms | 114.6 ms | 1201.5 ms |
| RegExp | 1209.5 ms | 1169.7 ms | 1171.8 ms | 198.6 ms | 707.4 ms |
| Splay | 162.7 ms | 161.6 ms | 166.8 ms | 26.2 ms | 66.7 ms |
| NavierStokes | 1349.8 ms | 378.1 ms | 377.3 ms | 102.6 ms | 1792.6 ms |
| TOTAL | 8408.2 ms | 7345.7 ms | 6508.6 ms | 582.4 ms | 6570.0 ms |

The new native build saves 11.4% overall and 40.7% on Crypto compared with the
rebased native build. Its total is 22.6% lower than the existing tiers. These
longer runs use one normal binary layout per version; current native DeltaBlue
samples range from 141.9 to 172.3 ms, while the balanced placement comparison
is within 1% of the baseline. Allocations are 4407.7 MB existing and 4420.0 MB
native, with 9.4 and 9.4-9.5 MB live after collection. RSS ranges overlap:
298.5-307.4 MiB existing, 298.6-331.7 MiB rebased native, and 287.3-328.8 MiB
current native. Node is 184.5-204.0 MiB and Node `--jitless` 190.5-191.9 MiB.
With JIT opted out, the eight-placement comparison is level overall
(651.2 to 651.6 ms, +0.1%).

The 5-10x target remains active. The longer actual Crypto limb loop now joins
the dense kernels in that range, but the typical 32-limb call is still 3.5x,
and Node remains 11.2x faster on the mixed suite. Native/Go boundary cost and
coverage of the surrounding object and call paths remain optimization targets.

Default and tagged full tests and vet pass. Native differential tests cover
every bitwise operator, aliased destinations, arbitrary-PC entry, budget exits,
nonfinite values, exponent boundaries, random IEEE doubles, and cached array
views. Native fuzzing passes 4.6 million inputs. Callback tests cover coercion
order, BigInt fallback, exact errors, proxy and accessor operations, cache
release, GC, array resizing, root-table refresh, and cancellation. Race,
checkptr, Go 1.24, Linux/amd64 execution under emulation, Windows cross-builds,
and the Linux/386 length test pass. Actual Windows execution remains unverified.
Test262 reports 92,869 passed, zero failures, and 342 existing skips, with
2758.7 MiB peak RSS; debugger compilation reports the same results, with
2994.0 MiB peak RSS.

### Bitwise conversion and native entry costs

Numeric literal masks and shifts now use architecture immediate instructions.
Ordinary integer conversions fall through a short truncation/overflow check;
the full IEEE significand conversion lives in cold blocks. ARM64 accepts the
signed 62-bit range and handles saturation in the cold path; amd64 detects
CVTTSD2SI's sentinel using subtraction overflow. Both retain complete
ToInt32/ToUint32 behavior for fractional, large, and nonfinite doubles.

External entries at unchecked host instructions return their host exit in Go
without loading and spilling native registers. Zero-budget entries still
return a budget exit; checked host instructions retain their native TDZ guard.
The VM's encoder/native output entry skips a redundant scalar-validity scan,
while the checked entry retains it. Both check storage shape, reachable PC,
closed code, and the budget. Encoding only clears inactive operands, and
primitive native returns avoid publishing locals that cannot be observed after
the eligible frame leaves. Callback and interpreter boundaries still publish
state and clear/reacquire borrowed storage.

Eight balanced placements, three fresh rounds per placement, compare the
previous bitwise/property commit with this change:

| Suite | Previous native | Current native | Change |
|---|---|---|---|
| Richards | 5.6 ms | 5.5 ms | -0.6% |
| DeltaBlue | 8.6 ms | 8.5 ms | -0.4% |
| Crypto | 75.6 ms | 67.1 ms | -11.2% |
| RayTrace | 48.2 ms | 48.0 ms | -0.2% |
| EarleyBoyer | 143.6 ms | 145.0 ms | +1.0% |
| RegExp | 98.6 ms | 98.0 ms | -0.6% |
| Splay | 102.1 ms | 101.9 ms | -0.1% |
| NavierStokes | 29.7 ms | 29.6 ms | -0.3% |
| TOTAL | 525.9 ms | 516.6 ms | -1.8% |

The JIT opt-out comparison is level (618.4 to 616.2 ms, -0.4%); individual
changes range from -0.7% to +0.6%. Three alternating fresh processes per
measurement on the same macOS/arm64 M5 Max and Go 1.27 give these warm means:

| Kernel | Interpreter | Existing tree | Native | Interpreter speedup |
|---|---|---|---|---|
| Vector, 8192 elements | 147.2 us | 100.0 us | 24.8 us | 5.9x |
| Stencil, 8192 elements | 264.6 us | 180.5 us | 37.2 us | 7.1x |
| Stencil with helper calls | 1063.8 us | 707.8 us | 145.6 us | 7.3x |
| Crypto am3, 32 limbs | 2425.3 ns | 1676.0 ns | 607.0 ns | 4.0x |
| Crypto am3, 8192 limbs | 538.1 us | 353.7 us | 76.9 us | 7.0x |

The previous native limb means in this paired measurement are 677.2 ns and
82.3 us. Allocations remain 432 bytes/3 allocations for am3 and 368 bytes/3
allocations for dense kernels; am3 code and metadata remain 17568 bytes.
Kernel-process RSS is 31.4-35.0 MiB current, 30.0-34.3 MiB previous.
First-use means (100 independent runtimes per process, three fresh processes,
excluding source compilation/setup) are:

| Kernel | Existing first call | Automatic first call | Automatic bytes / allocations |
|---|---|---|---|
| Vector | 103.9 us | 82.0 us | 100605 / 66 |
| Stencil | 182.2 us | 107.1 us | 127668 / 69 |
| Helper | 706.1 us | 217.4 us | 186378 / 72 |

The automatic stencil samples vary from 91.0 to 126.4 us; the previous native
first-use mean in these processes is 97.0 us. Construction still allocates no
native code. First-use RSS is 27.9-31.7 MiB. The qjs binaries are 40,503,538
bytes default and 40,726,946 bytes tagged (+223408 bytes, 0.55%).

Fifty-iteration fixed-work measurements alternate fresh processes of bytecode
(`QJS_NOTREE=1`), existing tiers, previous native, current native, and Node
v26.8.1. Three-process means are:

| Suite | Bytecode | Existing tiers | Previous native | Current native | Node |
|---|---|---|---|---|---|
| Richards | 128.4 ms | 91.9 ms | 92.9 ms | 95.1 ms | 4.1 ms |
| DeltaBlue | 145.1 ms | 141.9 ms | 142.0 ms | 149.6 ms | 6.0 ms |
| Crypto | 3145.0 ms | 2045.8 ms | 1214.5 ms | 1084.0 ms | 64.3 ms |
| RayTrace | 865.7 ms | 801.1 ms | 783.1 ms | 796.9 ms | 26.9 ms |
| EarleyBoyer | 3022.0 ms | 2384.4 ms | 2388.3 ms | 2412.0 ms | 110.9 ms |
| RegExp | 1106.1 ms | 1100.5 ms | 1117.1 ms | 1135.4 ms | 197.1 ms |
| Splay | 168.2 ms | 149.5 ms | 150.7 ms | 155.4 ms | 22.7 ms |
| NavierStokes | 1875.3 ms | 1294.0 ms | 408.5 ms | 368.9 ms | 102.3 ms |
| TOTAL | 10468.1 ms | 8019.2 ms | 6308.8 ms | 6207.3 ms | 568.0 ms |

Current native Crypto is 10.8% faster than the previous native build, 1.9x
the existing tiers, and 2.9x bytecode. The mixed total saves 1.6% versus the
previous native build, 22.6% versus existing tiers, and 40.7% versus bytecode.
These longer runs use one ordinary binary layout: current DeltaBlue is 5.4%
slower here but level in the balanced comparison. Current native process RSS
is 306.4-330.3 MiB; previous native is 308.7-321.8 MiB, existing tiers
280.6-313.0 MiB, bytecode 286.8-321.3 MiB, and Node 185.9-203.9 MiB.
Current native allocates 4419.5-4420.0 MB and retains 9.4-9.5 MB live after
collection; the previous build allocates 4419.9-4421.1 MB and retains 9.5 MB.

Three alternating fresh score-mode processes average 4831 for the previous
native build and 4889 for the current build (+1.2%). Current samples are 4880,
4893, and 4894. Separate single working-tree runs scored 5212 and 4739, so a
single score is unsuitable for attributing this relatively small overall gain.

`BenchmarkJITCryptoWorkload` times a whole external RSA encrypt/decrypt pair,
whose decryption checks the plaintext each time. Each tier compiles fresh
bytecode because its first tree decision is cached. Three fresh processes
give 59.2 ms bytecode, 41.4 ms existing, and 21.3 ms native per pair: 2.8x
bytecode and 1.9x existing, with 887 allocations and about 238 KB per pair in
every tier. Native code plus metadata is 475848 bytes; process RSS is about
30 MiB. This is a separate workload from the V8 driver's fixed work.

The 5-10x target remains incomplete, and 5x is a minimum usefulness threshold.
Short limb calls remain 4.0x, whole Crypto 2.8-2.9x bytecode, and Node is
16.9x faster on fixed Crypto and 10.9x faster on the mixed total. The native
Crypto profile spends only 28.7% of CPU samples in generated code; encoding,
property boundaries, selection, and tree execution remain substantial.
Guard diagnostics show output-array growth disabling native execution in
copy, add, subtract, multiply, square, and Montgomery reduction. Nullish
comparisons stop division; opaque returns stop exponentiation. Remainder
also prevents lowering the bit-shift routines. These are coverage targets,
not evidence that the performance goal has been reached.

Full default/tagged suites, vet, native race/checkptr tests, Go 1.24,
Linux/amd64 execution under emulation, Windows amd64/arm64 and Linux/386
builds, and the Linux/386 length test pass. Native fuzzing passes 4.4 million
inputs after the entry changes (6.1 million beforehand). Differential tests
enumerate every ARM64 logical-immediate mask and shifted count, arbitrary-PC
and tiny-budget entries, aliased destinations, and checked/unchecked host
entries. Test262 normal and debugger runs each report 92,869 passed, zero
failures, and 342 existing skips. The isolated native conformance run peaks
at 2659.0 MiB; the debugger run peaks at 3119.5 MiB while other validation
processes also run. Actual Windows execution remains unverified.
