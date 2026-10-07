# Optional native numeric executor

This package implements the experimental numeric executor in [the JIT plan](../../docs/jit-plan.md).
It emits numeric JavaScript functions directly as amd64 or arm64 machine code.
Its `compile` subpackage lowers bytecode into a slot IR, with a Go evaluator in
`ir` for verifying native exits. `WithJIT()` selects the optional VM tier.
The original bounded leaf kernel remains a bridge and allocation probe.

Ordinary builds exclude the machine-code emitters, OS allocation backends,
and assembly bridges. With `-tags quickjs_jit`, a backend is included only on
`linux/amd64`, `windows/amd64`, or `darwin/arm64`. `Supported()` only describes
the build; `NewLoop()` allocates lazily and reports `ErrUnavailable` when
executable-memory policy prevents construction. Native JavaScript execution
additionally requires explicit runtime opt-in.

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
operations. Captured locals, upvalues, arguments objects, direct eval, non-simple
parameters, calls, handlers, `with`, and other unsupported opcodes are refused,
including in unreachable code. Work is bounded to 4096 bytecode instructions and
256 total local/operand slots. It never changes `Function.VMCode`.

The analysis checks local/constant operands, branch targets, stack underflow,
stack capacity, and agreement at control-flow joins. Each reachable instruction
has a map of the next bytecode PC and live operand depth. Short-circuit branches
preserve their operand on the taken edge and pop it on the fallthrough edge.
The IR keeps locals first and operands next, in the interpreter's order.

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
suspend native selection for that cached function. Shared bytecode templates
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

Native instruction bodies keep the remaining budget and pre-instruction PC in
registers, publishing them only at shared guard, budget, and return exits.
Every reachable external entry has a trampoline that loads the budget;
internal branches target bodies directly. Straight-line and conditional
fallthroughs need no extra jump. Budget checks still happen before every
instruction, preserving the exact exit state and 4096-instruction bound.

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
go run ./internal/cmd/v8bench/placements compare -jit -dir /tmp/v8-v7 base change
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
