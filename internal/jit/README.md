# Optional native boundary prototype

This package implements the first experiments in [the JIT plan](../../docs/jit-plan.md).
It emits a bounded numeric kernel directly as amd64 or arm64 machine code.
It does not yet compile JavaScript bytecode, select a VM execution tier, or
provide the proposed `WithJIT()` option.

Ordinary builds exclude the machine-code emitters, OS allocation backends,
and assembly bridges. With `-tags quickjs_jit`, a backend is included only on
`linux/amd64`, `windows/amd64`, or `darwin/arm64`. `Supported()` only describes
the build; `NewLoop()` allocates lazily and reports `ErrUnavailable` when
executable-memory policy prevents construction. Nothing enables native
JavaScript execution in either build yet.

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

Local snapshot on October 7, 2026, with Go 1.27 on an Apple M5 Max:

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

The next milestone adds VM-specific eligibility, bytecode state/exit maps,
and bounded runtime ownership. Preserve the existing `Function.VMCode` tree
cache and handle both `runFD` and cached `callTree` dispatch when integration
begins. Add a runtime option only once it controls actual JavaScript execution.
