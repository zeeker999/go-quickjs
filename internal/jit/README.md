# Native JIT tier

This package is the optional native tier's machine-code side. The VM side
lives in `internal/vm/zjit_*.go`.

The tier is **experimental and opt-in**, and it needs two things:

1. a build with `-tags quickjs_jit` on `linux/amd64`, `windows/amd64` or
   `darwin/arm64`;
2. a runtime made with `quickjs.WithJIT()` (or `qjs --jit`,
   `-conformance.jit`, `v8bench -jit`).

Without either, every function runs in the interpreter and the tree tier,
exactly as before (see [Zero cost when off](#zero-cost-when-off)).

This file states the contracts the code keeps. Where the work is going is in
[the production plan](../../docs/jit-production-plan.md). Its status is in
[the progress tracker](../../docs/jit-progress.md), and measurements are in
[the results](../../docs/jit-results.md).

## Layout

| Package or file | Role |
|---|---|
| `ir` | The slot IR, one instruction per bytecode PC, with a state map at each. `Validate` checks a program, and `Evaluate`, a Go interpreter of the IR, is the oracle native code is tested against. |
| `compile` | Lowers bytecode into the IR, or refuses it (`Refusal`). It never imports the VM. |
| `program_*.go` | Analysis (`program_assembler.go`, `program_ranges.go`) and the two emitters: `program_amd64.go` and `program_arm64.go`. |
| `entry_*.s` | The bridges from Go into a program. |
| `memory_*.go`, `policy_darwin.go` | Executable memory. |
| `internal/vm/zjit_native.go` | The VM's side: selection, the code cache, frame encoding and publication, host exits. |
| `internal/vm/zjit_calls.go` | The call coordinator. |
| `internal/vm/zjit_disabled.go` | The stubs for builds without the tier. |

## The execution boundary

**Entry.** `enterProgram` is `NOSPLIT|NOFRAME`. It tail-jumps into the
program, which returns with a plain `RET` to `enterProgram`'s Go caller. Native
code:
- never calls anything;
- never touches SP or the frame pointer;
- uses Go's ABI0, not the platform C convention, on every OS.

**Registers.**

| | amd64 | arm64 |
|---|---|---|
| Never written | SP, BP, R14 (g) | SP, FP (R29), LR (R30), R18 (platform), R28 (g) |
| Clobbered | the rest, X15 included | the rest, R19-R25 included |

ABI0 callers expect clobbering: on amd64, Go restores X15 and R14 after any
ABI0 call. The emitter headers list each register's use.

**Budget.** Every entry runs at most `MaxIterations` (4096) committed IR
instructions, then returns to Go. Host operations handled in Go share that
budget. Go checks for cancellation, `Halt`, the memory limit and stale stack
slots before it re-enters.

**Not a safe point.** Generated code is never a safe point. SIGURG, Windows'
`SuspendThread` and SIGPROF see a PC outside Go's text:
- preemption retries until Go code runs again;
- a profile sample counts as external code.

The budget is what bounds how long the goroutine can delay a stop-the-world
pause.

**Faults kill the process.** Go cannot recover a fault at a PC outside its
text:
- on Windows the exception goes unhandled;
- on Unix it is fatal, except a nil-page access, which `Runtime`'s recover
  turns into an error.

So an emitter bug can take down the host. The defence is verification: the
IR oracle, differential tests at every entry and budget, and (planned) encoder
checks against a disassembler.

**Compiler panics are refusals.** `compile.Lower` and `Compile` recover a
panic in analysis or emission into a refusal (`Refusal`, `ErrProgram`), and
the function runs in the existing tiers.

## Values, roots and the garbage collector

**Native code stores no pointers.**
- A scalar slot (`ir.Value`) is 16 pointer-free bytes: bits and a kind.
- A reference is an `Opaque` or `String` handle: an index into Go-owned root
  tables that the VM fills.
- A handle never appears as a literal; `Validate` rejects one.

**Borrowed views.** A `ir.ArrayView` (40 bytes) grants native access to an
array's dense elements or to an object's property cells. It holds:
- a typed pointer that roots the storage;
- the storage's dense length;
- its JavaScript length;
- the tag boundary below which a cell holds a number;
- a permission to fill a pointer-free hole.

**What native code writes.** Only the number word of a cell that already holds
a number or a hole, whose pointer word is nil. It canonicalizes NaN first. It
never writes a Go pointer or a slice header, and never grows storage. So the
GC's write barrier is never needed.

**When Go invalidates views.** Before any operation that can run JavaScript or
the host (a call, a getter, a coercion, a proxy trap), Go:
1. publishes the frame;
2. clears every root and view;
3. runs the operation as the interpreter would;
4. re-encodes and rebuilds the views before re-entering.

**Lifetimes.** Owners (code, roots, views and their backing storage) stay
reachable from Go across every entry. Native code keeps no address between
entries.

## Deoptimization

**State maps.** Each reachable PC has a state map: the bytecode PC and the
live operand depth.

**Guards.** A guard fails before its instruction commits anything. The VM then:
1. publishes locals and operands at that PC;
2. resumes `executeAt` there, so the interpreter performs the operation itself.

Work already committed, earlier loop iterations included, is never replayed.

**Exits.** Every exit (guard, budget, host, return) reports the exact state
and the committed step count.

**Rest of the invocation.** A failed guard sends the invocation to the
interpreter for its remainder (`jitDeoptDepth`). Nested calls may still enter
native code.

## Selection and the code cache

**Function entry.** A closure counts framed calls. The eighth may compile its
function (`jitHotCalls`).

**Loops.** A loop may also promote at the back-edge check after a full budget
of 1024 backward jumps: on-stack replacement at the completed branch's target,
from the interpreter or the tree tier.

**The cache.** Each runtime keeps its own cache: at most 128 programs, weakly
keyed by the shared bytecode function. Shared bytecode carries no JIT state.

**Closure hints.** A closure remembers its program by a hint, a slot and tag
in the cache's hint table (not a pointer). The closure stays 128 bytes, and a
stale hint misses.

**Refusals.**
- Permanent: lowering refused it, or the emitter rejected it. These are
  cached, and the closure is marked refused.
- Deferred: the code budget was full. These are cached until the cache
  releases code.
- Temporary (no budget at all, executable memory unavailable): retried after
  another warmup. When the OS denies executable memory, the runtime marks the
  JIT unavailable for good.

**Giving up.** Eight guard misses refuse a program. A program whose
native-work-per-host-exit ratio stays low is kept for long invocations only
(`entrySlow`).

**Debugger.** Runtimes made `WithDebugger` never run native code.

## Memory

**Allocation.**
- Code is allocated writable, filled, then sealed read/execute. Pages are
  never writable and executable at once, and published code is never
  patched.
- Windows flushes the instruction cache with `FlushInstructionCache`.
- Darwin uses `ic ivau` with barriers.
- Each program is its own mapping (a chunked arena is planned).

**Darwin policy.** Darwin refuses executable memory to hardened or
code-signing-enforced processes (`ErrUnavailable`). There is no `MAP_JIT`
path yet.

**Not the script's memory.** The memory meter never counts the JIT, so a
script's measure, and whether it fails with `ErrMemoryLimit`, is the same with
the JIT on and off. The JIT has a budget of its own (`jitBudget`):
- 8 MB, or an eighth of the memory limit;
- it covers code pages, metadata, and the reference and call arenas.

**Release.**
- `ReleaseClosed` releases code once native frames have unwound.
- A failed release keeps ownership for a retry.
- A finalizer on `Code` reclaims what a host abandoned.

Runtime construction allocates nothing for the JIT.

## Zero cost when off

**Without the tag.**
- `jitBuilt` is false, and every hook in the interpreter and the tree tier is
  either an empty stub or inside `if jitBuilt`, so it is compiled out.
- `internal/cmd/hotdiff` checks the result. Compare an untagged `cmd/qjs` at
  main with one at the change: `executeAt`, `runTree`, `runTreeNested`,
  `callObject`, `callTree`, `runFD` and `backEdgeCheck` must be the same,
  instruction by instruction.

**With the tag, JIT off.**
- The JIT's fields fit existing padding: `closure` stays 128 bytes.
- `Runtime`'s and `Realm`'s JIT fields sit at the end of their structs, so no
  offset the interpreter reads moves. `TestJITFieldLayout` checks both builds.
- Measure the remaining flag tests with `placements`.

**With the JIT on.** Code the JIT never runs keeps the tree tier's fast nested
calls: only a frame whose loop may still promote keeps `runTree`'s recover
(`jitTreeRecovery`).

## Validation

```sh
go test ./internal/jit/... ./internal/vm                      # untagged
QUICKJS_REQUIRE_JIT=1 CGO_ENABLED=0 go test -tags quickjs_jit ./internal/jit/... ./internal/vm .
go test -tags quickjs_jit -race -count=1 ./internal/jit/...
go test -tags quickjs_jit -gcflags=all=-d=checkptr=2 ./internal/jit/... ./internal/vm
go test -tags quickjs_jit ./internal/jit/compile -run '^$' -fuzz '^FuzzLower$' -fuzztime=30s
TEST262_DIR=/path/to/test262 go test -tags quickjs_jit ./conformance \
  -run TestConformance -count=1 -timeout 60m -args -conformance.jit
```

- `QUICKJS_REQUIRE_JIT=1` fails a native test job that would otherwise pass
  by falling back.
- The VM's JIT tests force promotion and assert native entry, so they can't
  pass in the existing tiers.
- `-conformance.jit` keeps the normal thresholds, under which few test262
  tests reach native code. A stress mode that reports native coverage is
  Phase 1 of the plan.

`disasm -jit` reports a function's eligibility and IR without native support:

```sh
go run ./internal/cmd/disasm -jit -func sum -e \
  'function sum(n) { let s=0; for(let i=0;i<n;i++) s+=i; return s }'
```
