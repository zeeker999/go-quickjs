# JIT Phase 2: the new pipeline

Phase 2 of [the production plan](jit-production-plan.md) replaces the slot-IR
emitters with a typed SSA pipeline and a new execution boundary. This
document fixes the design before the code. Its status is tracked in
[the progress tracker](jit-progress.md).

## What Phase 1 measured, and what it changes

Phase 1's round-trip benchmarks (`docs/jit-results.md`) show that the
boundary, not the generated code, is where the JIT loses to the tree tier:

| Per loop iteration | Tree tier | JIT today |
|---|---|---|
| Pure arithmetic | 36.5 ns | 6.1 ns |
| Plus one `%` (a host exit) | +7 ns | +43 ns |
| Plus one Go call | +30 ns | +114 ns |

**The bare round trip.** One entry, one instruction, one exit costs 23.7 ns,
whatever the frame's size. Most of it is Go-side checking and set-up in
`Code.runArrays`, not the bridge.

**A Go call.** The profile of the Go-call benchmark shows:
- The fallback heuristic hands most iterations to the interpreter
  (`executeAt` is 26% of the time).
- About 26% goes to `jitPublishCalls` and `jitRebuildCalls`.
- About 22% goes to `nativeFrame`, `encodeFrame`, `jitHostFast` and
  `runArrays`.
- The machine code is a rounding error.

Three consequences for the design:

1. **No Go-side publish or re-encode.** Native code writes back the frame
   slots it changed when it exits, and loads the slots it needs when it
   resumes. Go does neither (D5).
2. **The exit and resume path is one lean loop.** It is one Go function, with
   one switch on the exit kind, and no per-entry validation of state that
   native code itself maintains (D7).
3. **Build a walking skeleton first.** One numeric loop goes through the
   whole new pipeline (SSA, machine IR, encoder, native stack, exit and
   resume) on both architectures. The round trip is measured *before* the
   pipeline widens. If it misses the target, the boundary is redesigned
   while it is still small.

## Packages

| Package | Role |
|---|---|
| `internal/jit/ir` | Kept: the slot IR, its state maps and its evaluator. Lowering from bytecode stays the front end; the slot IR becomes the frame-state contract and the oracle. |
| `internal/jit/compile` | Kept: bytecode to slot IR, eligibility. |
| `internal/jit/ssa` | New: typed SSA. Built from a slot-IR program, with optimization passes and a Go evaluator. |
| `internal/jit/mir` | New: machine IR over virtual registers, linear-scan register allocation, block layout. Instruction selection per architecture: `mir/amd64.go`, `mir/arm64.go`. |
| `internal/jit/asm/amd64`, `internal/jit/asm/arm64` | New: encoders, one function per instruction form, with golden tests in `internal/jit/verify`. |
| `internal/jit/rt` | New: the runtime ABI. It owns the context block, the native stack, the exit/resume bridges (`.s`) and the code arena. |
| `internal/vm/zzjit_*.go` | The VM side, rewritten for the new ABI. The old one stays until parity. |

Nothing new imports `internal/vm`. The VM imports the JIT.

## SSA (`internal/jit/ssa`)

### Representation

- **Values.** A function is a list of blocks. A block holds values in order,
  ending in a control value: `Jump`, `If`, `Return`, `Deopt` or `Exit`. A
  value has:
  - an op;
  - a type: `Int32`, `Float64`, `Bool`, `Tagged`, `Handle` or `Mem`;
  - up to three arguments, plus an auxiliary integer;
  - for guards and calls, a frame-state index.
- **Phis** sit at block heads, one argument per predecessor.
- **Effects** are ordered by a `Mem` chain: the memory state is an SSA value,
  threaded through every load, store and call. This is the textbook way to
  let later passes move pure code and leave effects in place.
- **Frame state.** A frame state is a bytecode PC, an operand depth, and one
  entry per slot (locals, captured bindings, operands), each an SSA value or
  a constant. Every guard, deopt and exit names one. This is the slot IR's
  state map, carried over: deoptimization stays exact by construction.

### Construction

The builder walks the slot IR once, block by block, keeping the current SSA
value of every slot. It uses Braun et al.'s on-the-fly SSA construction, with
incomplete phis at loop headers, sealed once their predecessors are known.
Each slot-IR instruction becomes:

- a **guard** on its operands' kinds where the slot IR checks them. In
  Phase 2 every guard keeps the slot IR's semantics; type speculation comes
  in Phase 3;
- the **operation** on unboxed values: `AddF64`, `ToInt32`, `ShlI32`,
  `CmpF64`, and so on;
- a **frame-state snapshot** at each instruction that can exit.

Constants, copies and stack shuffles (`Copy`, `CopyPair`, `Swap`,
`Insert2`/`Insert3`) disappear: they only rename values.

### Types and boxing

`Tagged` is the VM's 16-byte `Value`, as two words. A number is a tagged
value whose top 13 bits are not all set; `UnboxF64` is a guard and a move.
Booleans, undefined and null are tags. `Handle` is a root-table index, as in
the slot IR.

Phase 2 keeps numbers in `Float64` and integer conversions in `Int32` exactly
where the slot IR does. Speculating `Int32` for whole values is Phase 3.

### Passes in Phase 2

All are linear or near-linear, allocate from an arena, and are verified by
the SSA evaluator:

1. dead-code and dead-phi elimination;
2. copy and constant propagation, and folding of pure ops;
3. redundant guard elimination within a block: the same kind check on the
   same value.

GVN, LICM, range analysis and `Int32` speculation are Phase 3.

### Evaluator

`ssa.Evaluate` runs a function on a state given in slot-IR form, and returns
the same `ir.Exit` the slot-IR evaluator does. The differential chain is:

1. bytecode → slot IR: the slot-IR evaluator against the interpreter, which
   already exists;
2. slot IR → SSA: the SSA evaluator against the slot-IR evaluator, at every
   entry and budget;
3. SSA → machine code: native code against the SSA evaluator;
4. JavaScript → everything: `FuzzJITDifferential`.

So an optimizer bug and an encoder bug fail at different steps.

## Machine IR and allocation (`internal/jit/mir`)

**Instruction selection.** Each SSA value becomes a few machine
instructions over virtual registers, in two classes: integer and floating
point. Selection is per architecture, about 300 lines each. Everything after
it is shared:
- block layout, in reverse post-order with loops kept contiguous;
- liveness;
- linear-scan register allocation (Poletto and Sarkar, with lifetime holes).

**Spills** go to slots on the native stack (D7), never to Go memory.

**Registers.** Each architecture's set excludes the registers Go reserves,
so the verifier's rules hold by construction:
- amd64: SP, BP and R14;
- arm64: SP, R18, R28, R29 and R30.

Values live across a call to Go are spilled; the native stack survives the
call.

**Exit stubs** are emitted only for exits that something jumps to. This fixes
the old emitter's roughly 206 bytes per instruction.

## Encoders (`internal/jit/asm`)

Each encoder writes one instruction form per function, for example
`amd64.MovsdLoad(dst, base, disp)` or `arm64.FaddD(d, n, m)`. Branches go to
labels and are patched at the end of the function.

**Golden tests in `internal/jit/verify`.** For each form, every register and
a range of immediates are encoded and decoded with `x/arch`. The decoded text
must name the same operation and operands. Whole-program checks run as in
Phase 1.

## The runtime ABI (`internal/jit/rt`)

### Context block

One block per runtime: a typed Go struct, allocated once, and reached from
native code through a register. It holds:

| Field | Written by | Use |
|---|---|---|
| `locals`, `stack` (`unsafe.Pointer`) | Go, at every entry and resume | Base addresses of the frame's slots: native code reads and writes `Value`s there (D5) |
| `backEdges` (`*int`) | Go, once | The interpreter's back-edge counter (D6) |
| `exitKind`, `exitPC`, `exitDepth`, `exitArgs[4]` | native | The exit record |
| `resumeAddr`, `nativeSP`, `nativeFP` | native, at an exit that resumes | Where to come back (D7) |
| `roots`, `views` | Go | The root and view tables of the slot IR, unchanged |
| `constants` | Go, at compile | Pointers the code compares against (shapes, from Phase 4), kept alive by the program |

Its fields are typed (`unsafe.Pointer` or scalars), so the GC sees the
pointers. Native code only reads pointers from it. It writes scalars, and
writes `Value.num` only where `Value.ref` is nil (D5).

### Native stack

**What it is.** A pointer-free `[]byte` per runtime: 64 KB to start, grown by
an exit, as wazero does. Native frames live on it: spill slots, and the
return address of a native call once Phase 5 has them.

**Entry.** The entry bridge, `NOSPLIT|NOFRAME` asm:
1. saves Go's SP and the return address in the context block;
2. switches SP to the native stack;
3. jumps to the code.

**Exit.** An exit:
1. stores the record and its resume point;
2. switches SP back;
3. returns to the bridge's Go caller, as the slot-IR programs return today.

**Resume.** Go calls the resume bridge, which repeats the entry's first two
steps and jumps to `resumeAddr`.

**Why this is safe:**
- Go never sees the native stack: it is a byte slice with nothing to scan.
- A signal, preemption request or profiling sample that lands in native code
  finds a PC outside Go's text and treats it as it does today.
- BP is never written, so frame-pointer unwinding still walks the Go frames
  under the bridge.
- The goroutine's stack can move while Go handles an exit, because native code
  holds no pointer into it. The resume bridge saves the SP at the time it is
  called.

### Exits

| Kind | Native side | Go side |
|---|---|---|
| `Return` | value in the record | Return it. |
| `Deopt` | frame state stored into the frame (numbers natively; references recorded for Go), PC and depth set | `executeAt` from there. |
| `Poll` | `backEdges` reached zero at a back-edge | `backEdgeCheck`, then resume. |
| `Helper` | helper id and argument locations; resume point saved | Run the helper; resume. |
| `GrowStack` | native stack exhausted | Grow it, fix saved SP/FP, resume. |

**The helper loop.** It is one Go function: run, switch, act, resume. It has
no validation of state that native code maintains; that is the bridge's
contract, checked by tests and the verifier.

**Contained and reentrant helpers.** A helper is *contained* if no
JavaScript can run during it. A reentrant one (a call, a getter) needs the
frame complete before JavaScript runs. So the exit stub of a reentrant helper
stores every slot the frame state lists, not only those changed since entry.
On resume, every borrowed view and every hoisted fact is re-checked: the stub
resumes at a block that re-validates (D7). The view generation counter makes
the common case, where nothing changed, a single compare.

### Frame access (D5)

- **On entry,** native code loads the slots it reads from `locals` and
  `stack`. It tests each `Value`'s tag where the SSA has a guard.
- **On exit,** the stub stores each changed slot: a number into `num`, if
  `ref` is nil. A slot whose `ref` is not nil, or a value that is a handle, is
  written by Go from the record, with the write barrier. In a numeric loop
  that is none.
- **Go copies nothing.**

### Polling (D6)

Each back-edge decrements `*backEdges` and exits with `Poll` at zero or below.
This is the interpreter's counter, so `Halt`, cancellation and the memory
limit apply at the same points.

### The walking skeleton's simplifications (P2)

These were fixed while designing P2, and each is lifted by a later milestone.

**Tagged values are number words only.** Native code holds a slot's `num`
word and never its `ref` word, and writes the frame only when it exits, so at
every exit each slot still holds what it held at entry. A slot holding an
object may be read (its tag tested) and its value moved: moving it is Go's
job, which the exit asks for. Each tagged SSA value has an origin
(`ssa.Origins`): a primitive made natively, or a slot whose value at entry it
may be. At an exit, for each frame-state slot i whose value is not what was
loaded from i itself (P4, `mir.exitTo`):
- if its origin is slot k, and k's `ref` is not nil, and the word is k's
  (native code makes no word of a reference's kind), the value is k's
  reference: a record has Go copy slot k to slot i;
- otherwise it is a primitive, written natively into `num` when i's `ref` is
  nil, and by Go from a record when it is not, so that Go clears the
  pointer.

A phi can merge two slots' values (a loop doing `x=o`, entered at its header
with `x` holding a reference already), and two references can share a word,
as objects do. Such a phi has a *shadow*: an Int32 phi holding at run time
the slot its value came from, or -1 (`ssa.shadowMerges`). Its exits leave the
choice to Go: "slot r if it holds a reference, else this word". A return
works the same way through `RetFrom`. Go reads every record's source before
it writes any slot (`abi.Record`, `vm.jitApplyRecords`). The SSA evaluator
checks at every exit and return that each reference is where its origin, or
its shadow, says.

The skeleton instead checked at every entry that every slot held a
primitive, which sent any function holding a reference to the interpreter.

**Arrays in place (P4b).** An array operation finds its array the same way:
a value with the object word is its origin slot's object (every object has
one word), so `ArrayOf` reads that slot's pointer word, through the shadow
when there is one, and checks the class byte. Elements, the length and the
sparse flag are read at the offsets `abi.Encoding` gives. The semantics are
the slot IR's views': a read yields a number or exits, a store writes a
number over a number or leaves the instruction to Go, and keys are integers
below 2**32. Nothing native code reads of an object changes while it runs;
Go changes it only between entries.

**Kind tests are on the number word, with the VM's encoding passed in.** The
JIT never imports the VM: the VM passes the encoding as data (`rt.Encoding`).
- A number is a word whose top 13 bits are not all set.
- A tagged word at or above `heapBigBits` is a BigInt; otherwise its low byte
  is the kind and bits 8-15 its payload.
- Booleans, `undefined`, `null` and the uninitialized marker are exact
  words.
- Boxing a NaN writes the canonical NaN, because a negative NaN would collide
  with the tag range.

**No native stack yet.** A host exit stores the frame state and returns.
Go runs the one bytecode instruction and re-enters at the next PC's entry,
so an activation's spill area is dead once it exits, and a fixed area in the
context block serves. The native stack arrives with true resume (P5) and
native calls (Phase 5). The entry bridge stays a tail jump, as it is today.

## Migration

- **Side by side.** The new pipeline is chosen per runtime by an internal
  setting, `QJS_JIT_PIPELINE=ssa`, which the tests set. Until parity, the old
  pipeline stays the default.
- **Every test runs both.** The differential fuzzer, the stress corpus and
  stress test262 run against each.
- **Coverage grows op by op.** The SSA builder refuses what it doesn't handle
  yet, and the function stays in the old pipeline, or in the tree tier.
  Nothing is half-supported.
- **At parity** (Phase 2's gate), the default switches, and the old emitters
  and the Go-side encode and publish code go.

## Milestones

| ID | Deliverable | Done when |
|---|---|---|
| P1 | `ssa`: types, builder from slot IR, evaluator, the Phase 2 passes | The SSA evaluator matches the slot-IR evaluator on every program the slot-IR tests and fuzzers make, at every entry and budget. |
| P2 | Walking skeleton, amd64: selection, allocation and encoders for the numeric subset; `rt` with context block, native stack, entry/exit/resume, `Return`/`Deopt`/`Poll`; the VM runs a numeric loop through it | Native matches the SSA evaluator; **bare round trip ≤ 15 ns, helper round trip ≤ 25 ns, the sum loop at least as fast as the old pipeline.** If not, redesign before going on. |
| P3 | The skeleton on arm64 | The same, measured on a Mac or the macOS runner. |
| P4 | Coverage: arrays and views, properties, globals, captured bindings, strings, bitwise operations in full, calls as helpers | Every slot-IR operation is built. Stress corpus and fuzzer pass on the new pipeline. |
| P5 | Helpers: contained and reentrant classes, re-validation blocks, the generation counter | `BenchmarkJITHostRoundTrip`: `%` and a Go call cost no more than in the tree tier. |
| P6 | Code arena (R8): chunked allocation per runtime, batched sealing | Mappings bounded; release verified on all OSes. |
| P7 | Parity | Every benchmark and suite at least as fast as the old pipeline on both architectures; stress test262 and fuzzer clean; compile budget met. Then the old pipeline is deleted. |

P1 needs no machine code and runs on any host. P2 is the risky step, and it
gates the rest.
