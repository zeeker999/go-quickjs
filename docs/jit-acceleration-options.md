# Ways to make the JIT faster: options weighed

Written 2026-10-10 at e7752f1, before deciding what to build next. Nothing
here is implemented yet.

## Where the time goes

CPU profiles of each V8 suite on the new pipeline, construction on (the
configuration we mean to ship), `-mode fixed -n 20`. Flat time, grouped:

| Suite | Native code | Go GC + alloc | Tree tier | Other |
|---|---|---|---|---|
| Richards | 80% | – | – | parse 20% |
| NavierStokes | 79% | 4% | – | |
| Crypto | 61% | 26% | 5% | JIT glue 5% |
| EarleyBoyer | 45% | 46% | 2% | |
| RayTrace | 6% | 39% | 25% | VM built-ins 15%, JIT glue 9%, compiling 5% |
| Splay | 2% | 61% | 16% | string concatenation 9% |
| RegExp | – | 34% | 6% | regexp matcher 35%, VM 20% |
| DeltaBlue | 27% | 9% | 18% | VM 27% (few samples) |

Allocations per run: EarleyBoyer 1.54M, RegExp 1.42M, Splay 1.40M,
RayTrace 0.38M, Crypto 25k. GC frequency is not the lever: GOGC=400 leaves
the total level (Splay -17%, RayTrace and EarleyBoyer +10%). The cost
follows the number and size of allocations.

So there are three big buckets:
1. **Go's GC and allocator** (26-61% in five suites).
2. **Code that never reaches native code** (tree tier 16-25% in RayTrace,
   Splay and DeltaBlue), mostly through demotion, where native code leaves
   for Go too often.
3. **The quality of the native code itself** (45-80% where it runs).

## Options

### 1. Call Go from native code without leaving it (recommended first)

Today, anything native code can't do (string concatenation, `new RegExp`,
a built-in call, Math.pow outside its exact cases, a pointer store while
the collector marks) is an exit. The exit unwinds every native level into
VM frames, Go does the work, and the levels resume. Each such exit also
counts toward demoting the function (`jitUnwound`), and demotion spreads to
callers. That is RayTrace's 25% tree tier and Splay's GeneratePayloadTree
(5.1M tree entries).

The mechanism: native code is entered from Go assembly (`enterSSA`) and
never touches the machine stack (no CALL/PUSH/POP; the verify module checks
this). So the entry routine can become a real assembly frame with a
dispatch label. To call Go, native code saves its live registers where it
already saves them across native calls (scanned memory), stores a request
in its context, and jumps to that label. The routine CALLs one ordinary Go
function, `jitHostCall(ctx)`, through the ABI0 wrapper (which reloads g and
the zero register), then jumps back to the resume address.

The Go stack then shows only the assembly routine, with a known PC, frame
and stack map, and Go functions above it. Tracebacks, GC stack scans, stack
growth (native code holds no stack addresses) and async preemption (never
inside assembly or JIT code) all see a normal stack. It uses no runtime
internals, no linkname and no cgo, on amd64 and arm64.

- **Cost per call (estimate):** a few ns plus the dispatch, against 23.7 ns
  for a bare exit and re-entry (V6), and much more for an exit from inside
  native callees (unwinding, demotion).
- **Unlocks:** host operations that cost no demotion; stores that keep
  running natively while the collector marks (a Go helper does the store
  with its write barrier) instead of exiting; string concatenation;
  `new RegExp`; built-in calls in general; and later a baseline tier
  (option 7).
- **Risks:**
  - Re-entrancy: a Go helper that runs JS enters native code again, so
    nested entries must take contexts above the suspended levels.
  - Every live pointer must be in scanned memory across the call.
  - Exceptions thrown in the helper must come back as an exit.
- **Next step if chosen:** a prototype measuring the round trip and one
  user (string concatenation), before converting the host exits.

### 2. Coverage and the demotion policy

Reasons functions stay out of native code, by the survey:
- polymorphic stores (RayTrace's `best.hitCount = hits` meets two shapes);
- captured locals and closure creation (EarleyBoyer, 81k entries);
- the arguments object (58k);
- try/catch;
- for-in.

V8 never demotes a caller because a callee deoptimizes. After option 1,
host operations should not count toward demotion at all. Each item is small
to medium.

### 3. Fewer allocations from native code: escape analysis

V8's EscapeAnalysis plus scalar replacement. Constructions are already
inlined (receiver from a pool, constructor body inlined). An object that
never escapes the compiled function becomes its fields in registers, with
no allocation at all. RayTrace's Vector/Color temporaries are the case
(0.38M allocations per run). Allocation folding (one pool take for several
objects) is a smaller follow-up.

- **Effort:** medium-large, all in ssa (an analysis, rewriting field
  reads/writes, frame states that materialize the object at a deopt, as
  V8's do).
- **Risk:** deopt materialization correctness. Differential fuzzing covers
  it well.

### 4. Object representation (VM-wide; the biggest GC lever)

Each JS object is a Go `Object` (~96 bytes) plus a separate `props` slice.
Each `Property` is 24 bytes and repeats the key and flags its shape already
has. Arrays add an `elems` slice. `{x,y,z}` is about 170 bytes in two
allocations; V8's is about 30 in one.

Storing values only, inline for small objects, with keys and flags in the
shape, would roughly halve the bytes and allocations the GC handles in
EarleyBoyer, Splay, RegExp and RayTrace (34-61% of their time).

- **Cost:** it touches every tier and built-in, the debugger and the memory
  meter, and internal/vm's layout-sensitive hot paths.
- **Approach:** a project of its own with a plan and measurements, not a
  JIT change. Expect every tier to gain, not just the JIT.

### 5. Native code quality

For the native-dominated suites (Richards, NavierStokes, Crypto, half of
EarleyBoyer):
- load elimination and loop-invariant code motion in general (today only
  captured bindings and entry slots are handled specially);
- int32 representation where ranges allow (Crypto's bit operations; V8's
  representation selection);
- bounds-check elimination.

Each is a contained optimizer pass with differential tests. Gains are
probably 5-20% per suite where native code dominates.

### 6. Leaf machine-code routines

Math's transcendental functions (pow, exp, log, sin, …) and string
compare/hash. They are ported operation by operation from the Go code, so
results match bit for bit, and emitted into the arena. This is V8's C-call
model without calling Go. Little for these suites; useful for real code.
Option 1 makes it less urgent, since a Go call is then cheap.

### 7. A baseline tier (after option 1)

V8's Sparkplug: compile every function quickly, without speculation, to
machine code that does the simple cases inline and calls Go (option 1) for
the rest. It would replace the tree tier for warm code and for code the
optimizing tier refuses. It is large. Decide after options 1 and 2 show how
much tree time is left.

### 8. RegExp

35% of RegExp is the matcher, a separate engine. V8 compiles regexps to
machine code (irregexp). Profile and tune the Go matcher first; a regexp
compiler would be a project of its own.

### 9. Concurrent compilation

Compiling is 1-5% of time with construction on. The builder reads live VM
feedback, so it would need snapshots. Low priority.

## Not recommended

- **Making JIT code look like Go code to the runtime** (sonic-style:
  registering fake modules with pclntab, stack maps and SP tables, and
  calling `mallocgc` and the write barrier directly). It depends on runtime
  internals reached through linkname, which Go 1.23+ refuses without
  `-ldflags=-checklinkname=0` (sonic is grandfathered). The formats change
  between releases, and we test 1.24 and 1.27. A wrong stack map is silent
  heap corruption. Option 1 gives most of the benefit, calling Go cheaply,
  without any of that.
- **GC tuning by GOGC:** measured above, no net gain.

## Progress

**Option 1, first step (3ee0a14, d6455b0).** Native code calls Go
through `callGo` and goes on after it.

What it does:
- Arguments pass as words; no pointer is stored natively, so the call
  works while the collector marks.
- The result goes to the call's keep cell, written by Go.
- A refusal exits as before.
- `+` with a string, or a `+` whose speculation failed, is the first user
  (GoAdd). The helper never throws or runs script code.

How it was checked:
- Stack growth, GC and tracebacks across the call are tested on both
  architectures; CI is green, arm64 included.
- One bug was found and fixed on the way. A refused call exits with the
  state at the call, so the state's values must be saved across the call.

What it changed:
- A string-building loop is 12% faster. The V8 suite is level: string
  work itself dominates those loops.
- DeltaBlue with construction on lost 5 ms. Constraint functions now
  stay native and meet the next gap: `Constraint` has no native code (a
  store meeting several shapes), so each call to it from native code
  leaves for Go.

**Go calls, later steps (a1d4ca0, 124b7fc, 2a80874).**
- *Object pools* are filled again by a Go call (GoRefill), and the call
  is made again from its start. With construction on, RayTrace improved
  19% and EarleyBoyer 9%.
- *Pointer stores while the collector marks* are made by Go (GoStore),
  with its write barrier, instead of exiting. Keeps are written the same
  way. Their old "write nothing while marking" relied on every store
  exiting then; once Go made stores, it built cyclic lists
  (TestJITSSAKeepsWhileMarking).
- *The profit check* skips exits that marking caused. Native code flags
  them in `ExitMarking`. A function Go stopped entering now spends its
  recompile budget through the existing tiers' tries.
- *Measured.* Construction on: EarleyBoyer -8%, Splay -4%. Construction
  off: Splay +3.5%, within the noise. A Go store saves every register.

**Host exits that remain** (construction on, 10 iterations, by the
operation Go then runs):

| Suite | Exits |
|---|---|
| RayTrace | ~180,000 at `call_method` |
| Crypto | 26,800 at `set_index` |
| EarleyBoyer | 20,700 at `new`, 5,000 at `call` |
| DeltaBlue | 1,400 at `set_prop` |
| Splay | 50,700, almost all a literal pool running out |

RayTrace's calls are inlined callees that construct (`new Color(...)`
inside `Color.prototype.multiply`): inside an inlined callee a
construction exits, and Go makes the whole call again (Restart).

**What this suggests next.**
- *Re-entrant Go calls.* A call of Go that may run script code (a
  constructor, a callee without native code, a setter), with the native
  levels suspended, as V8's runtime calls may call JS. It would turn most
  remaining exits into calls. It needs three things:
  - `ctxTop` raised past the calling level for the call, since today
    `runSSA` takes `ctxTop` while native callee levels use the contexts
    past it without moving it;
  - the run's current-context state saved and restored;
  - a status for "threw", which exits to rethrow instead of making the
    operation again.

  Assumptions across the call are already re-checked, since native
  callees may run anything.
- *Pool refills by a Go call.* No script code runs, so it is small and
  safe. Literal pools running out are counted as leaving (unlike
  constructions'), which demotes functions such as Splay's
  GeneratePayloadTree.
- *Calls of Go built-ins by a Go call*: `String(x)` still leaves at
  every call, and that alone demoted a test's loop.
- *Polymorphic stores* (DeltaBlue's `Constraint`, RayTrace's
  `best.hitCount`), from option 2.

## Suggested order

1. Option 1, as a prototype first.
2. Option 2's items, with host operations no longer counting toward
   demotion.
3. Option 3 for RayTrace.
4. Option 5 passes as profiles of native code point to them.
5. Option 4 as its own planned project, when we choose to take on the VM's
   representation.
6. Options 6-9 when the measurements ask for them.
