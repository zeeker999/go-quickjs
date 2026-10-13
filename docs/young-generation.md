# Experimental Young Generation

The nursery is optional, portable Go code, and independent of the native JIT.
Build with `-tags quickjs_young` and opt in for each runtime:

```go
rt := quickjs.New(quickjs.WithYoungGeneration(4096))
defer rt.Close()
```

The argument bounds the number of tracked object slots, including reusable
slots. Zero or less selects 4096. This is an object count, not a byte limit.
The runtime allocates nursery metadata only on its first eligible allocation.
Without the build tag the option falls back to ordinary Go allocation, and
the interpreter's nursery hooks compile out.

The first allocation path covers ordinary two-property literals in the
bytecode interpreter. Tree and native allocations still use Go allocation.
To exercise the bytecode path explicitly with the command-line host:

```sh
QJS_NOTREE=1 go run -tags quickjs_young ./cmd/qjs \
  --young-generation=4096 script.js
```

`WithDebugger` and `WithMemoryLimit` disable the nursery for now. Functions
with captured locals or direct eval also keep ordinary allocation. These
restrictions avoid treating hidden host references, debugger state, mapped
arguments, or approximate memory accounting as exact collector roots.

## Lifetime Rules

Each slot owns an independent, typed Go allocation with inline properties.
The collector marks reachable young objects, clears dead allocations, and
reuses those slots. Objects surviving two minor collections are promoted
with their reachable young graph. Promotion releases nursery ownership at
the existing address; Go subsequently decides the allocation's lifetime.

Only the flat interpreter stack and plain young-object edges are supported
as roots. Before an operation can expose a value through a host callback,
closure capture, old-object store, weak registration, exotic storage, or
an unsupported operation, the VM conservatively promotes the nursery.
It also promotes before leaving an interpreter invocation. This preserves
the lifetime of public, copyable `Value` handles without adding a release
protocol to the embedding API.

Ordinary cancellation polls with `context.Background()` or `context.TODO()`
retain young objects. Custom contexts trigger promotion before their methods
run, as does possible entry into the native tier. A large nursery can therefore
see little reuse with custom contexts or JIT execution.

Go's collector continues to see and scan all nursery storage. Go write
barriers remain active. The nursery reduces repeated allocation; it does
not hide pointers from Go or replace Go's collector. The approximate
memory-meter walker never participates in reclamation.

## Validation And Measurement

`Runtime.YoungGenerationStats()` reports fresh allocations, reuse,
reclamation, promotion, minor collections, and occupied young slots.
Reclamation and reuse are separate counters: a reclaimed slot may remain
unused until another eligible allocation.

```sh
go test -tags quickjs_young ./...
go test -race -tags quickjs_young ./internal/vm -run '^TestYoungGeneration'
GODEBUG=gccheckmark=1 go test -tags quickjs_young \
  -gcflags=all=-d=checkptr=2 ./internal/vm -run '^TestYoungGeneration'
go test -tags quickjs_young ./internal/vm -run '^$' \
  -bench '^BenchmarkYoungGenerationObjects$' -benchmem
QJS_NOTREE=1 TEST262_DIR=/path/to/test262 go test -tags quickjs_young \
  ./conformance -run '^TestConformance$' -v -args \
  -conformance.dir=language/expressions/object,language/statements/for \
  -conformance.young-generation=2
```

The V8 runner accepts `-young-generation=N`. Compare fresh processes with
the same build and execution tier, including allocations, collection CPU,
first-use cost, live heap, and peak RSS. Lower allocation counts alone are
not evidence of faster execution. Conservative promotion and unsupported
allocation paths can substantially limit reuse.

### Initial Results

These are experimental results on macOS/arm64, Apple M5 Max, Go 1.27.0.
Each figure is the median of five fresh processes, alternating configurations,
with the tree tier disabled and no native JIT. The loop creates one million
two-property numeric objects, reads both fields, and checks the final sum.
Allocation totals include runtime construction and compilation; the loop timer
includes compilation but excludes runtime construction.

| Build / nursery | Loop (ms) | Go allocations | Go collections | Peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Ordinary build | 74.59 | 1,001,406 | 29 | 21.33 |
| Tagged build, off | 83.05 | 1,001,405 | 29 | 21.28 |
| Tagged build, 64 slots | 88.38 | 1,435 | 1 | 18.44 |
| Tagged build, 4096 slots | 89.38 | 5,570 | 1 | 18.45 |

The enabled nursery reduces allocations by over 99%, but takes about 6-8%
longer than the same tagged binary with it off. The extra checks also cost time
in tagged builds with the runtime option off. Ordinary builds omit those checks.
This implementation is not yet a throughput optimization.

In the same fresh-process measurements, one literal's first evaluation,
including compilation and nursery initialization, takes 47.8 microseconds in
the ordinary build, 57.2 with the tagged nursery off, 123.3 with 64 slots, and
76.8 with 4096 slots. These small first-use figures vary with process startup
and allocation state. Construction takes about 0.45-0.66 ms and does not create
nursery storage. With the runtime still alive after Go GC, the loop's process
has about 4.7 MiB live heap with the nursery off or at 64 slots, and 5.9 MiB at
4096 slots. The unstripped CLI grows from 41,054,370 to 41,072,866 bytes when
the tag is included, an increase of 18,496 bytes.

EarleyBoyer, run five times per fresh process, takes 378.4 ms with the tagged
nursery off, 374.5 ms with 64 slots, and 369.3 ms with 4096 slots. All three
allocate about 1.411 million times and 197.9 MiB; peak RSS stays about 55-56 MiB.
There is no meaningful allocation reduction or established speedup for this
workload. Captured locals, conservative escape promotion, and unsupported
allocation paths restrict coverage.

An eight-placement comparison of ordinary builds against the pre-nursery
revision finds no material V8-suite regression (597.8 versus 592.3 ms total,
three alternating rounds per placement). This checks code layout separately
from comparisons that enable the nursery in one binary.

Validation includes the full Go suites for ordinary, nursery, and combined
JIT/nursery builds; focused race and pointer checks; Go 1.24 nursery tests;
and pure-Go builds for Linux/amd64, Windows/amd64, Windows/arm64, Linux/386,
and macOS/arm64. A selected Test262 run passes 5,018 variants with two slots.
Those short conformance tests exercise allocation and escape promotion but
do not collect; the focused VM tests exercise reclamation, reuse, cycles,
mixed-age promotion, host retention, and concurrent Go collections.

The next step for broader workloads is an exact root protocol for the tree
and native tiers, followed by barriers for old-to-young references. Adding
allocation paths before that protocol would make reuse unsafe.
