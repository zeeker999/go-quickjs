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
