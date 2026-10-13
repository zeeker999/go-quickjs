package quickjs

import "github.com/go-quickjs/go-quickjs/internal/vm"

// WithYoungGeneration enables an experimental, non-moving nursery for ordinary
// two-property objects allocated by the bytecode interpreter. It requires the
// quickjs_young build tag, independently of WithJIT. Other builds use Go GC.
// Tree and native code currently keep using ordinary Go allocation.
//
// Capacity bounds the number of nursery slots, including reusable free slots;
// zero or less selects 4096. Storage is allocated on first eligible use.
// Survivors and objects crossing unsupported operations are promoted in place
// to Go ownership. The nursery is disabled with WithDebugger or WithMemoryLimit.
// This option is experimental and does not guarantee a speedup.
func WithYoungGeneration(capacity int) Option {
	if capacity <= 0 {
		capacity = 4096
	}
	return func(c *config) { c.youngGeneration = capacity }
}

// YoungGenerationStats counts nursery allocations, reuse, reclamation,
// promotion, minor collections, and currently occupied young slots.
type YoungGenerationStats = vm.YoungGenerationStats

// YoungGenerationStats reports the activity of this runtime's optional nursery.
// A disabled or unused nursery returns zero counters.
func (r *Runtime) YoungGenerationStats() YoungGenerationStats { return r.rt.YoungGenerationStats() }
