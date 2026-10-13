package vm

// YoungGenerationSupported reports whether this build includes the nursery.
const YoungGenerationSupported = youngBuilt

// YoungGenerationStats counts the experimental nursery's object slots.
type YoungGenerationStats struct {
	Allocated, Reused, Reclaimed, Promoted, Collections, Young uint64
}
