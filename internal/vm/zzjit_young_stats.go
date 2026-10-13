package vm

// YoungGenerationStats counts the experimental nursery's object slots.
type YoungGenerationStats struct {
	Allocated, Reused, Reclaimed, Promoted, Collections, Young uint64
}
