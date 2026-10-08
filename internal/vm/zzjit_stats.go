package vm

// JITStats counts what the optional native tier did in a runtime, for tests
// and the conformance runner: they must tell code that ran natively from code
// that only could have. Compiled counts programs made; Entries native
// entries; Guards guard failures; Hosts exits to Go for an operation;
// Budgets returns to Go when an entry's instruction budget ran out; and
// Interpreted invocations finished in the interpreter after native code;
// SSAEntries counts the entries into code the new pipeline compiled.
type JITStats struct {
	Compiled, Entries, Guards, Hosts, Budgets, Interpreted, SSAEntries uint64
}
