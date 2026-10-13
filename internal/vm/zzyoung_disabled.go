//go:build !quickjs_young

package vm

import "github.com/go-quickjs/go-quickjs/internal/bytecode"

const youngBuilt = false

type youngFields struct{}

func (*Runtime) initYoung(int) {}

func (*Runtime) releaseYoung()                                    {}
func (*Runtime) promoteYoung()                                    {}
func (*Runtime) youngBefore(*frame, bytecode.Instr, []Value, int) {}
func (*Runtime) youngLiteral(*frame, int, int, *shape) *Object    { return nil }

// YoungGenerationStats is zero in a build without quickjs_young.
func (*Runtime) YoungGenerationStats() YoungGenerationStats { return YoungGenerationStats{} }
