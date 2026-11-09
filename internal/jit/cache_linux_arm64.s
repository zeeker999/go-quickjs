//go:build quickjs_jit && !android && !ios

#include "textflag.h"

// flushInstructionCache makes code just written visible to instruction
// fetch: each data cache line cleaned to the point of unification, a
// barrier, each instruction cache line invalidated, a barrier, and a
// synchronization of the context. The lines' sizes are CTR_EL0's, which
// Linux lets EL0 read (or emulates).
TEXT ·flushInstructionCache(SB), NOSPLIT|NOFRAME, $0-16
	MOVD start+0(FP), R0
	MOVD size+8(FP), R1
	CBZ R1, done
	ADD R0, R1, R1
	WORD $0xd53b0022 // mrs x2, ctr_el0
	// The data cache's line: 4 << DminLine, bits 16 to 19, bytes.
	UBFX $16, R2, $4, R3
	MOVD $4, R4
	LSL R3, R4, R3
	SUB $1, R3, R5
	BIC R5, R0, R6
data:
	WORD $0xd50b7b26 // dc cvau, x6
	ADD R3, R6, R6
	CMP R1, R6
	BLO data
	DSB $11
	// The instruction cache's line: 4 << IminLine, bits 0 to 3, bytes.
	AND $15, R2, R3
	LSL R3, R4, R3
	SUB $1, R3, R5
	BIC R5, R0, R6
instruction:
	WORD $0xd50b7526 // ic ivau, x6
	ADD R3, R6, R6
	CMP R1, R6
	BLO instruction
	DSB $11
	ISB $15
done:
	RET
