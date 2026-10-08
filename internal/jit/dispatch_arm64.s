//go:build quickjs_jit && !android && !ios && darwin

#include "textflag.h"

TEXT ·enterDispatch(SB), NOSPLIT|NOFRAME, $0-32
	MOVD code+0(FP), R16
	MOVD state+8(FP), R0
	MOVD slots+16(FP), R2
	MOVD arrays+24(FP), R1
	MOVD $·dispatchProgram(SB), R3
	MOVD R3, 8(R0)
	JMP (R16)

// Tail transfers preserve Go's stack and original LR. All mutable metadata and
// frame cells are scalars; the Go owner prepares and roots immutable pointers.
TEXT ·dispatchProgram(SB), NOSPLIT|NOFRAME, $0
	MOVD 8(R0), R3
	CBZ R3, dispatchReturn
	JMP dispatchCall

dispatchReturn:
	MOVD 40(R0), R3
	MOVD 104(R0), R4
	LSL $7, R3, R6
	ADD R6, R4, R4
	MOVD 0(R0), R5
	MOVD 56(R4), R6
	SUB R5, R6, R6
	MOVD 48(R4), R7
	ADD R6, R7, R7
	MOVD R7, 48(R4)
	CBZ R3, dispatchDone
	MOVD 40(R4), R6
	MOVD 64(R0), R5
	SUB R6, R5, R5
	MOVD R5, 64(R0)
	MOVD 16(R4), R6
	SUB $1, R3, R3
	MOVD R3, 40(R0)
	MOVD 104(R0), R4
	LSL $7, R3, R5
	ADD R5, R4, R4
	MOVD 48(R4), R5
	ADD R7, R5, R5
	MOVD R5, 48(R4)
	MOVD 0(R0), R5
	MOVD R5, 56(R4)
	MOVD 0(R4), R17
	MOVD 8(R4), R9
	MOVD R9, 16(R0)
	MOVD 112(R0), R2
	LSL $12, R3, R3
	ADD R3, R2, R2
	LSL $4, R6, R6
	ADD R2, R6, R6
	MOVD 24(R0), R3
	MOVD R3, 0(R6)
	MOVD 32(R0), R3
	MOVD R3, 8(R6)
	MOVD ZR, 24(R0)
	MOVD ZR, 32(R0)
	MOVD 96(R0), R4
	LSL $6, R17, R17
	ADD R17, R4, R4
	MOVD 16(R4), R6
	LSL $3, R9, R9
	ADD R9, R6, R6
	MOVD (R6), R16
	CMP $0, R16
	BLT dispatchHost
	MOVD 8(R4), R6
	ADD R6, R16, R16
	MOVD $·dispatchProgram(SB), R3
	MOVD R3, 8(R0)
	JMP (R16)

dispatchCall:
	MOVD 40(R0), R17
	MOVD 104(R0), R4
	LSL $7, R17, R3
	ADD R3, R4, R4
	MOVD 0(R4), R6
	MOVD 96(R0), R7
	LSL $6, R6, R3
	ADD R3, R7, R7
	MOVD 24(R0), R3
	MOVD 56(R7), R5
	CMP R5, R3
	BHS dispatchHost
	LSL $4, R6, R6
	ADD R3, R6, R6
	LSL $7, R6, R6
	MOVD 120(R0), R16
	ADD R6, R16, R16
	MOVD 0(R16), R3
	MOVD 48(R0), R5
	CMP R5, R3
	BHS dispatchHost
	MOVD 8(R16), R6
	LSL $4, R6, R6
	ADD R2, R6, R6
	MOVD 0(R6), R3
	MOVD 48(R16), R5
	CMP R5, R3
	BNE dispatchHost
	MOVD 8(R6), R3
	MOVD 56(R16), R5
	CMP R5, R3
	BNE dispatchHost
	MOVD 40(R16), R6
	CMP $0, R6
	BLT dispatchReceiverOK
	LSL $4, R6, R6
	ADD R2, R6, R6
	MOVD 0(R6), R3
	MOVD 64(R16), R5
	CMP R5, R3
	BNE dispatchHost
	MOVD 8(R6), R3
	MOVD 72(R16), R5
	CMP R5, R3
	BNE dispatchHost
dispatchReceiverOK:
	MOVD 56(R0), R3
	CBZ R3, dispatchHost
	ADD $1, R17, R3
	MOVD 88(R0), R5
	CMP R5, R3
	BHS dispatchHost
	MOVD 0(R16), R3
	MOVD 96(R0), R7
	LSL $6, R3, R3
	ADD R3, R7, R7
	MOVD 48(R7), R3
	MOVD 72(R0), R6
	MOVD 64(R0), R5
	SUB R5, R6, R6
	CMP R6, R3
	BHI dispatchHost
	MOVD 0(R0), R6
	CBZ R6, dispatchHost

	MOVD 56(R4), R3
	SUB R6, R3, R3
	ADD $1, R3, R3
	MOVD 48(R4), R5
	ADD R3, R5, R5
	MOVD R5, 48(R4)
	MOVD 16(R0), R3
	ADD $1, R3, R3
	MOVD R3, 8(R4)
	SUB $1, R6, R6
	MOVD R6, 0(R0)
	MOVD 56(R0), R3
	SUB $1, R3, R3
	MOVD R3, 56(R0)
	MOVD 80(R0), R3
	ADD $1, R3, R3
	MOVD R3, 80(R0)
	MOVD 64(R0), R3
	MOVD 48(R7), R5
	ADD R5, R3, R3
	MOVD R3, 64(R0)
	MOVD 16(R16), R3
	LSL $4, R3, R3
	ADD R2, R3, R3
	ADD $1, R17, R17
	MOVD R17, 40(R0)
	MOVD 104(R0), R4
	LSL $7, R17, R5
	ADD R5, R4, R4
	MOVD 112(R0), R2
	LSL $12, R17, R5
	ADD R5, R2, R2
	MOVD R3, R17 // caller arguments survive template copying
	MOVD 0(R16), R3
	MOVD R3, 0(R4)
	MOVD ZR, 8(R4)
	MOVD 32(R16), R3
	MOVD R3, 16(R4)
	MOVD 16(R16), R3
	MOVD R3, 24(R4)
	MOVD 24(R16), R3
	MOVD R3, 32(R4)
	MOVD 48(R7), R3
	MOVD R3, 40(R4)
	MOVD ZR, 48(R4)
	MOVD R6, 56(R4)
	MOVD 80(R0), R3
	MOVD R3, 64(R4)
	MOVD ZR, 16(R0)
	MOVD ZR, 24(R0)
	MOVD ZR, 32(R0)
	MOVD 24(R7), R6
	MOVD 32(R7), R3
	MOVD R2, R4
	CBZ R3, dispatchTemplateDone
dispatchTemplate:
	MOVD 0(R6), R5
	MOVD 8(R6), R9
	MOVD R5, 0(R4)
	MOVD R9, 8(R4)
	ADD $16, R6, R6
	ADD $16, R4, R4
	SUB $1, R3, R3
	CBNZ R3, dispatchTemplate
dispatchTemplateDone:
	MOVD 24(R16), R3
	MOVD 40(R7), R6
	CMP R6, R3
	BLS dispatchArgCount
	MOVD R6, R3
dispatchArgCount:
	MOVD R2, R4
	CBZ R3, dispatchArgumentsDone
dispatchArguments:
	MOVD 0(R17), R5
	MOVD 8(R17), R9
	MOVD R5, 0(R4)
	MOVD R9, 8(R4)
	ADD $16, R17, R17
	ADD $16, R4, R4
	SUB $1, R3, R3
	CBNZ R3, dispatchArguments
dispatchArgumentsDone:
	MOVD 0(R7), R16
	CBZ R16, dispatchHost
	MOVD $·dispatchProgram(SB), R3
	MOVD R3, 8(R0)
	JMP (R16)

dispatchHost:
	MOVD $3, R3 // HostExit, or BudgetExit after a committed callee return
	MOVD 0(R0), R5
	CBNZ R5, dispatchHostKind
	MOVD $2, R3
dispatchHostKind:
	MOVD R3, 8(R0)
	MOVD ZR, 24(R0)
	MOVD ZR, 32(R0)
dispatchDone:
	RET
