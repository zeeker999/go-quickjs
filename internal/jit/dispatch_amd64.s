//go:build quickjs_jit && !android && !ios && (linux || windows)

#include "textflag.h"

TEXT ·enterDispatch(SB), NOSPLIT|NOFRAME, $0-32
	MOVQ code+0(FP), AX
	MOVQ state+8(FP), DI
	MOVQ slots+16(FP), SI
	MOVQ arrays+24(FP), CX
	LEAQ ·dispatchProgram(SB), DX
	MOVQ DX, 8(DI)
	JMP AX

// No native CALL or stack adjustment: SP, BP, g and Go's return address remain
// intact. Only scalar slots and scalar frame metadata are written.
TEXT ·dispatchProgram(SB), NOSPLIT|NOFRAME, $0
	CMPQ 8(DI), $0
	JE dispatchReturn
	JMP dispatchCall

dispatchReturn:
	MOVQ 40(DI), DX
	MOVQ 104(DI), R15
	MOVQ DX, AX
	SHLQ $7, AX
	ADDQ AX, R15
	MOVQ 0(DI), R10
	MOVQ 56(R15), R8
	SUBQ R10, R8
	ADDQ R8, 48(R15)
	TESTQ DX, DX
	JE dispatchDone
	MOVQ 40(R15), R8
	SUBQ R8, 64(DI)
	MOVQ 16(R15), R9
	MOVQ 48(R15), R8
	DECQ DX
	MOVQ DX, 40(DI)
	MOVQ 104(DI), R15
	MOVQ DX, AX
	SHLQ $7, AX
	ADDQ AX, R15
	ADDQ R8, 48(R15)
	MOVQ R10, 56(R15)
	MOVQ 0(R15), R8
	MOVQ 8(R15), R11
	MOVQ R11, 16(DI)
	MOVQ 112(DI), SI
	MOVQ DX, AX
	SHLQ $12, AX
	ADDQ AX, SI
	SHLQ $4, R9
	LEAQ (SI)(R9*1), AX
	MOVQ 24(DI), BX
	MOVQ BX, 0(AX)
	MOVQ 32(DI), BX
	MOVQ BX, 8(AX)
	MOVQ $0, 24(DI)
	MOVQ $0, 32(DI)
	MOVQ 96(DI), R12
	SHLQ $6, R8
	ADDQ R8, R12
	MOVQ 16(R12), R9
	MOVQ (R9)(R11*8), AX
	CMPQ AX, $0
	JL dispatchHost
	ADDQ 8(R12), AX
	LEAQ ·dispatchProgram(SB), DX
	MOVQ DX, 8(DI)
	JMP AX

dispatchCall:
	MOVQ 40(DI), DX
	MOVQ 104(DI), R15
	MOVQ DX, AX
	SHLQ $7, AX
	ADDQ AX, R15
	MOVQ 0(R15), R8
	MOVQ 96(DI), R12
	MOVQ R8, AX
	SHLQ $6, AX
	ADDQ AX, R12
	MOVQ 24(DI), R9
	CMPQ R9, 56(R12)
	JAE dispatchHost
	SHLQ $4, R8
	ADDQ R9, R8
	SHLQ $7, R8
	MOVQ 120(DI), R13
	ADDQ R8, R13
	MOVQ 0(R13), R8
	CMPQ R8, 48(DI)
	JAE dispatchHost
	MOVQ 8(R13), AX
	SHLQ $4, AX
	ADDQ SI, AX
	MOVQ 0(AX), R8
	CMPQ R8, 48(R13)
	JNE dispatchHost
	MOVQ 8(AX), R8
	CMPQ R8, 56(R13)
	JNE dispatchHost
	MOVQ 40(R13), AX
	CMPQ AX, $0
	JL dispatchReceiverOK
	SHLQ $4, AX
	ADDQ SI, AX
	MOVQ 0(AX), R8
	CMPQ R8, 64(R13)
	JNE dispatchHost
	MOVQ 8(AX), R8
	CMPQ R8, 72(R13)
	JNE dispatchHost
dispatchReceiverOK:
	CMPQ 56(DI), $0
	JE dispatchHost
	MOVQ DX, AX
	INCQ AX
	CMPQ AX, 88(DI)
	JAE dispatchHost
	MOVQ 0(R13), R8
	MOVQ 96(DI), R12
	SHLQ $6, R8
	ADDQ R8, R12
	MOVQ 48(R12), R8
	MOVQ 72(DI), R9
	SUBQ 64(DI), R9
	CMPQ R8, R9
	JA dispatchHost
	MOVQ 0(DI), R10
	TESTQ R10, R10
	JE dispatchHost

	MOVQ 56(R15), R9
	SUBQ R10, R9
	INCQ R9
	ADDQ R9, 48(R15)
	MOVQ 16(DI), AX
	INCQ AX
	MOVQ AX, 8(R15)
	DECQ R10
	MOVQ R10, 0(DI)
	DECQ 56(DI)
	INCQ 80(DI)
	ADDQ R8, 64(DI)
	MOVQ 16(R13), AX
	SHLQ $4, AX
	LEAQ (SI)(AX*1), R11
	INCQ DX
	MOVQ DX, 40(DI)
	MOVQ 104(DI), R15
	MOVQ DX, AX
	SHLQ $7, AX
	ADDQ AX, R15
	MOVQ 112(DI), SI
	MOVQ DX, AX
	SHLQ $12, AX
	ADDQ AX, SI
	MOVQ 0(R13), AX
	MOVQ AX, 0(R15)
	MOVQ $0, 8(R15)
	MOVQ 32(R13), AX
	MOVQ AX, 16(R15)
	MOVQ 16(R13), AX
	MOVQ AX, 24(R15)
	MOVQ 24(R13), AX
	MOVQ AX, 32(R15)
	MOVQ 48(R12), AX
	MOVQ AX, 40(R15)
	MOVQ $0, 48(R15)
	MOVQ R10, 56(R15)
	MOVQ 80(DI), AX
	MOVQ AX, 64(R15)
	MOVQ $0, 16(DI)
	MOVQ $0, 24(DI)
	MOVQ $0, 32(DI)
	MOVQ 24(R12), R9
	MOVQ 32(R12), R10
	MOVQ SI, R8
	TESTQ R10, R10
	JE dispatchTemplateDone
dispatchTemplate:
	MOVQ 0(R9), AX
	MOVQ 8(R9), BX
	MOVQ AX, 0(R8)
	MOVQ BX, 8(R8)
	ADDQ $16, R9
	ADDQ $16, R8
	DECQ R10
	JNZ dispatchTemplate
dispatchTemplateDone:
	MOVQ 24(R13), R10
	CMPQ R10, 40(R12)
	JBE dispatchArgCount
	MOVQ 40(R12), R10
dispatchArgCount:
	MOVQ SI, R8
	TESTQ R10, R10
	JE dispatchArgumentsDone
dispatchArguments:
	MOVQ 0(R11), AX
	MOVQ 8(R11), BX
	MOVQ AX, 0(R8)
	MOVQ BX, 8(R8)
	ADDQ $16, R11
	ADDQ $16, R8
	DECQ R10
	JNZ dispatchArguments
dispatchArgumentsDone:
	MOVQ 0(R12), AX
	TESTQ AX, AX
	JE dispatchHost
	LEAQ ·dispatchProgram(SB), DX
	MOVQ DX, 8(DI)
	JMP AX

dispatchHost:
	MOVQ $3, AX // HostExit, or BudgetExit at the next uncommitted instruction
	CMPQ 0(DI), $0
	JNE dispatchHostKind
	MOVQ $2, AX
dispatchHostKind:
	MOVQ AX, 8(DI)
	MOVQ $0, 24(DI)
	MOVQ $0, 32(DI)
dispatchDone:
	RET
