#include "funcdata.h"
#include "textflag.h"

// The native handlers' entry, and their store of a pointer with the write
// barrier (zzzyn_native_amd64.go): the JIT's enterRaw and storeRef, the
// package's own, so that the handlers need no JIT.

TEXT ·nativeEnter(SB), NOSPLIT|NOFRAME, $0-16
	MOVQ code+0(FP), AX
	MOVQ ctx+8(FP), DI
	JMP AX

TEXT ·nativeStoreRef(SB), NOSPLIT|NOFRAME, $0-0
	ADJSP $40
	NO_LOCAL_POINTERS
	MOVQ AX, 0(SP)
	MOVQ CX, 8(SP)
	MOVQ DI, 16(SP)
	MOVQ R11, 24(SP)
	CALL ·nativeStoreRefGo(SB)
	MOVQ 16(SP), DI
	MOVQ 24(SP), AX
	ADJSP $-40
	JMP AX

TEXT ·nativeStoreRefAddr(SB), NOSPLIT, $0-8
	LEAQ ·nativeStoreRef(SB), AX
	MOVQ AX, ret+0(FP)
	RET

TEXT ·nativeHandlersAddr(SB), NOSPLIT, $0-8
	LEAQ ·regNativeHandlers(SB), AX
	MOVQ AX, ret+0(FP)
	RET

TEXT ·nativeTableAddr(SB), NOSPLIT, $0-8
	LEAQ ·regNativeTable(SB), AX
	MOVQ AX, ret+0(FP)
	RET
