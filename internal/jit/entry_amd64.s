//go:build quickjs_jit && !android && !ios && (linux || windows)

#include "textflag.h"

TEXT ·enter(SB), NOSPLIT|NOFRAME, $0-16
	MOVQ code+0(FP), AX
	MOVQ state+8(FP), DI
	JMP AX

TEXT ·enterProgram(SB), NOSPLIT|NOFRAME, $0-24
	MOVQ code+0(FP), AX
	MOVQ state+8(FP), DI
	MOVQ slots+16(FP), SI
	JMP AX
