//go:build quickjs_jit && !android && !ios && (linux || windows)

#include "textflag.h"

TEXT ·enterProgram(SB), NOSPLIT|NOFRAME, $0-32
	MOVQ code+0(FP), AX
	MOVQ state+8(FP), DI
	MOVQ slots+16(FP), SI
	MOVQ arrays+24(FP), CX
	JMP AX
