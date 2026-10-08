//go:build quickjs_jit && !android && !ios && (linux || windows)

#include "textflag.h"

TEXT ·enterProgram(SB), NOSPLIT|NOFRAME, $0-32
	MOVQ code+0(FP), AX
	MOVQ state+8(FP), DI
	MOVQ slots+16(FP), SI
	MOVQ arrays+24(FP), CX
	JMP AX

// enterSSA tail-jumps to a function compiled by the new pipeline, with the
// context block in DI. The code returns directly to enterSSA's Go caller.
TEXT ·enterSSA(SB), NOSPLIT|NOFRAME, $0-16
	MOVQ code+0(FP), AX
	MOVQ ctx+8(FP), DI
	JMP AX
