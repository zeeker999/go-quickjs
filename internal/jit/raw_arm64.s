//go:build quickjs_jit && !android && !ios && (darwin || linux)

#include "textflag.h"

// enterRaw tail-jumps to code another compiler made (RawCode), with a
// context of its maker's own in R0. The code returns through the link
// register directly to enterRaw's Go caller.
TEXT ·enterRaw(SB), NOSPLIT|NOFRAME, $0-16
	MOVD code+0(FP), R16
	MOVD ctx+8(FP), R0
	JMP (R16)
