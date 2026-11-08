package ssa

import "testing"

// int32Phis counts f's int32 phis, and its phis of doubles a loop's
// header has.
func int32Phis(f *Func) (ints, doubles int) {
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op != OpPhi || !b.LoopHeader {
				continue
			}
			switch v.Type {
			case Int32:
				ints++
			case Float64:
				doubles++
			}
		}
	}
	return ints, doubles
}

// A loop's phi of doubles that only carries int32s is an int32 phi where
// its uses take it as one (intPhis, intPhiUses): Emscripten's counter and
// sums are; Crypto's am3 masks, x & 0x3fff and x >> 14, carried round its
// loop by its entry's phis for products with its digits, are not -- as
// int32s they had been converted to doubles at every digit, the V8
// suite's Crypto 27% slower. A sum read only after its loop is one too.
func TestIntPhisWhereTheyPay(t *testing.T) {
	for _, c := range []struct {
		src     string
		minInts int
		ints    bool
	}{
		{`function f(n){var j=0,a=1,b=2,s=0;for(j=0;(j|0)<(n|0);j=(j+1)|0){a=(a*1103+b)|0;b=(b^(a>>>7))|0;s=(s+(a&255)-(b>>29))|0}return s}`, 4, true},
		{`function f(t,w,x,n){var xl=x&0x3fff,xh=x>>14,c=0,i=0,j=0;while(--n>=0){var l=t[i]&0x3fff;var h=t[i++]>>14;var m=xh*l+h*xl;l=xl*l+((m&0x3fff)<<14)+w[j]+c;c=(l>>28)+(m>>14)+xh*h;w[j++]=l&0xfffffff}return c}`, 0, false},
		{`function f(n){var s=0;for(var i=0;i<n;i=(i+1)|0)s=(s+i)|0;return s}`, 2, true},
	} {
		f := lowerJS(t, c.src)
		Optimize(f)
		ints, _ := int32Phis(f)
		if c.ints && ints < c.minInts || !c.ints && ints != 0 {
			t.Errorf("%s: %d int32 phis, want %v (at least %d):\n%s", c.src, ints, c.ints, c.minInts, f)
		}
	}
}
