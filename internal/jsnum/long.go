package jsnum

import (
	"math"
	"math/bits"
	"strconv"
	"strings"
)

// maxDecimalDigits is the most significant digits a decimal literal is
// read with. A double nearest a decimal is decided by its first 768
// significant digits and whether any after them is not a zero: the
// midpoints between doubles, at which rounding turns, have at most 767. So
// the digits after the 800th are only looked at, as V8 looks at those
// after its 772nd (kMaxSignificantDigits): strconv reads every digit,
// which in a string of millions of them took longer than V8 by half again.
const maxDecimalDigits = 800

// ParseFloat is strconv.ParseFloat(s, 64), which a decimal literal of more
// than maxDecimalDigits characters is given in shortDecimal's shorter form.
func ParseFloat(s string) (float64, error) {
	if len(s) > maxDecimalDigits {
		if short, ok := shortDecimal(s); ok {
			s = short
		}
	}
	return strconv.ParseFloat(s, 64)
}

// ParseShort reads the number JSON's grammar gives at s[i:] -- an optional
// minus, digits with no leading zero, a fraction, an exponent -- and returns
// it and where it ends, if its digits fit a uint64 and its power of ten is in
// the range of powers, as nearly every number in a document does. ok is false
// for anything else, a number or not, for ParseFloat, which is exact for
// every number but reads the text again for it, which is most of what a
// number in a document costs.
func ParseShort(s string, i int) (f float64, end int, ok bool) {
	n := len(s)
	neg := false
	if i < n && s[i] == '-' {
		neg = true
		i++
	}
	start := i
	var m uint64
	for i < n {
		d := s[i] - '0'
		if d > 9 {
			break
		}
		m = m*10 + uint64(d)
		i++
	}
	digits := i - start
	if digits == 0 || s[start] == '0' && digits > 1 {
		return 0, 0, false
	}
	exp := 0
	if i < n && s[i] == '.' {
		i++
		from := i
		for i < n {
			d := s[i] - '0'
			if d > 9 {
				break
			}
			m = m*10 + uint64(d)
			i++
		}
		if i == from {
			return 0, 0, false
		}
		exp = from - i
		digits += i - from
	}
	if digits > maxShortDigits {
		return 0, 0, false
	}
	if i < n && (s[i] == 'e' || s[i] == 'E') {
		i++
		expNeg := false
		if i < n && (s[i] == '+' || s[i] == '-') {
			expNeg = s[i] == '-'
			i++
		}
		from, e := i, 0
		for i < n && i-from < 4 {
			d := s[i] - '0'
			if d > 9 {
				break
			}
			e = e*10 + int(d)
			i++
		}
		if i == from || i < n && s[i]-'0' <= 9 {
			return 0, 0, false
		}
		if expNeg {
			e = -e
		}
		exp += e
	}
	// The digits and a power of ten up to 10^22 are each exactly a double, so
	// that their product or quotient is the double nearest the number. This
	// is what is left to do for most numbers; it is also what decides one
	// that a double holds exactly, which scaledDecimal, whose powers below 1
	// are not exact, cannot tell from a midpoint.
	if m < 1<<53 && exp >= -maxExactPower && exp <= maxExactPower {
		f = float64(m)
		if exp < 0 {
			f /= exactPowers[-exp]
		} else if exp > 0 {
			f *= exactPowers[exp]
		}
		if neg {
			f = -f
		}
		return f, i, true
	}
	if m == 0 {
		if neg {
			return math.Copysign(0, -1), i, true
		}
		return 0, i, true
	}
	if f, ok = scaledDecimal(m, exp, neg); !ok {
		return 0, 0, false
	}
	return f, i, true
}

// maxExactPower is the highest power of ten a double holds exactly.
const maxExactPower = 22

// exactPowers is the powers of ten a double holds exactly.
var exactPowers = [maxExactPower + 1]float64{1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10,
	1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22}

// maxShortDigits is the most digits a uint64 is sure to hold.
const maxShortDigits = 19

// scaledDecimal is the double nearest m times ten to the exp, for an m that is
// not zero, by the Eisel-Lemire algorithm: m times a 128-bit approximation
// of the power, enough of whose bits are known to be right to say how it
// rounds, or ok false where they are not, or the result would be subnormal
// or infinite. strconv had it until Go 1.27, and no longer has what it takes,
// so that a program gets it here whichever Go built it.
func scaledDecimal(m uint64, exp int, neg bool) (f float64, ok bool) {
	if exp < minShortPower || exp > maxShortPower {
		return 0, false
	}
	power := &shortPowers[exp-minShortPower]
	clz := bits.LeadingZeros64(m)
	m <<= uint(clz)
	const bias = 1023
	exp2 := uint64(217706*exp>>16+64+bias) - uint64(clz)

	xHi, xLo := bits.Mul64(m, power[1])
	if xHi&0x1FF == 0x1FF && xLo+m < m {
		yHi, yLo := bits.Mul64(m, power[0])
		mergedHi, mergedLo := xHi, xLo+yHi
		if mergedLo < xLo {
			mergedHi++
		}
		if mergedHi&0x1FF == 0x1FF && mergedLo+1 == 0 && yLo+m < m {
			return 0, false
		}
		xHi, xLo = mergedHi, mergedLo
	}

	// To 54 bits.
	msb := xHi >> 63
	mant := xHi >> (msb + 9)
	exp2 -= 1 ^ msb
	// Half way between two doubles, which only exact arithmetic decides.
	if xLo == 0 && xHi&0x1FF == 0 && mant&3 == 1 {
		return 0, false
	}
	// To 53.
	mant += mant & 1
	mant >>= 1
	if mant>>53 > 0 {
		mant >>= 1
		exp2++
	}
	if exp2-1 >= 0x7FF-1 {
		return 0, false
	}
	b := exp2<<52 | mant&(1<<52-1)
	if neg {
		b |= 1 << 63
	}
	return math.Float64frombits(b), true
}

// The powers of ten scaledDecimal knows.
const (
	minShortPower = -64
	maxShortPower = 64
)

// shortPowers is, for each power of ten from minShortPower, its significand
// to 128 bits, the top one set, rounded down as the algorithm has it: the low
// word, then the high.
var shortPowers = [maxShortPower - minShortPower + 1][2]uint64{
	{0x3F2398D747B36224, 0xA87FEA27A539E9A5}, // 1e-64
	{0x8EEC7F0D19A03AAD, 0xD29FE4B18E88640E}, // 1e-63
	{0x1953CF68300424AC, 0x83A3EEEEF9153E89}, // 1e-62
	{0x5FA8C3423C052DD7, 0xA48CEAAAB75A8E2B}, // 1e-61
	{0x3792F412CB06794D, 0xCDB02555653131B6}, // 1e-60
	{0xE2BBD88BBEE40BD0, 0x808E17555F3EBF11}, // 1e-59
	{0x5B6ACEAEAE9D0EC4, 0xA0B19D2AB70E6ED6}, // 1e-58
	{0xF245825A5A445275, 0xC8DE047564D20A8B}, // 1e-57
	{0xEED6E2F0F0D56712, 0xFB158592BE068D2E}, // 1e-56
	{0x55464DD69685606B, 0x9CED737BB6C4183D}, // 1e-55
	{0xAA97E14C3C26B886, 0xC428D05AA4751E4C}, // 1e-54
	{0xD53DD99F4B3066A8, 0xF53304714D9265DF}, // 1e-53
	{0xE546A8038EFE4029, 0x993FE2C6D07B7FAB}, // 1e-52
	{0xDE98520472BDD033, 0xBF8FDB78849A5F96}, // 1e-51
	{0x963E66858F6D4440, 0xEF73D256A5C0F77C}, // 1e-50
	{0xDDE7001379A44AA8, 0x95A8637627989AAD}, // 1e-49
	{0x5560C018580D5D52, 0xBB127C53B17EC159}, // 1e-48
	{0xAAB8F01E6E10B4A6, 0xE9D71B689DDE71AF}, // 1e-47
	{0xCAB3961304CA70E8, 0x9226712162AB070D}, // 1e-46
	{0x3D607B97C5FD0D22, 0xB6B00D69BB55C8D1}, // 1e-45
	{0x8CB89A7DB77C506A, 0xE45C10C42A2B3B05}, // 1e-44
	{0x77F3608E92ADB242, 0x8EB98A7A9A5B04E3}, // 1e-43
	{0x55F038B237591ED3, 0xB267ED1940F1C61C}, // 1e-42
	{0x6B6C46DEC52F6688, 0xDF01E85F912E37A3}, // 1e-41
	{0x2323AC4B3B3DA015, 0x8B61313BBABCE2C6}, // 1e-40
	{0xABEC975E0A0D081A, 0xAE397D8AA96C1B77}, // 1e-39
	{0x96E7BD358C904A21, 0xD9C7DCED53C72255}, // 1e-38
	{0x7E50D64177DA2E54, 0x881CEA14545C7575}, // 1e-37
	{0xDDE50BD1D5D0B9E9, 0xAA242499697392D2}, // 1e-36
	{0x955E4EC64B44E864, 0xD4AD2DBFC3D07787}, // 1e-35
	{0xBD5AF13BEF0B113E, 0x84EC3C97DA624AB4}, // 1e-34
	{0xECB1AD8AEACDD58E, 0xA6274BBDD0FADD61}, // 1e-33
	{0x67DE18EDA5814AF2, 0xCFB11EAD453994BA}, // 1e-32
	{0x80EACF948770CED7, 0x81CEB32C4B43FCF4}, // 1e-31
	{0xA1258379A94D028D, 0xA2425FF75E14FC31}, // 1e-30
	{0x096EE45813A04330, 0xCAD2F7F5359A3B3E}, // 1e-29
	{0x8BCA9D6E188853FC, 0xFD87B5F28300CA0D}, // 1e-28
	{0x775EA264CF55347D, 0x9E74D1B791E07E48}, // 1e-27
	{0x95364AFE032A819D, 0xC612062576589DDA}, // 1e-26
	{0x3A83DDBD83F52204, 0xF79687AED3EEC551}, // 1e-25
	{0xC4926A9672793542, 0x9ABE14CD44753B52}, // 1e-24
	{0x75B7053C0F178293, 0xC16D9A0095928A27}, // 1e-23
	{0x5324C68B12DD6338, 0xF1C90080BAF72CB1}, // 1e-22
	{0xD3F6FC16EBCA5E03, 0x971DA05074DA7BEE}, // 1e-21
	{0x88F4BB1CA6BCF584, 0xBCE5086492111AEA}, // 1e-20
	{0x2B31E9E3D06C32E5, 0xEC1E4A7DB69561A5}, // 1e-19
	{0x3AFF322E62439FCF, 0x9392EE8E921D5D07}, // 1e-18
	{0x09BEFEB9FAD487C2, 0xB877AA3236A4B449}, // 1e-17
	{0x4C2EBE687989A9B3, 0xE69594BEC44DE15B}, // 1e-16
	{0x0F9D37014BF60A10, 0x901D7CF73AB0ACD9}, // 1e-15
	{0x538484C19EF38C94, 0xB424DC35095CD80F}, // 1e-14
	{0x2865A5F206B06FB9, 0xE12E13424BB40E13}, // 1e-13
	{0xF93F87B7442E45D3, 0x8CBCCC096F5088CB}, // 1e-12
	{0xF78F69A51539D748, 0xAFEBFF0BCB24AAFE}, // 1e-11
	{0xB573440E5A884D1B, 0xDBE6FECEBDEDD5BE}, // 1e-10
	{0x31680A88F8953030, 0x89705F4136B4A597}, // 1e-9
	{0xFDC20D2B36BA7C3D, 0xABCC77118461CEFC}, // 1e-8
	{0x3D32907604691B4C, 0xD6BF94D5E57A42BC}, // 1e-7
	{0xA63F9A49C2C1B10F, 0x8637BD05AF6C69B5}, // 1e-6
	{0x0FCF80DC33721D53, 0xA7C5AC471B478423}, // 1e-5
	{0xD3C36113404EA4A8, 0xD1B71758E219652B}, // 1e-4
	{0x645A1CAC083126E9, 0x83126E978D4FDF3B}, // 1e-3
	{0x3D70A3D70A3D70A3, 0xA3D70A3D70A3D70A}, // 1e-2
	{0xCCCCCCCCCCCCCCCC, 0xCCCCCCCCCCCCCCCC}, // 1e-1
	{0x0000000000000000, 0x8000000000000000}, // 1e0
	{0x0000000000000000, 0xA000000000000000}, // 1e1
	{0x0000000000000000, 0xC800000000000000}, // 1e2
	{0x0000000000000000, 0xFA00000000000000}, // 1e3
	{0x0000000000000000, 0x9C40000000000000}, // 1e4
	{0x0000000000000000, 0xC350000000000000}, // 1e5
	{0x0000000000000000, 0xF424000000000000}, // 1e6
	{0x0000000000000000, 0x9896800000000000}, // 1e7
	{0x0000000000000000, 0xBEBC200000000000}, // 1e8
	{0x0000000000000000, 0xEE6B280000000000}, // 1e9
	{0x0000000000000000, 0x9502F90000000000}, // 1e10
	{0x0000000000000000, 0xBA43B74000000000}, // 1e11
	{0x0000000000000000, 0xE8D4A51000000000}, // 1e12
	{0x0000000000000000, 0x9184E72A00000000}, // 1e13
	{0x0000000000000000, 0xB5E620F480000000}, // 1e14
	{0x0000000000000000, 0xE35FA931A0000000}, // 1e15
	{0x0000000000000000, 0x8E1BC9BF04000000}, // 1e16
	{0x0000000000000000, 0xB1A2BC2EC5000000}, // 1e17
	{0x0000000000000000, 0xDE0B6B3A76400000}, // 1e18
	{0x0000000000000000, 0x8AC7230489E80000}, // 1e19
	{0x0000000000000000, 0xAD78EBC5AC620000}, // 1e20
	{0x0000000000000000, 0xD8D726B7177A8000}, // 1e21
	{0x0000000000000000, 0x878678326EAC9000}, // 1e22
	{0x0000000000000000, 0xA968163F0A57B400}, // 1e23
	{0x0000000000000000, 0xD3C21BCECCEDA100}, // 1e24
	{0x0000000000000000, 0x84595161401484A0}, // 1e25
	{0x0000000000000000, 0xA56FA5B99019A5C8}, // 1e26
	{0x0000000000000000, 0xCECB8F27F4200F3A}, // 1e27
	{0x4000000000000000, 0x813F3978F8940984}, // 1e28
	{0x5000000000000000, 0xA18F07D736B90BE5}, // 1e29
	{0xA400000000000000, 0xC9F2C9CD04674EDE}, // 1e30
	{0x4D00000000000000, 0xFC6F7C4045812296}, // 1e31
	{0xF020000000000000, 0x9DC5ADA82B70B59D}, // 1e32
	{0x6C28000000000000, 0xC5371912364CE305}, // 1e33
	{0xC732000000000000, 0xF684DF56C3E01BC6}, // 1e34
	{0x3C7F400000000000, 0x9A130B963A6C115C}, // 1e35
	{0x4B9F100000000000, 0xC097CE7BC90715B3}, // 1e36
	{0x1E86D40000000000, 0xF0BDC21ABB48DB20}, // 1e37
	{0x1314448000000000, 0x96769950B50D88F4}, // 1e38
	{0x17D955A000000000, 0xBC143FA4E250EB31}, // 1e39
	{0x5DCFAB0800000000, 0xEB194F8E1AE525FD}, // 1e40
	{0x5AA1CAE500000000, 0x92EFD1B8D0CF37BE}, // 1e41
	{0xF14A3D9E40000000, 0xB7ABC627050305AD}, // 1e42
	{0x6D9CCD05D0000000, 0xE596B7B0C643C719}, // 1e43
	{0xE4820023A2000000, 0x8F7E32CE7BEA5C6F}, // 1e44
	{0xDDA2802C8A800000, 0xB35DBF821AE4F38B}, // 1e45
	{0xD50B2037AD200000, 0xE0352F62A19E306E}, // 1e46
	{0x4526F422CC340000, 0x8C213D9DA502DE45}, // 1e47
	{0x9670B12B7F410000, 0xAF298D050E4395D6}, // 1e48
	{0x3C0CDD765F114000, 0xDAF3F04651D47B4C}, // 1e49
	{0xA5880A69FB6AC800, 0x88D8762BF324CD0F}, // 1e50
	{0x8EEA0D047A457A00, 0xAB0E93B6EFEE0053}, // 1e51
	{0x72A4904598D6D880, 0xD5D238A4ABE98068}, // 1e52
	{0x47A6DA2B7F864750, 0x85A36366EB71F041}, // 1e53
	{0x999090B65F67D924, 0xA70C3C40A64E6C51}, // 1e54
	{0xFFF4B4E3F741CF6D, 0xD0CF4B50CFE20765}, // 1e55
	{0xBFF8F10E7A8921A4, 0x82818F1281ED449F}, // 1e56
	{0xAFF72D52192B6A0D, 0xA321F2D7226895C7}, // 1e57
	{0x9BF4F8A69F764490, 0xCBEA6F8CEB02BB39}, // 1e58
	{0x02F236D04753D5B4, 0xFEE50B7025C36A08}, // 1e59
	{0x01D762422C946590, 0x9F4F2726179A2245}, // 1e60
	{0x424D3AD2B7B97EF5, 0xC722F0EF9D80AAD6}, // 1e61
	{0xD2E0898765A7DEB2, 0xF8EBAD2B84E0D58B}, // 1e62
	{0x63CC55F49F88EB2F, 0x9B934C3B330C8577}, // 1e63
	{0x3CBF6B71C76B25FB, 0xC2781F49FFCFA6D5}, // 1e64
}

// shortDecimal writes s, a decimal literal -- a sign, digits with a point
// among or around them, and an exponent, as strconv.ParseFloat reads it --
// with its first maxDecimalDigits significant digits, a 1 after them when a
// digit it leaves out is not a zero, and the exponent that keeps the point
// where it was: a string that strconv rounds to the same double. ok is
// false for anything else.
func shortDecimal(s string) (short string, ok bool) {
	i := 0
	neg := false
	if i < len(s) && (s[i] == '+' || s[i] == '-') {
		neg = s[i] == '-'
		i++
	}
	intStart := i
	i = digitsEnd(s, i)
	intEnd, fracStart, fracEnd := i, i, i
	if i < len(s) && s[i] == '.' {
		fracStart = i + 1
		fracEnd = digitsEnd(s, fracStart)
		i = fracEnd
	}
	if intEnd == intStart && fracEnd == fracStart {
		return "", false
	}
	// The exponent, held to a bound far past any double's, beyond which
	// the value is infinite or zero however the digits move it; an int64,
	// which an int is not on 386.
	const expBound = 1 << 30
	var exp int64
	if i < len(s) && (s[i] == 'e' || s[i] == 'E') {
		i++
		expNeg := false
		if i < len(s) && (s[i] == '+' || s[i] == '-') {
			expNeg = s[i] == '-'
			i++
		}
		end := digitsEnd(s, i)
		if end == i {
			return "", false
		}
		for ; i < end && exp < expBound; i++ {
			exp = exp*10 + int64(s[i]-'0')
		}
		i = end
		if expNeg {
			exp = -exp
		}
	}
	if i != len(s) {
		return "", false
	}
	// The significant digits are the integer part's from its first that is
	// not a zero, then the fraction's, or, with no such digit in the
	// integer part, the fraction's from its first; point is how many of
	// them come before the point, which is negative for the zeros after it.
	var first, second string
	var point int64
	if lead := zerosEnd(s, intStart, intEnd); lead < intEnd {
		first, second = s[lead:intEnd], s[fracStart:fracEnd]
		point = int64(intEnd - lead)
	} else {
		lead := zerosEnd(s, fracStart, fracEnd)
		if lead == fracEnd {
			// Nothing but zeros: zero, of the sign written.
			if neg {
				return "-0", true
			}
			return "0", true
		}
		first = s[lead:fracEnd]
		point = -int64(lead - fracStart)
	}
	var b strings.Builder
	b.Grow(maxDecimalDigits + 32)
	if neg {
		b.WriteByte('-')
	}
	b.WriteString("0.")
	sticky := false
	if len(first) >= maxDecimalDigits {
		b.WriteString(first[:maxDecimalDigits])
		sticky = !allZeros(first[maxDecimalDigits:]) || !allZeros(second)
	} else {
		b.WriteString(first)
		rest := maxDecimalDigits - len(first)
		if len(second) > rest {
			b.WriteString(second[:rest])
			sticky = !allZeros(second[rest:])
		} else {
			b.WriteString(second)
		}
	}
	if sticky {
		b.WriteByte('1')
	}
	b.WriteByte('e')
	b.WriteString(strconv.FormatInt(point+exp, 10))
	return b.String(), true
}

// digitsEnd is the end of the run of decimal digits in s that starts at
// i, found eight bytes at a time.
func digitsEnd(s string, i int) int {
	for ; i+8 <= len(s); i += 8 {
		x := load64(s, i)
		// A byte below '0' sets its high bit in x-0x30 (and may borrow from
		// the bytes above it), and a byte above '9' sets it in x+0x46 or, past
		// 0xb9, in x-0x30: a word of digits sets none, nor carries.
		if ((x-0x3030303030303030)|(x+0x4646464646464646))&0x8080808080808080 != 0 {
			break
		}
	}
	for i < len(s) && s[i]-'0' < 10 {
		i++
	}
	return i
}

// zerosEnd is where the run of '0' in s[i:end] that starts at i ends.
func zerosEnd(s string, i, end int) int {
	for ; i+8 <= end && load64(s, i) == 0x3030303030303030; i += 8 {
	}
	for i < end && s[i] == '0' {
		i++
	}
	return i
}

// allZeros reports whether s, of digits, is nothing but '0'.
func allZeros(s string) bool { return zerosEnd(s, 0, len(s)) == len(s) }

// load64 is the eight bytes of s at i, the first the lowest.
func load64(s string, i int) uint64 {
	_ = s[i+7]
	return uint64(s[i]) | uint64(s[i+1])<<8 | uint64(s[i+2])<<16 | uint64(s[i+3])<<24 |
		uint64(s[i+4])<<32 | uint64(s[i+5])<<40 | uint64(s[i+6])<<48 | uint64(s[i+7])<<56
}
