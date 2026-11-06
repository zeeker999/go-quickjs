package jsnum

import (
	"fmt"
	"math"
	"math/big"
	"math/rand/v2"
	"strconv"
	"testing"
)

// checkShort compares ParseShort on s with strconv, which is exact, wherever
// it answers: it may always decline.
func checkShort(t *testing.T, s string) bool {
	t.Helper()
	f, end, ok := ParseShort(s, 0)
	if !ok {
		return false
	}
	if end != len(s) {
		t.Fatalf("ParseShort(%q) ended at %d", s, end)
	}
	want, err := strconv.ParseFloat(s, 64)
	if err != nil {
		t.Fatalf("ParseShort(%q) = %v, strconv: %v", s, f, err)
	}
	if math.Float64bits(f) != math.Float64bits(want) {
		t.Fatalf("ParseShort(%q) = %v (%#x), want %v (%#x)", s, f, math.Float64bits(f), want, math.Float64bits(want))
	}
	return true
}

func TestParseShortAgainstStrconv(t *testing.T) {
	for _, s := range []string{
		"0", "-0", "0.0", "-0.0", "0e0", "-0e5", "0.000", "1", "-1", "12", "1.5", "-1.5",
		"0.1", "0.2", "0.3", "1e23", "1e22", "1e-22", "8.41e21", "2.0e-308",
		"9007199254740991", "9007199254740992", "9007199254740993", "9007199254740994",
		"18446744073709551615", "1844674407370955161", "9999999999999999999",
		"1.7976931348623157e308", "4.9e-324", "2.2250738585072014e-308",
		"123456789012345678", "0.000001", "1E5", "1e+5", "1e-5", "1.0E+2",
		"3.141592653589793", "2.718281828459045", "0.30000000000000004",
	} {
		checkShort(t, s)
	}
	r := rand.New(rand.NewPCG(1, 2))
	answered, n := 0, 2000000
	if testing.Short() {
		n = 100000
	}
	for range n {
		m := r.Uint64() >> uint(r.IntN(64))
		e := r.IntN(160) - 80
		var s string
		switch r.IntN(4) {
		case 0:
			s = fmt.Sprintf("%de%d", m, e)
		case 1:
			d := strconv.FormatUint(m, 10)
			at := r.IntN(len(d)) + 1
			s = d[:at] + "." + d[at:]
			if at == len(d) {
				s += "0"
			}
			if len(d) > 1 && d[0] == '0' {
				continue
			}
		case 2:
			s = strconv.FormatFloat(r.Float64()*math.Pow(10, float64(e)), 'g', -1, 64)
		case 3:
			s = strconv.FormatFloat(math.Float64frombits(r.Uint64()), 'e', r.IntN(18), 64)
		}
		if r.IntN(2) == 0 {
			s = "-" + s
		}
		if checkShort(t, s) {
			answered++
		}
	}
	if answered < n/3 {
		t.Errorf("answered %d of %d: it should answer most numbers", answered, n)
	}
}

// Decimals on either side of the midpoint of two neighbouring doubles, which
// is where a rounding that is only nearly right goes wrong.
func TestParseShortAtMidpoints(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	n := 300000
	if testing.Short() {
		n = 20000
	}
	for range n {
		exp := r.IntN(300) - 150
		f := math.Ldexp(float64(1<<52|r.Uint64()&(1<<52-1)), exp)
		next := math.Nextafter(f, math.Inf(1))
		mid := new(big.Float).SetPrec(2000).SetFloat64(f)
		mid.Add(mid, new(big.Float).SetPrec(2000).SetFloat64(next))
		mid.Quo(mid, big.NewFloat(2))
		for _, digits := range []int{17, 18, 19} {
			s := mid.Text('e', digits-1)
			checkShort(t, s)
		}
	}
}

func TestParseShortDeclines(t *testing.T) {
	for _, s := range []string{
		"", "-", "01", "-01", "1.", ".5", "1e", "1e+", "1.e5", "--1",
		"123456789012345678901", "1.2345678901234567890", "1e400", "1e-400", "1e99999",
	} {
		if f, end, ok := ParseShort(s, 0); ok {
			t.Errorf("ParseShort(%q) = %v, %d", s, f, end)
		}
	}
	// A number is read up to what cannot be in it.
	if f, end, ok := ParseShort("[12.5,3]", 1); !ok || f != 12.5 || end != 5 {
		t.Errorf("got %v, %d, %v", f, end, ok)
	}
}

// A number of few places is written from its digits: what strconv writes is
// what a shorter way must come to.
func TestAppendFloatPlaces(t *testing.T) {
	check := func(v float64) {
		t.Helper()
		got := string(AppendFloat(nil, v))
		want := strconv.FormatFloat(v, 'f', -1, 64)
		if a := math.Abs(v); a < 1e-6 || a >= 1e21 {
			want = FormatFloat(v)
		}
		if got != want {
			t.Fatalf("AppendFloat(%v) = %q, want %q", v, got, want)
		}
	}
	for _, v := range []float64{
		0.5, -0.5, 0.25, 1.5, 20.5, 0.1, 0.2, 0.3, 0.1 + 0.2, 1.1, 2.2, 3.3, 0.000001, 0.000002,
		999999999.5, 999999999.999999, 1000000000.5, 123456.789, 0.05, 0.005, 0.0005, 100.01,
		1.000001, 1.0000001, 4.35, 2.675, 1.005, 1.255, 1e-7, 8.5e-6,
	} {
		check(v)
	}
	r := rand.New(rand.NewPCG(5, 6))
	n := 3000000
	if testing.Short() {
		n = 100000
	}
	for range n {
		// Decimals of a few places, and their neighbours, which are not.
		v := float64(r.IntN(2000000000)) / math.Pow10(r.IntN(9))
		check(v)
		check(math.Nextafter(v, 0))
		check(math.Nextafter(v, math.Inf(1)))
		check(r.Float64() * 1000)
	}
}
