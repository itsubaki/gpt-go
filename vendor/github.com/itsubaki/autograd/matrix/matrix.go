package matrix

import (
	"math"
	"time"
	"sort"
	randv2 "math/rand/v2"
	"strings"
	"fmt"
	"sync"

	"github.com/itsubaki/autograd/rand"
)

type Matrix [][]float64


// ProfileStats stores the profiling statistics
type ProfileStats struct {
	TotalTime  time.Duration
	CallCount  int
	MaxTime    time.Duration
	MinTime    time.Duration
}

// ProfileData is a global map that stores profiling data for all functions
var ProfileData = struct {
	sync.RWMutex
	Stats map[string]*ProfileStats
}{
	Stats: make(map[string]*ProfileStats),
}

// RecordFunctionCall profiles a function call and records statistics
func RecordFunctionCall(funcName string, startTime time.Time) {
	elapsed := time.Since(startTime)

	ProfileData.Lock()
	defer ProfileData.Unlock()

	stats, exists := ProfileData.Stats[funcName]
	if !exists {
		ProfileData.Stats[funcName] = &ProfileStats{
			TotalTime: elapsed,
			CallCount: 1,
			MaxTime:   elapsed,
			MinTime:   elapsed,
		}
		return
	}

	stats.TotalTime += elapsed
	stats.CallCount++

	if elapsed > stats.MaxTime {
		stats.MaxTime = elapsed
	}

	if elapsed < stats.MinTime {
		stats.MinTime = elapsed
	}
}

// PrintProfileStats prints the recorded profiling statistics
func PrintProfileStats() {
	ProfileData.RLock()
	defer ProfileData.RUnlock()

	fmt.Println("\n--- Matrix Package Profiling Statistics ---")
	fmt.Printf("%-15s %-15s %-15s %-15s %-15s %-15s\n",
		"Function", "Total Time", "Calls", "Avg Time", "Min Time", "Max Time")
	fmt.Println(strings.Repeat("-", 90))

	for funcName, stats := range ProfileData.Stats {
		avgTime := stats.TotalTime / time.Duration(stats.CallCount)
		fmt.Printf("%-15s %-15s %-15d %-15s %-15s %-15s\n",
			funcName,
			stats.TotalTime.String(),
			stats.CallCount,
			avgTime.String(),
			stats.MinTime.String(),
			stats.MaxTime.String())
	}
	fmt.Println(strings.Repeat("-", 90))
}

// GetSortedProfileStats returns the profiling statistics sorted by the specified field
func GetSortedProfileStats(sortBy string) []struct {
	FuncName string
	Stats    ProfileStats
} {
	ProfileData.RLock()
	defer ProfileData.RUnlock()

	result := make([]struct {
		FuncName string
		Stats    ProfileStats
	}, 0, len(ProfileData.Stats))

	for funcName, stats := range ProfileData.Stats {
		result = append(result, struct {
			FuncName string
			Stats    ProfileStats
		}{
			FuncName: funcName,
			Stats:    *stats,
		})
	}

	switch sortBy {
	case "totalTime":
		sort.Slice(result, func(i, j int) bool {
			return result[i].Stats.TotalTime > result[j].Stats.TotalTime
		})
	case "callCount":
		sort.Slice(result, func(i, j int) bool {
			return result[i].Stats.CallCount > result[j].Stats.CallCount
		})
	case "avgTime":
		sort.Slice(result, func(i, j int) bool {
			return result[i].Stats.TotalTime/time.Duration(result[i].Stats.CallCount) >
				result[j].Stats.TotalTime/time.Duration(result[j].Stats.CallCount)
		})
	case "maxTime":
		sort.Slice(result, func(i, j int) bool {
			return result[i].Stats.MaxTime > result[j].Stats.MaxTime
		})
	default:
		// Default sort by total time
		sort.Slice(result, func(i, j int) bool {
			return result[i].Stats.TotalTime > result[j].Stats.TotalTime
		})
	}

	return result
}

// PrintSortedProfileStats prints the profiling statistics sorted by the specified field
func PrintSortedProfileStats(sortBy string) {
	sortedStats := GetSortedProfileStats(sortBy)

	fmt.Println("\n--- Matrix Package Profiling Statistics (Sorted by " + sortBy + ") ---")
	fmt.Printf("%-15s %-15s %-15s %-15s %-15s %-15s\n",
		"Function", "Total Time", "Calls", "Avg Time", "Min Time", "Max Time")
	fmt.Println(strings.Repeat("-", 90))

	for _, item := range sortedStats {
		avgTime := item.Stats.TotalTime / time.Duration(item.Stats.CallCount)
		fmt.Printf("%-15s %-15s %-15d %-15s %-15s %-15s\n",
			item.FuncName,
			item.Stats.TotalTime.String(),
			item.Stats.CallCount,
			avgTime.String(),
			item.Stats.MinTime.String(),
			item.Stats.MaxTime.String())
	}
	fmt.Println(strings.Repeat("-", 90))
}

// Now wrap all of the original matrix functions with profiling

func New(v ...[]float64) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("New", startTime)

	out := make(Matrix, len(v))
	copy(out, v)
	return out
}

func Zero(m, n int) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("Zero", startTime)

	out := make(Matrix, m)
	for i := range m {
		out[i] = make([]float64, n)
	}

	return out
}

func ZeroLike(m Matrix) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("ZeroLike", startTime)

	return Zero(Dim(m))
}

func OneLike(m Matrix) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("OneLike", startTime)

	return AddC(1.0, ZeroLike(m))
}

func From(x [][]int) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("From", startTime)

	out := Zero(len(x), len(x[0]))
	for i := range x {
		for j := range x[i] {
			out[i][j] = float64(x[i][j])
		}
	}

	return out
}

// rnd returns a pseudo-random number generator.
func rnd(s ...randv2.Source) *randv2.Rand {
	if len(s) == 0 || s[0] == nil {
		return randv2.New(rand.NewSource(rand.MustRead()))
	}

	return randv2.New(s[0])
}

func Rand(m, n int, s ...randv2.Source) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("Rand", startTime)

	return F(Zero(m, n), func(_ float64) float64 { return rnd(s...).Float64() })
}

func Randn(m, n int, s ...randv2.Source) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("Randn", startTime)

	return F(Zero(m, n), func(_ float64) float64 { return rnd(s...).NormFloat64() })
}

func Size(m Matrix) int {
	startTime := time.Now()
	defer RecordFunctionCall("Size", startTime)

	s := 1
	for _, v := range Shape(m) {
		s = s * v
	}

	return s
}

func Shape(m Matrix) []int {
	startTime := time.Now()
	defer RecordFunctionCall("Shape", startTime)

	a, b := Dim(m)
	return []int{a, b}
}

func Dim(m Matrix) (int, int) {
	startTime := time.Now()
	defer RecordFunctionCall("Dim", startTime)

	return len(m), len(m[0])
}

func AddC(c float64, m Matrix) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("AddC", startTime)

	return F(m, func(v float64) float64 { return c + v })
}

func SubC(c float64, m Matrix) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("SubC", startTime)

	return F(m, func(v float64) float64 { return c - v })
}

func MulC(c float64, m Matrix) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("MulC", startTime)

	return F(m, func(v float64) float64 { return c * v })
}

func Exp(m Matrix) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("Exp", startTime)

	return F(m, func(v float64) float64 { return math.Exp(v) })
}

func Log(m Matrix) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("Log", startTime)

	return F(m, func(v float64) float64 { return math.Log(v) })
}

func Sin(m Matrix) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("Sin", startTime)

	return F(m, func(v float64) float64 { return math.Sin(v) })
}

func Cos(m Matrix) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("Cos", startTime)

	return F(m, func(v float64) float64 { return math.Cos(v) })
}

func Tanh(m Matrix) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("Tanh", startTime)

	return F(m, func(v float64) float64 { return math.Tanh(v) })
}

func Pow(c float64, m Matrix) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("Pow", startTime)

	return F(m, func(v float64) float64 { return math.Pow(v, c) })
}

func Add(m, n Matrix) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("Add", startTime)

	return F2(m, n, func(a, b float64) float64 { return a + b })
}

func Sub(m, n Matrix) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("Sub", startTime)

	return F2(m, n, func(a, b float64) float64 { return a - b })
}

func Mul(m, n Matrix) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("Mul", startTime)

	return F2(m, n, func(a, b float64) float64 { return a * b })
}

func Div(m, n Matrix) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("Div", startTime)

	return F2(m, n, func(a, b float64) float64 { return a / b })
}

func Mean(m Matrix) float64 {
	startTime := time.Now()
	defer RecordFunctionCall("Mean", startTime)

	return Sum(m) / float64(Size(m))
}

func Sum(m Matrix) float64 {
	startTime := time.Now()
	defer RecordFunctionCall("Sum", startTime)

	var sum float64
	for _, v := range Flatten(m) {
		sum = sum + v
	}

	return sum
}

func Max(m Matrix) float64 {
	startTime := time.Now()
	defer RecordFunctionCall("Max", startTime)

	max := m[0][0]
	for _, v := range Flatten(m) {
		if v > max {
			max = v
		}
	}

	return max
}

func Min(m Matrix) float64 {
	startTime := time.Now()
	defer RecordFunctionCall("Min", startTime)

	min := m[0][0]
	for _, v := range Flatten(m) {
		if v < min {
			min = v
		}
	}

	return min
}

func Argmax(m Matrix) []int {
	startTime := time.Now()
	defer RecordFunctionCall("Argmax", startTime)

	p, q := Dim(m)

	out := make([]int, p)
	for i := range p {
		max := m[i][0]
		for j := range q {
			if m[i][j] > max {
				max, out[i] = m[i][j], j
			}
		}
	}

	return out
}

func Dot(m, n Matrix) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("Dot", startTime)

	a, b := Dim(m)
	_, p := Dim(n)

	out := Zero(a, p)
	for i := range a {
		for j := range p {
			for k := 0; k < b; k++ {
				out[i][j] = out[i][j] + m[i][k]*n[k][j]
			}
		}
	}

	return out
}

func Clip(m Matrix, min, max float64) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("Clip", startTime)

	return F(m, func(v float64) float64 {
		if v < min {
			return min
		}

		if v > max {
			return max
		}

		return v
	})
}

func Mask(m Matrix, f func(x float64) bool) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("Mask", startTime)

	return F(m, func(v float64) float64 {
		if f(v) {
			return 1
		}

		return 0
	})
}

func Broadcast(m, n Matrix) (Matrix, Matrix) {
	startTime := time.Now()
	defer RecordFunctionCall("Broadcast", startTime)

	return BroadcastTo(Shape(n), m), BroadcastTo(Shape(m), n)
}

func BroadcastTo(shape []int, m Matrix) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("BroadcastTo", startTime)

	a, b := shape[0], shape[1]

	if len(m) == 1 && len(m[0]) == 1 {
		out := make([]float64, a*b)
		for i := range a * b {
			out[i] = m[0][0]
		}

		return Reshape(shape, New(out))
	}

	if len(m) == 1 {
		// b is ignored
		out := make(Matrix, a)
		for i := range a {
			out[i] = m[0]
		}

		return out
	}

	if len(m[0]) == 1 {
		// a is ignored
		out := Zero(len(m), b)
		for i := range m {
			for j := range b {
				out[i][j] = m[i][0]
			}
		}

		return out
	}

	return m
}

func SumTo(shape []int, m Matrix) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("SumTo", startTime)

	if shape[0] == 1 && shape[1] == 1 {
		return New([]float64{Sum(m)})
	}

	if shape[0] == 1 {
		return SumAxis0(m)
	}

	if shape[1] == 1 {
		return SumAxis1(m)
	}

	return m
}

func SumAxis0(m Matrix) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("SumAxis0", startTime)

	p, q := Dim(m)

	v := make([]float64, 0, q)
	for j := range q {
		var sum float64
		for i := range p {
			sum = sum + m[i][j]
		}

		v = append(v, sum)
	}

	return New(v)
}

func SumAxis1(m Matrix) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("SumAxis1", startTime)

	p, q := Dim(m)

	v := make([]float64, 0, p)
	for i := range p {
		var sum float64
		for j := range q {
			sum = sum + m[i][j]
		}

		v = append(v, sum)
	}

	return Transpose(New(v))
}

func MaxAxis1(m Matrix) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("MaxAxis1", startTime)

	p, q := Dim(m)

	v := make([]float64, 0, p)
	for i := range p {
		max := m[i][0]
		for j := range q {
			if m[i][j] > max {
				max = m[i][j]
			}
		}

		v = append(v, max)
	}

	return Transpose(New(v))
}

func Transpose(m Matrix) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("Transpose", startTime)

	p, q := Dim(m)

	out := Zero(q, p)
	for i := range q {
		for j := range p {
			out[i][j] = m[j][i]
		}
	}

	return out
}

func Reshape(shape []int, m Matrix) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("Reshape", startTime)

	p, q := Dim(m)
	a, b := shape[0], shape[1]

	v := Flatten(m)
	if a < 1 {
		a = p * q / b
	}

	if b < 1 {
		b = p * q / a
	}

	out := make(Matrix, a)
	for i := range a {
		out[i] = v[i*b : (i+1)*b]
	}

	return out
}

func Flatten(m Matrix) []float64 {
	startTime := time.Now()
	defer RecordFunctionCall("Flatten", startTime)

	out := make([]float64, 0, Size(m))
	for _, r := range m {
		out = append(out, r...)
	}

	return out
}

func F(m Matrix, f func(a float64) float64) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("F", startTime)

	p, q := Dim(m)

	out := Zero(p, q)
	for i := range p {
		for j := range q {
			out[i][j] = f(m[i][j])
		}
	}

	return out
}

func F2(m, n Matrix, f func(a, b float64) float64) Matrix {
	startTime := time.Now()
	defer RecordFunctionCall("F2", startTime)

	x, y := Broadcast(m, n)
	p, q := Dim(x)

	out := Zero(p, q)
	for i := range p {
		for j := range q {
			out[i][j] = f(x[i][j], y[i][j])
		}
	}

	return out
}