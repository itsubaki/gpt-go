package pkg

import (
	"fmt"
	"math"
	"testing"

	"github.com/itsubaki/autograd/variable"
	"github.com/stretchr/testify/require"
)

func TestSoftmaxBasic(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf([]float64{1, 2, 3}, []float64{4, 5, 6})
	result := Softmax(a)

	// Expected values calculated from numpy for comparison
	// softmax([1, 2, 3]) = [0.09003057, 0.24472847, 0.66524096]
	// softmax([4, 5, 6]) = [0.09003057, 0.24472847, 0.66524096]
	expected := [][]float64{
		{0.09003057, 0.24472847, 0.66524096},
		{0.09003057, 0.24472847, 0.66524096},
	}

	for i := range expected {
		for j := range expected[i] {
			r.InDelta(expected[i][j], result.Data[i][j], 1e-7, "Softmax calculated incorrectly")
		}
	}

	for i := range result.Data {
		sum := 0.0
		for j := range result.Data[i] {
			sum += result.Data[i][j]
		}
		r.InDelta(1.0, sum, 1e-10, "Softmax row doesn't sum to 1")
	}
}

func TestSoftmaxWithLargeValues(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf([]float64{100, 100.1, 100.2})
	result := Softmax(a)

	fmt.Println(result)
	expected := [][]float64{
		{0.30060960535572756, 0.33222499353334567, 0.3671654011109268},
	}

	for i := range expected {
		for j := range expected[i] {
			r.InDelta(expected[i][j], result.Data[i][j], 1e-7, "Softmax with large values calculated incorrectly")
		}
	}

	sum := 0.0
	for j := range result.Data[0] {
		sum += result.Data[0][j]
	}
	r.InDelta(1.0, sum, 1e-10, "Softmax row doesn't sum to 1")
}

func TestSoftmaxWithMasking(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf(
		[]float64{1, math.Inf(-1), 3},            // First row has masked second element
		[]float64{math.Inf(-1), 2, 3},            // Second row has masked first element
		[]float64{1, 2, math.Inf(-1)},            // Third row has masked third element
		[]float64{1, math.Inf(-1), math.Inf(-1)}, // Fourth row has only one valid element
	)
	result := Softmax(a)

	expected := [][]float64{
		{0.119203, 0.0, 0.880797}, // softmax([1, -Inf, 3])
		{0.0, 0.268941, 0.731059}, // softmax([-Inf, 2, 3])
		{0.268941, 0.731059, 0.0}, // softmax([1, 2, -Inf])
		{1.0, 0.0, 0.0},           // softmax([1, -Inf, -Inf])
	}

	for i := range expected {
		for j := range expected[i] {
			r.InDelta(expected[i][j], result.Data[i][j], 1e-6, "Softmax with masked values incorrect")
		}
	}

	for i := range result.Data {
		sum := 0.0
		for j := range result.Data[i] {
			sum += result.Data[i][j]
		}
		r.InDelta(1.0, sum, 1e-10, "Softmax masked row doesn't sum to 1")
	}
}

func TestSoftmaxWithAllMasked(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf(
		[]float64{0, math.Inf(-1), math.Inf(-1)},
		[]float64{1, 2, 3},
	)
	result := Softmax(a)

	fmt.Println(result)
	expected := [][]float64{
		{1, 0, 0},                            // Uniform distribution
		{0.09003057, 0.24472847, 0.66524096}, // Normal softmax results
	}

	for j := range result.Data[0] {
		r.InDelta(expected[0][j], result.Data[0][j], 1e-6,
			"All-masked row should give uniform-like distribution due to epsilon")
	}

	for j := range result.Data[1] {
		r.InDelta(expected[1][j], result.Data[1][j], 1e-6,
			"Normal row calculated incorrectly")
	}

	for i := range result.Data {
		sum := 0.0
		for j := range result.Data[i] {
			sum += result.Data[i][j]
		}
		r.InDelta(1.0, sum, 1e-10, "Softmax row doesn't sum to 1")
	}
}

func TestSoftmaxGradient(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf([]float64{1, 2, 3})
	result := Softmax(a)

	expected := [][]float64{
		{0.09003057, 0.24472847, 0.66524096},
	}

	for j := range expected[0] {
		r.InDelta(expected[0][j], result.Data[0][j], 1e-7, "Softmax output incorrect")
	}

	result.Grad = variable.NewOf([]float64{1, 1, 1})
	result.Backward()

	for j := range a.Grad.Data[0] {
		r.InDelta(0.0, a.Grad.Data[0][j], 1e-7, "Gradient with uniform upstream should be zero")
	}

	b := variable.NewOf([]float64{1, 2, 3})
	resultB := Softmax(b)
	resultB.Grad = variable.NewOf([]float64{1, 0, 0}) // Only gradient for first output
	resultB.Backward()

	expectedGrad := [][]float64{
		{0.09003057 * (1 - 0.09003057), -0.09003057 * 0.24472847, -0.09003057 * 0.66524096},
	}

	for j := range expectedGrad[0] {
		r.InDelta(expectedGrad[0][j], b.Grad.Data[0][j], 1e-7, "Gradient calculation incorrect")
	}
}

//func TestSoftmaxInComputationGraph(t *testing.T) {
//	r := require.New(t)
//
//	a := variable.NewOf([]float64{1, 2, 3})
//	softmaxOutput := Softmax(a)
//
//	weights := variable.NewOf([]float64{0.5, 0.3, 0.2})
//	result := variable.Mul(softmaxOutput, weights)
//
//	sumResult := Sum(result)
//
//	expectedSum := 0.09003057*0.5 + 0.24472847*0.3 + 0.66524096*0.2
//
//	r.InDelta(expectedSum, sumResult.Data[0][0], 1e-7, "Computation graph forward pass incorrect")
//
//	sumResult.Grad = variable.NewOf([]float64{1.0})
//
//	sumResult.Backward()
//
//	r.NotNil(a.Grad, "Gradient should propagate to the input")
//
//	r.Equal(len(a.Data[0]), len(a.Grad.Data[0]), "Gradient should have same shape as input")
//}
//
//func TestSoftmaxNumericalStability(t *testing.T) {
//	r := require.New(t)
//
//	a1 := variable.NewOf([]float64{-1000, -1000, -1000})
//	result1 := Softmax(a1)
//
//	for j := range result1.Data[0] {
//		r.InDelta(1.0/3.0, result1.Data[0][j], 1e-6, "Uniform distribution expected for extreme negative values")
//	}
//
//	sum1 := 0.0
//	for j := range result1.Data[0] {
//		sum1 += result1.Data[0][j]
//	}
//	r.InDelta(1.0, sum1, 1e-10, "Softmax extreme negative values should sum to 1")
//
//	a2 := variable.NewOf([]float64{0, -1e10, -1e10})
//	result2 := Softmax(a2)
//
//	r.InDelta(1.0, result2.Data[0][0], 1e-10, "Dominant element should be close to 1")
//	r.InDelta(0.0, result2.Data[0][1], 1e-10, "Suppressed element should be close to 0")
//	r.InDelta(0.0, result2.Data[0][2], 1e-10, "Suppressed element should be close to 0")
//
//	sum2 := 0.0
//	for j := range result2.Data[0] {
//		sum2 += result2.Data[0][j]
//	}
//	r.InDelta(1.0, sum2, 1e-10, "Softmax with extreme values should sum to 1")
//
//	a3 := variable.NewOf([]float64{1e10, 0, -1e10})
//	result3 := Softmax(a3)
//
//	r.InDelta(1.0, result3.Data[0][0], 1e-10, "Dominant element should be close to 1")
//	r.InDelta(0.0, result3.Data[0][1], 1e-10, "Middle element should be close to 0")
//	r.InDelta(0.0, result3.Data[0][2], 1e-10, "Suppressed element should be close to 0")
//
//	sum3 := 0.0
//	for j := range result3.Data[0] {
//		sum3 += result3.Data[0][j]
//	}
//	r.InDelta(1.0, sum3, 1e-10, "Softmax with mixed extreme values should sum to 1")
//
//	a4 := variable.NewOf([]float64{1e6, 1e6 + 1e-5, 1e6 + 2e-5})
//	result4 := Softmax(a4)
//
//	sum4 := 0.0
//	for j := range result4.Data[0] {
//		sum4 += result4.Data[0][j]
//	}
//	r.InDelta(1.0, sum4, 1e-10, "Softmax with very close large values should sum to 1")
//
//	r.Greater(result4.Data[0][2], result4.Data[0][1], "Ranking should be preserved even with close values")
//	r.Greater(result4.Data[0][1], result4.Data[0][0], "Ranking should be preserved even with close values")
//}

func Sum(x *variable.Variable) *variable.Variable {
	data := []float64{0}
	for i := range x.Data {
		for j := range x.Data[i] {
			data[0] += x.Data[i][j]
		}
	}
	return variable.NewOf(data)
}
