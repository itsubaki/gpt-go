package pkg

import (
	"fmt"
	"testing"

	"github.com/itsubaki/autograd/variable"
	"github.com/stretchr/testify/require"
)

func TestVarianceBasic(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf([]float64{1, 2, 3}, []float64{4, 5, 6})
	result := Variance(a)

	expected := [][]float64{
		{2.0 / 3.0}, // variance of [1,2,3] is ((1-2)² + (2-2)² + (3-2)²)/3 = (1 + 0 + 1)/3 = 2/3
		{2.0 / 3.0}, // variance of [4,5,6] is ((4-5)² + (5-5)² + (6-5)²)/3 = (1 + 0 + 1)/3 = 2/3
	}

	// Allow small floating point differences
	for i := range expected {
		for j := range expected[i] {
			r.InDelta(expected[i][j], result.Data[i][j], 1e-10, "Variance calculated incorrectly")
		}
	}
}

func TestVarianceOfConstants(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf([]float64{5, 5, 5}, []float64{-3, -3, -3})
	result := Variance(a)

	expected := [][]float64{
		{0}, // variance of constant values is 0
		{0},
	}

	r.Equal(expected, result.Data, "Variance of constant values should be 0")
}

func TestVarianceWithNegatives(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf([]float64{-1, 0, 1}, []float64{-10, 0, 10})
	result := Variance(a)

	expected := [][]float64{
		{2.0 / 3.0},   // variance of [-1,0,1] = 2/3
		{200.0 / 3.0}, // variance of [-10,0,10] = 100/3
	}

	// Allow small floating point differences
	for i := range expected {
		for j := range expected[i] {
			fmt.Println(result.Data[i][j])
			r.InDelta(expected[i][j], result.Data[i][j], 1e-10, "Variance calculated incorrectly")
		}
	}
}

func TestVarianceGradient(t *testing.T) {
	r := require.New(t)

	// Values [1, 3, 5] have a mean of 3
	a := variable.NewOf([]float64{1, 3, 5})
	result := Variance(a)

	// Expected variance: ((1-3)² + (3-3)² + (5-3)²)/3 = (4 + 0 + 4)/3 = 8/3
	r.InDelta(8.0/3.0, result.Data[0][0], 1e-10, "Variance calculated incorrectly")

	// Set gradient to 1.0
	result.Grad = variable.NewOf([]float64{1.0})
	result.Backward()

	// For variance, the gradient for each input element is:
	// (2/n) * (x_i - mean)
	// For this example:
	// x_1 = 1: (2/3) * (1 - 3) = (2/3) * (-2) = -4/3
	// x_2 = 3: (2/3) * (3 - 3) = (2/3) * (0) = 0
	// x_3 = 5: (2/3) * (5 - 3) = (2/3) * (2) = 4/3
	expectedGrad := [][]float64{
		{-4.0 / 3.0, 0, 4.0 / 3.0},
	}

	// Allow small floating point differences
	for i := range expectedGrad {
		for j := range expectedGrad[i] {
			r.InDelta(expectedGrad[i][j], a.Grad.Data[i][j], 1e-10, "Incorrect gradient computation")
		}
	}
}

func TestVarianceInComputationGraph(t *testing.T) {
	r := require.New(t)

	// Create input with a single row
	a := variable.NewOf([]float64{2, 4, 6})

	// Calculate variance
	v := Variance(a)

	// Multiply by scalar
	k := variable.NewOf([]float64{0.5})
	result := variable.Mul(v, k)

	// Variance of [2,4,6] is ((2-4)² + (4-4)² + (6-4)²)/3 = (4 + 0 + 4)/3 = 8/3
	// So result should be 8/3 * 0.5 = 4/3
	r.InDelta(4.0/3.0, result.Data[0][0], 1e-10, "Calculation in graph incorrect")

	// Backpropagate
	result.Backward()

	// Expected gradients for a:
	// For each input i, the gradient is:
	// 0.5 * (2/3) * (a_i - mean)
	// Mean of [2,4,6] is 4, so:
	// a_1 = 2: 0.5 * (2/3) * (2-4) = 0.5 * (2/3) * (-2) = -2/3
	// a_2 = 4: 0.5 * (2/3) * (4-4) = 0.5 * (2/3) * (0) = 0
	// a_3 = 6: 0.5 * (2/3) * (6-4) = 0.5 * (2/3) * (2) = 2/3
	expectedGrad := [][]float64{
		{-2.0 / 3.0, 0, 2.0 / 3.0},
	}

	// Allow small floating point differences
	for i := range expectedGrad {
		for j := range expectedGrad[i] {
			r.InDelta(expectedGrad[i][j], a.Grad.Data[i][j], 1e-10, "Incorrect gradient in computation graph")
		}
	}
}
