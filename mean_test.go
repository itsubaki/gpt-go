package main

import (
	"testing"

	"github.com/itsubaki/autograd/variable"
	"github.com/stretchr/testify/require"
)

func TestMeanBasic(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf([]float64{1, 2, 3}, []float64{4, 5, 6})
	result := Mean(a)

	expected := [][]float64{
		{2},
		{5},
	}

	r.Equal(expected, result.Data, "Mean failed to calculate correctly")
}

func TestMeanWithZero(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf([]float64{1, 0, 1})
	result := Mean(a)

	r.InDelta(2/3.0, result.Data[0][0], 1e-10)
}

func TestMeanWithZeros(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf([]float64{0, 0, 0}, []float64{0, 0, 0})
	result := Mean(a)

	expected := [][]float64{
		{0},
		{0},
	}

	r.Equal(expected, result.Data, "Mean of zeros should be zero")
}

func TestMeanWithNegatives(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf([]float64{-1, 2, -3}, []float64{4, -5, 6})
	result := Mean(a)

	expected := [][]float64{
		{-2.0 / 3.0},
		{5.0 / 3.0},
	}

	r.Equal(expected, result.Data, "Mean failed with negative values")
}

func TestMeanGradient(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf([]float64{1, 2, 3}, []float64{4, 5, 6})
	result := Mean(a)

	result.Grad = variable.NewOf([]float64{0.1}, []float64{0.2})
	result.Backward()

	r.NotNil(a.Grad, "Gradient for a should not be nil")

	// Each element in a row should receive gradient / n
	expectedGrad := [][]float64{
		{0.1 / 3.0, 0.1 / 3.0, 0.1 / 3.0},
		{0.2 / 3.0, 0.2 / 3.0, 0.2 / 3.0},
	}

	r.Equal(expectedGrad, a.Grad.Data, "Incorrect gradient for mean operation")
}

func TestMeanWithScalarGradient(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf([]float64{1, 2}, []float64{3, 4})
	result := Mean(a)

	// Set gradient to 1.0
	result.Grad = variable.NewOf([]float64{1.0}, []float64{1.0})
	result.Backward()

	// Each element should receive 1.0/2.0 = 0.5
	expectedGrad := [][]float64{
		{0.5, 0.5},
		{0.5, 0.5},
	}

	r.Equal(expectedGrad, a.Grad.Data, "Gradient should be distributed evenly")
}

func TestMeanInComputationGraph(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf([]float64{1, 3}, []float64{2, 4})
	meanA := Mean(a)

	// Multiply the means by 2
	result := variable.Mul(meanA, variable.NewOf([]float64{2}, []float64{2}))

	expected := [][]float64{
		{4}, // (1+3)/2 * 2 = 4
		{6}, // (2+4)/2 * 2 = 6
	}

	r.Equal(expected, result.Data, "Mean in computation graph gave incorrect result")

	// Backpropagate
	result.Backward()

	// Each input should receive (2 * 1/2) = 1.0 gradient
	expectedGrad := [][]float64{
		{1.0, 1.0},
		{1.0, 1.0},
	}

	r.Equal(expectedGrad, a.Grad.Data, "Incorrect gradient through computation graph")
}
