package main

import (
	"testing"

	"github.com/itsubaki/autograd/variable"
	"github.com/stretchr/testify/require"
)

func TestCatBasic(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf([]float64{1, 2}, []float64{3, 4})
	b := variable.NewOf([]float64{5, 6}, []float64{7, 8})

	result := Cat(a, b)

	expected := [][]float64{
		{1, 2, 5, 6},
		{3, 4, 7, 8},
	}

	r.Equal(expected, result.Data, "Cat failed to concatenate columns correctly")
}

func TestCatSingleInput(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf([]float64{1, 2}, []float64{3, 4})

	result := Cat(a)

	r.Equal(a.Data, result.Data, "Single input should return unchanged")
}

func TestCatThreeTensors(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf([]float64{1, 2}, []float64{3, 4})
	b := variable.NewOf([]float64{5, 6}, []float64{7, 8})
	c := variable.NewOf([]float64{9, 10}, []float64{11, 12})

	result := Cat(a, b, c)

	expected := [][]float64{
		{1, 2, 5, 6, 9, 10},
		{3, 4, 7, 8, 11, 12},
	}

	r.Equal(expected, result.Data, "Cat failed to concatenate three tensors")
}

func TestCatGradient(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf([]float64{1, 2}, []float64{3, 4})
	b := variable.NewOf([]float64{5, 6}, []float64{7, 8})

	result := Cat(a, b)

	result.Grad = variable.NewOf([]float64{0.1, 0.2, 0.3, 0.4}, []float64{0.5, 0.6, 0.7, 0.8})
	result.Backward()

	r.NotNil(a.Grad, "Gradient for a should not be nil")
	r.NotNil(b.Grad, "Gradient for b should not be nil")

	expectedGradA := [][]float64{{0.1, 0.2}, {0.5, 0.6}}
	expectedGradB := [][]float64{{0.3, 0.4}, {0.7, 0.8}}

	r.Equal(expectedGradA, a.Grad.Data, "Incorrect gradient for a")
	r.Equal(expectedGradB, b.Grad.Data, "Incorrect gradient for b")
}

func TestCatWithThreeDifferentMatrices(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf([]float64{1, 2}, []float64{3, 4})
	b := variable.NewOf([]float64{5, 6}, []float64{7, 8})
	c := variable.NewOf([]float64{9, 10}, []float64{11, 12})

	result := Cat(a, b, c)

	// Set gradient for the 2x6 result
	result.Grad = variable.NewOf(
		[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
		[]float64{0.7, 0.8, 0.9, 1.0, 1.1, 1.2},
	)
	result.Backward()

	// Check gradients
	expectedGradA := [][]float64{
		{0.1, 0.2},
		{0.7, 0.8},
	}
	expectedGradB := [][]float64{
		{0.3, 0.4},
		{0.9, 1.0},
	}
	expectedGradC := [][]float64{
		{0.5, 0.6},
		{1.1, 1.2},
	}

	r.Equal(expectedGradA, a.Grad.Data, "Incorrect gradient for a")
	r.Equal(expectedGradB, b.Grad.Data, "Incorrect gradient for b")
	r.Equal(expectedGradC, c.Grad.Data, "Incorrect gradient for c")
}

func TestMatrixMultiplicationWithCat(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf([]float64{1, 2}, []float64{3, 4})
	b := variable.NewOf([]float64{5, 6}, []float64{7, 8})
	c := Cat(a, b) // This will be a 2x4 matrix
	d := variable.NewOf([]float64{0.1}, []float64{0.2}, []float64{0.3}, []float64{0.4})

	result := variable.MatMul(c, d)

	expected := [][]float64{
		{4.4},
		{6.4},
	}
	r.Equal(expected, result.Data, "Matrix multiplication failed")

	result.Backward()

	expectedGradA := [][]float64{
		{0.1, 0.2},
		{0.1, 0.2},
	}
	expectedGradB := [][]float64{
		{0.3, 0.4},
		{0.3, 0.4},
	}

	r.Equal(expectedGradA, a.Grad.Data, "Incorrect gradient for a")
	r.Equal(expectedGradB, b.Grad.Data, "Incorrect gradient for b")
}
