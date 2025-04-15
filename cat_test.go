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
		{1, 2},
		{3, 4},
		{5, 6},
		{7, 8},
	}

	r.Equal(expected, result.Data, "Cat failed to concatenate correctly")
}

func TestCatSingleInput(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf([]float64{1, 2}, []float64{3, 4})

	result := Cat(a)

	r.Equal(a.Data, result.Data, "Single input should return unchanged")
}

func TestCatThreeTensors(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf([]float64{1, 2})
	b := variable.NewOf([]float64{3, 4})
	c := variable.NewOf([]float64{5, 6})

	result := Cat(a, b, c)

	expected := [][]float64{
		{1, 2},
		{3, 4},
		{5, 6},
	}

	r.Equal(expected, result.Data, "Cat failed to concatenate three tensors")
}

func TestCatDifferentRowCounts(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf([]float64{1, 2})                  // 1 row
	b := variable.NewOf([]float64{3, 4}, []float64{5, 6}) // 2 rows

	result := Cat(a, b)

	expected := [][]float64{
		{1, 2},
		{3, 4},
		{5, 6},
	}

	r.Equal(expected, result.Data, "Cat failed with different row counts")
}

func TestCatGradient(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf([]float64{1, 2})
	b := variable.NewOf([]float64{3, 4})

	result := Cat(a, b)

	result.Grad = variable.NewOf([]float64{0.1, 0.2}, []float64{0.3, 0.4})
	result.Backward()

	r.NotNil(a.Grad, "Gradient for a should not be nil")
	r.NotNil(b.Grad, "Gradient for b should not be nil")

	expectedGradA := [][]float64{{0.1, 0.2}}
	expectedGradB := [][]float64{{0.3, 0.4}}

	r.Equal(expectedGradA, a.Grad.Data, "Incorrect gradient for a")
	r.Equal(expectedGradB, b.Grad.Data, "Incorrect gradient for b")
}

func TestCatGradientDifferentRowCounts(t *testing.T) {
	r := require.New(t)

	a := variable.NewOf([]float64{1, 2})                  // 1 row
	b := variable.NewOf([]float64{3, 4}, []float64{5, 6}) // 2 rows

	result := Cat(a, b)

	result.Grad = variable.NewOf([]float64{0.1, 0.2}, []float64{0.3, 0.4}, []float64{0.5, 0.6})
	result.Backward()

	expectedGradA := [][]float64{{0.1, 0.2}}
	expectedGradB := [][]float64{{0.3, 0.4}, {0.5, 0.6}}

	r.Equal(expectedGradA, a.Grad.Data, "Incorrect gradient for a")
	r.Equal(expectedGradB, b.Grad.Data, "Incorrect gradient for b")
}

func TestCatCyclicComputationalGraph(t *testing.T) {
	r := require.New(t)

	// Create input variables
	a := variable.NewOf([]float64{1, 2})
	b := variable.NewOf([]float64{3, 4})

	// Create computation: y = a + cat(a, b)
	c := Cat(a, b)
	y := variable.Add(a, c)

	// Set gradient
	y.Backward()

	// a should receive gradient from two paths: directly from a, and through concatenation
	expectedGradA := [][]float64{{3, 3}}
	r.Equal(expectedGradA, a.Grad.Data, "Gradient for a should be the sum from both paths")

	// b should receive gradient only through concatenation
	expectedGradB := [][]float64{{1, 1}}
	r.Equal(expectedGradB, b.Grad.Data, "Gradient for b should come only from cat path")
}

func TestCatCyclicComputationalGraphWithMul(t *testing.T) {
	r := require.New(t)

	// Create input variables
	a := variable.NewOf([]float64{1, 2})
	b := variable.NewOf([]float64{3, 4})

	// Create computation: y = a * cat(a, b)
	c := Cat(a, b)
	y := MatMul(a, c)

	// Set gradient
	y.Backward()

	// a should receive gradient from two paths:
	// 1. Through direct multiplication (gradient = c)
	// 2. Through being part of cat(a,b) (gradient = a)
	expectedGradA := [][]float64{{4, 8}} // a gets c as grad (which is [[1,2],[3,4]]) and its first row gets multiplied by a ([1,2])
	r.Equal(expectedGradA, a.Grad.Data, "Gradient for a should include contributions from both paths")

	// b should receive gradient from being multiplied by a
	expectedGradB := [][]float64{{2, 2}} // b's rows each get multiplied by a ([1,2])
	r.Equal(expectedGradB, b.Grad.Data, "Gradient for b should come from multiplication with a")
}
