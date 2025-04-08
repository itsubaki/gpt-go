package main

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestAtScalar(t *testing.T) {
	r := require.New(t)

	tensor := Tensor1D([]float64{1})
	r.Equal(tensor.At(0), 1.0)
}

func TestAtVector(t *testing.T) {
	r := require.New(t)

	tensor := Tensor1D([]float64{1, 2, 3})
	r.Equal(tensor.At(0), 1.0)
	r.Equal(tensor.At(1), 2.0)
	r.Equal(tensor.At(2), 3.0)
}

func TestAtMatrix(t *testing.T) {
	r := require.New(t)

	tensor := Tensor2D([][]float64{
		{1, 2},
		{3, 4},
	})
	r.Equal(tensor.At(0, 0), 1.0)
	r.Equal(tensor.At(0, 1), 2.0)
	r.Equal(tensor.At(1, 0), 3.0)
	r.Equal(tensor.At(1, 1), 4.0)
}

func TestMulVector(t *testing.T) {
	r := require.New(t)

	tensorA := Tensor1D([]float64{1, 2, 3})
	tensorB := Tensor2D(
		[][]float64{
			{4},
			{5},
			{6},
		})
	result := tensorA.Mul(tensorB)

	expected := Tensor1D([]float64{32})
	r.Equal(expected.Data, result.Data)
}

func TestMulMatrix(t *testing.T) {
	r := require.New(t)

	tensorA := Tensor2D([][]float64{
		{1, 2},
		{3, 4},
	})
	tensorB := Tensor2D([][]float64{
		{5, 6},
		{7, 8},
	})

	result := tensorA.Mul(tensorB)
	expected := Tensor2D([][]float64{
		{19, 22},
		{43, 50},
	})

	r.Equal(result.Data, expected.Data)
}

func TestOffset(t *testing.T) {
	r := require.New(t)

	tensor := Tensor2D([][]float64{
		{1, 2},
		{3, 4},
	})

	r.Equal(tensor.offset(0, 0), 0)
	r.Equal(tensor.offset(0, 1), 1)
	r.Equal(tensor.offset(1, 0), 2)
	r.Equal(tensor.offset(1, 1), 3)
}
