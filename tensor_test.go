package main

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestAtScalar(t *testing.T) {
	r := require.New(t)

	tensor := Tensor1D([]float64{1})
	r.Equal(tensor.At(0), 1)
}

func TestAtVector(t *testing.T) {
	r := require.New(t)

	tensor := Tensor1D([]float64{1, 2, 3})
	r.Equal(tensor.At(0), 1.0)
	r.Equal(tensor.At(1), 2.0)
	r.Equal(tensor.At(2), 3.0)
}
