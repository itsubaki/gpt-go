package main

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func Test3input1OutputModel(t *testing.T) {
	r := require.New(t)

	input := Tensor1D([]float64{1, 0, 1})
	layer := NewLinear(3, 1)

	output := layer.Forward(input)
	r.Equal(6.0, output)

	grad := layer.Backward(input, Scalar(1))
	r.Equal([]float64{2.0, 2.0, 4.0}, grad.Data)

	r.Equal([]float64{1.0, 0.0, 1.0}, layer.WeightGrad.Data)
}
