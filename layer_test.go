package main

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func Test3to1Model(t *testing.T) {
	r := require.New(t)

	input := Tensor1D(1, 0, 1)
	layer := NewLinear(3, 1)
	layer.Weight = Tensor2D(
		[][]float64{
			{2},
			{2},
			{4},
		})

	output, _ := layer.Forward(input, nil)
	r.Equal([]float64{6.0}, output.Data)

	inputGrad := layer.Backward(input, Scalar(1.0))
	r.Equal([]float64{2.0, 2.0, 4.0}, inputGrad.Data)

	r.Equal([]float64{1.0, 0.0, 1.0}, layer.WeightGrad.Data)
}

func Test3to2Model(t *testing.T) {
	r := require.New(t)

	input := Tensor1D(1, 0, 1)
	layer := NewLinear(3, 2)
	layer.Weight = Tensor2D(
		[][]float64{
			{2, 3},
			{2, 2},
			{4, 1},
		})

	output, _ := layer.Forward(input, nil)
	r.Equal([]float64{6.0, 4.0}, output.Data)

	grad := layer.Backward(input, Tensor1D(1, 1))
	r.Equal([]float64{5.0, 4.0, 5.0}, grad.Data)
	//
	//r.Equal([]float64{1.0, 0.0, 1.0}, layer.WeightGrad.Data)
}
