package main

import (
	"github.com/itsubaki/autograd/variable"
)

// Cat concatenates variables along the columns
func Cat(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{Forwarder: &CatT{NumInputs: len(x)}}).First(x...)
}

type CatT struct {
	NumInputs int
	ColSize   int
}

// Concatenate along the columns dimension (dim=1)
func (f *CatT) Forward(x ...*variable.Variable) []*variable.Variable {
	rows := len(x[0].Data)
	f.ColSize = len(x[0].Data[0])
	totalCols := f.ColSize * len(x)

	result := make([][]float64, rows)
	for i := range result {
		result[i] = make([]float64, totalCols)
		colOffset := 0
		for _, v := range x {
			copy(result[i][colOffset:], v.Data[i])
			colOffset += f.ColSize
		}
	}

	return []*variable.Variable{
		variable.NewOf(result...),
	}
}

func (f *CatT) Backward(gy ...*variable.Variable) []*variable.Variable {
	grads := make([]*variable.Variable, f.NumInputs)

	// Split along columns
	for i := 0; i < f.NumInputs; i++ {
		colOffset := i * f.ColSize
		colData := make([][]float64, len(gy[0].Data))

		for j := range colData {
			colData[j] = make([]float64, f.ColSize)
			copy(colData[j], gy[0].Data[j][colOffset:colOffset+f.ColSize])
		}

		grads[i] = variable.NewOf(colData...)
	}

	return grads
}
