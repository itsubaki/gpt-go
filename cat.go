package main

import (
	"github.com/itsubaki/autograd/variable"
)

// Cat concatenates variables along dimension 0 (rows)
// Only inputs of the same shape are supported
func Cat(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{Forwarder: &CatT{NumInputs: len(x)}}).First(x...)
}

type CatT struct {
	NumInputs int
	RowCounts []int
}

func (f *CatT) Forward(x ...*variable.Variable) []*variable.Variable {
	// Store row counts for backward pass
	f.RowCounts = make([]int, len(x))

	// Initialize result with capacity for all rows
	totalRows := 0
	for i, v := range x {
		rows := len(v.Data)
		f.RowCounts[i] = rows
		totalRows += rows
	}

	// Concatenate along rows
	result := make([][]float64, 0, totalRows)
	for _, v := range x {
		result = append(result, v.Data...)
	}

	return []*variable.Variable{
		variable.NewOf(result...),
	}
}

func (f *CatT) Backward(gy ...*variable.Variable) []*variable.Variable {
	grads := make([]*variable.Variable, f.NumInputs)

	// Split gradient by row counts
	rowOffset := 0
	for i, rows := range f.RowCounts {
		// Extract the portion of gradient for this input
		inputGrad := gy[0].Data[rowOffset : rowOffset+rows]

		grads[i] = variable.NewOf(inputGrad...)

		rowOffset += rows
	}

	return grads
}
