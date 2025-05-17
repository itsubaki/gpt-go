package pkg

import (
	"github.com/itsubaki/autograd/variable"
)

// Cat concatenates matrices horizontally
func Cat(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{Forwarder: &CatT{NumInputs: len(x)}}).First(x...)
}

type CatT struct {
	NumInputs int
	ColSize   int
}

// Concatenate along the columns dimension (dim=1)
func (f *CatT) Forward(x ...*variable.Variable) []*variable.Variable {
	rows := x[0].Data.Rows
	f.ColSize = x[0].Data.Cols
	totalCols := f.ColSize * len(x)

	result := make([][]float64, rows)
	for i := range result {
		result[i] = make([]float64, totalCols)
		colOffset := 0
		for _, v := range x {
			copy(result[i][colOffset:], v.Data.Row(i))
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
	for i := range f.NumInputs {
		colOffset := i * f.ColSize
		colData := make([][]float64, gy[0].Data.Rows)

		for j := range colData {
			colData[j] = make([]float64, f.ColSize)
			copy(colData[j], gy[0].Data.Row(j)[colOffset:colOffset+f.ColSize])
		}

		grads[i] = variable.NewOf(colData...)
	}

	return grads
}
