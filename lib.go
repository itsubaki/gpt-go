package main

import (
	"fmt"
	"math"

	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
	"gonum.org/v1/gonum/stat/distuv"
)

type Model struct {
	params layer.Parameters
}

func (m Model) Params() layer.Parameters {
	return m.params
}

func Rows(x *variable.Variable, indexes ...float64) *variable.Variable {
	var intIndexes []int
	for _, index := range indexes {
		intIndexes = append(intIndexes, int(index))
	}

	return (&variable.Function{Forwarder: &variable.GetItemT{Slices: intIndexes}}).First(x)
}

// Add tests
func RandKaiming(dims ...int) *variable.Variable {
	sigma := math.Sqrt(2.0 / float64(dims[1]))
	dist := distuv.Normal{Mu: 0, Sigma: sigma}
	result := matrix.F(matrix.Zero(dims[0], dims[1]), func(_ float64) float64 { return dist.Rand() })

	return variable.NewOf(result...)
}

// Only works with 2D tensors
func Tril(m *variable.Variable) *variable.Variable {
	result := variable.ZeroLike(m)
	for i := 0; i < len(m.Data); i++ {
		for j := 0; j < len(m.Data[i]); j++ {
			if j <= i {
				result.Data[i][j] = m.Data[i][j]
			}
		}
	}

	return result
}

// The result would be added to computation graph and tied to m
func MaskedInfFill(m, mask *variable.Variable) *variable.Variable {
	negInfMaskedData := matrix.F2(m.Data, mask.Data, func(a, b float64) float64 {
		if b == 0 {
			return math.Inf(-1)
		}

		return a
	})
	mMasked := Add(variable.Mul(m, mask), variable.NewOf(negInfMaskedData...))

	return mMasked
}

// Arange creates a new slice containing a sequence of values from start to end (exclusive) with the given step.
// If step is not provided, it defaults to 1.
func Arange(end int) []float64 {
	step := 1.0

	// Calculate the number of elements
	n := int(math.Ceil((float64(end)) / step))
	if n <= 0 {
		return []float64{}
	}

	result := make([]float64, n)
	for i := 0; i < n; i++ {
		result[i] = float64(i) * step
	}

	return result
}

func PrintShape(v *variable.Variable) {
	fmt.Printf("(%d, %d)\n", len(v.Data), len(v.Data[0]))
}
