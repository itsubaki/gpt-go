package pkg

import (
	"github.com/itsubaki/autograd/variable"
)

func Variance(x ...*variable.Variable) *variable.Variable {
	means := Mean(x[0])

	diffs := variable.Sub(x[0], means)
	squaredDiffs := variable.Pow(2)(diffs)
	variance := Mean(squaredDiffs)

	return variance
}
