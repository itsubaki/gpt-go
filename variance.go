package main

import (
	"github.com/itsubaki/autograd/variable"
)

func Variance(x ...*variable.Variable) *variable.Variable {
	// Calculate mean per row
	means := Mean(x[0])
	zeros := variable.ZeroLike(x[0])

	// means is [rows, 1], ones is [1, cols], result is [rows, cols]
	broadcastedMeans := variable.Add(zeros, means)

	diffs := variable.Sub(x[0], broadcastedMeans)
	squaredDiffs := variable.Pow(2)(diffs)
	variance := Mean(squaredDiffs)

	return variance
}
