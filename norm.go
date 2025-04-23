package main

import (
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"

	"gptgo/pkg"
)

var (
	Mean     = pkg.Mean
	Variance = pkg.Variance
	Div      = pkg.Div
	Mul      = variable.Mul
	Pow      = variable.Pow
)

type LayerNorm struct {
	Scale *variable.Variable
	Shift *variable.Variable
	eps   float64
}

func NewLayerNorm(dim int) *LayerNorm {
	return &LayerNorm{
		eps:   1e-05,
		Scale: Ones(1, dim),
		Shift: Zeros(1, dim),
	}
}

// It is implemented using existing primitives, so backprop will work
func (ln *LayerNorm) Forward(x *variable.Variable) *variable.Variable {
	xmean := Mean(x)
	xvar := Variance(x)
	eps := variable.New(ln.eps)
	xhat := Div(Sub(x, xmean), Pow(0.5)(Add(xvar, eps)))
	out := Add(Mul(ln.Scale, xhat), ln.Shift)

	return out
}

func (ln *LayerNorm) Params() []layer.Parameter {
	return []layer.Parameter{
		ln.Scale,
		ln.Shift,
	}
}
