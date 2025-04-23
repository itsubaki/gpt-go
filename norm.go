package main

import (
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"

	"gptgo/pkg"
)

type LayerNorm struct {
	Scale *variable.Variable
	Shift *variable.Variable
	eps   float64
}

func NewLayerNorm(dim int) *LayerNorm {
	return &LayerNorm{
		eps:   1e-05,
		Scale: pkg.Ones(1, dim),
		Shift: pkg.Zeros(1, dim),
	}
}

// It is implemented using existing primitives, so backprop will work
func (ln *LayerNorm) Forward(x *variable.Variable) *variable.Variable {
	xmean := pkg.Mean(x)
	xvar := pkg.Variance(x)
	xhat := pkg.Div(pkg.Sub(x, xmean), pkg.Pow(0.5)(pkg.Add(xvar, variable.New(ln.eps))))
	out := pkg.Add(pkg.Mul(ln.Scale, xhat), ln.Shift)

	return out
}

func (ln *LayerNorm) Params() []layer.Parameter {
	return []layer.Parameter{
		ln.Scale,
		ln.Shift,
	}
}
