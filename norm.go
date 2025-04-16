package main

import (
	"github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

type LayerNorm struct {
	eps   float64
	scale *variable.Variable
	shift *variable.Variable
}

func NewLayerNorm(dim int) *LayerNorm {
	return &LayerNorm{
		eps:   1e-05,
		scale: OneLike(Zeros(1, dim)),
		shift: Zeros(1, dim),
	}
}

// It is implemented using existing primitives, so backprop will work
func (ln *LayerNorm) Forward(x *variable.Variable) *variable.Variable {
	xmean := Mean(x)
	xvar := Variance(x)
	xhat := function.Div(function.Sub(x, xmean), function.Pow(0.5)(Add(xvar, variable.New(ln.eps))))
	out := function.Add(function.Mul(ln.scale, xhat), ln.shift)

	return out
}

func (ln *LayerNorm) Params() []layer.Parameter {
	return []layer.Parameter{
		ln.scale,
		ln.shift,
	}
}
