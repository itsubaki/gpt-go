package pkg

import (
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

type LayerNorm struct {
	Scale *variable.Variable
	Shift *variable.Variable
	eps   float64
}

func NewLayerNorm(dim int) *LayerNorm {
	return &LayerNorm{
		eps:   1e-05,
		Scale: OneLike(Zeros(1, dim)),
		Shift: Zeros(1, dim),
	}
}

// It is implemented using existing primitives, so backprop will work
func (ln *LayerNorm) Forward(x *variable.Variable) *variable.Variable {
	xmean := Mean(x)
	xvar := Variance(x)
	xhat := Div(Sub(x, xmean), Pow(0.5)(Add(xvar, variable.New(ln.eps))))
	out := Add(Mul(ln.Scale, xhat), ln.Shift)

	return out
}

func (ln *LayerNorm) Params() []layer.Parameter {
	return []layer.Parameter{
		ln.Scale,
		ln.Shift,
	}
}
