package main

import (
	"math"

	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"

	"github.com/zakirullin/gpt-go/pkg"
)

var (
	Mean        = pkg.Mean
	Variance    = pkg.Variance
	Div         = pkg.Div
	Mul         = variable.Mul
	Pow         = variable.Pow
	RandWeights = xavier
)

type Linear struct {
	In, Out  int
	Weight   *variable.Variable
	Biased   bool
	Bias     *variable.Variable
	BiasGrad *variable.Variable
}

func NewLinear(in, out int, opts ...LinearOption) *Linear {
	l := &Linear{
		In:     in,
		Out:    out,
		Weight: RandWeights(in, out),
		Biased: true,
		Bias:   variable.Zero(1, out),
	}

	for _, opt := range opts {
		opt(l)
	}

	return l
}

// Forward computes the output based on the input (forward pass)
func (l *Linear) Forward(input *variable.Variable) *variable.Variable {
	logits := MatMul(input, l.Weight)

	if l.Biased {
		logits = Add(logits, l.Bias)
	}

	return logits
}

func (l *Linear) Params() []layer.Parameter {
	params := []layer.Parameter{
		l.Weight,
	}

	if l.Biased {
		params = append(params, l.Bias)
	}

	return params
}

type LinearOption func(*Linear)

func NoBias() LinearOption {
	return func(l *Linear) {
		l.Biased = false
		// Set bias tensors to nil or zero-sized tensors
		l.Bias = nil
		l.BiasGrad = nil
	}
}

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

func xavier(inSize, outSize int) *variable.Variable {
	w := matrix.Randn(inSize, outSize)
	xavier := 1.0 / math.Sqrt(float64(inSize))
	return variable.NewOf(matrix.MulC(xavier, w)...)
}
