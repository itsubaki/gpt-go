package main

import (
	"math"
	"math/rand/v2"

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
	RandWeights = uniform
)

type Linear struct {
	In, Out int
	Weight  *variable.Variable
	Biased  bool
	Bias    *variable.Variable
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

// It is implemented using existing primitives, so back propagation will work
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

// Sample values from uniform(-1/sqrt(in_features), 1/sqrt(in_features)).
// Same weights initialization is used in PyTorch.
func uniform(inSize, outSize int) *variable.Variable {
	bound := 1 / math.Sqrt(float64(inSize))
	rnd := func(_ float64) float64 {
		return (2 * bound * rand.Float64()) - bound
	}

	m := matrix.Zero(inSize, outSize)
	m = matrix.F(m, rnd)

	return variable.NewOf(m...)
}
