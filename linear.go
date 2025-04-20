// Add embeddings
package main

import (
	"math"

	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"

	"gptgo/pkg"
)

type Linear struct {
	In, Out  int
	Weight   *variable.Variable
	Biased   bool
	Bias     *variable.Variable
	BiasGrad *variable.Variable
}

var randWeights = xavier

type LinearOption func(*Linear)

// TODO rename from new Linear to something other?
func NewLinear(in, out int, opts ...LinearOption) *Linear {
	l := &Linear{
		In:     in,
		Out:    out,
		Weight: randWeights(in, out),
		Biased: true,
		Bias:   variable.Zero(1, out),
	}

	for _, opt := range opts {
		opt(l)
	}

	return l
}

func NoBias() LinearOption {
	return func(l *Linear) {
		l.Biased = false
		// Set bias tensors to nil or zero-sized tensors
		l.Bias = nil
		l.BiasGrad = nil
	}
}

// Forward computes the output based on the input (forward pass)
func (l *Linear) Forward(input *variable.Variable) *variable.Variable {
	logits := pkg.MatMul(input, l.Weight)

	if l.Biased {
		logits = pkg.Add(logits, l.Bias)
	}

	return logits
}

func xavier(inSize, outSize int) *variable.Variable {
	w := matrix.Randn(inSize, outSize)
	xavier := 1.0 / math.Sqrt(float64(inSize))
	return variable.NewOf(matrix.MulC(xavier, w)...)
}
