// Add embeddings
package main

import (
	"math"

	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

type Linear struct {
	In, Out  int
	Weight   *variable.Variable
	Biased   bool
	Bias     *variable.Variable
	BiasGrad *variable.Variable
}

type LinearOption func(*Linear)

// TODO rename from new Linear to something other?
func NewLinear(in, out int, opts ...LinearOption) *Linear {
	l := &Linear{
		In:     in,
		Out:    out,
		Weight: initw(in, out),
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
	logits := MatMul(input, l.Weight)

	if l.Biased {
		logits = Add(logits, l.Bias)
	}

	return logits
}

func initw(inSize, outSize int) *variable.Variable {
	w := matrix.Randn(inSize, outSize)
	xavier := 1.0 / math.Sqrt(float64(inSize))
	return variable.NewOf(matrix.MulC(xavier, w)...)
}

// Backward computes the gradient of the loss with respect to the input (backward pass).
// Returns the gradient of the loss with respect to the input.
// We must compute:
// 1. Gradient of the loss with respect to the weights
// 2. Gradient of the loss with respect to the bias
// 3. Gradient of the loss with respect to the input
// (x * w + b)
// dL/dx = (x * w + b)' * dL/dy = w * dL/dy
// dL/dw = x * dL/dy
// dL/db = dL/dy (addition just continues gradient, it flows)
func (l *Linear) Backward(input *Tensor, gradOutput *Tensor) *Tensor {
	return nil
	//// Calculate the gradients for this example
	//// TODO simplify this
	//// TODO don't calc bias for non-biased layers
	//weightsGradForExample := input.T().Mul(gradOutput)
	//biasGradForExample := gradOutput
	//
	//// Accumulate gradients instead of overwriting
	//// TODO tensor.add?
	//if l.WeightGrad == nil {
	//	l.WeightGrad = weightsGradForExample
	//	l.BiasGrad = biasGradForExample
	//} else {
	//	for i := 0; i < len(l.WeightGrad.Data); i++ {
	//		l.WeightGrad.Data[i] += weightsGradForExample.Data[i]
	//	}
	//
	//	for i := 0; i < len(l.BiasGrad.Data); i++ {
	//		l.BiasGrad.Data[i] += biasGradForExample.Data[i]
	//	}
	//}
	//
	//inputGrad := gradOutput.Mul(l.Weight.T())
	//
	//return inputGrad
}

func (l *Linear) ZeroGrad() {
	l.Weight.Cleargrad()
	l.Bias.Cleargrad()
}
