// Add embeddings
package main

type Linear struct {
	In, Out    int
	Weight     *Tensor
	WeightGrad *Tensor
	Biased     bool
	Bias       *Tensor
	BiasGrad   *Tensor
}

// TODO rename from new Linear to something other?
func NewLinear(in, out int, biased bool) *Linear {
	return &Linear{
		In:         in,
		Out:        out,
		Weight:     RandN(in, out),
		WeightGrad: Zeros(in, out),
		Biased:     biased,
		Bias:       Zeros(out),
		BiasGrad:   Zeros(out),
	}
}

// Forward computes the output based on the input (forward pass)
// Targets parameter is used in learning.
// Returns logits and loss
// TODO add guards
func (l *Linear) Forward(input *Tensor) *Tensor {
	logits := input.Mul(l.Weight)
	logits.Add(l.Bias)

	return logits
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
	// Calculate the gradients for this example
	weightsGradForExample := input.T().Mul(gradOutput)
	biasGradForExample := gradOutput

	// Accumulate gradients instead of overwriting
	// TODO tensor.add?
	if l.WeightGrad == nil {
		l.WeightGrad = weightsGradForExample
		l.BiasGrad = biasGradForExample
	} else {
		for i := 0; i < len(l.WeightGrad.Data); i++ {
			l.WeightGrad.Data[i] += weightsGradForExample.Data[i]
		}

		for i := 0; i < len(l.BiasGrad.Data); i++ {
			l.BiasGrad.Data[i] += biasGradForExample.Data[i]
		}
	}

	inputGrad := gradOutput.Mul(l.Weight.T())

	return inputGrad
}

func (l *Linear) ZeroGrad() {
	l.WeightGrad = Zeros(l.Out, l.In)
	l.BiasGrad = Zeros(l.Out)
}
