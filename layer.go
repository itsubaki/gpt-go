package main

type Linear struct {
	In, Out    int
	Weight     *Tensor
	WeightGrad *Tensor
	Bias       *Tensor
	BiasGrad   *Tensor
}

// TODO rename from new Linear to something other?
func NewLinear(in, out int) *Linear {
	return &Linear{
		In:         in,
		Out:        out,
		Weight:     RandN(in, out),
		WeightGrad: Zeros(in, out),
		Bias:       Zeros(out),
		BiasGrad:   Zeros(out),
	}
}

// Forward computes the output based on the input (forward pass)
// Targets parameter is used in learning.
// Returns logits and loss
// input is B*T*C
// targets is B*T
func (l *Linear) Forward(input *Tensor, targets *Tensor) (*Tensor, float64) {
	logits := input.Mul(l.Weight)
	for i := 0; i < logits.Shape[0]; i++ { // Batch
		for j := 0; j < len(logits.At(i, j).Data); i++ {
			logits.Data[i] += l.Bias.At(i).First()
		}
	}

	loss := 0.0
	if targets != nil {
		loss = CrossEntropyLoss(logits, targets)
	}

	return logits, loss
}

// Backward computes the gradient of the loss with respect to the input (backward pass).
// Returns the gradient of the loss with respect to the input.
// We must compute:
// 1. Gradient of the loss with respect to the weights
// 2. Gradient of the loss with respect to the bias
// 3. Gradient of the loss with respect to the input
// input is B*T*C
func (l *Linear) Backward(input *Tensor, gradOutput *Tensor) *Tensor {
	// TODO function to check shapes?

	// Gradient of the loss with respect to the weights
	l.WeightGrad = input.T().Mul(gradOutput)

	// Gradient of the loss with respect to the bias
	l.BiasGrad = gradOutput

	// Gradient of the loss with respect to the input
	inputGrad := gradOutput.Mul(l.Weight.T())

	return inputGrad
}

func (l *Linear) ZeroGrad() {
	l.WeightGrad = Zeros(l.Out, l.In)
	l.BiasGrad = Zeros(l.Out)
}
