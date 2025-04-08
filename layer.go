package main

type Linear struct {
	In, Out    int
	Weight     *Tensor
	WeightGrad *Tensor
	Bias       float64
}

func NewLinear(in, out int) *Linear {
	return &Linear{
		In:         in,
		Out:        out,
		Weight:     RandN(in, out),
		WeightGrad: Zeros(in, out),
		Bias:       0,
	}
}

// Forward computes the output based on the input (forward pass)
// TODO add guards
// TODO add bias
func (l *Linear) Forward(input *Tensor) float64 {
	// Fix move to tests
	l.Weight = Tensor2D(
		[][]float64{
			{2},
			{2},
			{4},
		})
	result := input.Mul(l.Weight)

	return result.Sum()
}

// Backward computes the gradient of the loss with respect to the input (backward pass).
// Returns the gradient of the loss with respect to the input.
// We must compute:
// 1. Gradient of the loss with respect to the weights
// 2. Gradient of the loss with respect to the input
// 3. Gradient of the loss with respect to the bias (skip for now)
func (l *Linear) Backward(input *Tensor, gradOutput *Tensor) *Tensor {
	// TODO function to check shapes?
	//if len(gradOutput.Shape) != l.Out {
	//	panic("Gradient output size does not match layer output size")
	//}

	// Calculate gradient with respect to the weights
	// WeightGrad is [Out, In]
	for i := 0; i < l.Out; i++ {
		for j := 0; j < l.In; j++ {
			existingGrad := l.WeightGrad.At(j, i)
			// TODO generalize for grad with a few outputs
			newGrad := gradOutput.At(0, i) * input.At(i, j)
			l.WeightGrad.Set(existingGrad+newGrad, j, i)
		}
	}

	inputGrad := Zeros(input.Shape...)
	for i := 0; i < l.Out; i++ {
		for j := 0; j < l.In; j++ {
			existingGrad := inputGrad.At(j)
			newGrad := gradOutput.At(0, i) * l.Weight.At(j, i)
			inputGrad.Set(existingGrad+newGrad, j)
		}
	}

	return inputGrad
}

func (l *Linear) ZeroGrad() {
	l.WeightGrad = Zeros(l.Out, l.In)
}
