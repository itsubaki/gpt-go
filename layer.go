package main

type Linear struct {
	In, Out int
	Weight  Tensor
	Bias    float64
}

func NewLinear(in, out int) *Linear {
	return &Linear{
		In:     in,
		Out:    out,
		Weight: Tensor{Shape: []int{out, in}, Data: make([]float64, out*in)},
		Bias:   0,
	}
}

func (l *Linear) Forward(input []float64) []float64 {
	output := make([]float64, l.Out)
	for i := 0; i < l.Out; i++ {
		sum := l.Bias
		for j := 0; j < l.In; j++ {
			sum += input[j] * l.Weight.At(i*l.In+j)
		}
		output[i] = sum
	}
	return output
}

func (l *Linear) Backward(gradOutput []float64) {

}
