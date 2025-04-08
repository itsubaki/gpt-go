package main

import "fmt"

const (
	vocabSize = 64
	embedSize = 32
	blockSize = 8
)

// Embeddings are basically tensors under the hood
// What if we codegenerate files for different tensors/linear layers
func main() {
	//tokenEmbeds := RandN(vocabSize, embedSize)
	//positionEmbeds := RandN(blockSize, embedSize)

	input := Tensor1D([]float64{1, 0, 1})
	layer := NewLinear(3, 1)

	output := layer.Forward(input)
	fmt.Println("Sum:", output)

	grad := layer.Backward(input, Scalar(1))
	fmt.Println("Gradient:", grad.Data)
}
