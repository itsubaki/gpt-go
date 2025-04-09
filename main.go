package main

import (
	"fmt"
)

const (
	epochs       = 100
	learningRate = 0.1
	//vocabSize    = 64
	//embedSize    = 32
	//blockSize    = 8
)

//var tokenEmbeds = RandN(vocabSize, embedSize)
//var layer = NewLinear(embedSize, vocabSize)

func forward(indexes *Tensor, targets *Tensor) {
	//layer.ZeroGrad()

	//for _, index := range indexes.Data {
	//
	//}
	//
	//layer.Forward()
}

// Embeddings are basically tensors under the hood
// What if we codegenerate files for different tensors/linear layers
func main() {
	// I want my nn to capture two outcomes:
	// 1. Sum of numbers is >= 5
	// 2. Sum of numbers is < 5
	layer := NewLinear(3, 2)

	// Training loop
	for i := 0; i < epochs; i++ {
		input := Tensor1D(1, 2, 3)
		targets := []int{0}

		// Forward pass
		logits, loss := layer.Forward(input, targets)
		fmt.Printf("Epoch %d: Loss: %f\n", i, loss)
		fmt.Printf("Logits: %v\n", Softmax(logits))

		// Backward pass
		layer.ZeroGrad()
		gradOutput := Tensor1D(1, 1) // GradOutput is currently 1, the derivative of the end graph
		layer.Backward(input, gradOutput)

		// Update weights
		for i, _ := range layer.Weight.Data {
			layer.Weight.Data[i] += learningRate * layer.WeightGrad.Data[i]
		}
	}

}
