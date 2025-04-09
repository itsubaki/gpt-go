package main

import (
	"fmt"
	"math/rand"
)

const (
	vocabSize = 64
	embedSize = 32
	blockSize = 8
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
	rand.Seed(42)

	// I want my nn to capture two outcomes:
	// 1. Sum of numbers is >= 5
	// 2. Sum of numbers is < 5
	layer := NewLinear(3, 2)

	// Training loop
	epochs := 10
	for i := 0; i < epochs; i++ {
		// Forward pass
		output, loss := layer.Forward(Tensor1D(1, 2, 3), []int{0})
		fmt.Printf("Epoch %d: Loss: %f\n", i, loss)
		fmt.Printf("Output: %v\n", output.Data)
	}

}
