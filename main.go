package main

import (
	"fmt"
)

const (
	epochs       = 1000
	learningRate = 0.01
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
	layer.Weight = T2{
		{2, 3},
		{2, 2},
		{4, 1},
	}.Tensor()

	// Training loop
	input := Tensor1D(1, 0, 1)
	targets := []int{1} // Our sum is less than 5, so the output should be 1
	for i := 0; i < epochs; i++ {
		// Forward pass
		logits, loss := layer.Forward(input, targets)
		fmt.Printf("Epoch %d: Loss: %f\n", i, loss)
		//layer.WeightGrad.Print()
		//layer.Weight.Print()
		//fmt.Printf("Logits: %v\n", Softmax(logits).Data)

		// Backward pass
		layer.ZeroGrad()

		gradOut1 := logits.At(0).First() - 0
		gradOut2 := logits.At(1).First() - 1
		gradOutput := Tensor1D(gradOut1, gradOut2)
		layer.Backward(input, gradOutput)
		//layer.WeightGrad.Print()

		// Update weights
		for i, _ := range layer.Weight.Data {
			// Model won't converge because we skipped bias
			layer.Weight.Data[i] -= learningRate * layer.WeightGrad.Data[i]
		}
	}

}
