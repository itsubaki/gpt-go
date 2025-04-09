package main

const (
	blockSize    = 8
	batchSize    = 4
	epochs       = 1
	learningRate = 0.01
	embedSize    = 32
)

// Embeddings are basically tensors under the hood
// What if we code-generate files for different tensors/linear layers
func main() {
	data, vocabSize := Data()

	layer := NewLinear(vocabSize, embedSize)
	_ = layer

	// Main training loop
	for i := 0; i < epochs; i++ {
		inputs, targets := Batch(data.Data, batchSize, blockSize)
		inputs.Print()
		targets.Print()

		//logits, loss := layer.Forward(inputs, targets)
	}

	return

	//	// Backward pass
	//	layer.ZeroGrad()
	//	probs := Softmax(logits)
	//	gradOut1 := probs.At(0).First() - 0
	//	gradOut2 := probs.At(1).First() - 1
	//	gradOutput := Tensor1D(gradOut1, gradOut2)
	//	layer.Backward(input, gradOutput)
}
