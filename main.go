package main

import "fmt"

const (
	//blockSize    = 8
	epochs       = 1
	learningRate = 0.01
	embedSize    = 4
)

// Embeddings are basically tensors under the hood
// What if we code-generate files for different tensors/linear layers
func main() {
	data, vocabSize := Data()

	embeds := RandN(vocabSize, embedSize)
	layer := NewLinear(embedSize, vocabSize)
	_ = layer

	// Main training loop
	for i := 0; i < epochs; i++ {
		x, y := Batch(data.Data, 1, 1)
		// No batches for now
		x = x.At(0)
		y = y.At(0)
		fmt.Printf("%v\n", y)

		embed := embeds.At(int(x.First()))
		y.Print()
		embed.Print()

		//if targets != nil {
		//	loss := CrossEntropyLoss(logits, targets)
		//}

		//logits, loss := layer.Forward(embed, y)
		//logits.Print()
		//fmt.Printf("Epoch %d, Loss: %f\n", i, loss)
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
