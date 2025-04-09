package main

import "fmt"

const (
	epochs       = 1000
	learningRate = 0.001
	embedSize    = 32
)

// Embeddings are basically tensors under the hood
// What if we code-generate files for different tensors/linear layers
func main() {
	data, vocabSize := Data()

	embeds := RandN(vocabSize, embedSize)
	layer := NewLinear(embedSize, vocabSize)
	_ = layer

	xs, ys := Batch(data.Data, 1, 10)

	// Main training loop
	for i := 0; i < len(ys.Data); i++ {
		// Forward pass
		// No batches for now
		x := xs.At(0).At(i)
		y := ys.At(0).At(i)

		embed := embeds.At(int(x.First()))
		logits := layer.Forward(embed)

		// Backward pass
		layer.ZeroGrad()
		probs := Softmax(logits)
		grads := make([]float64, vocabSize)
		for j := 0; j < vocabSize; j++ {
			oneHot := 0.0
			if y.First() == float64(j) {
				oneHot = 1.0
			}
			grads[j] = probs.At(j).First() - oneHot
		}
		gradOutput := Tensor1D(grads...)
		layer.Backward(embed, gradOutput)

		// Update weights
		for j := 0; j < len(layer.Weight.Data); j++ {
			layer.Weight.Data[j] -= learningRate * layer.WeightGrad.Data[j]
		}

		// Update bias
		for j := 0; j < len(layer.Bias.Data); j++ {
			//layer.Bias.Data[j] -= learningRate * layer.BiasGrad.Data[j]
		}

		//if (i % 100) == 0 {
		loss := CrossEntropyLoss(logits, y.First())
		fmt.Printf("Epoch %d, Loss: %f\n", i, loss)
		//}
	}

}
