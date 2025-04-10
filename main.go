package main

import "fmt"

const (
	epochs       = 100
	learningRate = 0.001
	embedSize    = 32
	batchSize    = 100
)

// Embeddings are basically tensors under the hood
// What if we code-generate files for different tensors/linear layers
func main() {
	data, vocabSize := Data()

	embeds := RandN(vocabSize, vocabSize)
	embedsGrad := Zeros(vocabSize, vocabSize)
	//layer := NewLinear(vocabSize, vocabSize)
	//_ = layer

	xs, ys := Batch(data.Data, 1, 1000000)

	// Main training loop
	averageLost := 0.0
	embedsGrad = Zeros(vocabSize, vocabSize)
	for i := 0; i < len(ys.Data); i++ {

		// Forward pass
		// No batches for now
		x := xs.At(0).At(i)
		y := ys.At(0).At(i)

		embed := embeds.At(int(x.First()))

		// Backward pass
		probs := Softmax(embed)
		for j := 0; j < len(probs.Data); j++ {
			// Calculate gradient: (probability - one_hot_target)
			targetValue := 0.0
			if j == int(y.First()) {
				targetValue = 1.0
			}

			gradVal := probs.Data[j] - targetValue

			tokenIdx := int(x.First())
			embedsGrad.Data[tokenIdx*vocabSize+j] += gradVal
		}

		averageLost += CrossEntropyLoss(embed, y.First())

		// Check if we've completed a batch
		if (i%batchSize) == 0 || i == len(ys.Data)-1 {
			// Update weights using accumulated gradients
			for j := 0; j < len(embeds.Data); j++ {
				averageGrad := embedsGrad.Data[j] / float64(batchSize)
				embeds.Data[j] -= learningRate * averageGrad
			}

			// Print the average loss for this batch
			fmt.Printf("Epoch %d, Loss: %f\n", i, averageLost/float64(batchSize))

			embedsGrad = Zeros(vocabSize, vocabSize)
			averageLost = 0.0
		}

		//logits := layer.Forward(embed)
		//
		//// Backward pass
		//layer.ZeroGrad()
		//probs := Softmax(logits)
		//grads := make([]float64, vocabSize)
		//for j := 0; j < vocabSize; j++ {
		//	oneHot := 0.0
		//	if y.First() == float64(j) {
		//		oneHot = 1.0
		//	}
		//	grads[j] = probs.At(j).First() - oneHot
		//}
		//gradOutput := Tensor1D(grads...)
		//layer.Backward(embed, gradOutput)
		//
		//// Update weights
		//for j := 0; j < len(layer.Weight.Data); j++ {
		//	layer.Weight.Data[j] -= learningRate * layer.WeightGrad.Data[j]
		//}
		//
		//// Update bias
		//for j := 0; j < len(layer.Bias.Data); j++ {
		//	//layer.Bias.Data[j] -= learningRate * layer.BiasGrad.Data[j]
		//}

		//if (i % 100) == 0 {
		//loss := CrossEntropyLoss(logits, y.First())
		//fmt.Printf("Epoch %d, Loss: %f\n", i, loss)
		//}
	}

}
