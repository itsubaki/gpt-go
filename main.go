package main

import "fmt"

const (
	learningRate = 0.01
	batchSize    = 100
	epochs       = 100
	embedSize    = 32
)

// Embeddings are basically tensors under the hood
// What if we code-generate files for different tensors/linear layers
func main() {
	data, vocabSize := Data()

	embeds := RandKaiming(vocabSize, vocabSize)
	embedsGrad := Zeros(vocabSize, vocabSize)
	//layer := NewLinear(vocabSize, vocabSize)
	//_ = layer

	xs, ys := Batch(data.Data, 1, 1000000)
	//xs.Print()
	//ys.Print()

	// Main training loop
	lostSum := 0.0
	embedsGrad = Zeros(vocabSize, vocabSize)
	for i := 0; i < len(ys.Data); i++ {

		// Forward pass
		// No batches for now
		x := xs.At(0).At(i)
		y := ys.At(0).At(i)

		logits := embeds.At(int(x.First()))

		// Backward pass
		probs := Softmax(logits)
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

		lostSum += CrossEntropyLoss(logits, y.First())

		// Check if we've completed a batch
		if ((i%batchSize) == 0 && i != 0) || i == len(ys.Data)-1 {
			// Update weights using accumulated gradients
			for j := 0; j < len(embeds.Data); j++ {
				//averageGrad := embedsGrad.Data[j] / float64(batchSize)
				embeds.Data[j] -= learningRate * embedsGrad.Data[j]
			}

			// Print the average loss for this batch
			// Random loss is 4.174
			fmt.Printf("Epoch %d, Loss: %f\n", i, lostSum/float64(batchSize))

			embedsGrad = Zeros(vocabSize, vocabSize)
			lostSum = 0.0
		}
	}
}
