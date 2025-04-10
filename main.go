package main

import "fmt"

const (
	batchSize    = 16
	learningRate = 0.01
	embedSize    = 32
)

// Embeddings are basically tensors under the hood
// What if we code-generate files for different tensors/linear layers
func main() {
	data, vocabSize := Data()

	embeds := RandKaiming(vocabSize, embedSize)
	layer := NewLinear(embedSize, vocabSize)

	// It's not really batch, both inputs and targets are vectors.
	// We don't use batches
	// Inputs are indexes for embeds table
	inputs, targets := Batch(data.Data, 1, len(data.Data)-1)
	inputs, targets = inputs.At(0), targets.At(0)

	// Main training loop
	lossSum := 0.0
	for i := 0; i < len(targets.Data); i++ {
		// Forward pass
		input := inputs.At(i).First()
		target := targets.At(i).First()

		embed := embeds.At(int(input))
		logits := layer.Forward(embed)

		// Backward pass
		layer.ZeroGrad()
		probs := Softmax(logits)
		grads := make([]float64, vocabSize)
		for j := 0; j < vocabSize; j++ {
			oneHot := 0.0
			if target == float64(j) {
				oneHot = 1.0
			}
			grads[j] = probs.At(j).First() - oneHot
		}
		gradOutput := Tensor1D(grads...)
		layer.Backward(embed, gradOutput)

		// Loss calculation
		lossSum += CrossEntropyLoss(logits, target)

		// We only update weights once in a while.
		// Kinda "emulating" batches
		if (i % batchSize) == 0 {
			// Update weights
			for j := 0; j < len(layer.Weight.Data); j++ {
				layer.Weight.Data[j] -= learningRate * layer.WeightGrad.Data[j]
			}

			// Update bias
			for j := 0; j < len(layer.Bias.Data); j++ {
				layer.Bias.Data[j] -= learningRate * layer.BiasGrad.Data[j]
			}

			if (i % (batchSize * 1000)) == 0 {
				fmt.Printf("Loss: %f\n", lossSum/float64(batchSize))
			}

			lossSum = 0.0
			layer.ZeroGrad()
		}
	}

	// Generate text
	context := "A"
	maxTokens := 500
	temperature := 0.6 // The higher the temperature, the more random the output
	token := int(Encode(context).First())
	fmt.Println("\nGenerated text after training:")
	for i := 0; i < maxTokens; i++ {
		embed := embeds.At(token)
		output := layer.Forward(embed)
		probs := Softmax(output)
		token = SampleTemp(probs, temperature)
		decodedToken := Decode([]int{token})
		fmt.Printf(decodedToken)
	}
}
