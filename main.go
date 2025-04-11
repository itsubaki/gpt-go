package main

import (
	"fmt"
	"math"
)

const (
	batchSize    = 16
	learningRate = 0.01
	embedSize    = 32
	headSize     = 16
)

// Embeddings are basically tensors under the hood
// What if we code-generate files for different tensors/linear layers
func main() {
	t := Tril(Ones(3, 3))
	t = t.MaskedFill(t, 0, math.Inf(-1))
	t = Softmax(t)

	t.Print()

	return
	data, vocabSize := Data()

	embeds := RandKaiming(vocabSize, embedSize)
	embedsGrad := Zeros(vocabSize, embedSize)

	lmHead := NewLinear(embedSize, vocabSize)

	key := NewLinear(embedSize, headSize, NoBias())
	_ = key
	query := NewLinear(embedSize, headSize, NoBias())
	_ = query
	value := NewLinear(embedSize, headSize, NoBias())
	_ = value

	// It's not really batch, both inputs and targets are vectors.
	// We don't use batches
	// Inputs are indexes for embeds table
	inputs, targets := Batch(data.Data, 1, len(data.Data)-1)
	inputs, targets = inputs.At(0), targets.At(0)

	// Main training loop
	lossSum := 0.0
	for i := 0; i < len(inputs.Data); i++ {
		// Forward pass
		input := inputs.At(i).First()
		target := targets.At(i).First()

		embed := embeds.At(int(input))
		logits := lmHead.Forward(embed)

		// Backward pass
		lmHead.ZeroGrad()
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
		lmHead.Backward(embed, gradOutput)

		// Calculate gradient for embed
		grad := gradOutput.Mul(lmHead.Weight.T())
		embedGrad := embedsGrad.At(int(input))
		grad = embedGrad.Add(grad)
		for j := 0; j < len(embedGrad.Data); j++ {
			embedGrad.Data[j] += grad.At(j).First()
		}

		// Loss calculation
		lossSum += CrossEntropyLoss(logits, target)

		// We only update weights once in a while.
		// Kinda "emulating" batches
		if (i % batchSize) == 0 {
			// Update weights
			for j := 0; j < len(lmHead.Weight.Data); j++ {
				lmHead.Weight.Data[j] -= learningRate * lmHead.WeightGrad.Data[j]
			}

			// Update bias
			for j := 0; j < len(lmHead.Bias.Data); j++ {
				lmHead.Bias.Data[j] -= learningRate * lmHead.BiasGrad.Data[j]
			}

			// Update embeds
			for j := 0; j < len(embeds.Data); j++ {
				embeds.Data[j] -= learningRate * embedsGrad.Data[j]
			}

			if (i % (batchSize * 1000)) == 0 {
				fmt.Printf("Loss: %f\n", lossSum/float64(batchSize))
			}

			lossSum = 0.0
			lmHead.ZeroGrad()
			embedsGrad = Zeros(vocabSize, embedSize)
		}
	}

	// Generate text
	context := "A"
	maxTokens := 500
	token := int(Encode(context).First())
	fmt.Println("\nGenerated text after training:")
	for i := 0; i < maxTokens; i++ {
		embed := embeds.At(token)
		output := lmHead.Forward(embed)
		probs := Softmax(output)
		token = Sample(probs)
		decodedToken := Decode([]int{token})
		fmt.Printf(decodedToken)
	}
}
