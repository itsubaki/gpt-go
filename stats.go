// Maybe move to tensor?
package main

import (
	"fmt"
	"math"
)

// TODO only supports 1D tensors
func Softmax(tensor *Tensor) []float64 {
	vec := tensor.Data
	result := make([]float64, len(vec))

	var sum float64 = 0
	var maxVal float64 = vec[0]

	// Find max for numerical stability
	// TODO replace with generic
	for _, v := range vec {
		if v > maxVal {
			maxVal = v
		}
	}

	// Compute exp and sum
	for i, v := range vec {
		expVal := math.Exp(v - maxVal)
		result[i] = expVal
		sum += expVal
	}

	// Normalize
	for i := range result {
		result[i] /= sum
	}

	return result
}

// CrossEntropyLoss computes the cross-entropy loss between logits and targets.
// Rating is calculated for every batch, then summed and divided by batch size.
// Average loss across all batches is returned.
func CrossEntropyLoss(logits *Tensor, targets []int) float64 {
	// Make it not hardcoded
	// TODO add checks
	batchSize := len(targets)

	rating := 0.0
	// One rating per every batch
	for i := 0; i < batchSize; i++ {
		fmt.Printf("%v\n", logits.At(i).Data)
		normalizedLogits := Softmax(logits.At(i))
		fmt.Printf("%v\n", normalizedLogits)
		// Compute log likelihood
		rating += math.Log(normalizedLogits[targets[i]])
		fmt.Printf("%v\n", rating)
	}

	// The higher rating the better, the loss is opposite :)
	loss := -rating / float64(batchSize)

	return loss
}
