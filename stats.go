// Maybe move to tensor?
package main

import (
	"math"
)

// TODO only supports 1D tensors
func Softmax(tensor *Tensor) *Tensor {
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

	return Tensor1D(result...)
}

// CrossEntropyLoss computes the cross-entropy loss between logits and targets.
func CrossEntropyLoss(logits *Tensor, target float64) float64 {
	rating := 0.0
	normalizedLogits := Softmax(logits)
	// Compute log likelihood
	logitIndex := int(target)
	rating += math.Log(normalizedLogits.At(logitIndex).First())

	// The higher rating the better, the loss is opposite :)
	loss := -rating

	return loss
}
