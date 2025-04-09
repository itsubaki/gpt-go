// Maybe move to tensor?
package main

import "math"

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

// CrossEntropyLoss computes the cross-entropy loss between logits () and targets.
// TODO batch support
func CrossEntropyLoss(logits *Tensor, targets []int) float64 {
	//batchSize := len(targets)
	normalizedLogits := Softmax(logits)
	rating := 0.0
	for _, target := range targets {
		rating += math.Log(normalizedLogits[target])
	}
	// The higher rating the better, the loss is opposite :)
	loss := -rating // mean

	//for i, target := range targets {
	//	if target >= 0 && target < logits.cols {
	//		// Extract logits for this sample
	//		rowLogits := make([]float64, logits.cols)
	//		for j := 0; j < logits.cols; j++ {
	//			rowLogits[j] = logits.Get(i, j)
	//		}
	//
	//		// Apply softmax
	//		probs := Softmax(rowLogits)
	//
	//		// Compute negative log likelihood
	//		loss -= math.Log(probs[target])
	//	}
	//}

	return loss // / float64(batchSize)
}
