// Maybe move to tensor?
package main

import (
	"math"
	"math/rand"
)

// TODO only supports 1D tensors
func Softmax(tensor *Tensor) *Tensor {
	vec := tensor.Data
	result := make([]float64, len(vec))

	var sum = 0.0
	var maxVal = vec[0]

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

func Sample(probs *Tensor) int {
	// Get probabilities as slice
	// Generate random number in [0,1)
	r := rand.Float64()

	// Find the first index where cumulative probability exceeds r
	cumulativeProb := 0.0
	for i, p := range probs.Data {
		cumulativeProb += p
		if r < cumulativeProb {
			return i
		}
	}

	// Fallback (should rarely happen due to floating point precision)
	return len(probs.Data) - 1
}

func SampleGreedy(probs *Tensor) int {
	// Get the index of the maximum probability
	maxIndex := 0
	maxValue := probs.Data[0]
	for i, p := range probs.Data {
		if p > maxValue {
			maxValue = p
			maxIndex = i
		}
	}
	return maxIndex
}

// The higher the temperature, the more random the output, 0.7 is usually default
func SampleTemp(probs *Tensor, temperature float64) int {
	// Make a copy of the probabilities to avoid modifying the original
	adjustedProbs := make([]float64, len(probs.Data))
	copy(adjustedProbs, probs.Data)

	// Apply temperature scaling
	if temperature != 1.0 {
		// Lower temperature = more deterministic (higher values become more dominant)
		// Higher temperature = more random (probabilities become more uniform)
		sum := 0.0
		for i, p := range adjustedProbs {
			// Apply temperature by raising to power of 1/temperature
			adjustedProbs[i] = math.Pow(p, 1.0/temperature)
			sum += adjustedProbs[i]
		}

		// Re-normalize
		for i := range adjustedProbs {
			adjustedProbs[i] /= sum
		}
	}

	// Sample using the adjusted probabilities (same as your original Sample function)
	r := rand.Float64()
	cumulativeProb := 0.0
	for i, p := range adjustedProbs {
		cumulativeProb += p
		if r < cumulativeProb {
			return i
		}
	}

	return len(adjustedProbs) - 1
}
