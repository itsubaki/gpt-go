// Maybe move to tensor?
package main

import (
	"math"
	"math/rand"

	"github.com/itsubaki/autograd/variable"
)

// Softmax applies the softmax function to a tensor
// For 1D tensors, it applies softmax to the entire vector
// For 2D tensors, it applies softmax to each row (dim=-1 in PyTorch)
func Softmax(tensor *Tensor) *Tensor {
	// Check dimensions
	dims := len(tensor.Shape)
	if dims != 1 && dims != 2 {
		panic("Softmax only supports 1D and 2D tensors")
	}

	result := &Tensor{
		Shape: make([]int, len(tensor.Shape)),
		Data:  make([]float64, len(tensor.Data)),
	}
	copy(result.Shape, tensor.Shape)

	if dims == 1 {
		// Apply softmax to 1D tensor
		vec := tensor.Data
		var sum = 0.0
		var maxVal = vec[0]

		// Find max for numerical stability
		for _, v := range vec {
			if v > maxVal {
				maxVal = v
			}
		}

		// Compute exp and sum
		for i, v := range vec {
			expVal := math.Exp(v - maxVal)
			result.Data[i] = expVal
			sum += expVal
		}

		// Normalize
		for i := range result.Data {
			result.Data[i] /= sum
		}
	} else { // dims == 2
		// Apply softmax to each row of 2D tensor
		rows, cols := tensor.Shape[0], tensor.Shape[1]

		for i := 0; i < rows; i++ {
			// Find max in this row for numerical stability
			rowStart := i * cols
			maxVal := tensor.Data[rowStart]
			for j := 0; j < cols; j++ {
				if tensor.Data[rowStart+j] > maxVal {
					maxVal = tensor.Data[rowStart+j]
				}
			}

			// Compute exp and sum for this row
			sum := 0.0
			for j := 0; j < cols; j++ {
				expVal := math.Exp(tensor.Data[rowStart+j] - maxVal)
				result.Data[rowStart+j] = expVal
				sum += expVal
			}

			// Normalize this row
			for j := 0; j < cols; j++ {
				result.Data[rowStart+j] /= sum
			}
		}
	}

	return result
}

// CrossEntropyLoss computes the cross-entropy loss between logits and targets.
// THIS is wrong and numerically non-stable implementation
func CrossEntropyLoss(logits *Tensor, target float64) float64 {
	normalizedLogits := Softmax(logits)
	// Compute log likelihood
	logitIndex := int(target)
	rating := math.Log(normalizedLogits.At(logitIndex).First())

	// The higher rating the better, the loss is opposite :)
	loss := -rating

	return loss
}

func Sample(probs *variable.Variable) int {
	// Get probabilities as slice
	// Generate random number in [0,1)
	r := rand.Float64()

	// Find the first index where cumulative probability exceeds r
	cumulativeProb := 0.0
	for i, p := range probs.Data[0] {
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
