// Maybe move to tensor?
package main

import (
	"math"
	"math/rand"
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
func CrossEntropyLoss(logits *Tensor, target float64) float64 {
	normalizedLogits := Softmax(logits)
	// Compute log likelihood
	logitIndex := int(target)
	rating := math.Log(normalizedLogits.At(logitIndex).First())

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

//// SoftmaxWithDim applies softmax along the specified dimension
//// dim=0 applies softmax to each column, dim=1 (or -1) applies softmax to each row
//func SoftmaxWithDim(tensor *Tensor, dim int) *Tensor {
//	// Check dimensions
//	if len(tensor.Shape) != 2 {
//		panic("SoftmaxWithDim only supports 2D tensors")
//	}
//
//	// Handle negative dimensions
//	if dim < 0 {
//		dim = len(tensor.Shape) + dim
//	}
//
//	// Validate dimension
//	if dim != 0 && dim != 1 {
//		panic("Dimension must be 0 or 1 for 2D tensors")
//	}
//
//	rows, cols := tensor.Shape[0], tensor.Shape[1]
//	result := &Tensor{
//		Shape: []int{rows, cols},
//		Data:  make([]float64, rows*cols),
//	}
//
//	if dim == 1 { // Apply softmax to each row (equivalent to Softmax for 2D)
//		for i := 0; i < rows; i++ {
//			// Find max in this row for numerical stability
//			rowStart := i * cols
//			maxVal := tensor.Data[rowStart]
//			for j := 0; j < cols; j++ {
//				if tensor.Data[rowStart+j] > maxVal {
//					maxVal = tensor.Data[rowStart+j]
//				}
//			}
//
//			// Compute exp and sum for this row
//			sum := 0.0
//			for j := 0; j < cols; j++ {
//				expVal := math.Exp(tensor.Data[rowStart+j] - maxVal)
//				result.Data[rowStart+j] = expVal
//				sum += expVal
//			}
//
//			// Normalize this row
//			for j := 0; j < cols; j++ {
//				result.Data[rowStart+j] /= sum
//			}
//		}
//	} else { // dim == 0, apply softmax to each column
//		for j := 0; j < cols; j++ {
//			// Find max in this column for numerical stability
//			maxVal := tensor.Data[j]
//			for i := 0; i < rows; i++ {
//				if tensor.Data[i*cols+j] > maxVal {
//					maxVal = tensor.Data[i*cols+j]
//				}
//			}
//
//			// Compute exp and sum for this column
//			sum := 0.0
//			expVals := make([]float64, rows) // Temporary storage for exp values
//			for i := 0; i < rows; i++ {
//				expVal := math.Exp(tensor.Data[i*cols+j] - maxVal)
//				expVals[i] = expVal
//				sum += expVal
//			}
//
//			// Normalize this column
//			for i := 0; i < rows; i++ {
//				result.Data[i*cols+j] = expVals[i] / sum
//			}
//		}
//	}
//
//	return result
//}
