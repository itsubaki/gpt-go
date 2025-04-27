package pkg

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

var (
	Add   = variable.Add
	Div   = variable.Div
	Zeros = variable.Zero
)

// Sample returns a random index based on the given probabilities.
func Sample(probs *variable.Variable) float64 {
	r := rand.Float64()

	// Find the first index where cumulative probability exceeds r
	cumulativeProb := 0.0
	for i, p := range probs.Data[0] {
		cumulativeProb += p
		if r < cumulativeProb {
			return float64(i)
		}
	}

	// Fallback (should rarely happen due to floating point precision)
	return float64(len(probs.Data)) - 1
}

// SampleTemp returns a random index based on the given probabilities and temperature.
// The higher the temperature, the more random the sampling.
// Usually, temperature is between 0.5 and 0.8.
func SampleTemp(probs *variable.Variable, temperature float64) float64 {
	adjustedProbs := make([]float64, len(probs.Data[0]))
	copy(adjustedProbs, probs.Data[0])
	if temperature != 1.0 {
		// Lower temperature: higher probs amplified, lower reduced, more deterministic
		// Higher temperature: probabilities become more uniform, more random
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

	return Sample(variable.NewOf(adjustedProbs))
}

// SampleMax returns the index of the maximum value.
// I use that to verify that the model has remembered some patterns.
func SampleMax(probs *variable.Variable) float64 {
	maxProb := -1.0
	maxIndex := 0
	for i, p := range probs.Data[0] {
		if p > maxProb {
			maxProb = p
			maxIndex = i
		}
	}

	return float64(maxIndex)
}

// Returns rows at specified indexes. Negative indexes return rows from the end.
func Rows(x *variable.Variable, indexes ...float64) *variable.Variable {
	size := len(x.Data)

	var intIndexes []int
	for _, index := range indexes {
		intIndex := int(index)
		if intIndex < 0 {
			intIndex = size + intIndex
		}

		intIndexes = append(intIndexes, intIndex)
	}

	return (&variable.Function{Forwarder: &variable.GetItemT{Slices: intIndexes}}).First(x)
}

// Add tests
func RandKaiming(rows, cols int) *variable.Variable {
	// Calculate the standard deviation based on Kaiming/He formula
	// For ReLU activation, we use a factor of 2.0 divided by fan-in (input dimensions)
	sigma := math.Sqrt(2.0 / float64(cols))

	// Define a transformation function that samples from a normal distribution
	// and scales values by the calculated standard deviation
	rnd := func(_ float64) float64 {
		return rand.NormFloat64() * sigma
	}

	result := matrix.F(matrix.Zero(rows, cols), rnd)

	return variable.NewOf(result...)
}

// Only works with 2D tensors
func Tril(m *variable.Variable) *variable.Variable {
	result := variable.ZeroLike(m)
	for i := 0; i < len(m.Data); i++ {
		for j := 0; j < len(m.Data[i]); j++ {
			if j <= i {
				result.Data[i][j] = m.Data[i][j]
			}
		}
	}

	return result
}

// The result would be added to computation graph and tied to m
func MaskedInfFill(m, mask *variable.Variable) *variable.Variable {
	negInfMaskedData := matrix.F2(m.Data, mask.Data, func(a, b float64) float64 {
		if b == 0 {
			return math.Inf(-1)
		}

		return a
	})
	mMasked := Add(variable.Mul(m, mask), variable.NewOf(negInfMaskedData...))

	return mMasked
}

func PrintShape(v *variable.Variable) {
	fmt.Printf("(%d, %d)\n", len(v.Data), len(v.Data[0]))
}

// Returns a matrix of 1s
func Ones(m, n int) *variable.Variable {
	out := make([][]float64, m)
	for i := range m {
		out[i] = make([]float64, n)
		for j := range n {
			out[i][j] = 1.0
		}
	}

	return variable.NewOf(out...)
}

func DisableDropout() {
	variable.Config.Train = false // Prevent dropout
}

// Returns the first element of the variable
func Val(x *variable.Variable) float64 {
	return x.Data[0][0]
}
