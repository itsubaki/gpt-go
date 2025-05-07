package pkg

import (
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

	// Find the first index where cumulative probability exceeds r.
	cumulativeProb := 0.0
	for i, p := range probs.Data[0] {
		cumulativeProb += p
		if r < cumulativeProb {
			return float64(i)
		}
	}

	return float64(len(probs.Data)) - 1
}

// SampleTemp returns a random index based on the given probabilities and temperature.
// The higher the temperature, the more random the sampling.
// Usually, temperature is between 0.5 and 0.8.
func SampleTemp(probs *variable.Variable, temperature float64) float64 {
	adjustedProbs := make([]float64, len(probs.Data[0]))
	copy(adjustedProbs, probs.Data[0])
	if temperature != 1.0 {
		// Lower temperature: higher probs amplified, lower reduced, more deterministic.
		// Higher temperature: probabilities become more uniform, more random.
		sum := 0.0
		for i, p := range adjustedProbs {
			// Apply temperature by raising to power of 1/temperature.
			adjustedProbs[i] = math.Pow(p, 1.0/temperature)
			sum += adjustedProbs[i]
		}

		for i := range adjustedProbs {
			adjustedProbs[i] /= sum
		}
	}

	return Sample(variable.NewOf(adjustedProbs))
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

// Returns a matrix of random values from a normal distribution.
func Normal(rows, cols int) *variable.Variable {
	rnd := func(_ float64) float64 {
		// Standard deviation = 0.02 is widely used in transformer models like GPT-2.
		// It prevents too large values in the beginning of training.
		std := 0.02
		return rand.NormFloat64() * std
	}

	m := matrix.Zero(rows, cols)
	m = matrix.F(m, rnd)

	return variable.NewOf(m...)
}

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

// The result would be added to computation graph and tied to m.
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

// Returns a matrix of ones.
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
	variable.Config.Train = false // disables dropout
}

// Returns the first element of the variable.
func Val(x *variable.Variable) float64 {
	return x.Data[0][0]
}

func Flat(x *variable.Variable) []float64 {
	return matrix.Flatten(x.Data)
}

func Millions(num int) float64 {
	return float64(num) / 1e6
}
