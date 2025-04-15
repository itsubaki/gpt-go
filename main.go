package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/autograd/variable"
	"gonum.org/v1/gonum/stat/distuv"
)

const (
	blockSize    = 64 // We don't have batches, so increase blockSize instead for better convergence
	learningRate = 0.001
	embedSize    = 32
	headSize     = 8
	numHeads     = 4
	epochs       = 10000
)

var (
	Add          = variable.Add
	MatMul       = variable.MatMul
	RandN        = variable.Randn
	Zeros        = variable.Zero
	ZeroLike     = variable.ZeroLike
	OneLike      = variable.OneLike
	Softmax      = function.Softmax
	CrossEntropy = function.SoftmaxCrossEntropy
)

// Embeddings are basically tensors under the hood
// What if we code-generate files for different tensors/linear layers
func main() {
	rand.Seed(42)

	data, vocabSize := Data()

	embeds := RandKaiming(vocabSize, embedSize)
	posEmbeds := RandKaiming(blockSize, embedSize)
	mulHead := NewMultiHeadAttention(numHeads, embedSize, headSize)
	lmHead := NewLinear(embedSize, vocabSize)

	params := make(layer.Parameters)
	params.Add("weights", lmHead.Weight)
	params.Add("bias", lmHead.Bias)
	params.Add("embeds", embeds)
	params.Add("posEmbeds", posEmbeds)
	for i, param := range mulHead.Params() {
		params.Add(fmt.Sprintf("%d#mulHead", i), param)
	}

	optimize := optimizer.Adam{
		Alpha: learningRate,
		Beta1: 0.9,
		Beta2: 0.999,
	}

	// Main training loop
	for i := 0; i < epochs; i++ {
		// Inputs are indexes for embeds table
		inputs, targets := GetSequence(data.Data[0], blockSize)

		// Forward pass
		inputEmbeds := Rows(embeds, inputs.Data[0]...)
		inputPosEmbeds := Rows(posEmbeds, Arange(blockSize)...)
		x := Add(inputEmbeds, inputPosEmbeds)

		features := mulHead.Forward(x)
		logits := lmHead.Forward(features)

		// Backward pass
		loss := CrossEntropy(logits, targets)
		loss.Backward()
		if (i % 100) == 0 {
			fmt.Println(loss.Data[0][0])
		}

		// Update weights
		optimize.Update(Model{params})
		params.Cleargrads()
	}

	// Generate text
	context := "A"
	maxTokens := 500
	contextTokens := Encode(context).Data[0]
	fmt.Println("\nGenerated text after training:")
	for i := 0; i < maxTokens; i++ {
		if len(contextTokens) > blockSize {
			contextTokens = contextTokens[len(contextTokens)-blockSize:]
		}

		// Get embeddings for all tokens in context
		inputEmbeds := Rows(embeds, contextTokens...)

		features := mulHead.Forward(inputEmbeds)
		output := lmHead.Forward(features)

		// We only care about the prediction for the next token, which is the last position
		lastTokenOutput := variable.GetItem([]int{len(contextTokens) - 1})(output)

		probs := function.Softmax(lastTokenOutput)
		nextToken := Sample(probs)

		decodedToken := Decode(nextToken)
		fmt.Printf(decodedToken)

		contextTokens = append(contextTokens, float64(nextToken))
	}
}

type Model struct {
	params layer.Parameters
}

func (m Model) Params() layer.Parameters {
	return m.params
}

func Rows(x *variable.Variable, indexes ...float64) *variable.Variable {
	var intIndexes []int
	for _, index := range indexes {
		intIndexes = append(intIndexes, int(index))
	}

	return (&variable.Function{Forwarder: &variable.GetItemT{Slices: intIndexes}}).First(x)
}

// Add tests
func RandKaiming(dims ...int) *variable.Variable {
	sigma := math.Sqrt(2.0 / float64(dims[1]))
	dist := distuv.Normal{Mu: 0, Sigma: sigma}
	result := matrix.F(matrix.Zero(dims[0], dims[1]), func(_ float64) float64 { return dist.Rand() })

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

// Arange creates a new slice containing a sequence of values from start to end (exclusive) with the given step.
// If step is not provided, it defaults to 1.
func Arange(end int) []float64 {
	step := 1.0

	// Calculate the number of elements
	n := int(math.Ceil((float64(end)) / step))
	if n <= 0 {
		return []float64{}
	}

	result := make([]float64, n)
	for i := 0; i < n; i++ {
		result[i] = float64(i) * step
	}

	return result
}

func PrintShape(v *variable.Variable) {
	fmt.Printf("(%d, %d)\n", len(v.Data), len(v.Data[0]))
}
