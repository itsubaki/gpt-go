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
	blockSize    = 8 * batchSize
	batchSize    = 32
	learningRate = 0.001
	embedSize    = 32
	epochs       = 10000
)

var (
	Add          = variable.Add
	MatMul       = variable.MatMul
	CrossEntropy = function.SoftmaxCrossEntropy
)

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

// Embeddings are basically tensors under the hood
// What if we code-generate files for different tensors/linear layers
func main() {
	rand.Seed(42)

	data, vocabSize := Data()

	embeds := RandKaiming(vocabSize, embedSize)
	lmHead := NewLinear(embedSize, vocabSize)

	params := make(layer.Parameters)
	params.Add("weights", lmHead.Weight)
	params.Add("bias", lmHead.Bias)
	params.Add("embeds", embeds)

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
		logits := lmHead.Forward(inputEmbeds)

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
	token := Encode(context).Data[0][0]
	fmt.Println("\nGenerated text after training:")
	for i := 0; i < maxTokens; i++ {
		embed := Rows(embeds, token)
		output := lmHead.Forward(embed)
		probs := function.Softmax(output)
		token = Sample(probs)
		decodedToken := Decode(token)
		fmt.Printf(decodedToken)
	}
}
