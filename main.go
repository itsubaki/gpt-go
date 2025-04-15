package main

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/autograd/variable"
)

const (
	blockSize    = 8 * 32 // We don't have batches, so we increase blockSize for convergence
	learningRate = 0.001
	embedSize    = 32
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
	mulHead := NewMultiHeadAttention(numHeads, embedSize, embedSize/numHeads)
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
	context := "Ali"
	maxTokens := 500
	contextTokens := Encode(context).Data[0]
	fmt.Println("\nGenerated text after training:")
	fmt.Printf(context)
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
