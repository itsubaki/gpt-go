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
	blockSize    = 64 // We don't have batches, so we increase blockSize for convergence
	embedSize    = 64
	numHeads     = 4
	epochs       = 40000
	learningRate = 0.005
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
	ReLU         = function.ReLU
)

// Embeddings are basically tensors under the hood
// What if we code-generate files for different tensors/linear layers
func main() {
	rand.Seed(42)

	data, vocabSize := Data()

	embeds := RandKaiming(vocabSize, embedSize)
	posEmbeds := RandKaiming(blockSize, embedSize)
	blocks := []*Block{
		NewBlock(embedSize, numHeads),
	}
	lmHead := NewLinear(embedSize, vocabSize)

	params := make(layer.Parameters)
	params.Add("weights", lmHead.Weight)
	params.Add("bias", lmHead.Bias)
	params.Add("embeds", embeds)
	params.Add("posEmbeds", posEmbeds)
	for i, block := range blocks {
		for j, param := range block.Params() {
			params.Add(fmt.Sprintf("%d-%d#block", i, j), param)
		}
	}

	optimize := optimizer.Adam{
		Alpha: learningRate,
		Beta1: 0.9,
		Beta2: 0.999,
	}

	// Parameters for gradient accumulation
	virtualBatchSize := 32 // Target batch size to emulate
	accumCount := 0

	// Main training loop
	for i := 0; i < epochs; i++ {
		// Inputs are indexes for embeds table
		inputs, targets := GetSequence(data.Data[0], blockSize)

		// Forward pass
		inputEmbeds := Rows(embeds, inputs.Data[0]...) // Get embed for every input token
		input := Add(inputEmbeds, posEmbeds)           // Add positional embedding, (blockSize, embedSize)
		features := blocks[0].Forward(input)
		logits := lmHead.Forward(features) // Get a list of final probabilities for the next token

		// Loss calculation
		loss := CrossEntropy(logits, targets)
		scaledLoss := variable.MulC(1.0/float64(virtualBatchSize), loss)
		if (i % 100) == 0 {
			fmt.Println(loss.Data[0][0])
		}

		// Backward pass
		scaledLoss.Backward()
		accumCount++
		if accumCount == virtualBatchSize {
			optimize.Update(Model{params})
			accumCount = 0
			params.Cleargrads()
		}
	}

	// Generate text
	context := "Alibab i"
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
		input := Add(inputEmbeds, posEmbeds)
		features := blocks[0].Forward(input)
		output := lmHead.Forward(features)

		// We only care about the prediction for the next token, which is the last position
		lastTokenOutput := variable.GetItem([]int{len(contextTokens) - 1})(output)

		probs := function.Softmax(lastTokenOutput)
		nextToken := Sample(probs)

		decodedToken := Decode(nextToken)
		fmt.Printf(decodedToken)

		contextTokens = append(contextTokens, nextToken)
	}
}
