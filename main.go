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
	blockSize    = 32 // We don't have batches, so we increase blockSize for convergence
	batchSize    = 32 // Emulated batch
	embedSize    = 64
	numHeads     = 4
	numLayers    = 4
	epochs       = 22000
	learningRate = 0.0003
	evalIters    = 100
	dropout      = 0.0
)

var (
	Add          = variable.Add
	MatMul       = variable.MatMul
	Zeros        = variable.Zero
	OneLike      = variable.OneLike
	Softmax      = function.Softmax
	CrossEntropy = function.SoftmaxCrossEntropy
	ReLU         = function.ReLU
	Dropout      = function.DropoutSimple
)

// Embeddings are basically tensors under the hood
// What if we code-generate files for different tensors/linear layers
func main() {
	rand.Seed(42)
	data, vocabSize := Data()

	embeds := RandKaiming(vocabSize, embedSize)
	posEmbeds := RandKaiming(blockSize, embedSize)
	var blocks []*Block
	for range numLayers {
		blocks = append(blocks, NewBlock(embedSize, numHeads))
	}
	norm := NewLayerNorm(embedSize)
	lmHead := NewLinear(embedSize, vocabSize)

	params := make(layer.Parameters)
	params.Add("embeds", embeds)
	params.Add("posEmbeds", posEmbeds)
	for i, block := range blocks {
		for j, param := range block.Params() {
			params.Add(fmt.Sprintf("%d-%d#block", i, j), param)
		}
	}
	params.Add("normScale", norm.Scale)
	params.Add("normShift", norm.Shift)
	params.Add("weights", lmHead.Weight)
	params.Add("bias", lmHead.Bias)

	numParams := 0
	for _, param := range params {
		numParams += len(param.Data) * len(param.Data[0])
	}
	fmt.Printf("%.3fM parameters\n", float64(numParams)/1e6)

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
		inputEmbeds := Rows(embeds, inputs.Data[0]...) // Get embed for every input token
		input := Add(inputEmbeds, posEmbeds)           // Add positional embedding, (blockSize, embedSize)
		for _, block := range blocks {
			input = block.Forward(input)
		}
		input = norm.Forward(input)     // Normalize inputs
		logits := lmHead.Forward(input) // Get a list of final logits for the next token

		// Loss calculation
		loss := CrossEntropy(logits, targets)
		if (i % evalIters) == 0 {
			fmt.Println(loss.Data[0][0])
		}

		// Backward pass
		loss.Backward()
		optimize.Update(Model{params})
		params.Cleargrads()
	}

	// Generate text
	variable.Config.Train = false // Prevent dropout
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
		for _, block := range blocks {
			input = block.Forward(input)
		}
		input = norm.Forward(input)
		logits := lmHead.Forward(input) // Get a list of final logits for the next token

		// We only care about the prediction for the next token, which is the last position
		lastTokenOutput := variable.GetItem([]int{len(contextTokens) - 1})(logits)

		probs := Softmax(lastTokenOutput)
		nextToken := Sample(probs)

		decodedToken := Decode(nextToken)
		fmt.Printf(decodedToken)

		contextTokens = append(contextTokens, nextToken)
	}
}
