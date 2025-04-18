package main

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"

	"gptgo/pkg"
)

// Hyperparameters
const (
	blockSize    = 32 // We don't have batches, so we increase blockSize for convergence
	embedSize    = 64
	numHeads     = 4
	numLayers    = 4
	epochs       = 20000
	learningRate = 0.005
	evalIters    = 1000
	dropout      = 0
	lossScale    = 1.0
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

	embeds := pkg.RandKaiming(vocabSize, embedSize)
	posEmbeds := pkg.RandKaiming(blockSize, embedSize)
	var blocks []*Block
	for range numLayers {
		blocks = append(blocks, NewBlock(embedSize, numHeads))
	}
	norm := pkg.NewLayerNorm(embedSize)
	lmHead := NewLinear(embedSize, vocabSize)

	params := pkg.NewParams()
	params.Add(embeds, posEmbeds)
	for _, block := range blocks {
		params.Add(block.Params()...)
	}
	params.Add(norm.Scale, norm.Shift)
	params.Add(lmHead.Weight, lmHead.Bias)
	fmt.Println(params)

	optimize := pkg.AdamW{
		Alpha: learningRate,
		Beta1: 0.9,
		Beta2: 0.999,
	}

	// Main training loop
	fmt.Printf("bs=%d, es=%d, lr=%.4f, ls=%.2f, epochs=%d\n", blockSize, embedSize, learningRate, lossScale, epochs)
	for i := 0; i < epochs; i++ {
		// Inputs are indexes for embeds table
		inputs, targets := TrainingSequence(data.Data[0], blockSize)

		// Forward pass
		inputEmbeds := pkg.Rows(embeds, inputs.Data[0]...) // Get embed for every input token
		input := Add(inputEmbeds, posEmbeds)               // Add positional embedding, (blockSize, embedSize)
		for _, block := range blocks {
			input = block.Forward(input)
		}
		input = norm.Forward(input)     // Normalize inputs
		logits := lmHead.Forward(input) // Get a list of final logits for the next token

		// Loss calculation
		loss := CrossEntropy(logits, targets)
		scaledLoss := variable.MulC(lossScale, loss)
		if (i % evalIters) == 0 {
			fmt.Printf("%.5f, epoch: %d\n", loss.Data[0][0], i)
		}

		// Backward pass
		scaledLoss.Backward()
		optimize.Update(params)
		params.ZeroGrad()
	}

	// Generate text
	variable.Config.Train = false // Prevent dropout
	context := "Mysterious Island"
	maxTokens := 500
	contextTokens := Encode(context).Data[0]
	fmt.Printf(context)
	for i := 0; i < maxTokens; i++ {
		if len(contextTokens) > blockSize {
			contextTokens = contextTokens[len(contextTokens)-blockSize:]
		}

		// Get embeddings for all tokens in context
		inputEmbeds := pkg.Rows(embeds, contextTokens...)
		input := Add(inputEmbeds, posEmbeds)
		for _, block := range blocks {
			input = block.Forward(input)
		}
		input = norm.Forward(input)
		logits := lmHead.Forward(input) // Get a list of final logits for the next token

		// We only care about the prediction for the next token, which is the last position
		lastTokenOutput := variable.GetItem([]int{len(contextTokens) - 1})(logits)

		probs := Softmax(lastTokenOutput)
		nextToken := pkg.Sample(probs)

		decodedToken := Decode(nextToken)
		fmt.Printf(decodedToken)

		contextTokens = append(contextTokens, nextToken)
	}
}
