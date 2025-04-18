package main

import (
	"fmt"
	"strings"

	"github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"

	"gptgo/data"
	"gptgo/pkg"
)

// Hyperparameters
const (
	blockSize        = 32
	embedSize        = 64
	heads            = 4
	layers           = 4
	epochs           = 50000
	learningRate     = 0.0001
	evalIters        = 1000
	dropout          = 0
	lossScale        = 1.00
	pretrainedTokens = 4000
)

var (
	Add          = variable.Add
	Softmax      = function.Softmax
	CrossEntropy = function.SoftmaxCrossEntropy
)

func main() {
	fmt.Println("Loading dataset...")
	dataset, vocabSize := data.Tokenize(pretrainedTokens)
	fmt.Printf("First 100 characters:\n%s\n", strings.TrimSpace(data.Decode(dataset[:100]...)))
	fmt.Printf("Vocabulary: %s\n", data.Characters())

	// Basic transformer components
	embeds := pkg.RandKaiming(vocabSize, embedSize)
	posEmbeds := pkg.RandKaiming(blockSize, embedSize)
	var blocks []*Block
	for range layers {
		blocks = append(blocks, NewBlock(embedSize, heads))
	}
	norm := pkg.NewLayerNorm(embedSize)
	lmHead := NewLinear(embedSize, vocabSize)

	// Collecting all the parameters
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
	fmt.Printf("bs=%d, es=%d, lr=%.4f, ls=%.2f, vs=%d, epochs=%d\n", blockSize, embedSize, learningRate, lossScale, vocabSize, epochs)
	for i := 0; i < epochs; i++ {
		// Inputs are indexes for embeds table
		inputs, targets := data.Sample(dataset, blockSize)

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
			fmt.Printf("epoch: %5d, loss: %.5f\n", i, loss.Data[0][0])
		}

		// Backward pass
		scaledLoss.Backward()

		// Weights update
		optimize.Update(params)
		params.ZeroGrad()
	}

	// Generate text
	variable.Config.Train = false // Prevent dropout
	context := "Magic forest"
	maxTokens := 500
	contextTokens := data.Encode(context)
	fmt.Printf("\n%s", context)
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
		decodedToken := data.Decode(nextToken)
		fmt.Printf(decodedToken)
		contextTokens = append(contextTokens, nextToken)
	}
}
