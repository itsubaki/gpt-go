package main

import (
	"fmt"
	"strings"

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
	epochs           = 40000
	learningRate     = 0.0005
	evalIters        = 1000
	dropout          = 0.2  // disable some % of our neurons to prevent overfitting, model is likely to generalize
	lossScale        = 1.00 // we don't use batches, so scaling loss down may help better convergence
	pretrainedTokens = 3500
)

func main() {
	fmt.Println("Loading dataset...")
	dataset, vocabSize := data.Tokenize(pretrainedTokens)
	fmt.Printf("First 100 characters:\n%s\n", strings.TrimSpace(data.Decode(dataset[:100]...)))
	fmt.Printf("Vocabulary: %s\n", data.Characters())

	// Basic transformer components
	tokEmbeds := pkg.RandKaiming(vocabSize, embedSize)
	posEmbeds := pkg.RandKaiming(blockSize, embedSize)
	var blocks []*Block
	for range layers {
		blocks = append(blocks, NewBlock(embedSize, heads))
	}
	norm := pkg.NewLayerNorm(embedSize)
	lmHead := NewLinear(embedSize, vocabSize)

	// Collecting all the parameters
	params := pkg.NewParams()
	params.Add(tokEmbeds, posEmbeds)
	for _, block := range blocks {
		params.Add(block.Params()...)
	}
	params.Add(norm.Scale, norm.Shift)
	params.Add(lmHead.Weight, lmHead.Bias)
	fmt.Println(params)

	optimize := pkg.AdamW{
		Alpha:       learningRate,
		Beta1:       0.9,
		Beta2:       0.999,
		WeightDecay: 0.01,
	}

	// Main training loop
	fmt.Printf("bs=%d, es=%d, lr=%.4f, ls=%.2f, vs=%d, epochs=%d \n", blockSize, embedSize, learningRate, lossScale, vocabSize, epochs)
	for i := 0; i < epochs; i++ {
		// Input contains blockSize consecutive tokens.
		// Targets contains the target for each input token.
		// Example: for input=[0,1], targets=[1,2], meaning
		// that next token after 0 is 1, next after 1 is 2.
		input, targets := data.Sample(dataset, blockSize)

		// Forward pass, calculate predictions for every input token.
		// embeds is:
		// [
		//   [vector for tok=1],
		//   [vector for tok=2],
		//   ...
		// ]
		embeds := pkg.Rows(tokEmbeds, input.Data[0]...) // Get embed for every input token
		embeds = Add(embeds, posEmbeds)                 // Add positional embedding
		for _, block := range blocks {
			embeds = block.Forward(embeds)
		}
		embeds = norm.Forward(embeds) // Normalize embeds
		// logits is:
		// [
		//   [score for tok=0, ..., score for tok=3], // for input tok=0
		//   [score for tok=0, ..., score for tok=4], // for input tok=1
		//   ...
		// ]
		logits := lmHead.Forward(embeds)

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
		inputEmbeds := pkg.Rows(tokEmbeds, contextTokens...)
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
