package main

import (
	"fmt"
	"strings"

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
	dropout          = 0.0  // disable some % of our neurons to prevent overfitting, model is likely to generalize
	lossScale        = 1.0  // we don't use batches, so scaling loss down may help better convergence
	pretrainedTokens = 3000 // how many of subword pretrained tokens to add on top of default character-based tokens
)

func main() {
	fmt.Println("Loading dataset...")
	dataset, vocabSize := data.Tokenize(pretrainedTokens)
	fmt.Printf("First characters:\n%s\n", strings.TrimSpace(data.Decode(dataset[:45]...)))
	fmt.Printf("Vocabulary: %s\n", data.Characters())

	// Basic transformer components.
	tokEmbeds := RandEmbeds(vocabSize, embedSize)
	posEmbeds := RandEmbeds(blockSize, embedSize)
	var blocks []*Block
	for range layers {
		blocks = append(blocks, NewBlock(embedSize, heads))
	}
	norm := NewLayerNorm(embedSize)
	lmHead := NewLinear(embedSize, vocabSize)

	// Collecting all the parameters.
	params := pkg.NewParams()
	params.Add(tokEmbeds, posEmbeds)
	for _, block := range blocks {
		params.Add(block.Params()...)
	}
	params.Add(norm.Scale, norm.Shift)
	params.Add(lmHead.Weight, lmHead.Bias)
	fmt.Println(params)

	optimizer := pkg.NewAdamW(learningRate)

	// Main training loop.
	fmt.Printf("bs=%d, es=%d, lr=%.4f, ls=%.2f, vs=%d, epochs=%d \n", blockSize, embedSize, learningRate, lossScale, vocabSize, epochs)
	for i := 0; i < epochs; i++ {
		// Targets contain the expected next token for each input token.
		input, targets := data.Sample(dataset, blockSize)

		// Forward pass, calculate predictions for every input token.
		embeds := Rows(tokEmbeds, input.Data[0]...) // get embed for every input token
		embeds = Add(embeds, posEmbeds)             // add positional embedding
		for _, block := range blocks {
			embeds = block.Forward(embeds)
		}
		embeds = norm.Forward(embeds)
		logits := lmHead.Forward(embeds) // converts contextual embeddings to next-token predictions

		// Loss calculation, how much our predicted targets differ from the expected targets?
		loss := CrossEntropy(logits, targets)
		loss = MulC(lossScale, loss)
		if (i % evalIters) == 0 {
			fmt.Printf("epoch: %5d, loss: %.5f\n", i, Val(loss)/lossScale)
		}

		// Backward pass, calculate the gradients (how much each parameter contributes to the loss)
		// for all the parameters (weights, biases, embeds). Loss is the tail of a computation graph.
		loss.Backward()
		// Nudge the parameters in the direction of the gradients, so to minimize the loss.
		optimizer.Update(params)
		params.ZeroGrad()
	}

	// Generate text
	pkg.DisableDropout()
	prompt := "Mysterious island"
	maxTokens := 500
	contextTokens := data.Encode(prompt)
	fmt.Printf("\n%s", prompt)
	for i := 0; i < maxTokens; i++ {
		if len(contextTokens) > blockSize {
			contextTokens = contextTokens[len(contextTokens)-blockSize:]
		}

		// Get embeddings for all tokens in context
		embeds := Rows(tokEmbeds, contextTokens...)
		embeds = Add(embeds, posEmbeds)
		for _, block := range blocks {
			embeds = block.Forward(embeds)
		}
		embeds = norm.Forward(embeds)
		logits := lmHead.Forward(embeds) // Get a list of final logits for the next token

		// We only care about the prediction for the next token, which is the last position
		lastTokenOutput := GetItem([]int{len(contextTokens) - 1})(logits)
		probs := Softmax(lastTokenOutput)
		nextToken := pkg.Sample(probs)
		decodedToken := data.Decode(nextToken)
		fmt.Printf(decodedToken)
		contextTokens = append(contextTokens, nextToken)
	}
}
