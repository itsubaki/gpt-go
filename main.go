package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
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
	epochs           = 20000
	learningRate     = 0.001
	evalIters        = 1000
	dropout          = 0.0  // disable some % of our neurons to prevent overfitting, model is likely to generalize
	lossScale        = 1.0  // we don't use batches, so scaling loss down may help better convergence
	pretrainedTokens = 5000 // how many of subword pretrained tokens to add on top of default character-based tokens
	maxTokens        = 200  // tokens limit for generation
)

func main() {
	// Skip training if -chat flag is set.
	epochs := epochs
	chat := flag.Bool("chat", false, "Skip training and jump straight to chat")
	flag.Parse()
	if *chat {
		epochs = -1
	}

	// Loading dataset and building vocabulary.
	fmt.Println("Tokenizing dataset...")
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
	params.Add(norm.Params()...)
	params.Add(lmHead.Params()...)
	params.LoadPretrainedIfExists()
	fmt.Printf("Model size: %s\n", params)

	// Training loop.
	fmt.Printf("bs=%d, es=%d, lr=%.4f, ls=%.2f, vs=%d, epochs=%d \n", blockSize, embedSize, learningRate, lossScale, vocabSize, epochs)
	optimizer := pkg.NewAdamW(learningRate)
	for i := 0; i <= epochs; i++ {
		// Targets contain the ground truth nextToken token for each input token.
		input, targets := data.Sample(dataset, blockSize)

		// Forward pass, calculate predictions for every input token.
		embeds := Rows(tokEmbeds, input.Data[0]...) // get embed for every input token
		embeds = Add(embeds, posEmbeds)             // add positional embedding
		for _, block := range blocks {
			embeds = block.Forward(embeds)
		}
		embeds = norm.Forward(embeds)
		logits := lmHead.Forward(embeds) // converts contextual embeddings to nextToken-token predictions

		// Loss calculation, how much our predicted targets differ from the ground truth targets?
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
	params.Save()
	pkg.DisableDropout()

	// Sample from the model.
	nextToken := func(tokens []float64) float64 {
		tokens = tokens[max(0, len(tokens)-blockSize):]

		// Get embeddings for all tokens in context.
		embeds := Rows(tokEmbeds, tokens...)
		embeds = Add(embeds, posEmbeds)
		for _, block := range blocks {
			embeds = block.Forward(embeds)
		}
		embeds = norm.Forward(embeds)
		logits := lmHead.Forward(embeds) // get a list of final logits for the nextToken token

		// We only care about the probabilities of the nextToken token for the last token.
		logitsForNextToken := Rows(logits, -1)
		probs := Softmax(logitsForNextToken)
		nextToken := pkg.Sample(probs)
		decodedToken := data.Decode(nextToken)
		fmt.Print(decodedToken)

		return nextToken
	}

	prompt := "Mysterious Island"
	for {
		fmt.Printf("\n%s", prompt)
		context := data.Encode(prompt)
		for i := 0; i < maxTokens; i++ {
			context = append(context, nextToken(context))
		}
		fmt.Print("\n$ ")
		scanner := bufio.NewScanner(os.Stdin)
		scanner.Scan()
		prompt = scanner.Text()
		if prompt == "exit" {
			fmt.Println("Bye!")
			break
		}
	}
}
