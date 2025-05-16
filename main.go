package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/zakirullin/gpt-go/data"
	"github.com/zakirullin/gpt-go/pkg"
)

// Hyperparameters
const (
	blockSize        = 32
	embedSize        = 64
	heads            = 4
	layers           = 4
	learningRate     = 0.0005
	steps            = 40000 // number of training steps, increase for better results
	evalSteps        = 1000  // evaluate loss once per every evalSteps
	dropout          = 0.0   // disable some % of our neurons to prevent overfitting, model is likely to generalize
	pretrainedTokens = 6000  // number of pretrained tokens to add on top of auto-detected characters
	maxTokens        = 50    // tokens limit for generation
)

func main() {
	// Skip training if "-chat" flag is provided.
	steps := steps
	chat := flag.Bool("chat", false, "Skip training and jump straight to chat")
	flag.Parse()
	if *chat {
		steps = -1
	}

	// Loading dataset and building vocabulary.
	fmt.Println("Tokenizing dataset...")
	dataset, vocabSize := data.Tokenize(pretrainedTokens)
	fmt.Printf("First characters:\n%s\n", strings.TrimSpace(data.Decode(dataset[:45]...)))
	fmt.Printf("Vocabulary: %s\n", data.Chars())
	fmt.Printf("Tokens in dataset: %.3fM\n", pkg.Millions(len(dataset)))

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
	params.TryLoadPretrained()
	fmt.Printf("Model size: %.3fM\n", pkg.Millions(params.Count()))

	// Training loop.
	losses := 0.0
	optimizer := pkg.NewAdamW(learningRate)
	fmt.Printf("bs=%d, es=%d, lr=%.4f, vs=%d, steps=%d\n", blockSize, embedSize, learningRate, vocabSize, steps)
	for i := range steps {
		// Targets contain the ground truth next token for each input token.
		input, targets := data.Sample(dataset, blockSize)

		// Forward pass, calculate predictions for every input token.
		embeds := Rows(tokEmbeds, Flat(input)...) // get embed for every input token
		embeds = Add(embeds, posEmbeds)           // add positional embedding
		for _, block := range blocks {            // self-attention and feed-forward
			embeds = block.Forward(embeds)
		}
		embeds = norm.Forward(embeds)
		logits := lmHead.Forward(embeds) // get scores for the next token for every context-enriched embed

		// Loss calculation, "how much our predicted targets differ from the ground truth targets?"
		loss := SoftmaxCrossEntropy(logits, targets)
		losses += Val(loss)
		fmt.Printf("\r%s", strings.Repeat("·", (i%evalSteps)*26/evalSteps)) // progress bar
		if i%evalSteps == 0 {
			avgLoss := losses / float64(min(i+1, evalSteps))
			fmt.Printf("\rstep: %5d, loss: %.4f\n", i, avgLoss)
			losses = 0
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
	// Training is done.

	// Predicts the next token based on the context of tokens.
	nextTok := func(context []float64) float64 {
		context = context[max(0, len(context)-blockSize):]

		// Feed context tokens to the model.
		embeds := Rows(tokEmbeds, context...)
		embeds = Add(embeds, posEmbeds)
		for _, block := range blocks {
			embeds = block.Forward(embeds)
		}
		embeds = norm.Forward(embeds)
		logits := lmHead.Forward(embeds) // get a list of final logits for the next token

		// We only care about the probabilities of the next token for the last token.
		logitsForNextToken := Rows(logits, -1)
		probs := Softmax(logitsForNextToken)
		tok := pkg.SampleTemp(probs, 0.8)

		return tok
	}

	// Sample from the model.
	prompt := "mysterious island"
	for {
		fmt.Printf("\n%s", prompt)
		context := data.Encode(prompt)
		for range maxTokens {
			nextToken := nextTok(context)
			fmt.Print(data.Decode(nextToken))
			context = append(context, nextToken)
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
