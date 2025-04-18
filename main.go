package main

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"

	"gptgo/data"
	"gptgo/pkg"
)

// Hyperparameters
const (
	blockSize        = 64
	embedSize        = 32
	heads            = 4
	layers           = 4
	epochs           = 4000
	learningRate     = 0.0005
	evalIters        = 1000
	dropout          = 0
	lossScale        = 1.00
	pretrainedTokens = 5000
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
	timings := make(map[string]time.Duration)
	counts := make(map[string]int)

	for i := 0; i < epochs; i++ {
		// Inputs are indexes for embeds table
		startTime := time.Now()
		inputs, targets := data.Sample(dataset, blockSize)
		timings["Sample"] += time.Since(startTime)
		counts["Sample"]++

		// Forward pass
		startTime = time.Now()
		inputEmbeds := pkg.Rows(embeds, inputs.Data[0]...) // Get embed for every input token
		timings["Rows"] += time.Since(startTime)
		counts["Rows"]++

		startTime = time.Now()
		input := Add(inputEmbeds, posEmbeds) // Add positional embedding, (blockSize, embedSize)
		timings["Add"] += time.Since(startTime)
		counts["Add"]++

		for j, block := range blocks {
			startTime = time.Now()
			input = block.Forward(input)
			blockName := fmt.Sprintf("Block-%d", j)
			timings[blockName] += time.Since(startTime)
			counts[blockName]++
		}

		startTime = time.Now()
		input = norm.Forward(input) // Normalize inputs
		timings["Norm"] += time.Since(startTime)
		counts["Norm"]++

		startTime = time.Now()
		logits := lmHead.Forward(input) // Get a list of final logits for the next token
		timings["LMHead"] += time.Since(startTime)
		counts["LMHead"]++

		// Loss calculation
		startTime = time.Now()
		loss := CrossEntropy(logits, targets)
		timings["CrossEntropy"] += time.Since(startTime)
		counts["CrossEntropy"]++

		startTime = time.Now()
		scaledLoss := variable.MulC(lossScale, loss)
		timings["MulC"] += time.Since(startTime)
		counts["MulC"]++

		if (i % evalIters) == 0 {
			fmt.Printf("%.5f, epoch: %d\n", loss.Data[0][0], i)
		}

		// Backward pass
		startTime = time.Now()
		scaledLoss.Backward()
		timings["Backward"] += time.Since(startTime)
		counts["Backward"]++

		// Weights update
		startTime = time.Now()
		optimize.Update(params)
		timings["Update"] += time.Since(startTime)
		counts["Update"]++

		params.ZeroGrad()
	}

	fmt.Println("\n--- Timing Statistics (milliseconds) ---")

	// Convert to sorted slice for output
	type TimingEntry struct {
		Name    string
		TotalMs int64
		AvgMs   float64
		Percent float64
	}

	var entries []TimingEntry
	totalTime := time.Duration(0)
	for _, timing := range timings {
		totalTime += timing
	}

	for name, timing := range timings {
		entries = append(entries, TimingEntry{
			Name:    name,
			TotalMs: timing.Milliseconds(),
			AvgMs:   float64(timing.Milliseconds()) / float64(counts[name]),
			Percent: float64(timing) / float64(totalTime) * 100,
		})
	}

	// Sort by total time (descending)
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].TotalMs > entries[j].TotalMs
	})

	// Print formatted table
	fmt.Printf("%-15s %-15s %-15s %-15s\n", "Operation", "Total (ms)", "Avg (ms)", "Percent (%)")
	fmt.Println("---------------------------------------------------------------")
	for _, entry := range entries {
		fmt.Printf("%-15s %-15d %-15.2f %-15.2f\n",
			entry.Name,
			entry.TotalMs,
			entry.AvgMs,
			entry.Percent)
	}
	fmt.Printf("\nTotal training time: %d ms\n", totalTime.Milliseconds())

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
