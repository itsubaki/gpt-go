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
	learningRate = 0.001
	embedSize    = 32
	numHeads     = 4
	epochs       = 100000
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

	// Parameters for gradient accumulation
	virtualBatchSize := 32 // Target batch size to emulate
	actualBatchSize := 1   // Current batch size (single sample)
	accumSteps := virtualBatchSize / actualBatchSize
	accumCount := 0

	// Main training loop
	for i := 0; i < epochs; i++ {
		// Zero gradients only at the start of a virtual batch
		if accumCount == 0 {
			params.Cleargrads()
		}

		// Inputs are indexes for embeds table
		inputs, targets := GetSequence(data.Data[0], blockSize)

		// Forward pass
		inputEmbeds := Rows(embeds, inputs.Data[0]...)
		input := Add(inputEmbeds, posEmbeds)

		features := mulHead.Forward(input)
		logits := lmHead.Forward(features)

		// Compute loss
		loss := CrossEntropy(logits, targets)

		// Scale the loss to maintain proper gradient magnitudes
		scaledLoss := variable.MulC(1.0/float64(accumSteps), loss)

		// Backward pass with scaled loss
		scaledLoss.Backward()

		// Print original loss (not scaled) for monitoring
		if (i % 100) == 0 {
			fmt.Println(loss.Data[0][0])
		}

		// Update accumulation counter
		accumCount++

		// Only update weights after accumulating gradients from accumSteps samples
		if accumCount == accumSteps {
			// Update weights using accumulated gradients
			optimize.Update(Model{params})
			// Reset counter
			accumCount = 0
		}
	}

	// Handle case where epochs doesn't divide evenly by accumSteps
	if accumCount > 0 {
		optimize.Update(Model{params})
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

		features := mulHead.Forward(input)
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
