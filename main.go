package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"

	"gptgo/pkg"
)

const (
	blockSize    = 64 // We don't have batches, so we increase blockSize for convergence
	batchSize    = 32 // Emulated batch
	embedSize    = 64
	numHeads     = 4
	numLayers    = 4
	epochs       = 20000
	learningRate = 0.003
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

	embeds := pkg.RandKaiming(vocabSize, embedSize)
	posEmbeds := pkg.RandKaiming(blockSize, embedSize)
	var blocks []*Block
	for range numLayers {
		blocks = append(blocks, NewBlock(embedSize, numHeads))
	}
	norm := pkg.NewLayerNorm(embedSize)
	lmHead := pkg.NewLinear(embedSize, vocabSize)

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
	for i := 0; i < epochs; i++ {
		// Inputs are indexes for embeds table
		inputs, targets := GetSequence(data.Data[0], blockSize)

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
		//scaledLoss := variable.MulC(1.0/float64(batchSize), loss)
		if (i % evalIters) == 0 {
			fmt.Println(loss.Data[0][0])
		}

		// Backward pass
		loss.Backward()
		//ClipGradByNorm(0.3, params)
		optimize.Update(params)
		params.ZeroGrad()
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

// 2. Add gradient clipping
func ClipGradByNorm(maxNorm float64, p layer.Parameters) {
	for _, param := range p {
		if param.Grad == nil {
			continue
		}

		// Calculate gradient norm
		gradNormSquared := 0.0
		for i := range param.Grad.Data {
			for j := range param.Grad.Data[i] {
				gradNormSquared += param.Grad.Data[i][j] * param.Grad.Data[i][j]
			}
		}
		gradNorm := math.Sqrt(gradNormSquared)

		// Clip if needed
		if gradNorm > maxNorm {
			scale := maxNorm / gradNorm
			for i := range param.Grad.Data {
				for j := range param.Grad.Data[i] {
					param.Grad.Data[i][j] *= scale
				}
			}
		}
	}
}
