package main

import (
	"fmt"

	"github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

const (
	batchSize    = 16
	learningRate = 0.01
	embedSize    = 32
	headSize     = 16
)

var (
	AddC         = variable.AddC
	Add          = variable.Add
	SubC         = variable.SubC
	Sub          = variable.Sub
	MulC         = variable.MulC
	Mul          = variable.Mul
	DivC         = variable.DivC
	Div          = variable.Div
	Sin          = variable.Sin
	Cos          = variable.Cos
	Tanh         = variable.Tanh
	Exp          = variable.Exp
	Log          = variable.Log
	Pow          = variable.Pow
	Square       = variable.Square
	Neg          = variable.Neg
	Sum          = variable.Sum
	SumTo        = variable.SumTo
	BroadcastTo  = variable.BroadcastTo
	Reshape      = variable.Reshape
	Transpose    = variable.Transpose
	MatMul       = variable.MatMul
	Max          = variable.Max
	Min          = variable.Min
	Clip         = variable.Clip
	GetItem      = variable.GetItem
	CrossEntropy = function.SoftmaxCrossEntropy
)

// Embeddings are basically tensors under the hood
// What if we code-generate files for different tensors/linear layers
func main() {
	// Training loop
	sgd := func(a, b float64) float64 { return a - learningRate*b }

	data, vocabSize := Data()

	embeds := make([]*variable.Variable, vocabSize)
	for i := range embeds {
		embeds[i] = variable.Randn(1, embedSize)
	}

	lmHead := NewLinear(embedSize, vocabSize)

	// Inputs are indexes for embeds table
	inputs, targets := GetSequence(data.Data, len(data.Data)-1)

	// Main training loop
	lossSum := 0.0
	for i := 0; i < len(inputs.Data[0]); i++ {
		// Forward pass
		input := inputs.Data[0][i]
		target := targets.Data[0][i]

		embed := embeds[int(input)]
		logits := lmHead.Forward(embed)

		// Backward pass
		loss := CrossEntropy(logits, variable.New(target))
		loss.Backward()
		lossSum += loss.Data[0][0]

		if (i%batchSize) == 0 && i != 0 {
			// Update weights
			lmHead.Weight = variable.NewOf(matrix.F2(lmHead.Weight.Data, lmHead.WeightGrad.Data, sgd)...)
			lmHead.Bias = variable.NewOf(matrix.F2(lmHead.Bias.Data, lmHead.BiasGrad.Data, sgd)...)
			embeds[int(input)] = variable.NewOf(matrix.F2(embed.Data, embed.Grad.Data, sgd)...)

			if (i % (batchSize * 1000)) == 0 {
				fmt.Printf("Loss: %f\n", lossSum/float64(batchSize))
			}

			lossSum = 0.0
			lmHead.ZeroGrad()
			embeds[int(input)].Cleargrad()
		}
	}
	//}
	//
	//// Generate text
	//context := "A"
	//maxTokens := 500
	//token := int(Encode(context).First())
	//fmt.Println("\nGenerated text after training:")
	//for i := 0; i < maxTokens; i++ {
	//	embed := embeds.At(token)
	//	output := lmHead.Forward(embed)
	//	probs := Softmax(output)
	//	token = Sample(probs)
	//	decodedToken := Decode([]int{token})
	//	fmt.Printf(decodedToken)
	//}
}
