package main

import (
	"fmt"

	"github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

const (
	blockSize    = 8
	batchSize    = 16
	learningRate = 0.005
	embedSize    = 32
	epochs       = 1000000
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
	RandN        = variable.Randn
	CrossEntropy = function.SoftmaxCrossEntropy
)

// Embeddings are basically tensors under the hood
// What if we code-generate files for different tensors/linear layers
func main() {
	sgd := func(a, b float64) float64 { return a - (learningRate*(b))/batchSize }

	data, vocabSize := Data()

	embeds := RandN(vocabSize, embedSize)
	lmHead := NewLinear(embedSize, vocabSize)

	// Main training loop
	lossSum := 0.0
	for i := 0; i < epochs; i++ {
		// Inputs are indexes for embeds table
		inputs, targets := GetSequence(data.Data, blockSize)

		// Forward pass
		inputEmbeds := variable.Zero(len(inputs.Data[0]), embedSize)
		for j := range inputEmbeds.Data {
			inputEmbeds.Data[j] = embeds.Data[int(inputs.Data[0][j])]
		}

		logits := lmHead.Forward(inputEmbeds)

		// Backward pass
		loss := CrossEntropy(logits, targets)
		loss.Backward()
		fmt.Println(loss)
		lossSum += loss.Data[0][0]

		if (i*len(inputs.Data[0])%batchSize) == 0 && i != 0 {
			// Update weights
			lmHead.Weight = variable.NewOf(matrix.F2(lmHead.Weight.Data, lmHead.Weight.Grad.Data, sgd)...)
			lmHead.Bias = variable.NewOf(matrix.F2(lmHead.Bias.Data, lmHead.Bias.Grad.Data, sgd)...)
			//for j := range embeds {
			//	if embeds[j].Grad != nil {
			//		embeds[j].Grad.Data = matrix.Clip(embeds[j].Grad.Data, -0.5, 0.5)
			//		embeds[j] = variable.NewOf(matrix.F2(embeds[j].Data, embeds[j].Grad.Data, sgd)...)
			//		embeds[j].Cleargrad()
			//	}
			//}

			if (i * len(inputs.Data[0]) % (1000)) == 0 {
				fmt.Printf("Loss: %f\n", lossSum/float64(batchSize))
			}

			lossSum = 0.0
			lmHead.ZeroGrad()
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
