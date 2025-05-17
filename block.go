package main

import (
	"github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
	"github.com/zakirullin/gpt-go/pkg"
)

var (
	Zeros               = variable.Zero
	Ones                = pkg.Ones
	ReLU                = function.ReLU
	Dropout             = function.DropoutSimple
	MatMul              = pkg.MatMul
	Add                 = variable.Add
	Sub                 = variable.Sub
	MulC                = variable.MulC
	Transpose           = variable.Transpose
	Softmax             = function.Softmax
	SoftmaxCrossEntropy = function.SoftmaxCrossEntropy
	RandEmbeds          = pkg.Normal
	Rows                = pkg.Rows
	Val                 = pkg.Val
	Flat                = pkg.Flat
)

type Block struct {
	embedSize int
	headCount int
	saHead    *MultiHeadAttention
	ffwd      *Linear
	ffwdProj  *Linear
	norm1     *LayerNorm
	norm2     *LayerNorm
}

func NewBlock(embedSize, numHeads int) *Block {
	return &Block{
		embedSize: embedSize,
		headCount: numHeads,
		saHead:    NewMultiHeadAttention(embedSize, numHeads),
		ffwd:      NewLinear(embedSize, embedSize*4),
		ffwdProj:  NewLinear(embedSize*4, embedSize),
		norm1:     NewLayerNorm(embedSize),
		norm2:     NewLayerNorm(embedSize),
	}
}

func (b *Block) Forward(input *variable.Variable) *variable.Variable {
	// Self-attention with residual connections. Input is our highway, we allow the gradient to flow back unimpeded.
	input = b.norm1.Forward(input)   // Normalize input (mean=0, var=1), i.e. normalize every token's embed
	saOut := b.saHead.Forward(input) // Encode relationships between positions, (blockSize, embedSize)
	input = Add(input, saOut)        // Add residual attention output back to main path

	// Feed-forward network with residual connection
	input = b.norm2.Forward(input)                  // Normalize input
	ffwdExpanded := b.ffwd.Forward(input)           // Expand to higher dimension
	ffwdActivated := ReLU(ffwdExpanded)             // Apply activation function
	ffwdOutput := b.ffwdProj.Forward(ffwdActivated) // Project back to original dimension
	ffwdOutput = Dropout(dropout)(ffwdOutput)       // Dropping out some neurons to prevent overfitting
	input = Add(input, ffwdOutput)                  // Add feed-forward residual output to main path

	return input
}

func (b *Block) Params() []layer.Parameter {
	var params []layer.Parameter
	params = append(params, b.saHead.Params()...)
	params = append(params, b.ffwd.Weight, b.ffwd.Bias)
	params = append(params, b.ffwdProj.Weight, b.ffwdProj.Bias)
	params = append(params, b.norm1.Scale, b.norm1.Shift)
	params = append(params, b.norm2.Scale, b.norm2.Shift)

	return params
}
