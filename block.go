package main

import (
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

type Block struct {
	embedSize int
	headCount int
	saHead    *MultiHeadAttention
	ffwd      *Linear
	ffwdProj  *Linear
}

func NewBlock(embedSize, numHeads int) *Block {
	return &Block{
		embedSize: embedSize,
		headCount: numHeads,
		saHead:    NewMultiHeadAttention(embedSize, numHeads),
		ffwd:      NewLinear(embedSize, embedSize*4),
		ffwdProj:  NewLinear(embedSize*4, embedSize),
	}
}

func (b *Block) Forward(input *variable.Variable) *variable.Variable {
	// Self-attention with residual connections. Input is our highway, we allow the gradient to flow back unimpeded.
	saOut := b.saHead.Forward(input) // Encode relationships between positions, (blockSize, embedSize)
	input = Add(input, saOut)        // Add residual attention output back to main path

	// Feed-forward network with residual connection
	ffwdExpanded := b.ffwd.Forward(input)           // Expand to higher dimension
	ffwdActivated := ReLU(ffwdExpanded)             // Apply activation function
	ffwdOutput := b.ffwdProj.Forward(ffwdActivated) // Project back to original dimension
	input = Add(input, ffwdOutput)                  // Add feed-forward residual output to main path

	return input
}

func (b *Block) Params() []layer.Parameter {
	var params []layer.Parameter
	for _, param := range b.saHead.Params() {
		params = append(params, param)
	}
	params = append(params, b.ffwd.Weight, b.ffwd.Bias)
	params = append(params, b.ffwdProj.Weight, b.ffwdProj.Bias)

	return params
}
