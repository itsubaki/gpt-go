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
}

func NewBlock(embedSize, numHeads int) *Block {
	return &Block{
		embedSize: embedSize,
		headCount: numHeads,
		saHead:    NewMultiHeadAttention(embedSize, numHeads),
		ffwd:      NewLinear(embedSize, embedSize),
	}
}

func (b *Block) Forward(input *variable.Variable) *variable.Variable {
	// Skip (Residual) connections. Input is our highway, we allow the gradient to flow back unimpeded
	input = Add(input, b.saHead.Forward(input))         // Encode relationships between positions, (blockSize, embedSize)
	features := Add(input, ReLU(b.ffwd.Forward(input))) // Learn more complex patterns, which linear projections can't

	return features
}

func (b *Block) Params() []layer.Parameter {
	var params []layer.Parameter
	for _, param := range b.saHead.Params() {
		params = append(params, param)
	}
	params = append(params, b.ffwd.Weight)
	params = append(params, b.ffwd.Bias)

	return params
}
