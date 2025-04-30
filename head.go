package main

import (
	"math"

	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"

	"github.com/zakirullin/gpt-go/pkg"
)

var (
	Tril          = pkg.Tril
	MaskedInfFill = pkg.MaskedInfFill
)

type MultiHeadAttention struct {
	numHeads  int
	embedSize int
	headSize  int
	Heads     []*Head
	proj      *Linear
}

func NewMultiHeadAttention(embedSize, numHeads int) *MultiHeadAttention {
	heads := make([]*Head, numHeads)
	headSize := embedSize / numHeads
	for i := range heads {
		heads[i] = NewHead(embedSize, headSize)
	}

	return &MultiHeadAttention{
		Heads:     heads,
		numHeads:  numHeads,
		embedSize: embedSize,
		headSize:  headSize,
		proj:      NewLinear(embedSize, embedSize),
	}
}

func (mh *MultiHeadAttention) Forward(input *variable.Variable) *variable.Variable {
	var features []*variable.Variable
	for _, head := range mh.Heads {
		features = append(features, head.Forward(input))
	}

	out := pkg.Cat(features...)
	out = mh.proj.Forward(out)  // Project back to (embedSize, embedSize)
	out = Dropout(dropout)(out) // Dropping out some neurons to prevent overfitting

	return out
}

func (mh *MultiHeadAttention) Params() []layer.Parameter {
	var params []layer.Parameter
	for _, head := range mh.Heads {
		params = append(params, head.Query.Weight, head.Key.Weight, head.Value.Weight)
	}
	params = append(params, mh.proj.Weight, mh.proj.Bias)

	return params
}

type Head struct {
	embedSize int
	headSize  int
	Key       *Linear
	Query     *Linear
	Value     *Linear
}

// Number of embeds
func NewHead(embedSize, headSize int) *Head {
	key := NewLinear(embedSize, headSize, NoBias())
	query := NewLinear(embedSize, headSize, NoBias())
	value := NewLinear(embedSize, headSize, NoBias())

	return &Head{embedSize, headSize, key, query, value}
}

// Self-attention is happening here
func (h *Head) Forward(input *variable.Variable) *variable.Variable {
	key := h.Key.Forward(input)
	query := Transpose(h.Query.Forward(input))
	wei := MatMul(key, query)

	T := len(input.Data) // number of tokens
	tril := Tril(Ones(T, T))
	wei = MaskedInfFill(wei, tril)
	wei = Softmax(wei)
	wei = Dropout(dropout)(wei)

	v := h.Value.Forward(input)
	weightedSum := MatMul(wei, v)
	normalizedSum := MulC(math.Pow(float64(h.embedSize), -0.5), weightedSum)

	return normalizedSum
}
