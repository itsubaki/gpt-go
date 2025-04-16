package main

import (
	"math"

	"github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
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

	out := Cat(features...)

	return mh.proj.Forward(out) // Project back to (embedSize, embedSize)
}

func (mh *MultiHeadAttention) Params() []layer.Parameter {
	var params []layer.Parameter
	for _, head := range mh.Heads {
		params = append(params, head.Query.Weight, head.Key.Weight, head.Value.Weight)
	}

	return params
}

func (mh *MultiHeadAttention) ZeroGrad() {
	for _, head := range mh.Heads {
		head.ZeroGrad()
	}
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

func (h *Head) Forward(input *variable.Variable) *variable.Variable {
	T := len(input.Data)
	wei := MatMul(h.Key.Forward(input), variable.Transpose(h.Query.Forward(input)))

	tril := Tril(OneLike(Zeros(T, T)))
	wei = MaskedInfFill(wei, tril)
	wei = function.Softmax(wei)

	v := h.Value.Forward(input)

	weightedSum := MatMul(wei, v)
	normalizedSum := function.MulC(math.Pow(float64(h.embedSize), -0.5), weightedSum)

	return normalizedSum
}

func (h *Head) ZeroGrad() {
	h.Key.ZeroGrad()
	h.Query.ZeroGrad()
	h.Value.ZeroGrad()
}
