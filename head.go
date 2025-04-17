package main

import (
	"math"

	"github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"

	"gptgo/pkg"
)

type MultiHeadAttention struct {
	numHeads  int
	embedSize int
	headSize  int
	Heads     []*Head
	proj      *pkg.Linear
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
		proj:      pkg.NewLinear(embedSize, embedSize),
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
	Key       *pkg.Linear
	Query     *pkg.Linear
	Value     *pkg.Linear
}

// Number of embeds
func NewHead(embedSize, headSize int) *Head {
	key := pkg.NewLinear(embedSize, headSize, pkg.NoBias())
	query := pkg.NewLinear(embedSize, headSize, pkg.NoBias())
	value := pkg.NewLinear(embedSize, headSize, pkg.NoBias())

	return &Head{embedSize, headSize, key, query, value}
}

func (h *Head) Forward(input *variable.Variable) *variable.Variable {
	T := len(input.Data)
	key := h.Key.Forward(input)
	query := variable.Transpose(h.Query.Forward(input))
	wei := MatMul(key, query)

	tril := pkg.Tril(OneLike(Zeros(T, T)))
	wei = pkg.MaskedInfFill(wei, tril)
	wei = function.Softmax(wei)
	wei = Dropout(dropout)(wei)

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
