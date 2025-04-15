package main

import (
	"math"

	"github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

type Head struct {
	embedSize int
	headSize  int
	key       *Linear
	query     *Linear
	value     *Linear
}

// Number of embeds
func NewHead(embedSize, headSize int) *Head {
	key := NewLinear(headSize, headSize, NoBias())
	query := NewLinear(headSize, headSize, NoBias())
	value := NewLinear(headSize, headSize, NoBias())

	return &Head{embedSize, headSize, key, query, value}
}

func (h *Head) Forward(input *variable.Variable) *variable.Variable {
	T := len(input.Data)
	wei := MatMul(h.key.Forward(input), variable.Transpose(h.query.Forward(input)))

	tril := Tril(OneLike(Zeros(T, T)))
	wei = MaskedInfFill(wei, tril)
	wei = function.Softmax(wei)

	v := h.value.Forward(input)

	weightedSum := MatMul(wei, v)
	normalizedSum := function.MulC(math.Pow(float64(h.embedSize), -0.5), weightedSum)

	return normalizedSum
}

func (h *Head) ZeroGrad() {
	h.key.ZeroGrad()
	h.query.ZeroGrad()
	h.value.ZeroGrad()
}
