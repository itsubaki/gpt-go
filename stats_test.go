package main

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestSoftmax(t *testing.T) {
	r := require.New(t)

	logits := Tensor1D(2.0, 1.0, 0.1)

	result := Softmax(logits)

	// Came from pytorch
	expected := []float64{0.6590011388859679, 0.24243297070471392, 0.09856589040931818}

	r.InDeltaf(expected[0], result.First(), 1e-6, "First softmax value incorrect")
	r.InDeltaf(expected[1], result.At(1).First(), 1e-6, "Second softmax value incorrect")
	r.InDeltaf(expected[2], result.At(2).First(), 1e-6, "Third softmax value incorrect")

	r.InDeltaf(1.0, result.First()+result.At(1).First()+result.At(2).First(), 1e-6, "Softmax values should sum to 1")
}

func TestCrossEntropyLoss(t *testing.T) {
	r := require.New(t)

	logits := T2{
		{2.0, 1.0, 0.1},
	}.Tensor()
	targets := []int{0}

	loss := CrossEntropyLoss(logits, targets)

	// Came from pytorch
	expected := 0.4170299470424652

	r.InDeltaf(expected, loss, 1e-6, "Cross entropy loss incorrect")
}
