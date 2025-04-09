package main

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestSoftmax(t *testing.T) {
	r := require.New(t)

	logits := Tensor1D(2.0, 1.0, 0.1)

	result := Softmax(logits)

	sum := math.Exp(2.0) + math.Exp(1.0) + math.Exp(0.1)
	expected := []float64{
		math.Exp(2.0) / sum,
		math.Exp(1.0) / sum,
		math.Exp(0.1) / sum,
	}

	r.Len(result, 3)
	r.InDeltaf(expected[0], result[0], 1e-6, "First softmax value incorrect")
	r.InDeltaf(expected[1], result[1], 1e-6, "Second softmax value incorrect")
	r.InDeltaf(expected[2], result[2], 1e-6, "Third softmax value incorrect")

	r.InDeltaf(1.0, result[0]+result[1]+result[2], 1e-6, "Softmax values should sum to 1")
}

func TestCrossEntropyLoss(t *testing.T) {
	r := require.New(t)

	logits := T2{
		{2.0, 1.0, 0.1},
	}.Tensor()
	targets := []int{0}

	loss := CrossEntropyLoss(logits, targets)

	sum1 := math.Exp(2.0) + math.Exp(1.0) + math.Exp(0.1)
	prob := math.Exp(2.0) / sum1

	expected := -(math.Log(prob))

	r.InDeltaf(expected, loss, 1e-6, "Cross entropy loss incorrect")
}
