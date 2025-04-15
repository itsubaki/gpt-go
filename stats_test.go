package main

//
//import (
//	"math"
//	"testing"
//
//	"github.com/stretchr/testify/require"
//)
//
//func TestSoftmax(t *testing.T) {
//	r := require.New(t)
//
//	logits := Tensor1D(2.0, 1.0, 0.1)
//
//	result := Softmax(logits)
//
//	// Came from pytorch
//	expected := []float64{0.6590011388859679, 0.24243297070471392, 0.09856589040931818}
//
//	r.InDeltaf(expected[0], result.First(), 1e-6, "First softmax Value incorrect")
//	r.InDeltaf(expected[1], result.At(1).First(), 1e-6, "Second softmax Value incorrect")
//	r.InDeltaf(expected[2], result.At(2).First(), 1e-6, "Third softmax Value incorrect")
//
//	r.InDeltaf(1.0, result.First()+result.At(1).First()+result.At(2).First(), 1e-6, "Softmax values should sum to 1")
//}
//
//func TestCrossEntropyLoss(t *testing.T) {
//	r := require.New(t)
//
//	logits := Tensor1D(2.0, 1.0, 0.1)
//	loss := CrossEntropyLoss(logits, 0)
//
//	// Came from pytorch
//	expected := 0.4170299470424652
//
//	r.InDeltaf(expected, loss, 1e-6, "Cross entropy loss incorrect")
//}
//
//func TestSoftmax1D(t *testing.T) {
//	r := require.New(t)
//
//	// Test with 1D tensor
//	input := &Tensor{
//		Shape: []int{4},
//		Data:  []float64{1.0, 2.0, 3.0, 4.0},
//	}
//
//	result := Softmax(input)
//
//	// Calculate expected values manually
//	maxVal := 4.0
//	expSum := math.Exp(1.0-maxVal) + math.Exp(2.0-maxVal) + math.Exp(3.0-maxVal) + math.Exp(4.0-maxVal)
//	expected := &Tensor{
//		Shape: []int{4},
//		Data: []float64{
//			math.Exp(1.0-maxVal) / expSum,
//			math.Exp(2.0-maxVal) / expSum,
//			math.Exp(3.0-maxVal) / expSum,
//			math.Exp(4.0-maxVal) / expSum,
//		},
//	}
//
//	// Check shape
//	r.Equal(expected.Shape, result.Shape)
//
//	// Check values (using approximate equality due to floating point)
//	for i := range expected.Data {
//		r.InDelta(expected.Data[i], result.Data[i], 1e-10)
//	}
//
//	// Check sum is approximately 1.0
//	sum := 0.0
//	for _, v := range result.Data {
//		sum += v
//	}
//	r.InDelta(1.0, sum, 1e-10)
//}
//
//func TestSoftmax2D(t *testing.T) {
//	r := require.New(t)
//
//	// Test with 2D tensor
//	input := &Tensor{
//		Shape: []int{2, 3},
//		Data:  []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
//	}
//
//	result := Softmax(input)
//
//	// Calculate expected values manually
//	// Row 1: [1.0, 2.0, 3.0]
//	maxVal1 := 3.0
//	expSum1 := math.Exp(1.0-maxVal1) + math.Exp(2.0-maxVal1) + math.Exp(3.0-maxVal1)
//
//	// Row 2: [4.0, 5.0, 6.0]
//	maxVal2 := 6.0
//	expSum2 := math.Exp(4.0-maxVal2) + math.Exp(5.0-maxVal2) + math.Exp(6.0-maxVal2)
//
//	expected := &Tensor{
//		Shape: []int{2, 3},
//		Data: []float64{
//			math.Exp(1.0-maxVal1) / expSum1, math.Exp(2.0-maxVal1) / expSum1, math.Exp(3.0-maxVal1) / expSum1,
//			math.Exp(4.0-maxVal2) / expSum2, math.Exp(5.0-maxVal2) / expSum2, math.Exp(6.0-maxVal2) / expSum2,
//		},
//	}
//
//	// Check shape
//	r.Equal(expected.Shape, result.Shape)
//
//	// Check values (using approximate equality due to floating point)
//	for i := range expected.Data {
//		r.InDelta(expected.Data[i], result.Data[i], 1e-10)
//	}
//
//	// Check each row sums to approximately 1.0
//	sum1 := result.Data[0] + result.Data[1] + result.Data[2]
//	sum2 := result.Data[3] + result.Data[4] + result.Data[5]
//	r.InDelta(1.0, sum1, 1e-10)
//	r.InDelta(1.0, sum2, 1e-10)
//}
