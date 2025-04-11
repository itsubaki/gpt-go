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

	// Came from pytorch
	expected := []float64{0.6590011388859679, 0.24243297070471392, 0.09856589040931818}

	r.InDeltaf(expected[0], result.First(), 1e-6, "First softmax value incorrect")
	r.InDeltaf(expected[1], result.At(1).First(), 1e-6, "Second softmax value incorrect")
	r.InDeltaf(expected[2], result.At(2).First(), 1e-6, "Third softmax value incorrect")

	r.InDeltaf(1.0, result.First()+result.At(1).First()+result.At(2).First(), 1e-6, "Softmax values should sum to 1")
}

func TestCrossEntropyLoss(t *testing.T) {
	r := require.New(t)

	logits := Tensor1D(2.0, 1.0, 0.1)
	loss := CrossEntropyLoss(logits, 0)

	// Came from pytorch
	expected := 0.4170299470424652

	r.InDeltaf(expected, loss, 1e-6, "Cross entropy loss incorrect")
}

func TestSoftmax1D(t *testing.T) {
	r := require.New(t)

	// Test with 1D tensor
	input := &Tensor{
		Shape: []int{4},
		Data:  []float64{1.0, 2.0, 3.0, 4.0},
	}

	result := Softmax(input)

	// Calculate expected values manually
	maxVal := 4.0
	expSum := math.Exp(1.0-maxVal) + math.Exp(2.0-maxVal) + math.Exp(3.0-maxVal) + math.Exp(4.0-maxVal)
	expected := &Tensor{
		Shape: []int{4},
		Data: []float64{
			math.Exp(1.0-maxVal) / expSum,
			math.Exp(2.0-maxVal) / expSum,
			math.Exp(3.0-maxVal) / expSum,
			math.Exp(4.0-maxVal) / expSum,
		},
	}

	// Check shape
	r.Equal(expected.Shape, result.Shape)

	// Check values (using approximate equality due to floating point)
	for i := range expected.Data {
		r.InDelta(expected.Data[i], result.Data[i], 1e-10)
	}

	// Check sum is approximately 1.0
	sum := 0.0
	for _, v := range result.Data {
		sum += v
	}
	r.InDelta(1.0, sum, 1e-10)
}

func TestSoftmax2D(t *testing.T) {
	r := require.New(t)

	// Test with 2D tensor
	input := &Tensor{
		Shape: []int{2, 3},
		Data:  []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
	}

	result := Softmax(input)

	// Calculate expected values manually
	// Row 1: [1.0, 2.0, 3.0]
	maxVal1 := 3.0
	expSum1 := math.Exp(1.0-maxVal1) + math.Exp(2.0-maxVal1) + math.Exp(3.0-maxVal1)

	// Row 2: [4.0, 5.0, 6.0]
	maxVal2 := 6.0
	expSum2 := math.Exp(4.0-maxVal2) + math.Exp(5.0-maxVal2) + math.Exp(6.0-maxVal2)

	expected := &Tensor{
		Shape: []int{2, 3},
		Data: []float64{
			math.Exp(1.0-maxVal1) / expSum1, math.Exp(2.0-maxVal1) / expSum1, math.Exp(3.0-maxVal1) / expSum1,
			math.Exp(4.0-maxVal2) / expSum2, math.Exp(5.0-maxVal2) / expSum2, math.Exp(6.0-maxVal2) / expSum2,
		},
	}

	// Check shape
	r.Equal(expected.Shape, result.Shape)

	// Check values (using approximate equality due to floating point)
	for i := range expected.Data {
		r.InDelta(expected.Data[i], result.Data[i], 1e-10)
	}

	// Check each row sums to approximately 1.0
	sum1 := result.Data[0] + result.Data[1] + result.Data[2]
	sum2 := result.Data[3] + result.Data[4] + result.Data[5]
	r.InDelta(1.0, sum1, 1e-10)
	r.InDelta(1.0, sum2, 1e-10)
}

//func TestSoftmaxWithDim(t *testing.T) {
//	r := require.New(t)
//
//	// Create a 2x3 tensor for testing
//	input := &Tensor{
//		Shape: []int{2, 3},
//		Data:  []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
//	}
//
//	// Test softmax along dim=1 (rows)
//	resultRows := SoftmaxWithDim(input, 1)
//
//	// Calculate expected values manually for dim=1
//	// Row 1: [1.0, 2.0, 3.0]
//	maxVal1 := 3.0
//	expSum1 := math.Exp(1.0-maxVal1) + math.Exp(2.0-maxVal1) + math.Exp(3.0-maxVal1)
//
//	// Row 2: [4.0, 5.0, 6.0]
//	maxVal2 := 6.0
//	expSum2 := math.Exp(4.0-maxVal2) + math.Exp(5.0-maxVal2) + math.Exp(6.0-maxVal2)
//
//	expectedRows := &Tensor{
//		Shape: []int{2, 3},
//		Data: []float64{
//			math.Exp(1.0-maxVal1) / expSum1, math.Exp(2.0-maxVal1) / expSum1, math.Exp(3.0-maxVal1) / expSum1,
//			math.Exp(4.0-maxVal2) / expSum2, math.Exp(5.0-maxVal2) / expSum2, math.Exp(6.0-maxVal2) / expSum2,
//		},
//	}
//
//	// Check shape and values for dim=1
//	r.Equal(expectedRows.Shape, resultRows.Shape)
//	for i := range expectedRows.Data {
//		r.InDelta(expectedRows.Data[i], resultRows.Data[i], 1e-10)
//	}
//
//	// Check each row sums to approximately 1.0
//	sum1 := resultRows.Data[0] + resultRows.Data[1] + resultRows.Data[2]
//	sum2 := resultRows.Data[3] + resultRows.Data[4] + resultRows.Data[5]
//	r.InDelta(1.0, sum1, 1e-10)
//	r.InDelta(1.0, sum2, 1e-10)
//
//	// Test softmax along dim=0 (columns)
//	resultCols := SoftmaxWithDim(input, 0)
//
//	// Calculate expected values manually for dim=0
//	// Col 1: [1.0, 4.0]
//	maxValCol1 := 4.0
//	expSumCol1 := math.Exp(1.0-maxValCol1) + math.Exp(4.0-maxValCol1)
//
//	// Col 2: [2.0, 5.0]
//	maxValCol2 := 5.0
//	expSumCol2 := math.Exp(2.0-maxValCol2) + math.Exp(5.0-maxValCol2)
//
//	// Col 3: [3.0, 6.0]
//	maxValCol3 := 6.0
//	expSumCol3 := math.Exp(3.0-maxValCol3) + math.Exp(6.0-maxValCol3)
//
//	expectedCols := &Tensor{
//		Shape: []int{2, 3},
//		Data: []float64{
//			math.Exp(1.0-maxValCol1) / expSumCol1, math.Exp(2.0-maxValCol2) / expSumCol2, math.Exp(3.0-maxValCol3) / expSumCol3,
//			math.Exp(4.0-maxValCol1) / expSumCol1, math.Exp(5.0-maxValCol2) / expSumCol2, math.Exp(6.0-maxValCol3) / expSumCol3,
//		},
//	}
//
//	// Check shape and values for dim=0
//	r.Equal(expectedCols.Shape, resultCols.Shape)
//	for i := range expectedCols.Data {
//		r.InDelta(expectedCols.Data[i], resultCols.Data[i], 1e-10)
//	}
//
//	// Check each column sums to approximately 1.0
//	col1Sum := resultCols.Data[0] + resultCols.Data[3]
//	col2Sum := resultCols.Data[1] + resultCols.Data[4]
//	col3Sum := resultCols.Data[2] + resultCols.Data[5]
//	r.InDelta(1.0, col1Sum, 1e-10)
//	r.InDelta(1.0, col2Sum, 1e-10)
//	r.InDelta(1.0, col3Sum, 1e-10)
//
//	// Test with negative dimension (dim=-1 should be same as dim=1)
//	resultNegDim := SoftmaxWithDim(input, -1)
//	r.Equal(expectedRows.Shape, resultNegDim.Shape)
//	for i := range expectedRows.Data {
//		r.InDelta(expectedRows.Data[i], resultNegDim.Data[i], 1e-10)
//	}
//}
