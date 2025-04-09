package main

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestShape(t *testing.T) {
	r := require.New(t)

	scalar := Scalar(1)
	r.Equal([]int{}, scalar.Shape)

	tensor1d := Tensor1D(1, 2, 3)
	r.Equal([]int{3}, tensor1d.Shape)

	tensor2d := Tensor2D([][]float64{
		{1, 2},
		{3, 4},
	})

	r.Equal([]int{2, 2}, tensor2d.Shape)
}

func TestAtScalar(t *testing.T) {
	r := require.New(t)

	tensor := Tensor1D(1)
	r.Equal(tensor.At(0).First(), 1.0)
}

func TestAtVector(t *testing.T) {
	r := require.New(t)

	tensor := Tensor1D(1, 2, 3)
	r.Equal(tensor.At(0).First(), 1.0)
	r.Equal(tensor.At(1).First(), 2.0)
	r.Equal(tensor.At(2).First(), 3.0)
}

func TestAtMatrix(t *testing.T) {
	r := require.New(t)

	tensor := Tensor2D([][]float64{
		{1, 2},
		{3, 4},
	})
	r.Equal(tensor.At(0, 0).First(), 1.0)
	r.Equal(tensor.At(0, 1).First(), 2.0)
	r.Equal(tensor.At(1, 0).First(), 3.0)
	r.Equal(tensor.At(1, 1).First(), 4.0)
}

func TestMulVector(t *testing.T) {
	r := require.New(t)

	tensorA := Tensor1D(1, 2, 3)
	tensorB := T2{
		{4},
		{5},
		{6},
	}.Tensor()
	result := tensorA.Mul(tensorB)

	expected := Tensor1D(32)
	r.Equal(expected.Data, result.Data)
}

func TestMulVerticalVector(t *testing.T) {
	r := require.New(t)

	tensorA := T2{
		{1},
		{2},
		{3},
	}.Tensor()

	tensorB := Tensor1D(4, 5, 6)

	result := tensorA.Mul(tensorB)

	expected := T2{
		{4, 5, 6},
		{8, 10, 12},
		{12, 15, 18},
	}.Tensor()

	r.True(result.Equal(expected))
}

func TestMulMatrix(t *testing.T) {
	r := require.New(t)

	tensorA := Tensor2D([][]float64{
		{1, 2},
		{3, 4},
	})
	tensorB := Tensor2D([][]float64{
		{5, 6},
		{7, 8},
	})

	result := tensorA.Mul(tensorB)
	expected := Tensor2D([][]float64{
		{19, 22},
		{43, 50},
	})

	r.Equal(result.Data, expected.Data)
}

func TestMul3D(t *testing.T) {
	r := require.New(t)

	tensorA := Tensor3D([][][]float64{
		{
			{1, 2},
			{3, 4},
		},
		{
			{5, 6},
			{7, 8},
		},
	})

	tensorB := Tensor3D([][][]float64{
		{
			{5, 6},
			{7, 8},
		},
		{
			{9, 10},
			{11, 12},
		},
	})

	result := tensorA.Mul(tensorB)
	expected := Tensor3D([][][]float64{
		{
			{19, 22},
			{43, 50},
		},
		{
			{111, 122},
			{151, 166},
		},
	})

	r.Equal(expected.Data, result.Data)
}

func TestOffset(t *testing.T) {
	r := require.New(t)

	tensor := Tensor2D([][]float64{
		{1, 2},
		{3, 4},
	})

	x, y := tensor.offset(0, 0)
	r.Equal(0, x)
	r.Equal(1, y)

	x, y = tensor.offset(0, 1)
	r.Equal(1, x)
	r.Equal(2, y)

	x, y = tensor.offset(1, 0)
	r.Equal(2, x)
	r.Equal(3, y)

	x, y = tensor.offset(1, 1)
	r.Equal(3, x)
	r.Equal(4, y)
}

func TestOffsetPluckOutRow1D(t *testing.T) {
	r := require.New(t)

	tensor := T2{
		{1, 2},
	}.Tensor()

	row := tensor.At(0)
	r.Equal(row.Data, []float64{1, 2})
}

func TestOffsetPluckOutRow2D(t *testing.T) {
	r := require.New(t)

	tensor := Tensor2D([][]float64{
		{1, 2},
		{3, 4},
	})

	row := tensor.At(0)
	r.Equal(row.Data, []float64{1, 2})

	row = tensor.At(1)
	r.Equal(row.Data, []float64{3, 4})
}

func TestTransposeVector(t *testing.T) {
	r := require.New(t)

	tensor := Tensor1D(1, 2, 3)
	transposed := tensor.T()

	expected := T2{
		{1},
		{2},
		{3},
	}.Tensor()
	r.Equal(expected.Data, transposed.Data)
	r.Equal([]int{3, 1}, transposed.Shape)

	tensor2 := Tensor2D([][]float64{
		{1},
		{2},
		{3},
	})

	r.Equal([]float64{1, 2, 3}, tensor2.T().Data)
}

func TestTransposeMatrix(t *testing.T) {
	r := require.New(t)

	row := T2{
		{1, 2, 3},
	}.Tensor()

	col := row.T()
	expected := T2{
		{1},
		{2},
		{3},
	}.Tensor()
	r.True(col.Equal(expected))

	tensor := T2{
		{1, 2},
		{3, 4},
	}.Tensor()

	result := tensor.T()
	expected = T2{
		{1, 3},
		{2, 4},
	}.Tensor()

	r.Equal(result.Data, expected.Data)
}

func TestTransposeDeepMatrix(t *testing.T) {
	r := require.New(t)

	matrix := T2{
		{2, 3},
		{2, 2},
		{4, 1},
	}.Tensor()

	transposed := T2{
		{2, 2, 4},
		{3, 2, 1},
	}.Tensor()

	r.True(matrix.T().Equal(transposed))
}

func TestT2Builder(t *testing.T) {
	r := require.New(t)

	tensor := T2{
		{1, 2},
		{3, 4},
	}.Tensor()
	r.Equal(tensor.Shape, []int{2, 2})

	expected := Tensor2D([][]float64{
		{1, 2},
		{3, 4},
	})

	r.Equal(tensor.Data, expected.Data)
}
