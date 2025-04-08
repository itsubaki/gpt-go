package main

import (
	"fmt"

	"gonum.org/v1/gonum/stat/distuv"
)

type Tensor struct {
	Shape []int
	Data  []float64
}

func Scal(value float64) *Tensor {
	return Tensor1D([]float64{value})
}

func (t *Tensor) At(indexes ...int) float64 {
	if len(indexes) != len(t.Shape) {
		panic("index out of range")
	}

	offset := 0
	for i, index := range indexes {
		if index >= t.Shape[i] {
			panic("index out of range")
		}
		offset += index * t.Shape[i]
	}

	return t.Data[offset]
}

func (t *Tensor) Mul(other *Tensor) *Tensor {
	if len(t.Shape) != len(other.Shape) {
		panic("Tensor shapes do not match")
	}

	if len(t.Shape) == 1 {
		result := make([]float64, len(t.Data))
		for i := range t.Data {
			result[i] = t.Data[i] * other.Data[i]
		}
		return Tensor1D(result)
	}

	if len(t.Shape) == 2 {
		result := make([]float64, t.Shape[0]*other.Shape[1])
		for i := 0; i < t.Shape[0]; i++ {
			for j := 0; j < other.Shape[1]; j++ {
				for k := 0; k < t.Shape[1]; k++ {
					result[i*other.Shape[1]+j] += t.Data[i*t.Shape[1]+k] * other.Data[k*other.Shape[1]+j]
				}
			}
		}
		return Tensor1D(result)
	}

	panic("unsupported Tensor Shape")
}

// Print in human-readable form
func (t *Tensor) Print() {
	if len(t.Shape) == 1 {
		for _, v := range t.Data {
			fmt.Printf("%.3f ", v)
		}
		fmt.Println()
		return
	}

	if len(t.Shape) == 2 {
		for i := 0; i < t.Shape[0]; i++ {
			for j := 0; j < t.Shape[1]; j++ {
				fmt.Printf("%.3f ", t.Data[i*t.Shape[1]+j])
			}
			fmt.Println()
		}
		return
	}

	panic("unsupported Tensor Shape for print")
}

// Zero creates zero-filled tensor
func Zero(dims ...int) *Tensor {
	shape := make([]int, len(dims))
	copy(shape, dims)

	size := 1
	for _, dim := range dims {
		size *= dim
	}
	data := make([]float64, size)

	return &Tensor{
		Shape: shape,
		Data:  data,
	}
}

// RandN creates a tensor with normally distributed random values
func RandN(dims ...int) *Tensor {
	shape := make([]int, len(dims))
	copy(shape, dims)

	size := 1
	for _, dim := range dims {
		size *= dim
	}
	data := make([]float64, size)

	dist := distuv.Normal{Mu: 0, Sigma: 1}
	for i := 0; i < size; i++ {
		data[i] = dist.Rand()
	}

	return &Tensor{
		Shape: shape,
		Data:  data,
	}
}

func Tensor1D(data []float64) *Tensor {
	return &Tensor{
		Shape: []int{len(data)},
		Data:  data,
	}
}

func Tensor2D(data [][]float64) *Tensor {
	rows := len(data)
	cols := len(data[0])
	flatData := make([]float64, rows*cols)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			flatData[i*cols+j] = data[i][j]
		}
	}

	return &Tensor{
		Shape: []int{rows, cols},
		Data:  flatData,
	}
}

func Tensor3D(data [][][]float64) *Tensor {
	rows := len(data)
	cols := len(data[0])
	depth := len(data[0][0])
	flatData := make([]float64, rows*cols*depth)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			for k := 0; k < depth; k++ {
				flatData[i*cols*depth+j*depth+k] = data[i][j][k]
			}
		}
	}

	return &Tensor{
		Shape: []int{rows, cols, depth},
		Data:  flatData,
	}
}
