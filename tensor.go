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
	return Tensor1d([]float64{value})
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

func (t *Tensor) Set(val float64, indexes ...int) {
	if len(indexes) != len(t.Shape) {
		panic("index out of range")
	}

	// TODO reuse
	offset := 0
	for i, index := range indexes {
		if index >= t.Shape[i] {
			panic("index out of range")
		}
		offset += index * t.Shape[i]
	}

	t.Data[offset] = val
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
		return Tensor1d(result)
	}

	if len(t.Shape) == 2 {
		if t.Shape[1] != other.Shape[0] {
			panic(fmt.Sprintf("Tensor shapes do not match for multiplication: %v and %v", t.Shape, other.Shape))
		}

		result := make([]float64, t.Shape[0]*other.Shape[1])
		for i := 0; i < t.Shape[0]; i++ {
			for j := 0; j < other.Shape[1]; j++ {
				for k := 0; k < t.Shape[1]; k++ {
					result[i*other.Shape[1]+j] += t.Data[i*t.Shape[1]+k] * other.Data[k*other.Shape[1]+j]
				}
			}
		}
		return Tensor1d(result)
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

func (t *Tensor) Sum() float64 {
	if len(t.Shape) == 1 {
		sum := 0.0
		for _, v := range t.Data {
			sum += v
		}
		return sum
	}

	if len(t.Shape) == 2 {
		sum := 0.0
		for _, v := range t.Data {
			sum += v
		}
		return sum
	}

	panic("unsupported Tensor Shape for sum")
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

func Tensor1d(data []float64) *Tensor {
	return &Tensor{
		Shape: []int{1, len(data)},
		Data:  data,
	}
}

func Tensor2d(data [][]float64) *Tensor {
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

func Tensor3d(data [][][]float64) *Tensor {
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
