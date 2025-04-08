package main

import (
	"fmt"

	"gonum.org/v1/gonum/stat/distuv"
)

type Tensor struct {
	Shape []int
	Data  []float64
}

// Zeros creates zero-filled tensor
func Zeros(dims ...int) *Tensor {
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
		Shape: []int{1, len(data)},
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

func Scal(val float64) *Tensor {
	return &Tensor{
		Shape: []int{1, 1},
		Data:  []float64{val},
	}
}

func (t *Tensor) At(indexes ...int) float64 {
	return t.Data[t.offset(indexes...)]
}

func (t *Tensor) Set(val float64, indexes ...int) {
	t.Data[t.offset(indexes...)] = val
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

// offset calculates the flat offset in the data array based on the provided indexes
func (t *Tensor) offset(indexes ...int) int {
	// Special case for scalar (1x1) tensor
	if len(t.Shape) == 2 && t.Shape[0] == 1 && t.Shape[1] == 1 {
		// Allow calling with an empty slice or [0] for scalars
		if len(indexes) == 0 || (len(indexes) == 1 && indexes[0] == 0) {
			return 0
		}
	}

	// Special case for 1xN vector tensor - use just column index
	if len(t.Shape) == 2 && t.Shape[0] == 1 && len(indexes) == 1 {
		if indexes[0] >= 0 && indexes[0] < t.Shape[1] {
			return indexes[0]
		}
	}

	// Standard case - require full indexing
	if len(indexes) != len(t.Shape) {
		msg := fmt.Sprintf("can't get value from tensor with shape %v at index %v", t.Shape, indexes)
		panic(msg)
	}

	// Calculate the linear offset
	offset := 0
	stride := 1

	// Iterate in reverse order (column-major)
	for i := len(t.Shape) - 1; i >= 0; i-- {
		if indexes[i] < 0 || indexes[i] >= t.Shape[i] {
			msg := fmt.Sprintf("can't get value from tensor with shape %v at index %v", t.Shape, indexes)
			panic(msg)
		}

		offset += indexes[i] * stride
		stride *= t.Shape[i]
	}

	return offset
}
