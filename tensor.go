package main

import (
	"fmt"

	"gonum.org/v1/gonum/stat/distuv"
)

// Maybe do this for more compact way?
type d2 [][]float64

var tensor = d2{
	{1.0},
	{2.0},
	{3.0},
}

// scalar, d1, d2, d3 would implement data
// The initialization would look like this:
// tensor := d1{1, 2, 3}
// tensor := d2{{1, 2}, {3, 4}}
// Challenge is to convert it to tensor (we can't embed type in non-struct)
// d2.Tensor() could be massive again
// Or just leave it at
// tensor := Tensor(d2{{1, 2}, {3, 4}})
// what if copy-paste (code-generate) code in d1, d2, d3?
// > IF we use a few different types, we would have to use <any>
// everywhere in the code instead Tensor, and that I don't want.
// So, we're left with flexible constructor then?
// Or maybe quick function like t() accepting all sorts of data
// > On the other hand, we can accept Tensor interfaces everywhere
// And maybe better t1{}, t2{{}}, t3{{}} etc instead of d1 d2, because
// we create tensors, not dimensions
// > On the other hand, we don't need to init matrices manually, soo it may be pointless
// On the other hand, in tests it will be good. And also in studying
// Because our purpose is not prod-ready, but study-reading, so clarity should come first?

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

func Tensor1D(data ...float64) *Tensor {
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

func Scalar(val float64) *Tensor {
	return &Tensor{
		Shape: []int{1, 1},
		Data:  []float64{val},
	}
}

func (t *Tensor) First() float64 {
	if len(t.Data) == 0 {
		panic("Tensor is empty")
	}

	return t.Data[0]
}

// TODO pluck out rows/cols
// TODO for 3d
func (t *Tensor) At(indexes ...int) *Tensor {
	x, y := t.offset(indexes...)

	result := Tensor1D(t.Data[x:y]...)

	return result
}

// Work only for individual elements
func (t *Tensor) Set(val float64, indexes ...int) {
	x, _ := t.offset(indexes...)

	t.Data[x] = val
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
		return Tensor1D(result...)
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
		return Tensor1D(result...)
	}

	panic("unsupported Tensor Shape")
}

// TODO Print in human-readable form
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

func (t *Tensor) T() *Tensor {
	if len(t.Shape) != 2 {
		panic("Transpose only supported for 2D tensors")
	}

	result := make([]float64, len(t.Data))
	for i := 0; i < t.Shape[0]; i++ {
		for j := 0; j < t.Shape[1]; j++ {
			result[j*t.Shape[0]+i] = t.Data[i*t.Shape[1]+j]
		}
	}

	return &Tensor{
		Shape: []int{t.Shape[1], t.Shape[0]},
		Data:  result,
	}
}

// offset calculates the start and limit offsets in the data array for slicing
func (t *Tensor) offset(indexes ...int) (int, int) {
	// Special case for scalar (1x1) tensor
	if len(t.Shape) == 2 && t.Shape[0] == 1 && t.Shape[1] == 1 {
		// Allow calling with an empty slice or [0] for scalars
		if len(indexes) == 0 || (len(indexes) == 1 && indexes[0] == 0) {
			return 0, 1
		}
	}

	// Special case for 1xN vector tensor - use just column index
	if len(t.Shape) == 2 && t.Shape[0] == 1 && len(indexes) == 1 {
		if indexes[0] >= 0 && indexes[0] < t.Shape[1] {
			// For a 1D vector, if we only get the row index, return that entire row
			return indexes[0], indexes[0] + 1
		}
	}

	// For rows or columns access with a single index (when tensor is 2D)
	if len(t.Shape) == 2 && len(indexes) == 1 {
		idx := indexes[0]
		// If accessing a row
		if idx >= 0 && idx < t.Shape[0] {
			// Return the start of the row and the start of the next row
			startOffset := idx * t.Shape[1]
			endOffset := (idx + 1) * t.Shape[1]
			return startOffset, endOffset
		}
	}

	// Standard case - require full indexing
	if len(indexes) != len(t.Shape) {
		msg := fmt.Sprintf("can't get value from tensor with shape %v at index %v", t.Shape, indexes)
		panic(msg)
	}

	// Calculate the linear offset for a specific element
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

	// For a specific element, the range is just 1 element
	return offset, offset + 1
}
