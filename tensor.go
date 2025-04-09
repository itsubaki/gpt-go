package main

import (
	"fmt"

	"gonum.org/v1/gonum/stat/distuv"
)

// Maybe do this for more compact way?
type T2 [][]float64

type Tensor struct {
	Shape []int
	Data  []float64
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
func (t2 T2) Tensor() *Tensor {
	rows := len(t2)
	cols := len(t2[0])
	shape := []int{rows, cols}

	size := rows * cols
	data := make([]float64, size)

	for i, v := range t2 {
		for j, val := range v {
			data[i*cols+j] = val
		}
	}

	return &Tensor{
		Shape: shape,
		Data:  data,
	}
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
		Shape: []int{len(data)}, // Vector: just number of columns
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
		Shape: []int{rows, cols}, // Matrix: rows and columns
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
		Shape: []int{rows, cols, depth}, // Keep 3D as is
		Data:  flatData,
	}
}

func Scalar(val float64) *Tensor {
	return &Tensor{
		Shape: []int{}, // Empty shape for scalar
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

	// Return scalar if selecting a single element
	if y-x == 1 {
		return Scalar(t.Data[x])
	}

	// Otherwise return a vector
	return &Tensor{
		Shape: []int{y - x},
		Data:  t.Data[x:y],
	}
}

// Work only for individual elements
func (t *Tensor) Set(val float64, indexes ...int) {
	x, _ := t.offset(indexes...)
	t.Data[x] = val
}

func (t *Tensor) Mul(other *Tensor) *Tensor {
	// Special case for scalar multiplication
	if len(t.Shape) == 0 || len(other.Shape) == 0 {
		if len(t.Shape) == 0 && len(other.Shape) == 0 {
			// Scalar * Scalar
			return Scalar(t.Data[0] * other.Data[0])
		} else if len(t.Shape) == 0 {
			// Scalar * Tensor
			scalar := t.Data[0]
			result := make([]float64, len(other.Data))
			for i, v := range other.Data {
				result[i] = scalar * v
			}
			return &Tensor{
				Shape: append([]int{}, other.Shape...),
				Data:  result,
			}
		} else {
			// Tensor * Scalar
			scalar := other.Data[0]
			result := make([]float64, len(t.Data))
			for i, v := range t.Data {
				result[i] = v * scalar
			}
			return &Tensor{
				Shape: append([]int{}, t.Shape...),
				Data:  result,
			}
		}
	}

	// Vector * Vector (element-wise)
	if len(t.Shape) == 1 && len(other.Shape) == 1 && t.Shape[0] == other.Shape[0] {
		result := make([]float64, len(t.Data))
		for i := range t.Data {
			result[i] = t.Data[i] * other.Data[i]
		}
		return &Tensor{
			Shape: []int{t.Shape[0]},
			Data:  result,
		}
	}

	// Vector * Matrix (treating vector as a row vector)
	if len(t.Shape) == 1 && len(other.Shape) == 2 {
		if t.Shape[0] != other.Shape[0] {
			panic(fmt.Sprintf("Tensor shapes do not match for multiplication: %v and %v", t.Shape, other.Shape))
		}

		cols := other.Shape[1]
		result := make([]float64, cols)

		for j := 0; j < cols; j++ {
			for k := 0; k < t.Shape[0]; k++ {
				result[j] += t.Data[k] * other.Data[k*cols+j]
			}
		}

		return &Tensor{
			Shape: []int{cols},
			Data:  result,
		}
	}

	// Column vector * Row vector (outer product)
	if len(t.Shape) == 2 && len(other.Shape) == 1 && t.Shape[1] == 1 {
		// Treating t as a column vector and other as a row vector
		rows := t.Shape[0]
		cols := other.Shape[0]
		result := make([]float64, rows*cols)

		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				result[i*cols+j] = t.Data[i] * other.Data[j]
			}
		}

		return &Tensor{
			Shape: []int{rows, cols},
			Data:  result,
		}
	}

	// Matrix * Vector
	if len(t.Shape) == 2 && len(other.Shape) == 1 {
		if t.Shape[1] != other.Shape[0] {
			panic(fmt.Sprintf("Tensor shapes do not match for multiplication: %v and %v", t.Shape, other.Shape))
		}

		rows := t.Shape[0]
		result := make([]float64, rows)

		for i := 0; i < rows; i++ {
			for k := 0; k < t.Shape[1]; k++ {
				result[i] += t.Data[i*t.Shape[1]+k] * other.Data[k]
			}
		}

		return &Tensor{
			Shape: []int{rows},
			Data:  result,
		}
	}

	// Matrix multiplication
	if len(t.Shape) == 2 && len(other.Shape) == 2 {
		if t.Shape[1] != other.Shape[0] {
			panic(fmt.Sprintf("Tensor shapes do not match for multiplication: %v and %v", t.Shape, other.Shape))
		}

		rows := t.Shape[0]
		cols := other.Shape[1]
		result := make([]float64, rows*cols)

		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				for k := 0; k < t.Shape[1]; k++ {
					result[i*cols+j] += t.Data[i*t.Shape[1]+k] * other.Data[k*cols+j]
				}
			}
		}

		return &Tensor{
			Shape: []int{rows, cols},
			Data:  result,
		}
	}

	if len(t.Shape) == 2 && len(other.Shape) == 1 {
		// Treat vector as Nx1 matrix
		if t.Shape[1] != other.Shape[0] {
			panic(fmt.Sprintf("Tensor shapes do not match for multiplication: %v and %v", t.Shape, other.Shape))
		}

		rows := t.Shape[0]
		result := make([]float64, rows)

		for i := 0; i < rows; i++ {
			for k := 0; k < other.Shape[0]; k++ {
				result[i] += t.Data[i*t.Shape[1]+k] * other.Data[k]
			}
		}

		return &Tensor{
			Shape: []int{rows},
			Data:  result,
		}
	}

	msg := fmt.Sprintf("Tensor multiplication not supported for shapes: %v and %v", t.Shape, other.Shape)
	panic(msg)
}

// Print in human-readable form
func (t *Tensor) Print() {
	// For scalar
	if len(t.Shape) == 0 {
		fmt.Printf("%.3f\n", t.Data[0])
		return
	}

	// For vector (1D)
	if len(t.Shape) == 1 {
		fmt.Print("[ ")
		for _, v := range t.Data {
			fmt.Printf("%.3f ", v)
		}
		fmt.Println("]")
		return
	}

	// For matrix (2D)
	if len(t.Shape) == 2 {
		rows, cols := t.Shape[0], t.Shape[1]
		for i := 0; i < rows; i++ {
			fmt.Print("[ ")
			for j := 0; j < cols; j++ {
				fmt.Printf("%.3f ", t.Data[i*cols+j])
			}
			fmt.Println("]")
		}
		return
	}

	// For 3D tensors and higher
	fmt.Printf("Tensor with shape %v and data %v\n", t.Shape, t.Data)
}

func (t *Tensor) Sum() float64 {
	sum := 0.0
	for _, v := range t.Data {
		sum += v
	}
	return sum
}

func (t *Tensor) T() *Tensor {
	// Scalar transpose is the same scalar
	if len(t.Shape) == 0 {
		return &Tensor{
			Shape: []int{},
			Data:  []float64{t.Data[0]},
		}
	}

	// Vector transpose: Convert between row and column vector
	if len(t.Shape) == 1 {
		// Convert 1D vector (shape [n]) to a column vector (shape [n, 1])
		return &Tensor{
			Shape: []int{t.Shape[0], 1},
			Data:  append([]float64{}, t.Data...),
		}
	}

	// Handle special case: Convert column vector to 1D vector
	if len(t.Shape) == 2 && t.Shape[1] == 1 {
		return &Tensor{
			Shape: []int{t.Shape[0]},
			Data:  append([]float64{}, t.Data...),
		}
	}

	// Handle special case: Convert row vector to 1D vector
	if len(t.Shape) == 2 && t.Shape[0] == 1 {
		return &Tensor{
			Shape: []int{t.Shape[1]},
			Data:  append([]float64{}, t.Data...),
		}
	}

	// Matrix transpose
	if len(t.Shape) == 2 {
		rows, cols := t.Shape[0], t.Shape[1]
		result := make([]float64, len(t.Data))

		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				result[j*rows+i] = t.Data[i*cols+j]
			}
		}

		return &Tensor{
			Shape: []int{cols, rows},
			Data:  result,
		}
	}

	panic("Transpose only supported for up to 2D tensors")
}

func (t *Tensor) Equal(other *Tensor) bool {
	if len(t.Shape) != len(other.Shape) {
		return false
	}

	for i := range t.Shape {
		if t.Shape[i] != other.Shape[i] {
			return false
		}
	}

	if len(t.Data) != len(other.Data) {
		return false
	}

	for i := range t.Data {
		if t.Data[i] != other.Data[i] {
			return false
		}
	}

	return true
}

// offset calculates the start and limit offsets in the data array for slicing
func (t *Tensor) offset(indexes ...int) (int, int) {
	// Case 1: Scalar
	if len(t.Shape) == 0 {
		if len(indexes) == 0 {
			return 0, 1
		}
		panic("Cannot index into a scalar tensor")
	}

	// Case 2: Vector (1D)
	if len(t.Shape) == 1 {
		if len(indexes) == 0 {
			// Return the whole vector
			return 0, len(t.Data)
		}

		if len(indexes) == 1 {
			idx := indexes[0]
			if idx < 0 || idx >= t.Shape[0] {
				panic(fmt.Sprintf("Index %d out of bounds for tensor with shape %v", idx, t.Shape))
			}
			// Return a specific element
			return idx, idx + 1
		}

		panic(fmt.Sprintf("Too many indexes (%v) for vector with shape %v", indexes, t.Shape))
	}

	// Case 3: Matrix (2D)
	if len(t.Shape) == 2 {
		rows, cols := t.Shape[0], t.Shape[1]

		if len(indexes) == 0 {
			// Return the whole matrix
			return 0, len(t.Data)
		}

		if len(indexes) == 1 {
			// Return a whole row
			idx := indexes[0]
			if idx < 0 || idx >= rows {
				panic(fmt.Sprintf("Row index %d out of bounds for tensor with shape %v", idx, t.Shape))
			}
			return idx * cols, (idx + 1) * cols
		}

		if len(indexes) == 2 {
			// Return a specific element
			row, col := indexes[0], indexes[1]
			if row < 0 || row >= rows || col < 0 || col >= cols {
				panic(fmt.Sprintf("Indexes %v out of bounds for tensor with shape %v", indexes, t.Shape))
			}
			idx := row*cols + col
			return idx, idx + 1
		}

		panic(fmt.Sprintf("Too many indexes (%v) for matrix with shape %v", indexes, t.Shape))
	}

	// Case 4: Higher dimensional tensors
	if len(indexes) == 0 {
		// Return the whole tensor
		return 0, len(t.Data)
	}

	if len(indexes) != len(t.Shape) {
		panic(fmt.Sprintf("Number of indexes (%d) doesn't match tensor dimensions (%d)",
			len(indexes), len(t.Shape)))
	}

	// Calculate the linear offset for a specific element
	offset := 0
	stride := 1

	// Calculate offset using row-major order (last dimension varies fastest)
	for i := len(t.Shape) - 1; i >= 0; i-- {
		if indexes[i] < 0 || indexes[i] >= t.Shape[i] {
			panic(fmt.Sprintf("Index %d out of bounds at dimension %d for tensor with shape %v",
				indexes[i], i, t.Shape))
		}

		offset += indexes[i] * stride
		stride *= t.Shape[i]
	}

	// For a specific element, the range is just 1 element
	return offset, offset + 1
}
