// Parallelized matrix multiplication
package pkg

import (
	"runtime"

	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
	"gonum.org/v1/gonum/mat"
)

const (
	minSizeForParallel = 32 * 128 // the biggest matrices are wide, usually weights (cols=vocabSize)
)

var numWorkers = 1
var pool chan struct{}

func init() {
	pool = make(chan struct{}, numWorkers)
	for i := 0; i < numWorkers; i++ {
		pool <- struct{}{}
	}
}

func SetMaxGoroutines(size int) {
	numWorkers = size

	// Drain the existing pool
	oldSize := cap(pool)
	for i := 0; i < oldSize; i++ {
		<-pool
	}

	if size <= 0 {
		size = runtime.NumCPU()
	}

	pool = make(chan struct{}, size)
	for i := 0; i < size; i++ {
		pool <- struct{}{}
	}
}

func MatMul(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{Forwarder: &MatMulT{}}).First(x...)
}

type MatMulT struct {
	x, w *variable.Variable
}

func (f *MatMulT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x, f.w = x[0], x[1]

	y := matmul(x[0].Data, x[1].Data)
	return []*variable.Variable{
		variable.NewOf(y...),
	}
}

func (f *MatMulT) Backward(gy ...*variable.Variable) []*variable.Variable {
	return []*variable.Variable{
		MatMul(gy[0], variable.Transpose(f.w)), // gy * w.T
		MatMul(variable.Transpose(f.x), gy[0]), // x.T * gy
	}
}

func matmul(m, n matrix.Matrix) matrix.Matrix {
	mRows, mCols := matrix.Dim(m)
	nRows, nCols := matrix.Dim(n)

	if mCols != nRows {
		panic("Incompatible matrix dimensions")
	}

	// Prepare flat slices for the Gonum matrices
	mFlat := make([]float64, mRows*mCols)
	nFlat := make([]float64, nRows*nCols)

	// Copy data to flat slices in one pass
	for i := 0; i < mRows; i++ {
		for j := 0; j < mCols; j++ {
			mFlat[i*mCols+j] = m[i][j]
		}
	}

	for i := 0; i < nRows; i++ {
		for j := 0; j < nCols; j++ {
			nFlat[i*nCols+j] = n[i][j]
		}
	}

	// Create Gonum matrices with pre-filled data
	mDense := mat.NewDense(mRows, mCols, mFlat)
	nDense := mat.NewDense(nRows, nCols, nFlat)

	// Perform multiplication
	result := mat.NewDense(mRows, nCols, nil)
	result.Mul(mDense, nDense)

	// Extract result data in one pass
	resultFlat := result.RawMatrix().Data

	// Convert back to our matrix format
	out := matrix.Zero(mRows, nCols)
	for i := 0; i < mRows; i++ {
		for j := 0; j < nCols; j++ {
			out[i][j] = resultFlat[i*nCols+j]
		}
	}

	return out
}
