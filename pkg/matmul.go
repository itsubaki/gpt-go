// Parallelized matrix multiplication
package pkg

import (
	"fmt"
	"runtime"
	"sync"

	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
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

	y := Dot(x[0].Data, x[1].Data)
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

func Dot(m, n matrix.Matrix) matrix.Matrix {
	mRows, mCols := matrix.Dim(m)
	nRows, nCols := matrix.Dim(n)

	out := matrix.Zero(mRows, nCols)

	shouldParallel := mRows*mCols >= minSizeForParallel || nRows*nCols >= minSizeForParallel
	if shouldParallel {
		fmt.Println("parallell...")
		var wg sync.WaitGroup

		for i := range mRows {
			wg.Add(1)
			worker := <-pool
			go func(row int) {
				defer wg.Done()
				defer func() { pool <- worker }() // Return worker to pool when done

				for j := range nCols {
					for k := 0; k < mCols; k++ {
						out[row][j] += m[row][k] * n[k][j]
					}
				}
			}(i)
		}

		wg.Wait()
	} else {
		for i := range mRows {
			for j := range nCols {
				for k := 0; k < mCols; k++ {
					out[i][j] += m[i][k] * n[k][j]
				}
			}
		}
	}

	return out
}
