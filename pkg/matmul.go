// Parallelized matrix multiplication
package pkg

import (
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

	// Number of available CPU cores
	numCPU := runtime.NumCPU()

	// Only parallelize if matrices are large enough
	shouldParallel := mRows*mCols >= minSizeForParallel || nRows*nCols >= minSizeForParallel

	if shouldParallel {
		var wg sync.WaitGroup

		// Divide work by rows based on CPU count
		rowsPerWorker := (mRows + numCPU - 1) / numCPU // Ceiling division

		// Launch one goroutine per CPU core
		for workerID := 0; workerID < numCPU; workerID++ {
			wg.Add(1)

			go func(id int) {
				defer wg.Done()

				// Calculate row range for this worker
				startRow := id * rowsPerWorker
				endRow := min((id+1)*rowsPerWorker, mRows)

				// Skip if no work for this worker
				if startRow >= mRows {
					return
				}

				// Block size for tiling (cache optimization)
				const BLOCK_SIZE = 32

				// Process assigned chunk using blocking algorithm
				for ii := startRow; ii < endRow; ii += BLOCK_SIZE {
					for kk := 0; kk < mCols; kk += BLOCK_SIZE {
						for jj := 0; jj < nCols; jj += BLOCK_SIZE {
							// Calculate bounds for current block
							iEnd := min(ii+BLOCK_SIZE, endRow)
							kEnd := min(kk+BLOCK_SIZE, mCols)
							jEnd := min(jj+BLOCK_SIZE, nCols)

							// Process the current block with cache-friendly access
							for i := ii; i < iEnd; i++ {
								for k := kk; k < kEnd; k++ {
									mik := m[i][k]
									for j := jj; j < jEnd; j++ {
										out[i][j] += mik * n[k][j]
									}
								}
							}
						}
					}
				}
			}(workerID)
		}

		wg.Wait()
	} else {
		// Sequential version with blocking for small matrices
		const BLOCK_SIZE = 32

		for ii := 0; ii < mRows; ii += BLOCK_SIZE {
			for kk := 0; kk < mCols; kk += BLOCK_SIZE {
				for jj := 0; jj < nCols; jj += BLOCK_SIZE {
					iEnd := min(ii+BLOCK_SIZE, mRows)
					kEnd := min(kk+BLOCK_SIZE, mCols)
					jEnd := min(jj+BLOCK_SIZE, nCols)

					for i := ii; i < iEnd; i++ {
						for k := kk; k < kEnd; k++ {
							mik := m[i][k]
							for j := jj; j < jEnd; j++ {
								out[i][j] += mik * n[k][j]
							}
						}
					}
				}
			}
		}
	}

	return out
}
