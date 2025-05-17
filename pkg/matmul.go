// MatMul performs parallelized matrix multiplication that minimizes
// CPU cache misses. It divides the computation into sequential chunks
// processed by multiple goroutines in parallel.
package pkg

import (
	"runtime"
	"sync"

	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

const (
	blockSize = 32 // Group calculations by blocks for better CPU cache utilization
)

func MatMul(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{Forwarder: &MatMulT{}}).First(x...)
}

type MatMulT struct {
	x, w *variable.Variable
}

func (f *MatMulT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.x, f.w = x[0], x[1]

	y := matmul(x[0].Data, x[1].Data)
	return []*variable.Variable{y}
}

func (f *MatMulT) Backward(gy ...*variable.Variable) []*variable.Variable {
	return []*variable.Variable{
		MatMul(gy[0], variable.Transpose(f.w)), // gy * w.T
		MatMul(variable.Transpose(f.x), gy[0]), // x.T * gy
	}
}

func matmul(m, n *matrix.Matrix) *variable.Variable {
	mRows, mCols := m.Rows, m.Cols
	_, nCols := n.Rows, n.Cols

	result := Zeros(mRows, nCols)
	var wg sync.WaitGroup
	numCPU := runtime.NumCPU()
	// Create more chunks than CPUs for better load balancing
	// Adjust the multiplier to find the optimal balance
	chunkSize := max(1, mRows/(numCPU*4))
	for startRow := 0; startRow < mRows; startRow += chunkSize {
		wg.Add(1)

		go func(firstRow, lastRow int) {
			defer wg.Done()

			// Process this chunk of rows with blocking for better cache utilization
			for ii := firstRow; ii < lastRow; ii += blockSize {
				for kk := 0; kk < mCols; kk += blockSize {
					for jj := 0; jj < nCols; jj += blockSize {
						// Calculate bounds for current block
						iEnd := min(ii+blockSize, lastRow)
						kEnd := min(kk+blockSize, mCols)
						jEnd := min(jj+blockSize, nCols)

						// Process the current block with cache-friendly access
						for i := ii; i < iEnd; i++ {
							for k := kk; k < kEnd; k++ {
								aik := m.Data[i*mCols+k]
								for j := jj; j < jEnd; j++ {
									result.Data.Data[i*nCols+j] += aik * n.Data[k*nCols+j]
								}
							}
						}
					}
				}
			}
		}(startRow, min(startRow+chunkSize, mRows))
	}
	wg.Wait()

	return result
}
