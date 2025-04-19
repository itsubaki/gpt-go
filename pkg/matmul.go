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

// Implementation 5: Hybrid parallel blocked (more goroutines than CPUs)
func matmul(a, b matrix.Matrix) matrix.Matrix {
	aRows, aCols := len(a), len(a[0])
	bRows, bCols := len(b), len(b[0])

	if aCols != bRows {
		panic("Incompatible matrix dimensions")
	}

	c := ZeroMatrix(aRows, bCols)

	var wg sync.WaitGroup

	// Number of available CPU cores
	numCPU := runtime.NumCPU()

	// Create more chunks than CPUs for better load balancing
	// Adjust the multiplier to find the optimal balance
	chunkSize := max(1, aRows/(numCPU*4))

	for startRow := 0; startRow < aRows; startRow += chunkSize {
		wg.Add(1)

		go func(firstRow, lastRow int) {
			defer wg.Done()

			// Process this chunk of rows with blocking for better cache utilization
			const BLOCK_SIZE = 32

			for ii := firstRow; ii < lastRow; ii += BLOCK_SIZE {
				for kk := 0; kk < aCols; kk += BLOCK_SIZE {
					for jj := 0; jj < bCols; jj += BLOCK_SIZE {
						// Calculate bounds for current block
						iEnd := min(ii+BLOCK_SIZE, lastRow)
						kEnd := min(kk+BLOCK_SIZE, aCols)
						jEnd := min(jj+BLOCK_SIZE, bCols)

						// Process the current block with cache-friendly access
						for i := ii; i < iEnd; i++ {
							for k := kk; k < kEnd; k++ {
								aik := a[i][k]
								for j := jj; j < jEnd; j++ {
									c[i][j] += aik * b[k][j]
								}
							}
						}
					}
				}
			}
		}(startRow, min(startRow+chunkSize, aRows))
	}

	wg.Wait()
	return c
}

func ZeroMatrix(rows, cols int) matrix.Matrix {
	m := make(matrix.Matrix, rows)
	for i := range m {
		m[i] = make([]float64, cols)
	}
	return m
}
