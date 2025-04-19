package main

import (
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat"
)

const (
	Iterations = 5000
)

// Original implementation for reference
func MatMulOriginal(m Matrix, n Matrix) Matrix {
	a, b := len(m), len(m[0])
	_, p := len(n), len(n[0])

	out := Zero(a, p)
	for i := range a {
		for j := range p {
			for k := 0; k < b; k++ {
				out[i][j] = out[i][j] + m[i][k]*n[k][j]
			}
		}
	}
	return out
}

// Zero creates a zero matrix of size a×p
func Zero(a, p int) [][]float64 {
	out := make([][]float64, a)
	for i := range out {
		out[i] = make([]float64, p)
	}
	return out
}

// Optimization 1: Loop reordering for better cache locality
// This reorders loops to access memory in a more sequential pattern
func MatMulLoopReordered(m Matrix, n Matrix) Matrix {
	a, b := len(m), len(m[0])
	_, p := len(n), len(n[0])

	out := Zero(a, p)
	for i := 0; i < a; i++ {
		for k := 0; k < b; k++ {
			for j := 0; j < p; j++ {
				out[i][j] += m[i][k] * n[k][j]
			}
		}
	}
	return out
}

// Optimization 2: Loop tiling/blocking
// This improves cache utilization by operating on small blocks at a time
func MatMulBlocked(m Matrix, n Matrix) Matrix {
	a, b := len(m), len(m[0])
	_, p := len(n), len(n[0])

	out := Zero(a, p)
	const BLOCK_SIZE = 32 // Tune this based on your CPU's cache size

	// Iterate over blocks
	for ii := 0; ii < a; ii += BLOCK_SIZE {
		for kk := 0; kk < b; kk += BLOCK_SIZE {
			for jj := 0; jj < p; jj += BLOCK_SIZE {
				// Bounds for current block
				iEnd := min(ii+BLOCK_SIZE, a)
				jEnd := min(jj+BLOCK_SIZE, p)
				kEnd := min(kk+BLOCK_SIZE, b)

				// Process current block
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
	return out
}

// Optimization 3: Temporary variable and loop unrolling
// This reduces memory accesses and helps the compiler optimize
func MatMulTempVar(m Matrix, n Matrix) Matrix {
	a, b := len(m), len(m[0])
	_, p := len(n), len(n[0])
	out := Zero(a, p)

	for i := 0; i < a; i++ {
		for j := 0; j < p; j++ {
			sum := 0.0
			// Main loop with manual unrolling by 4
			k := 0
			for ; k < b-3; k += 4 {
				sum += m[i][k]*n[k][j] +
					m[i][k+1]*n[k+1][j] +
					m[i][k+2]*n[k+2][j] +
					m[i][k+3]*n[k+3][j]
			}
			// Handle remaining elements
			for ; k < b; k++ {
				sum += m[i][k] * n[k][j]
			}
			out[i][j] = sum
		}
	}
	return out
}

// Optimization 4: Pre-transpose second matrix for better cache behavior
func MatMulTranspose(m Matrix, n Matrix) Matrix {
	a, b := len(m), len(m[0])
	_, p := len(n), len(n[0])

	out := Zero(a, p)

	// Transpose n matrix for better memory access pattern
	nT := make([][]float64, p)
	for j := range nT {
		nT[j] = make([]float64, b)
		for k := 0; k < b; k++ {
			nT[j][k] = n[k][j]
		}
	}

	// Now multiply with the transposed matrix
	for i := 0; i < a; i++ {
		for j := 0; j < p; j++ {
			sum := 0.0
			for k := 0; k < b; k++ {
				sum += m[i][k] * nT[j][k] // Sequential memory access for both m and nT
			}
			out[i][j] = sum
		}
	}
	return out
}

// Create random matrix of size a×b with float64 values
func RandomMatrixF64(a, b int) [][]float64 {
	m := make([][]float64, a)
	for i := range m {
		m[i] = make([]float64, b)
		for j := range m[i] {
			m[i][j] = rand.Float64()
		}
	}
	return m
}

// Create random matrix of size a×b with float32 values
func RandomMatrixF32(a, b int) [][]float32 {
	m := make([][]float32, a)
	for i := range m {
		m[i] = make([]float32, b)
		for j := range m[i] {
			m[i][j] = rand.Float32()
		}
	}
	return m
}

// Create flat random matrix (1D array) of size a×b
func RandomMatrixFlatF64(a, b int) []float64 {
	flat := make([]float64, a*b)
	for i := range flat {
		flat[i] = rand.Float64()
	}
	return flat
}

// Matrix types
type Matrix [][]float64

// Create a random matrix of size rows×cols
func RandomMatrix(rows, cols int) Matrix {
	m := make(Matrix, rows)
	for i := range m {
		m[i] = make([]float64, cols)
		for j := range m[i] {
			m[i][j] = rand.Float64() - 0.5
		}
	}
	return m
}

// Create a zero matrix of size rows×cols
func ZeroMatrix(rows, cols int) Matrix {
	m := make(Matrix, rows)
	for i := range m {
		m[i] = make([]float64, cols)
	}
	return m
}

// Implementation 1: Naive matrix multiplication
func NaiveMatMul(a, b Matrix) Matrix {
	aRows, aCols := len(a), len(a[0])
	bRows, bCols := len(b), len(b[0])

	if aCols != bRows {
		panic("Incompatible matrix dimensions")
	}

	c := ZeroMatrix(aRows, bCols)

	for i := 0; i < aRows; i++ {
		for j := 0; j < bCols; j++ {
			for k := 0; k < aCols; k++ {
				c[i][j] += a[i][k] * b[k][j]
			}
		}
	}

	return c
}

// Implementation 2: Simple parallel matrix multiplication (one goroutine per row)
func SimpleParallelMatMul(a, b Matrix) Matrix {
	aRows, aCols := len(a), len(a[0])
	bRows, bCols := len(b), len(b[0])

	if aCols != bRows {
		panic("Incompatible matrix dimensions")
	}

	c := ZeroMatrix(aRows, bCols)
	var wg sync.WaitGroup

	for i := 0; i < aRows; i++ {
		wg.Add(1)
		go func(row int) {
			defer wg.Done()
			for j := 0; j < bCols; j++ {
				sum := 0.0
				for k := 0; k < aCols; k++ {
					sum += a[row][k] * b[k][j]
				}
				c[row][j] = sum
			}
		}(i)
	}

	wg.Wait()
	return c
}

// Implementation 3: Blocked matrix multiplication (cache-friendly)
func BlockedMatMul(a, b Matrix) Matrix {
	aRows, aCols := len(a), len(a[0])
	bRows, bCols := len(b), len(b[0])

	if aCols != bRows {
		panic("Incompatible matrix dimensions")
	}

	c := ZeroMatrix(aRows, bCols)

	// Block size for better cache utilization
	const BLOCK_SIZE = 32

	for ii := 0; ii < aRows; ii += BLOCK_SIZE {
		for kk := 0; kk < aCols; kk += BLOCK_SIZE {
			for jj := 0; jj < bCols; jj += BLOCK_SIZE {
				// Calculate bounds for current block
				iEnd := min(ii+BLOCK_SIZE, aRows)
				kEnd := min(kk+BLOCK_SIZE, aCols)
				jEnd := min(jj+BLOCK_SIZE, bCols)

				// Process the current block
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

	return c
}

// Implementation 4: Parallel blocked matrix multiplication (worker per CPU)
func ParallelBlockedMatMul(a, b Matrix) Matrix {
	aRows, aCols := len(a), len(a[0])
	bRows, bCols := len(b), len(b[0])

	if aCols != bRows {
		panic("Incompatible matrix dimensions")
	}

	c := ZeroMatrix(aRows, bCols)

	// Number of available CPU cores
	numCPU := runtime.NumCPU()

	var wg sync.WaitGroup

	// Divide work by rows based on CPU count
	rowsPerWorker := (aRows + numCPU - 1) / numCPU // Ceiling division

	// Launch one goroutine per CPU core
	for workerID := 0; workerID < numCPU; workerID++ {
		wg.Add(1)

		go func(id int) {
			defer wg.Done()

			// Calculate row range for this worker
			startRow := id * rowsPerWorker
			endRow := min((id+1)*rowsPerWorker, aRows)

			// Skip if no work for this worker
			if startRow >= aRows {
				return
			}

			// Block size for tiling (cache optimization)
			const BLOCK_SIZE = 32

			// Process assigned chunk using blocking algorithm
			for ii := startRow; ii < endRow; ii += BLOCK_SIZE {
				for kk := 0; kk < aCols; kk += BLOCK_SIZE {
					for jj := 0; jj < bCols; jj += BLOCK_SIZE {
						// Calculate bounds for current block
						iEnd := min(ii+BLOCK_SIZE, endRow)
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
		}(workerID)
	}

	wg.Wait()
	return c
}

// Implementation 5: Hybrid parallel blocked (more goroutines than CPUs)
func HybridParallelMatMul(a, b Matrix) Matrix {
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

// Implementation 6: Transposed-based matrix multiplication for better memory access
func TransposedMatMul(a, b Matrix) Matrix {
	aRows, aCols := len(a), len(a[0])
	bRows, bCols := len(b), len(b[0])

	if aCols != bRows {
		panic("Incompatible matrix dimensions")
	}

	// Transpose b matrix for better cache behavior
	bT := ZeroMatrix(bCols, bRows)
	for i := 0; i < bRows; i++ {
		for j := 0; j < bCols; j++ {
			bT[j][i] = b[i][j]
		}
	}

	// Parallel multiplication with transposed matrix
	c := ZeroMatrix(aRows, bCols)
	var wg sync.WaitGroup

	numCPU := runtime.NumCPU()
	chunkSize := max(1, aRows/(numCPU*2))

	for startRow := 0; startRow < aRows; startRow += chunkSize {
		wg.Add(1)

		go func(firstRow, lastRow int) {
			defer wg.Done()

			for i := firstRow; i < lastRow; i++ {
				for j := 0; j < bCols; j++ {
					sum := 0.0
					// Both a and bT now have good cache locality
					for k := 0; k < aCols; k++ {
						sum += a[i][k] * bT[j][k]
					}
					c[i][j] = sum
				}
			}
		}(startRow, min(startRow+chunkSize, aRows))
	}

	wg.Wait()
	return c
}

// Implementation 7: Gonum matrix multiplication
func GonumMatMul(a, b Matrix) Matrix {
	aRows, aCols := len(a), len(a[0])
	bRows, bCols := len(b), len(b[0])

	// Convert our matrices to Gonum format
	denseA := mat.NewDense(aRows, aCols, nil)
	denseB := mat.NewDense(bRows, bCols, nil)

	// Copy data into Gonum matrices
	for i := 0; i < aRows; i++ {
		for j := 0; j < aCols; j++ {
			denseA.Set(i, j, a[i][j])
		}
	}

	for i := 0; i < bRows; i++ {
		for j := 0; j < bCols; j++ {
			denseB.Set(i, j, b[i][j])
		}
	}

	// Perform multiplication
	denseC := mat.NewDense(aRows, bCols, nil)
	denseC.Mul(denseA, denseB)

	// Convert back to our Matrix format
	c := ZeroMatrix(aRows, bCols)
	for i := 0; i < aRows; i++ {
		for j := 0; j < bCols; j++ {
			c[i][j] = denseC.At(i, j)
		}
	}

	return c
}

// In main(), add this line after the other benchmarks:

// Generic benchmarking function
func benchmarkMultiplication(name string, matMulFunc func(a, b Matrix) Matrix, iterations int, a, b Matrix) {
	// Warm-up
	matMulFunc(a, b)

	start := time.Now()

	// Run the matrix multiplication the specified number of times
	for i := 0; i < iterations; i++ {
		matMulFunc(a, b)
	}

	duration := time.Since(start)
	avgTime := duration.Seconds() / float64(iterations)

	fmt.Printf("%-30s %10.6f seconds/op (total: %.2f seconds for %d iterations)\n",
		name, avgTime, duration.Seconds(), iterations)
}

func main() {
	fmt.Printf("Running on %d CPUs\n", runtime.NumCPU())

	// Seed the random number generator
	rand.Seed(time.Now().UnixNano())

	// Configure the benchmark
	iterations := Iterations
	aRows, aCols := 32, 64
	bCols := 5000

	// Create test matrices with the specified dimensions
	a := RandomMatrix(aRows, aCols)
	b := RandomMatrix(aCols, bCols)

	fmt.Printf("Multiplying matrices: %dx%d @ %dx%d = %dx%d\n",
		aRows, aCols, aCols, bCols, aRows, bCols)
	fmt.Printf("Running %d iterations of each implementation\n\n", iterations)

	// Run all implementations and benchmark them
	fmt.Println("Algorithm                      Time per op")
	fmt.Println("---------------------------------------------")

	benchmarkMultiplication("Naive", NaiveMatMul, iterations, a, b)
	benchmarkMultiplication("Simple parallel (per row)", SimpleParallelMatMul, iterations, a, b)
	benchmarkMultiplication("Blocked", BlockedMatMul, iterations, a, b)
	benchmarkMultiplication("Parallel blocked (per CPU)", ParallelBlockedMatMul, iterations, a, b)
	benchmarkMultiplication("Hybrid parallel", HybridParallelMatMul, iterations, a, b)
	benchmarkMultiplication("MatMulOriginal", MatMulOriginal, iterations, a, b)
	benchmarkMultiplication("Blocked", MatMulBlocked, iterations, a, b)
	benchmarkMultiplication("Transposed", TransposedMatMul, iterations, a, b)
	benchmarkMultiplication("Reordedred", MatMulLoopReordered, iterations, a, b)
	benchmarkMultiplication("MatmulTempVar", MatMulTempVar, iterations, a, b)
	benchmarkMultiplication("Gonum", GonumMatMul, iterations, a, b)

	fmt.Println("\nDone! Results show average time per matrix multiplication operation.")
}
