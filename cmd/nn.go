package main

// linear represents a linear (fully connected) layer
type linear struct {
	inFeatures  int
	outFeatures int
	weights     *Matrix
	bias        *Matrix
	weightsGrad *Gradient
	biasGrad    *Gradient
}

// NewLinear creates a new linear layer
func Linear(inFeatures, outFeatures int, seed int64) *linear {
	return &linear{
		inFeatures:  inFeatures,
		outFeatures: outFeatures,
		weights:     RandomMatrix(inFeatures, outFeatures, seed),
		bias:        NewMatrix(1, outFeatures),
		weightsGrad: NewGradient(inFeatures, outFeatures),
		biasGrad:    NewGradient(1, outFeatures),
	}
}

// Forward performs a forward pass through the linear layer
func (l *linear) Forward(input *Matrix) *Matrix {
	batchSize := input.rows
	output := NewMatrix(batchSize, l.outFeatures)

	for b := 0; b < batchSize; b++ {
		for o := 0; o < l.outFeatures; o++ {
			val := 0.0
			for i := 0; i < l.inFeatures; i++ {
				val += input.Get(b, i) * l.weights.Get(i, o)
			}

			val += l.bias.Get(0, o)

			output.Set(b, o, val)
		}
	}

	return output
}

func (l *linear) Parameters() []*Matrix {
	return []*Matrix{l.weights, l.bias}
}

func (l *linear) Gradients() []*Gradient {
	return []*Gradient{l.weightsGrad, l.biasGrad}
}

func (l *linear) ZeroGrad() {
	l.weightsGrad.Zero()
	l.biasGrad.Zero()
}

type BackwardLinear struct {
	linear      *linear
	inputCache  *Matrix
	weightsGrad *Gradient
	biasGrad    *Gradient
}

func NewBackwardLinear(linear *linear) *BackwardLinear {
	return &BackwardLinear{
		linear:      linear,
		weightsGrad: linear.weightsGrad,
		biasGrad:    linear.biasGrad,
	}
}

func (bl *BackwardLinear) Forward(input *Matrix) *Matrix {
	bl.inputCache = input
	return bl.linear.Forward(input)
}

func (bl *BackwardLinear) Backward(outputGrad *Matrix) *Matrix {
	batchSize := outputGrad.rows

	// Zero out the gradients
	bl.weightsGrad.Zero()
	bl.biasGrad.Zero()

	// Create input gradient matrix
	inputGrad := NewMatrix(batchSize, bl.linear.inFeatures)

	for b := 0; b < batchSize; b++ {
		for o := 0; o < bl.linear.outFeatures; o++ {
			outputGrad := outputGrad.Get(b, o)

			bl.biasGrad.Add(0, o, outputGrad)

			// For each input feature
			for i := 0; i < bl.linear.inFeatures; i++ {
				// Add to weights gradient
				inputVal := bl.inputCache.Get(b, i)
				bl.weightsGrad.Add(i, o, outputGrad*inputVal)

				// Add to input gradient
				weightVal := bl.linear.weights.Get(i, o)
				inputGrad.Set(b, i, inputGrad.Get(b, i)+outputGrad*weightVal)
			}
		}
	}

	return inputGrad
}

// Enhanced embedding module with better integration
type embedding struct {
	vocabSize   int
	embedSize   int
	weights     *Matrix
	weightsGrad *Gradient
}

// NewEmbedding creates a new embedding layer
func Embedding(vocabSize, embedSize int, seed int64) *embedding {
	return &embedding{
		vocabSize:   vocabSize,
		embedSize:   embedSize,
		weights:     RandomMatrix(vocabSize, embedSize, seed),
		weightsGrad: NewGradient(vocabSize, embedSize),
	}
}

// Forward performs a forward pass through the embedding layer
func (e *embedding) Forward(indices []int) *Matrix {
	batchSize := len(indices)
	result := NewMatrix(batchSize, e.embedSize)

	for i, idx := range indices {
		if idx >= 0 && idx < e.vocabSize {
			for j := 0; j < e.embedSize; j++ {
				result.Set(i, j, e.weights.Get(idx, j))
			}
		}
	}

	return result
}

// Forward2D performs a forward pass with 2D indices (batch, sequence)
func (e *embedding) Forward2D(indices [][]int) *Tensor {
	batchSize := len(indices)
	seqLen := len(indices[0])
	result := NewTensor(batchSize, seqLen, e.embedSize)

	for b := 0; b < batchSize; b++ {
		for t := 0; t < seqLen; t++ {
			idx := indices[b][t]
			if idx >= 0 && idx < e.vocabSize {
				for j := 0; j < e.embedSize; j++ {
					result.Set(b, t, j, e.weights.Get(idx, j))
				}
			}
		}
	}

	return result
}

// Parameters returns the parameters for the optimizer
func (e *embedding) Parameters() []*Matrix {
	return []*Matrix{e.weights}
}

// Gradients returns the gradients for the optimizer
func (e *embedding) Gradients() []*Gradient {
	return []*Gradient{e.weightsGrad}
}

// ZeroGrad sets all gradients to zero
func (e *embedding) ZeroGrad() {
	e.weightsGrad.Zero()
}

// BackwardEmbedding computes gradients for the embedding layer
type BackwardEmbedding struct {
	embedding    *embedding
	indicesCache [][]int
}

// NewBackwardEmbedding creates a backward embedding layer
func NewBackwardEmbedding(embedding *embedding) *BackwardEmbedding {
	return &BackwardEmbedding{
		embedding: embedding,
	}
}

// Forward caches the indices for backpropagation
func (be *BackwardEmbedding) Forward2D(indices [][]int) *Tensor {
	be.indicesCache = indices
	return be.embedding.Forward2D(indices)
}

func (be *BackwardEmbedding) Backward(outputGrad *TensorGradient) {
	be.embedding.weightsGrad.Zero()

	// Accumulate gradients for each output position
	batchSize := len(be.indicesCache)
	seqLen := len(be.indicesCache[0])

	for b := 0; b < batchSize; b++ {
		for t := 0; t < seqLen; t++ {
			idx := be.indicesCache[b][t]

			if idx >= 0 && idx < be.embedding.weights.rows {
				for c := 0; c < be.embedding.embedSize; c++ {
					grad := outputGrad.Get(b, t, c)
					be.embedding.weightsGrad.Add(idx, c, grad)
				}
			}
		}
	}
}
