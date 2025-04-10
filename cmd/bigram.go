// Why using matrix when there's a tensor?
package main

import (
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"sort"
)

type Gradient struct {
	rows, cols int
	data       []float64
}

// NewGradient creates a new Gradient matrix with the given dimensions
func NewGradient(rows, cols int) *Gradient {
	return &Gradient{
		rows: rows,
		cols: cols,
		data: make([]float64, rows*cols),
	}
}

// Get retrieves the gradient at (i, j)
func (g *Gradient) Get(i, j int) float64 {
	return g.data[i*g.cols+j]
}

// Set sets the gradient at (i, j)
func (g *Gradient) Set(i, j int, val float64) {
	g.data[i*g.cols+j] = val
}

// Add adds a value to the gradient at (i, j)
func (g *Gradient) Add(i, j int, val float64) {
	g.data[i*g.cols+j] += val
}

// Zero sets all gradients to zero
func (g *Gradient) Zero() {
	for i := range g.data {
		g.data[i] = 0
	}
}

// TODO also replace with tensor
// TensorGradient represents the gradients for a Tensor
type TensorGradient struct {
	dim1, dim2, dim3 int
	data             []float64
}

func NewTensorGradient(dim1, dim2, dim3 int) *TensorGradient {
	return &TensorGradient{
		dim1: dim1,
		dim2: dim2,
		dim3: dim3,
		data: make([]float64, dim1*dim2*dim3),
	}
}

// Get retrieves the gradient at (i, j, k)
func (g *TensorGradient) Get(i, j, k int) float64 {
	return g.data[(i*g.dim2+j)*g.dim3+k]
}

// Set sets the gradient at (i, j, k)
func (g *TensorGradient) Set(i, j, k int, val float64) {
	g.data[(i*g.dim2+j)*g.dim3+k] = val
}

// Add adds a value to the gradient at (i, j, k)
func (g *TensorGradient) Add(i, j, k int, val float64) {
	g.data[(i*g.dim2+j)*g.dim3+k] += val
}

// Zero sets all gradients to zero
func (g *TensorGradient) Zero() {
	for i := range g.data {
		g.data[i] = 0
	}
}

// Optimizer represents the optimizer for the model parameters
type Optimizer interface {
	Step()
	ZeroGrad()
}

// AdamOptimizer implements the Adam optimizer
type AdamOptimizer struct {
	learningRate float64
	beta1        float64
	beta2        float64
	epsilon      float64
	parameters   []*Matrix
	gradients    []*Gradient
	m            []*Gradient // First moment estimates
	v            []*Gradient // Second moment estimates
	t            int         // Time step
}

// NewAdamOptimizer creates a new Adam optimizer
func NewAdamOptimizer(learningRate float64, beta1, beta2, epsilon float64) *AdamOptimizer {
	return &AdamOptimizer{
		learningRate: learningRate,
		beta1:        beta1,
		beta2:        beta2,
		epsilon:      epsilon,
		parameters:   make([]*Matrix, 0),
		gradients:    make([]*Gradient, 0),
		m:            make([]*Gradient, 0),
		v:            make([]*Gradient, 0),
		t:            0,
	}
}

// AddParameter adds a parameter and its gradient to the optimizer
func (o *AdamOptimizer) AddParameter(param *Matrix, grad *Gradient) {
	o.parameters = append(o.parameters, param)
	o.gradients = append(o.gradients, grad)

	// Initialize moment estimates
	m := NewGradient(param.rows, param.cols)
	v := NewGradient(param.rows, param.cols)

	o.m = append(o.m, m)
	o.v = append(o.v, v)
}

// Step updates all parameters based on their gradients using Adam
func (o *AdamOptimizer) Step() {
	o.t++

	for i, param := range o.parameters {
		grad := o.gradients[i]
		m := o.m[i]
		v := o.v[i]

		// Update each parameter value using Adam
		for r := 0; r < param.rows; r++ {
			for c := 0; c < param.cols; c++ {
				// Update biased first and second moment estimates
				mVal := m.Get(r, c)
				vVal := v.Get(r, c)
				gVal := grad.Get(r, c)

				mVal = o.beta1*mVal + (1-o.beta1)*gVal
				vVal = o.beta2*vVal + (1-o.beta2)*gVal*gVal

				m.Set(r, c, mVal)
				v.Set(r, c, vVal)

				// Compute bias-corrected moment estimates
				mCorrected := mVal / (1 - math.Pow(o.beta1, float64(o.t)))
				vCorrected := vVal / (1 - math.Pow(o.beta2, float64(o.t)))

				// Update parameter
				newValue := param.Get(r, c) - o.learningRate*mCorrected/(math.Sqrt(vCorrected)+o.epsilon)
				param.Set(r, c, newValue)
			}
		}
	}
}

// ZeroGrad sets all gradients to zero
func (o *AdamOptimizer) ZeroGrad() {
	for _, grad := range o.gradients {
		grad.Zero()
	}
}

// OldBackwardEmbedding computes gradients for the embedding layer
type OldBackwardEmbedding struct {
	embedding   *OldEmbedding
	weights     *Matrix
	weightsGrad *Gradient
}

// NewBackwardEmbedding creates a backward embedding layer
func NewOldBackwardEmbedding(embedding *OldEmbedding) *OldBackwardEmbedding {
	return &OldBackwardEmbedding{
		embedding:   embedding,
		weights:     embedding.weights,
		weightsGrad: NewGradient(embedding.weights.rows, embedding.weights.cols),
	}
}

// Backward computes gradients for the embedding layer
func (be *OldBackwardEmbedding) Backward(indices [][]int, outputGrad *TensorGradient) {
	// Zero out the gradients
	be.weightsGrad.Zero()

	// Accumulate gradients for each output position
	batchSize := len(indices)
	seqLen := len(indices[0])

	for b := 0; b < batchSize; b++ {
		for t := 0; t < seqLen; t++ {
			idx := indices[b][t]

			if idx >= 0 && idx < be.weights.rows {
				for c := 0; c < be.weights.cols; c++ {
					grad := outputGrad.Get(b, t, c)
					be.weightsGrad.Add(idx, c, grad)
				}
			}
		}
	}
}

// Parameters returns the parameters and gradients for the optimizer
func (be *OldBackwardEmbedding) Parameters() (*Matrix, *Gradient) {
	return be.weights, be.weightsGrad
}

// BackwardBigramLanguageModel implements backpropagation for the BigramLanguageModel
type BackwardBigramLanguageModel struct {
	model      *BigramLanguageModel
	embedding  *OldBackwardEmbedding
	outputGrad *TensorGradient
}

// NewBackwardBigramLanguageModel creates a backward bigram language model
func NewBackwardBigramLanguageModel(model *BigramLanguageModel) *BackwardBigramLanguageModel {
	return &BackwardBigramLanguageModel{
		model:     model,
		embedding: NewOldBackwardEmbedding(model.tokenEmbeddings),
	}
}

// Backward computes gradients for the model
func (bm *BackwardBigramLanguageModel) Backward(idx [][]int, targets [][]int, logits *Tensor, loss float64) {
	batchSize := len(idx)
	seqLen := len(idx[0])
	vocabSize := bm.model.vocabSize

	// Initialize output gradients tensor if needed
	if bm.outputGrad == nil || bm.outputGrad.dim1 != batchSize || bm.outputGrad.dim2 != seqLen || bm.outputGrad.dim3 != vocabSize {
		bm.outputGrad = NewTensorGradient(batchSize, seqLen, vocabSize)
	} else {
		bm.outputGrad.Zero()
	}

	// Compute gradient of cross entropy loss with respect to softmax(logits)
	for b := 0; b < batchSize; b++ {
		for t := 0; t < seqLen; t++ {
			target := targets[b][t]

			// Extract logits for this position
			posLogits := make([]float64, vocabSize)
			for c := 0; c < vocabSize; c++ {
				posLogits[c] = logits.Get(b, t, c)
			}

			// Apply softmax
			probs := Softmax(posLogits)

			// Gradient of cross entropy is (prob - target_one_hot)
			for c := 0; c < vocabSize; c++ {
				targetValue := 0.0
				if c == target {
					targetValue = 1.0
				}

				// Set gradient for this position
				grad := probs[c] - targetValue
				bm.outputGrad.Set(b, t, c, grad/float64(batchSize*seqLen))
			}
		}
	}

	// Backward through embedding layer
	bm.embedding.Backward(idx, bm.outputGrad)
}

// Parameters returns the parameters and gradients for the optimizer
func (bm *BackwardBigramLanguageModel) Parameters() []*Matrix {
	return []*Matrix{bm.embedding.weights}
}

// Gradients returns the gradients for the optimizer
func (bm *BackwardBigramLanguageModel) Gradients() []*Gradient {
	return []*Gradient{bm.embedding.weightsGrad}
}

// TrainModel trains the model for a specified number of iterations
func TrainModel(model *BigramLanguageModel, data []int, batchSize, blockSize, iterations int, learningRate float64) {
	// Create backward model
	backwardModel := NewBackwardBigramLanguageModel(model)

	// Create Adam optimizer
	optimizer := NewAdamOptimizer(learningRate, 0.9, 0.999, 1e-8)

	// Add model parameters to optimizer
	embedWeights, embedGrads := backwardModel.embedding.Parameters()
	optimizer.AddParameter(embedWeights, embedGrads)

	// Training loop
	for iter := 0; iter < iterations; iter++ {
		// Get batch
		xb, yb := GetBatch(data, batchSize, blockSize)

		// Forward pass
		logits, loss := model.Forward(xb, yb)

		// Zero gradients
		optimizer.ZeroGrad()

		// Backward pass
		backwardModel.Backward(xb, yb, logits, loss)

		// Update parameters
		optimizer.Step()

		// Print progress
		if (iter+1)%100 == 0 || iter == 0 {
			// Forward pass again to get updated loss
			_, updatedLoss := model.Forward(xb, yb)

			// Print loss
			if (iter+1)%1000 == 0 {
				print("\n")
			}
			if (iter+1)%100 == 0 {
				print(iter+1, " iterations, loss: ", updatedLoss, "\n")
			}
		}
	}
}

// Matrix represents a 2D matrix of float64 values
type Matrix struct {
	rows, cols int
	data       []float64
}

// NewMatrix creates a new Matrix with the given dimensions
func NewMatrix(rows, cols int) *Matrix {
	return &Matrix{
		rows: rows,
		cols: cols,
		data: make([]float64, rows*cols),
	}
}

// RandomMatrix creates a new Matrix with random values
func RandomMatrix(rows, cols int, seed int64) *Matrix {
	r := rand.New(rand.NewSource(seed))
	m := NewMatrix(rows, cols)
	for i := range m.data {
		m.data[i] = r.Float64()*2 - 1 // Random values between -1 and 1
	}
	return m
}

// Get retrieves the value at (i, j)
func (m *Matrix) Get(i, j int) float64 {
	return m.data[i*m.cols+j]
}

// Set sets the value at (i, j)
func (m *Matrix) Set(i, j int, val float64) {
	m.data[i*m.cols+j] = val
}

// Shape returns the dimensions of the matrix
func (m *Matrix) Shape() (int, int) {
	return m.rows, m.cols
}

// Tensor represents a 3D tensor of float64 values
type Tensor struct {
	dim1, dim2, dim3 int
	data             []float64
}

// NewTensor creates a new Tensor with the given dimensions
func NewTensor(dim1, dim2, dim3 int) *Tensor {
	return &Tensor{
		dim1: dim1,
		dim2: dim2,
		dim3: dim3,
		data: make([]float64, dim1*dim2*dim3),
	}
}

// Get retrieves the value at (i, j, k)
func (t *Tensor) Get(i, j, k int) float64 {
	return t.data[(i*t.dim2+j)*t.dim3+k]
}

// Set sets the value at (i, j, k)
func (t *Tensor) Set(i, j, k int, val float64) {
	t.data[(i*t.dim2+j)*t.dim3+k] = val
}

// Shape returns the dimensions of the tensor
func (t *Tensor) Shape() (int, int, int) {
	return t.dim1, t.dim2, t.dim3
}

// embedding represents an embedding layer
type OldEmbedding struct {
	vocabSize  int
	outputSize int
	weights    *Matrix
}

// NewEmbedding creates a new embedding layer
func NewEmbedding(vocabSize, outputSize int, seed int64) *OldEmbedding {
	return &OldEmbedding{
		vocabSize:  vocabSize,
		outputSize: outputSize,
		weights:    RandomMatrix(vocabSize, outputSize, seed),
	}
}

// Forward performs a forward pass through the embedding layer
func (e *OldEmbedding) Forward(indices []int) *Matrix {
	batchSize := len(indices)
	result := NewMatrix(batchSize, e.outputSize)

	for i, idx := range indices {
		if idx >= 0 && idx < e.vocabSize {
			for j := 0; j < e.outputSize; j++ {
				result.Set(i, j, e.weights.Get(idx, j))
			}
		}
	}

	return result
}

// Forward2D performs a forward pass with 2D indices (batch, sequence)
func (e *OldEmbedding) Forward2D(indices [][]int) *Tensor {
	batchSize := len(indices)
	seqLen := len(indices[0])
	result := NewTensor(batchSize, seqLen, e.outputSize)

	for b := 0; b < batchSize; b++ {
		for t := 0; t < seqLen; t++ {
			idx := indices[b][t]
			if idx >= 0 && idx < e.vocabSize {
				for j := 0; j < e.outputSize; j++ {
					result.Set(b, t, j, e.weights.Get(idx, j))
				}
			}
		}
	}

	return result
}

// Softmax applies the softmax function to a vector
func Softmax(vec []float64) []float64 {
	result := make([]float64, len(vec))
	var sum float64 = 0
	var maxVal float64 = vec[0]

	// Find max for numerical stability
	for _, v := range vec {
		if v > maxVal {
			maxVal = v
		}
	}

	// Compute exp and sum
	for i, v := range vec {
		expVal := math.Exp(v - maxVal)
		result[i] = expVal
		sum += expVal
	}

	// Normalize
	for i := range result {
		result[i] /= sum
	}

	return result
}

// CrossEntropyLoss computes the cross-entropy loss
func CrossEntropyLoss(logits *Matrix, targets []int) float64 {
	batchSize := len(targets)
	loss := 0.0

	for i, target := range targets {
		if target >= 0 && target < logits.cols {
			// Extract logits for this sample
			rowLogits := make([]float64, logits.cols)
			for j := 0; j < logits.cols; j++ {
				rowLogits[j] = logits.Get(i, j)
			}

			// Apply softmax
			probs := Softmax(rowLogits)

			// Compute negative log likelihood
			loss -= math.Log(probs[target])
		}
	}

	return loss / float64(batchSize)
}

// BigramLanguageModel represents a simple bigram language model
type BigramLanguageModel struct {
	vocabSize       int
	tokenEmbeddings *OldEmbedding
}

// NewBigramLanguageModel creates a new BigramLanguageModel
func NewBigramLanguageModel(vocabSize int) *BigramLanguageModel {
	return &BigramLanguageModel{
		vocabSize:       vocabSize,
		tokenEmbeddings: NewEmbedding(vocabSize, vocabSize, 1337),
	}
}

// Forward performs a forward pass through the model
func (m *BigramLanguageModel) Forward(idx [][]int, targets [][]int) (*Tensor, float64) {
	// idx and targets are both (B,T) arrays of integers
	logits := m.tokenEmbeddings.Forward2D(idx) // (B,T,C)

	// If targets is nil, return only logits
	if targets == nil {
		return logits, 0
	}

	// Flatten logits and targets for loss computation
	batchSize := len(idx)
	seqLen := len(idx[0])
	flatLogits := NewMatrix(batchSize*seqLen, m.vocabSize)
	flatTargets := make([]int, batchSize*seqLen)

	for b := 0; b < batchSize; b++ {
		for t := 0; t < seqLen; t++ {
			flatIdx := b*seqLen + t
			flatTargets[flatIdx] = targets[b][t]

			for c := 0; c < m.vocabSize; c++ {
				flatLogits.Set(flatIdx, c, logits.Get(b, t, c))
			}
		}
	}

	loss := CrossEntropyLoss(flatLogits, flatTargets)
	return logits, loss
}

// Generate generates new tokens
func (m *BigramLanguageModel) Generate(idx [][]int, maxNewTokens int) [][]int {
	batchSize := len(idx)
	currentSeqLen := len(idx[0])

	// Make a copy of the input to avoid modifying the original
	result := make([][]int, batchSize)
	for b := 0; b < batchSize; b++ {
		result[b] = make([]int, currentSeqLen)
		copy(result[b], idx[b])
	}

	for i := 0; i < maxNewTokens; i++ {
		// Get the predictions
		logits, _ := m.Forward(result, nil)

		// For each sequence in the batch
		for b := 0; b < batchSize; b++ {
			currentLen := len(result[b])

			// Extract logits for the last time step
			lastLogits := make([]float64, m.vocabSize)
			for c := 0; c < m.vocabSize; c++ {
				lastLogits[c] = logits.Get(b, currentLen-1, c)
			}

			// Apply softmax to get probabilities
			probs := Softmax(lastLogits)

			// Sample from the distribution
			nextIdx := sampleFromDistribution(probs)

			// Append the sampled index to the result
			result[b] = append(result[b], nextIdx)
		}
	}

	return result
}

// sampleFromDistribution samples an index from a probability distribution
func sampleFromDistribution(probs []float64) int {
	r := rand.Float64()
	cumulativeProb := 0.0

	for i, prob := range probs {
		cumulativeProb += prob
		if r < cumulativeProb {
			return i
		}
	}

	// Fallback to the last index (should rarely happen)
	return len(probs) - 1
}

func main() {
	//_ = MyTensor([]float64{1, 2, 3})
	//_ = Scalar(5)

	// Download Shakespeare Data if it doesn't exist
	filePath := "input.txt"

	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		fmt.Println("Missing Shakespeare dataset...")
		return
	}

	// Read the text file
	data, err := ioutil.ReadFile(filePath)
	if err != nil {
		fmt.Printf("Error reading file: %v\n", err)
		return
	}

	text := string(data)
	fmt.Printf("Length of text: %d characters\n", len(text))
	fmt.Printf("First 100 characters: %s\n", text[:100])

	// Create text processor for encoding/decoding
	processor := NewTextProcessor(text)
	fmt.Printf("Vocabulary size: %d\n", processor.VocabSize())
	fmt.Printf("Vocabulary: %s\n", string(processor.chars[:min(100, len(processor.chars))]))

	// Encode the entire dataset
	encodedText := processor.Encode(text)

	// Create model with the correct vocabulary size
	vocabSize := processor.VocabSize()
	model := NewBigramLanguageModel(vocabSize)

	// Training parameters
	batchSize := 4
	blockSize := 8

	// Training settings
	iterations := 5000
	learningRate := 0.001

	TrainModel(model, encodedText, batchSize, blockSize, iterations, learningRate)

	//// Generate text after training
	context := make([][]int, 1)
	context[0] = []int{encodedText[0]} // Start with first token

	generatedIndices := model.Generate(context, 500)
	generatedText := processor.Decode(generatedIndices[0])

	fmt.Println("\nGenerated text after training:")
	fmt.Println(generatedText)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

type TextProcessor struct {
	chars []rune
	stoi  map[rune]int
	itos  map[int]rune
}

func NewTextProcessor(text string) *TextProcessor {
	// Get unique characters
	charMap := make(map[rune]bool)
	for _, ch := range text {
		charMap[ch] = true
	}

	// Convert to sorted slice
	chars := make([]rune, 0, len(charMap))
	for ch := range charMap {
		chars = append(chars, ch)
	}
	sort.Slice(chars, func(i, j int) bool {
		return chars[i] < chars[j]
	})

	// Create mappings
	stoi := make(map[rune]int)
	itos := make(map[int]rune)
	for i, ch := range chars {
		stoi[ch] = i
		itos[i] = ch
	}

	return &TextProcessor{
		chars: chars,
		stoi:  stoi,
		itos:  itos,
	}
}

// Encode converts a string to a slice of integers
func (tp *TextProcessor) Encode(s string) []int {
	result := make([]int, 0, len(s))
	for _, ch := range s {
		if idx, ok := tp.stoi[ch]; ok {
			result = append(result, idx)
		}
	}
	return result
}

// Decode converts a slice of integers to a string
func (tp *TextProcessor) Decode(indices []int) string {
	result := make([]rune, 0, len(indices))
	for _, idx := range indices {
		if ch, ok := tp.itos[idx]; ok {
			result = append(result, ch)
		}
	}
	return string(result)
}

// VocabSize returns the size of the vocabulary
func (tp *TextProcessor) VocabSize() int {
	return len(tp.chars)
}

// GetBatch creates training batches from encoded Data
func GetBatch(data []int, batchSize, blockSize int) ([][]int, [][]int) {
	// Create random indices
	dataLen := len(data) - blockSize
	if dataLen <= 0 {
		panic("Not enough Data for the given block size")
	}

	// Create xb and yb arrays
	xb := make([][]int, batchSize)
	yb := make([][]int, batchSize)

	for i := 0; i < batchSize; i++ {
		// Pick a random starting point
		idx := rand.Intn(dataLen)

		// Extract block
		xb[i] = make([]int, blockSize)
		yb[i] = make([]int, blockSize)

		for j := 0; j < blockSize; j++ {
			xb[i][j] = data[idx+j]
			yb[i][j] = data[idx+j+1] // target is the next character
		}
	}

	return xb, yb
}
