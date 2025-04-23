package main

import (
	"math"
	"testing"

	"github.com/itsubaki/autograd/variable"
)

func TestNeuron(t *testing.T) {
	// Our neuron has 2 inputs and 1 output (number of columns in weight matrix).
	// Its goal is to predict next number in the sequence.
	input := V{1, 2} // {x1, x2}
	weight := M{
		{2}, // how much x1 contributes to the output
		{3}, // how much x2 contributes to the output
	}

	// We calculate the output by multiplying the input vector with the weight matrix.
	output := MatMul(input.Var(), weight.Var())
	// output[0] = 1*2 + 2*3 = 8
	areEqual(t, V{8}, output)

	// That's a bad prediction (not equal to 3), so we have to adjust the weight.
	weight = M{
		{1}, // nudge the first weight down by 1
		{1}, // nudge the second weight down by 2
	}

	output = MatMul(input.Var(), weight.Var())
	// output = 1*-1 + 2*2 = 3
	// Now our neuron's prediction matches the target, so the weight are correct.
	// In reality, though, we don't tune the weight manually.
	areEqual(t, V{3}, output)
}

func TestLoss(t *testing.T) {
	input := V{1, 2}.Var()
	weight := M{
		{2}, // w1
		{3}, // w2
	}.Var()
	output := MatMul(input, weight)
	target := V{3}.Var()

	// We need a single number to characterize how bad our prediction is.
	// For that we need a loss function.
	// Let's pick the simplest one:
	// loss = prediction - target
	loss := Sub(output, target)
	// 8 - 3 = 5
	areEqual(t, V{5}, loss)
	// So the loss is 5, which is a bad loss (we should strive to 0).

	// Both input (x1, x2) and weights (w1, w2) values contribute to the loss.
	// So, to minimize the loss we can tune the input or the weights.
	// Since we are training a model, we want to tune the weights. Inputs come from fixed dataset.
	// So we need to calculate how much w1 and w2 contribute to the loss.
	// "By what amount the loss would change if we change x1 and x2 by some tiny value".
	// That's derivative. The speed of change of the loss with respect to the weights.

	// dLoss/dW1 = (x1 * w1 + x2 * w2) = x1 = 1
	// It means that to make our loss bigger, we need to move our x1 into positive direction (+),  1.
	// We want the opposite - to make it smaller, so we need to - 1 from our weight.
	// dLoss/dW2 = (x1 * w1 + x2 * w2) = x2 = 2

	// We calculate how much each weight contributes to the predicted value.
	// So that we know in which direction to nudge the weight to minimize the loss.
	loss.Backward()
	areEqual(t, V{1, 2}, weight.Grad)
}

func TestGradientDescent(t *testing.T) {
}

// TODO add more in-between explanations

func TestTrainingLoop(t *testing.T) {
	RandEmbeds = func(rows, cols int) *variable.Variable {
		return Zeros(rows, cols)
	}

	RandWeights = func(rows, cols int) *variable.Variable {
		return Zeros(rows, cols)
	}
}

//func mockDataset() string {
//	return "a\n\n"
//}
//
//func mockVocab() string {
//	return "[\\u000a][\\u000a] -> [\\u000a\\u000a]"
//}
//
//// Mock random initialization to deterministic values
//func mockRandEmbeds(rows, cols int) *variable.Variable {
//	// Create a simple pattern: [[1,2],[3,4],[5,6],...] for testing
//	data := make([][]float64, rows)
//	for i := 0; i < rows; i++ {
//		data[i] = make([]float64, cols)
//		for j := 0; j < cols; j++ {
//			data[i][j] = float64(i*cols + j + 1)
//		}
//	}
//	return variable.NewOf(data...)
//}
//
//func mockRandWeights(rows, cols int) *variable.Variable {
//	// Return all ones for simple multiplication
//	data := make([][]float64, rows)
//	for i := 0; i < rows; i++ {
//		data[i] = make([]float64, cols)
//		for j := 0; j < cols; j++ {
//			data[i][j] = 1.0
//		}
//	}
//	return variable.NewOf(data...)
//}
//
//func TestTransformerTrainingLoop(t *testing.T) {
//	// Test hyperparameters
//	const (
//		blockSize    = 2    // Just enough for our test input
//		embedSize    = 2    // Small embedding size for simplicity
//		heads        = 1    // Minimum attention heads
//		layers       = 1    // Just one transformer layer
//		epochs       = 1    // Run just one epoch for testing
//		learningRate = 0.01 // Higher learning rate for visible movement
//		lossScale    = 1.0  // No scaling
//	)
//
//	// Mock functions
//	origRandEmbeds := RandEmbeds
//	origRandWeights := RandWeights
//	origDataset := data.Dataset
//	origVocab := data.Vocab
//
//	defer func() {
//		// Restore original functions after test
//		RandEmbeds = origRandEmbeds
//		RandWeights = origRandWeights
//		data.Dataset = origDataset
//		data.Vocab = origVocab
//	}()
//
//	// Set mocks
//	RandEmbeds = mockRandEmbeds
//	RandWeights = mockRandWeights
//	data.Dataset = mockDataset
//	data.Vocab = mockVocab
//
//	// Initial setup
//	dataset, vocabSize := data.Tokenize(100) // Number doesn't matter for our mock
//
//	// Print test data for verification
//	fmt.Printf("Test dataset: %v\n", dataset)
//	fmt.Printf("Scalar characters: %s\n", data.Decode(dataset[:2]...))
//	fmt.Printf("Vocabulary: %s\n", data.Characters())
//
//	// Initialize model components with manually created values
//	// Token embeddings: create distinct, intuitive vectors for each token
//	// - Token 0 ('a'): [1, 0] - points right, representing a standard character
//	// - Token 1 ('\n'): [0, 1] - points up, representing a line break
//	// - Token 2 ('\n\n'): [2, 2] - points up and right with larger magnitude, representing a paragraph break
//	tokEmbeds := pkg.M{
//		{1, 0}, // 'a' - horizontal direction
//		{0, 1}, // '\n' - vertical direction
//		{2, 2}, // '\n\n' - diagonal with larger magnitude (stronger signal)
//	}.Var()
//
//	// Position embeddings: create vectors that emphasize position in sequence
//	// - Position 0 (first token): [0.5, 0.1] - small values as base position
//	// - Position 1 (second token): [0.1, 0.9] - larger vertical component to emphasize it's later in sequence
//	posEmbeds := pkg.M{
//		{0.5, 0.1}, // first position - mostly horizontal
//		{0.1, 0.9}, // second position - mostly vertical, larger magnitude in second dimension
//	}.Var()
//	fmt.Printf("Token embeddings:\n%v\n", tokEmbeds.Data)
//	fmt.Printf("Position embeddings:\n%v\n", posEmbeds.Data)
//
//	var blocks []*Block
//	for range layers {
//		blocks = append(blocks, NewBlock(embedSize, heads))
//	}
//	norm := pkg.NewLayerNorm(embedSize)
//	lmHead := NewLinear(embedSize, vocabSize)
//
//	// Collecting parameters
//	params := pkg.NewParams()
//	params.Add(tokEmbeds, posEmbeds)
//	for _, block := range blocks {
//		params.Add(block.Params()...)
//	}
//	params.Add(norm.Scale, norm.Shift)
//	params.Add(lmHead.Weight, lmHead.Bias)
//
//	optimizer := pkg.NewAdamW(learningRate)
//
//	// Run a single training iteration for testing
//	input, targets := data.Sample(dataset, blockSize)
//	fmt.Printf("Input tokens: %v\n", input.Data)
//	fmt.Printf("Target tokens: %v\n", targets.Data)
//
//	// Forward pass with debugging
//	embeds := pkg.Rows(tokEmbeds, input.Data[0]...) // get embed for every input token
//	fmt.Printf("Initial token embeddings:\n%v\n", embeds.Data)
//
//	embeds = Add(embeds, posEmbeds) // add positional embedding
//	fmt.Printf("After adding positional embeddings:\n%v\n", embeds.Data)
//
//	for i, block := range blocks {
//		embeds = block.Forward(embeds)
//		fmt.Printf("After transformer block %d:\n%v\n", i, embeds.Data)
//	}
//
//	embeds = norm.Forward(embeds)
//	fmt.Printf("After layer normalization:\n%v\n", embeds.Data)
//
//	logits := lmHead.Forward(embeds) // converts contextual embeddings to next-token predictions
//	fmt.Printf("Final logits (next token predictions):\n%v\n", logits.Data)
//
//	// Loss calculation
//	loss := CrossEntropy(logits, targets)
//	loss = MulC(lossScale, loss)
//	fmt.Printf("Initial loss: %.5f\n", loss.Data[0][0])
//
//	// Capture the initial logits for comparison
//	initialLogits := make([][]float64, len(logits.Data))
//	for i := range logits.Data {
//		initialLogits[i] = make([]float64, len(logits.Data[i]))
//		copy(initialLogits[i], logits.Data[i])
//	}
//
//	// Backward pass and parameter update
//	loss.Backward()
//	optimizer.Update(params)
//	params.ZeroGrad()
//}

func areEqual[T V | M](t *testing.T, want T, got *variable.Variable) {
	t.Helper()

	// Convert want to a 2D slice of float64
	var wantData [][]float64
	switch v := any(want).(type) {
	case V:
		// Convert V (vector) to 1xN matrix
		wantData = make([][]float64, 1)
		wantData[0] = make([]float64, len(v))
		copy(wantData[0], v)
	case M:
		// Convert M (matrix) to [][]float64
		wantData = make([][]float64, len(v))
		for i, row := range v {
			wantData[i] = make([]float64, len(row))
			copy(wantData[i], row)
		}
	default:
		t.Fatalf("unexpected type %T", want)
	}

	if len(wantData) != len(got.Data) {
		t.Errorf("dimension mismatch: want rows=%d, got rows=%d", len(wantData), len(got.Data))
		return
	}

	for i := range wantData {
		if len(wantData[i]) != len(got.Data[i]) {
			t.Errorf("dimension mismatch at row %d: want cols=%d, got cols=%d", i, len(wantData[i]), len(got.Data[i]))
			return
		}
		for j := range wantData[i] {
			// Use a small epsilon for floating point comparison
			epsilon := 1e-9
			if math.Abs(wantData[i][j]-got.Data[i][j]) > epsilon {
				t.Errorf("value mismatch at [%d][%d]: want=%f, got=%f", i, j, wantData[i][j], got.Data[i][j])
			}
		}
	}
}

type M [][]float64

func (m M) Var() *variable.Variable {
	return variable.NewOf(m...)
}

type V []float64

func (v V) Var() *variable.Variable {
	return variable.NewOf(v)
}
