package main

import (
	"math"
	"testing"

	"github.com/itsubaki/autograd/variable"

	"gptgo/data"
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
	areEqual(t, 8, output)

	// That's a bad prediction (not equal to 3), so we have to adjust the weight.
	weight = M{
		{1}, // nudge the first weight down by 1
		{1}, // nudge the second weight down by 2
	}

	output = MatMul(input.Var(), weight.Var())
	// output = 1*-1 + 2*2 = 3
	// Now our neuron's prediction matches the target, so the weight are correct.
	// In reality, though, we don't tune the weight manually.
	areEqual(t, 3, output)
}

func TestLinear(t *testing.T) {
	// Linear layer is a collection of neurons.
	layer := NewLinear(2, 1)
	layer.Weight = Ones(2, 1)
	output := layer.Forward(V{1, 2}.Var())
	areEqual(t, 3, output)
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
	areEqual(t, 5, loss)
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
	areMatricesEqual(t, M{{1}, {2}}, weight.Grad) // derivatives also called gradients
}

func TestGradientDescent(t *testing.T) {
	input := V{1, 2}.Var() // {x1, x2}
	weight := M{
		{2}, // w1
		{3}, // w2
	}

	// Now we know in which direction we should nudge the weights to minimize the loss.
	// Gradient for w1 is 1, it means that w1 contributes to loss proportionally.
	// Gradient for w2 is 2, it means that w2 contributes to loss twice as strongly.
	// I.e. if we nudge w2 by some tiny value 0.1, the loss will change by 0.2.
	// If we want to minimize the loss, we nudge the weights in the opposite direction.
	learningRate := 0.5
	weightGrad := V{1, 2}
	weight[0][0] -= learningRate * weightGrad[0] // w1 -= w1 * learningRate * w1Grad
	weight[1][0] -= learningRate * weightGrad[1] // w2 -= w2 * learningRate * w2Grad

	output := MatMul(input, weight.Var())
	// Previously the neuron predicted 8, now it predicts 5.5.
	areEqual(t, 5.5, output)
	// Previously the loss was 5, now it is 2.5, so the model has learned something.
	loss := Sub(output, V{3}.Var())
	areEqual(t, 2.5, loss)

	// Repeat the process.
	weight[0][0] -= learningRate * weightGrad[0]
	weight[1][0] -= learningRate * weightGrad[1]
	output = MatMul(input, weight.Var()) // MatMul({1, 2}, weights) = 3

	// The neuron predicts 3 now, which is exactly what follows after 1 and 2!
	areEqual(t, 3, output)
	loss = Sub(output, V{3}.Var())
	// The loss should be 0 now, because the neuron predicts the target value.
	areEqual(t, 0, loss)

	// Our simple model is now trained.
	// If the input is {1, 2}, the output is 3.
	// Our learning weights are:
	areMatricesEqual(t, M{{1}, {1}}, weight.Var()) // w1 = 1, w2 = 2
}

func TestTransformer(t *testing.T) {
	// Mocking
	RandEmbeds = func(rows, cols int) *variable.Variable {
		return Zeros(rows, cols)
	}
	RandWeights = func(rows, cols int) *variable.Variable {
		return Ones(rows, cols)
	}
	data.RandInt = func(_ int) int {
		return 0
	}

	vocabSize := 10
	embedSize := 2
	blockSize := 2

	// Basic transformer components
	tokEmbeds := RandEmbeds(vocabSize, embedSize)
	posEmbeds := RandEmbeds(blockSize, embedSize)
	block := NewBlock(embedSize, 1)
	norm := NewLayerNorm(embedSize)
	lmHead := NewLinear(embedSize, vocabSize)

	// Input contains blockSize consecutive tokens.
	// Targets contain the expected next token for each input token.
	// Example: for input=[0,1], targets=[1,2], meaning
	// that next token (target) after 0 is 1, next after 1 is 2.
	input, targets := data.Sample([]float64{0, 1, 2}, blockSize)

	// [
	//   [vector for tok0],
	//   [vector for tok1],
	//   ... other embeds
	// ]
	embeds := Rows(tokEmbeds, input.Data[0]...) // get embed for every input token
	embeds = Add(embeds, posEmbeds)             // add positional embedding
	embeds = block.Forward(embeds)
	embeds = norm.Forward(embeds)
	// [
	//   [score for tok0, ..., score for tokN], // for input tok0
	//   [score for tok0, ..., score for tokN], // for input tok1
	//   ... other logits
	// ]
	logits := lmHead.Forward(embeds) // converts contextual embeddings to next-token predictions

	// Loss calculation, how much our predicted targets differ from the actual targets?
	loss := CrossEntropy(logits, targets)

	areEqual(t, 2.302585092994046, loss)
}

func areEqual(t *testing.T, want float64, got *variable.Variable) {
	t.Helper()
	if math.Abs(want-Val(got)) > 1e-9 {
		t.Errorf("value mismatch: want %v, got %v", want, Val(got))
	}
}

func areMatricesEqual(t *testing.T, want M, got *variable.Variable) {
	t.Helper()
	gotMatrix := got.Data

	if len(want) != len(gotMatrix) {
		t.Errorf("matrix length mismatch: want length=%d, got length=%d", len(want), len(gotMatrix))
		return
	}

	for i := range want {
		for j := range want[i] {
			if math.Abs(want[i][j]-gotMatrix[i][j]) > 1e-9 {
				t.Errorf("matrix mismatch at row %d, column %d: want %v, got %v", i, j, want[i][j], gotMatrix[i][j])
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
