package main

import (
	"math"
	"testing"

	"github.com/itsubaki/autograd/variable"

	"github.com/zakirullin/gpt-go/data"
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
	layer.Weight = M{
		{1},
		{1},
	}.Var()
	output := layer.Forward(V{1, 2}.Var())
	areEqual(t, 3, output)
}

func TestLinearWithTwoInputs(t *testing.T) {
	layer := NewLinear(2, 1)
	layer.Weight = M{
		{1},
		{1},
	}.Var()

	// For each input row the weighted output is calculated independently.
	input := M{
		{1, 2},
		{2, 3},
	}.Var()
	output := layer.Forward(input)
	areMatricesEqual(t, M{
		{3},
		{5},
	}, output)
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
	areMatricesEqual(t, M{
		{1},
		{1},
	}, weight.Var()) // w1 = 1, w2 = 2
}

func TestSelfAttention(t *testing.T) {
	// Suppose we have the following tokens (~words) in our vocabulary:
	// 0 - "cat"
	// 1 - ", "
	// 2 - "dog"
	// 3 - " and"

	// Embeddings are a way to represent words (tokens) as vectors. It encodes the meaning of
	// the word in a high-dimensional space. Similar words are close to each other.
	embeds := M{
		{4, 1, 6}, // embedding for "cat"
		{1, 8, 1}, // embedding for ", "
		{4, 1, 7}, // embedding for "dog"
		{1, 9, 3}, // embedding for " and"
	}.Var()
	// As we can see, embeddings for "cat" and "dog" are quite similar.

	// The input to transformer models are a sequence of token embeddings.
	// So if we feed "cat, dog and" string to our transformer, we must encode it.
	// First we tokenize it, i.e. split the sentence into the tokens:
	input := V{0, 1, 2, 3}.Var()

	// Then convert it to the list of embeddings:
	//{
	//	{4, 1, 6}, // embedding for "cat"
	//	{1, 8, 3}, // embedding for ", "
	//	{4, 1, 7}, // embedding for "dog"
	//	{1, 9, 3}, // embedding for " and"
	//}
	inputEmbeds := Rows(embeds, Flat(input)...)
	areMatricesEqual(t, M{
		{4, 1, 6}, // embedding for "cat"
		{1, 8, 1}, // embedding for ", "
		{4, 1, 7}, // embedding for "dog"
		{1, 9, 3}, // embedding for " and"
	}, inputEmbeds)

	// How do we predict next token from a given sequence of tokens?
	// We can naively predict the next token by looking only at the last token (bigram model does that).
	// However, by looking at the token "and" alone, we lose the context of the previous tokens (what does "and" refer to?).

	// So, we have to somehow combine the information from the current token and all the previous tokens.
	// "cat" -> "cat", no previous tokens to look at
	// ", " -> "cat" + ", "
	// "blue" -> "cat" + ", " + "dog"
	// " and" -> "cat" + ", " + "dog" + " and"
	// Since we're operating with numerical representations of the words (embeddings), we can just add them together.
	// I.e. for token " and" we'll do that:
	// {4, 1, 6} + {1, 8, 1} + {4, 1, 7} + {1, 9, 3} = {10, 19, 17}
	// Now our resulting vector " and" combines more information from the previous tokens. Now we can predict the next
	// token more accurately, because we have more context.

	// To calculate the sum of all previous tokens, we can multiply by this triangular matrix:
	tril := M{
		{1, 0, 0, 0}, // first token attends only at itself ("cat"), it can't look into the future
		{1, 1, 0, 0}, // second token attends at itself and the previous token ( "cat" + ", ")
		{1, 1, 1, 0}, // third token attends at itself and the two previous tokens ("cat" + ", " + "dog")
		{1, 1, 1, 1}, // fourth token attends at itself and all the previous tokens ("cat" + ", " + "dog" + " and")
	}.Var()

	// So, at this point each embedding is enriched with the information from all the previous tokens.
	// That's the crux of self-attention.
	enrichedEmbeds := MatMul(tril, inputEmbeds)
	areMatricesEqual(t, M{
		{4, 1, 6},
		{5, 9, 7},
		{9, 10, 14},
		{10, 19, 17},
	}, enrichedEmbeds)

}

func TestWeightedSelfAttention(t *testing.T) {
	// In reality, though, we don't pay equal attention to all the previous tokens.
	// Some previous tokens are more interested to us, some are less.

	// Let's look at token "and" and its {1, 2, 3} embedding.
	// We treat those "1", "2" and "3" components in same way (some features we don't know about).
	// But let's split them into 3 categories:
	// Query - "what I am looking for"
	// Key - "what I can communicate"
	// Value - "what I give you"

	// Since we don't know how to split them, we can use a linear layer to learn that for us.
	// We introduce 3 linear layers: query, key and value.
	query := NewLinear(3, 3)   // converts each embedding into a query vector "what I am looking for"
	query.Weight = Zeros(3, 3) // manually set values for a good example
	query.Weight.Data.Set(1, 0, 10)

	key := NewLinear(3, 3) // converts each embedding into a key vector "what I can communicate"
	key.Weight = Zeros(3, 3)
	key.Weight.Data.Set(0, 0, 1) // first neuron is paying attention to the first component of the embedding

	value := NewLinear(3, 3) // converts each embedding into a value vector "what I give you"
	value.Weight = Ones(3, 3)

	embeds := M{
		{4, 1, 6}, // embedding for "cat"
		{1, 8, 1}, // embedding for ", "
		{4, 1, 7}, // embedding for "dog"
		{1, 9, 3}, // embedding for " and"
	}.Var()

	// Let's now extract the key and query vectors for each embed.

	k := key.Forward(embeds)
	// For our case, let's imagine that first component of our key is responsible for
	// I am "enumerable token". The bigger the value, "the more enumerable" the token is.
	areMatricesEqual(t, M{
		{4, 0, 0}, // "cat" is quite enumerable
		{1, 0, 0}, // ", " is not quite enumerable
		{4, 0, 0}, // "dog" is quite enumerable
		{1, 0, 0}, // " and" is not quite enumerable
	}, k)

	q := query.Forward(embeds)
	areMatricesEqual(t, M{
		{10, 0, 0}, // token "cat" is not looking for something enumerable
		{80, 0, 0}, // token ", " is looking for something enumerable a lot
		{10, 0, 0}, // token "dog" is not looking for something enumerable
		{90, 0, 0}, // token " and" is looking for something enumerable a lot
	}, q)

	// If we multiply q * k vectors, for each token we would answer to
	// a question "what tokens are interesting for me?".
	// Big values would indicate high interest.
	attentionScores := MatMul(q, Transpose(k))
	areMatricesEqual(t, M{
		{40, 10, 40, 10},
		{320, 80, 320, 80},
		{40, 10, 40, 10},
		{360, 90, 360, 90}, // token " and" is interested in tokens "cat" (score=360) and "dog" (score=360)
	}, attentionScores)

	tril := M{
		{1, 0, 0, 0}, // first token attends only at itself ("cat"), it can't look into the future
		{1, 1, 0, 0}, // second token attends at itself and the previous token ( "cat" + ", ")
		{1, 1, 1, 0}, // third token attends at itself and the two previous tokens ("cat" + ", " + "dog")
		{1, 1, 1, 1}, // fourth token attends at itself and all the previous tokens ("cat" + ", " + "dog" + " and")
	}.Var()
	// Previously we attended to all the previous token with the help of this tril matrix.
	// Now we only attend to those tokens in which we are interested, which is basically:
	attentionScores = MaskedInfFill(attentionScores, tril)
	no := math.Inf(-1)
	areMatricesEqual(t, M{
		{80, no, no, no},
		{640, 160, no, no},
		{80, 20, 80, no},
		{720, 180, 720, 180}, // token " and" is interested in "cat" and "dog", not so much in the others
	}, attentionScores)

	attentionScores = Softmax(attentionScores) // fancy trick to turn {1, 1, no, no} to {0.5, 0.5, 0, 0}
	areMatricesEqual(t, M{
		{1, 0, 0, 0},
		{1, 0, 0, 0},
		{0.5, 0, 0.5, 0},
		{0.5, 0, 0.5, 0}, // token " and" is interested in "cat" and "dog", not interested into others at all
	}, attentionScores)
}

func TestTransformer(t *testing.T) {
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
	// Example: for input={0,1}, targets={1,2}, meaning
	// that next token (target) after 0 is 1, next after 1 is 2.
	input, targets := data.Sample([]float64{0, 1, 2}, blockSize)

	// {
	//   {vector for tok0},
	//   {vector for tok1},
	//   ... other embeds
	// }
	embeds := Rows(tokEmbeds, Flat(input)...) // get embed for every input token
	embeds = Add(embeds, posEmbeds)           // add positional embedding
	embeds = block.Forward(embeds)
	embeds = norm.Forward(embeds)
	// {
	//   {score for tok0, ..., score for tokN}, // for input tok0
	//   {score for tok0, ..., score for tokN}, // for input tok1
	//   ... other logits
	// }
	logits := lmHead.Forward(embeds) // converts contextual embeddings to next-token predictions

	// Loss calculation, how much our predicted targets differ from the actual targets?
	loss := SoftmaxCrossEntropy(logits, targets)

	areEqual(t, 2.302585092994046, loss)
}

func areEqual(t *testing.T, want float64, got *variable.Variable) {
	t.Helper()
	if got.Data.Rows != 1 {
		t.Errorf("expected a single value, got %d values", got.Data.Rows)
		return
	}

	if math.Abs(want-Val(got)) > 1e-9 {
		t.Errorf("value mismatch: want %v, got %v", want, Val(got))
	}
}

func areMatricesEqual(t *testing.T, want M, got *variable.Variable) {
	t.Helper()
	if len(want) != got.Data.Rows {
		t.Errorf("matrix length mismatch: want length=%d, got length=%d", len(want), got.Data.Rows)
		return
	}

	for i := range want {
		if len(want[i]) != len(got.Data.Row(i)) {
			t.Errorf("matrix row length mismatch at row %d: want length=%d, got length=%d", i, len(want[i]), len(got.Data.Row(i)))
			return
		}
	}

	for i := range want {
		for j := range want[i] {
			if math.Abs(want[i][j]-got.Data.At(i, j)) > 1e-9 {
				t.Errorf("matrix mismatch at row %d, column %d: want %v, got %v", i, j, want[i][j], got.Data.At(i, j))
			}
		}
	}
}

type V []float64

func (v V) Var() *variable.Variable {
	return variable.NewOf(v)
}

type M [][]float64

func (m M) Var() *variable.Variable {
	return variable.NewOf(m...)
}
