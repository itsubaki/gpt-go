package main

const (
	vocabSize = 64
	embedSize = 32
	blockSize = 8
)

// Embeddings are basically tensors under the hood
// What if we codegenerate files for different tensors/linear layers
func main() {

	tokenEmbeds := Zero(vocabSize, embedSize)
	positionEmbeds := Zero(blockSize, embedSize)

	tokenEmbeds.Print()
	positionEmbeds.Print()

}
