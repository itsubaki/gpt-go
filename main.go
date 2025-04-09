package main

const (
	vocabSize = 64
	embedSize = 32
	blockSize = 8
)

var tokenEmbeds = RandN(vocabSize, embedSize)
var layer = NewLinear(embedSize, vocabSize)

func forward(indexes *Tensor, targets *Tensor) {

	layer.ZeroGrad()

	//for _, index := range indexes.Data {
	//
	//}
	//
	//layer.Forward()
}

// Embeddings are basically tensors under the hood
// What if we codegenerate files for different tensors/linear layers
func main() {
	//ten := Tensor{nil}
	//positionEmbeds := RandN(blockSize, embedSize)

	//input := Tensor1D([]float64{1, 0, 1})
	//layer := NewLinear(3, 1)

}
