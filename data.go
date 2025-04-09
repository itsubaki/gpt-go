package main

import (
	"fmt"
	"math/rand"
	"os"
	"sort"
)

func Data() (*Tensor, int) {
	rand.Seed(42) // TODO remove

	filePath := "input.txt"

	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		panic("Missing Shakespeare dataset...")
	}

	data, err := os.ReadFile(filePath)
	if err != nil {
		panic("Error reading Shakespeare dataset...")
	}

	text := string(data)
	fmt.Printf("Length of text: %d characters\n", len(text))
	fmt.Printf("First 100 characters: %s\n", text[:100])

	processor := NewTextProcessor(text)
	fmt.Printf("Vocabulary size: %d\n", processor.VocabSize())
	fmt.Printf("Vocabulary: %s\n", string(processor.chars[:min(100, len(processor.chars))]))

	encodedText := processor.Encode(text)

	return encodedText, processor.VocabSize()
}

// batch creates training batches from encoded Data
// Returns inputs and targets
func Batch(data []float64, batchSize, blockSize int) (*Tensor, *Tensor) {
	// Create random indices
	dataLen := len(data) - blockSize
	if dataLen <= 0 {
		panic("Not enough Data for the given block size")
	}

	// Create xb and yb arrays
	xb := make([][]float64, batchSize)
	yb := make([][]float64, batchSize)

	for i := 0; i < batchSize; i++ {
		// Pick a random starting point
		idx := rand.Intn(dataLen)

		// Extract block
		xb[i] = make([]float64, blockSize)
		yb[i] = make([]float64, blockSize)

		for j := 0; j < blockSize; j++ {
			xb[i][j] = data[idx+j]
			yb[i][j] = data[idx+j+1] // target is the next character
		}
	}

	return Tensor2D(xb), Tensor2D(yb)
}

type TextProcessor struct {
	chars []rune
	stoi  map[rune]int
	itos  map[int]rune
}

func (tp *TextProcessor) Encode(s string) *Tensor {
	result := make([]float64, 0, len(s))
	for _, ch := range s {
		if idx, ok := tp.stoi[ch]; ok {
			result = append(result, float64(idx))
		}
	}

	return Tensor1D(result...)
}

func (tp *TextProcessor) Decode(indices []int) string {
	result := make([]rune, 0, len(indices))
	for _, idx := range indices {
		if ch, ok := tp.itos[idx]; ok {
			result = append(result, ch)
		}
	}
	return string(result)
}

func (tp *TextProcessor) VocabSize() int {
	return len(tp.chars)
}

func NewTextProcessor(text string) *TextProcessor {
	charMap := make(map[rune]bool)
	for _, ch := range text {
		charMap[ch] = true
	}

	chars := make([]rune, 0, len(charMap))
	for ch := range charMap {
		chars = append(chars, ch)
	}
	sort.Slice(chars, func(i, j int) bool {
		return chars[i] < chars[j]
	})

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
