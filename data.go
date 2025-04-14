package main

import (
	"fmt"
	"math/rand"
	"os"
	"sort"

	"github.com/itsubaki/autograd/variable"
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

	InitTextProcessor(text)
	fmt.Printf("Vocabulary size: %d\n", VocabSize())
	fmt.Printf("Vocabulary: %s\n", string(chars[:min(100, len(chars))]))

	encodedText := Encode(text)

	return encodedText, VocabSize()
}

func Encode(s string) *Tensor {
	result := make([]float64, 0, len(s))
	for _, ch := range s {
		if idx, ok := stoi[ch]; ok {
			result = append(result, float64(idx))
		}
	}

	return Tensor1D(result...)
}

func Decode(indices []int) string {
	result := make([]rune, 0, len(indices))
	for _, idx := range indices {
		if ch, ok := itos[idx]; ok {
			result = append(result, ch)
		}
	}
	return string(result)
}

// Static text processor variables at module level
var (
	chars []rune
	stoi  map[rune]int
	itos  map[int]rune
)

func VocabSize() int {
	return len(chars)
}

// Batch creates training batches from encoded Data
// Returns inputs and targets
func GetSequence(data []float64, blockSize int) (*variable.Variable, *variable.Variable) {
	// Check if we have enough data
	dataLen := len(data) - blockSize
	if dataLen <= 0 {
		panic("Not enough Data for the given block size")
	}

	// Pick a random starting point
	idx := rand.Intn(dataLen)

	// Create x and y arrays (single sequences, no batch dimension)
	x := make([]float64, blockSize)
	y := make([]float64, blockSize)

	for j := 0; j < blockSize; j++ {
		x[j] = data[idx+j]
		y[j] = data[idx+j+1] // target is the next character
	}

	// Convert to tensors (note: using Vector instead of Tensor2D)
	return variable.New(x...), variable.New(y...)
}

// InitTextProcessor initializes the static text processor with the given text
func InitTextProcessor(text string) {
	charMap := make(map[rune]bool)
	for _, ch := range text {
		charMap[ch] = true
	}

	chars = make([]rune, 0, len(charMap))
	for ch := range charMap {
		chars = append(chars, ch)
	}
	sort.Slice(chars, func(i, j int) bool {
		return chars[i] < chars[j]
	})

	stoi = make(map[rune]int)
	itos = make(map[int]rune)
	for i, ch := range chars {
		stoi[ch] = i
		itos[i] = ch
	}
}
