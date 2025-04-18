package data

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestAddCharactersToVocabulary(t *testing.T) {
	r := require.New(t)

	chars = nil
	stoi = nil
	itos = nil
	longestTokens = nil

	testText := "hello world"
	AddCharactersToVocabulary(testText)

	r.Equal(8, len(chars))
	r.Contains(stoi, 'h', "Vocabulary should contain 'h'")
	r.Contains(stoi, 'e', "Vocabulary should contain 'e'")
	r.Contains(stoi, 'l', "Vocabulary should contain 'l'")
	r.Contains(stoi, 'o', "Vocabulary should contain 'o'")
	r.Contains(stoi, ' ', "Vocabulary should contain space")
	r.Contains(stoi, 'w', "Vocabulary should contain 'w'")
	r.Contains(stoi, 'r', "Vocabulary should contain 'r'")
	r.Contains(stoi, 'd', "Vocabulary should contain 'd'")

	for i, ch := range chars {
		r.Equal(i, stoi[ch], "stoi[ch] should map to index")
		r.Equal(ch, itos[i], "itos[i] should map back to character")
	}
}

func TestAddSubwordTokensToVocabulary(t *testing.T) {
	r := require.New(t)

	chars = []rune{'a', 'b', 'c'}
	stoi = map[rune]int{'a': 0, 'b': 1, 'c': 2}
	itos = map[int]rune{0: 'a', 1: 'b', 2: 'c'}
	tokenStoi = nil
	tokenItos = nil
	longestTokens = nil

	AddSubwordTokensToVocabulary("test\ntoken\nlongest_token")

	// Check token mappings
	r.Equal(3, len(chars), "Characters vocabulary should remain unchanged")
	r.Equal(3, len(tokenStoi), "Should have 3 subword tokens")

	// Check token IDs (should be baseVocabSize + index)
	r.Equal(3, tokenStoi["test"], "First token should have ID 3")
	r.Equal(4, tokenStoi["token"], "Second token should have ID 4")
	r.Equal(5, tokenStoi["longest_token"], "Third token should have ID 5")

	// Check reverse mappings
	r.Equal("test", tokenItos[3], "ID 3 should map to 'test'")
	r.Equal("token", tokenItos[4], "ID 4 should map to 'token'")
	r.Equal("longest_token", tokenItos[5], "ID 5 should map to 'longest_token'")

	// Check longestTokens sorting
	r.Equal(3, len(longestTokens), "Should have 3 tokens in longestTokens")
	r.Equal("longest_token", longestTokens[0], "Longest token should be first")
}

func TestEncode(t *testing.T) {
	r := require.New(t)

	chars = []rune{'h', 'e', 'l', 'o', ' ', 'w', 'r', 'd'}
	stoi = map[rune]int{
		'h': 0, 'e': 1, 'l': 2, 'o': 3, ' ': 4, 'w': 5, 'r': 6, 'd': 7,
	}
	itos = map[int]rune{
		0: 'h', 1: 'e', 2: 'l', 3: 'o', 4: ' ', 5: 'w', 6: 'r', 7: 'd',
	}

	tokenStoi = map[string]int{
		"hello": 8, "world": 9,
	}
	tokenItos = map[int]string{
		8: "hello", 9: "world",
	}
	longestTokens = []string{"hello", "world"}

	// Test encoding with subword tokens
	encoded := Encode("hello world")
	expected := []float64{8.0, 4.0, 9.0}
	r.Equal(expected, encoded, "Should encode using subword tokens when possible")

	// Test encoding with single characters
	encoded = Encode("he world")
	expected = []float64{0.0, 1.0, 4.0, 9.0}
	r.Equal(expected, encoded, "Should fall back to character encoding when no token matches")
}

func TestDecode(t *testing.T) {
	r := require.New(t)

	chars = []rune{'h', 'e', 'l', 'o', ' ', 'w', 'r', 'd'}
	stoi = map[rune]int{
		'h': 0, 'e': 1, 'l': 2, 'o': 3, ' ': 4, 'w': 5, 'r': 6, 'd': 7,
	}
	itos = map[int]rune{
		0: 'h', 1: 'e', 2: 'l', 3: 'o', 4: ' ', 5: 'w', 6: 'r', 7: 'd',
	}

	tokenStoi = map[string]int{
		"hello": 8, "world": 9,
	}
	tokenItos = map[int]string{
		8: "hello", 9: "world",
	}

	// Test decoding with subword tokens and characters
	decoded := Decode(8.0, 4.0, 9.0)
	r.Equal("hello world", decoded, "Should decode mix of tokens and characters correctly")

	// Test decoding with only characters
	decoded = Decode(0.0, 1.0, 2.0, 2.0, 3.0)
	r.Equal("hello", decoded, "Should decode single characters correctly")
}

func TestVocabSize(t *testing.T) {
	r := require.New(t)

	chars = []rune{'a', 'b', 'c'}
	tokenStoi = map[string]int{
		"token1": 3, "token2": 4,
	}

	size := VocabSize()
	r.Equal(5, size)
}

func TestSample(t *testing.T) {
	r := require.New(t)

	testData := []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	blockSize := 3

	x, y := Sample(testData, blockSize)
	fmt.Println(x, y)

	// Test the shapes of returned variables
	r.Equal(blockSize, len(x.Data[0]))
	r.Equal(blockSize, len(y.Data[0]))

	// Test that y is shifted by one from x
	for i := 0; i < blockSize; i++ {
		idx := -1
		for j := 0; j < len(testData)-blockSize; j++ {
			if testData[j] == x.Data[0][0] &&
				testData[j+1] == x.Data[0][1] &&
				testData[j+2] == x.Data[0][2] {
				idx = j
				break
			}
		}

		r.NotEqual(-1, idx)
		if idx != -1 {
			r.Equal(testData[idx+1], y.Data[0][0])
			r.Equal(testData[idx+2], y.Data[0][1])
			r.Equal(testData[idx+3], y.Data[0][2])
		}
	}

	// Test panic condition
	r.Panics(func() {
		Sample([]float64{1, 2}, 3)
	}, "Should panic when data is smaller than blockSize+1")
}
