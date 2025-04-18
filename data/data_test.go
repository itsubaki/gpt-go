package data

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestAddCharactersToVocabulary(t *testing.T) {
	r := require.New(t)

	idToToken = make(map[int]string)
	tokenToID = make(map[string]int)
	longestTokens = nil

	testText := "hello world"
	addTokensFromText(testText)

	r.Len(longestTokens, 8)
	r.Len(tokenToID, 8)
	r.Len(idToToken, 8)
	r.Contains(tokenToID, "h")
	r.Contains(tokenToID, "e")
	r.Contains(tokenToID, "l")
	r.Contains(tokenToID, "o")
	r.Contains(tokenToID, " ")
	r.Contains(tokenToID, "w")
	r.Contains(tokenToID, "r")
	r.Contains(tokenToID, "d")
}

func TestAddPretrainedTokensToVocabulary(t *testing.T) {
	r := require.New(t)

	idToToken = make(map[int]string)
	tokenToID = make(map[string]int)
	longestTokens = nil
	addTokensFromText("abc")

	addPretrainedTokens("test\ntoken\nlongest_token", 3)

	r.Len(longestTokens, 6)
	r.Len(tokenToID, 6)
	r.Len(idToToken, 6)

	r.Equal(3, tokenToID["test"], "First token should have ID 3")
	r.Equal(4, tokenToID["token"], "Second token should have ID 4")
	r.Equal(5, tokenToID["longest_token"], "Third token should have ID 5")

	// Check reverse mappings
	r.Equal("test", idToToken[3])
	r.Equal("token", idToToken[4])
	r.Equal("longest_token", idToToken[5])

	r.Equal("longest_token", longestTokens[0])
}

func TestEncode(t *testing.T) {
	r := require.New(t)

	longestTokens = []string{"hello", "world", "h", "e", "l", "o", " ", "w", "r", "d"}
	tokenToID = map[string]int{
		"h": 0, "e": 1, "l": 2, "o": 3, " ": 4, "w": 5, "r": 6, "d": 7, "hello": 8, "world": 9,
	}
	idToToken = map[int]string{
		0: "h", 1: "e", 2: "l", 3: "o", 4: " ", 5: "w", 6: "r", 7: "d", 8: "hello", 9: "world",
	}

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

	longestTokens = []string{"hello", "world", "h", "e", "l", "o", " ", "w", "r", "d"}
	tokenToID = map[string]int{
		"h": 0, "e": 1, "l": 2, "o": 3, " ": 4, "w": 5, "r": 6, "d": 7, "hello": 8, "world": 9,
	}
	idToToken = map[int]string{
		0: "h", 1: "e", 2: "l", 3: "o", 4: " ", 5: "w", 6: "r", 7: "d", 8: "hello", 9: "world",
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

	longestTokens = []string{"hello", "world", "h", "e", "l", "o", " ", "w", "r", "d"}
	tokenToID = map[string]int{
		"h": 0, "e": 1, "l": 2, "o": 3, " ": 4, "w": 5, "r": 6, "d": 7, "hello": 8, "world": 9,
	}
	idToToken = map[int]string{
		0: "h", 1: "e", 2: "l", 3: "o", 4: " ", 5: "w", 6: "r", 7: "d", 8: "hello", 9: "world",
	}

	size := VocabSize()
	r.Equal(10, size)
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
	}, "Should panic when text is smaller than blockSize+1")
}
