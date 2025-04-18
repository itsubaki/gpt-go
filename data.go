package main

import (
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strings"

	"github.com/itsubaki/autograd/variable"
)

const (
	limitVocab = 3000
)

var (
	chars []rune
	stoi  map[rune]int
	itos  map[int]rune

	tokenStoi    map[string]int
	tokenItos    map[int]string
	sortedTokens []string
)

func Data() (*variable.Variable, int) {
	rand.Seed(42)

	filePath := "jules_verne.txt"

	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		panic("Missing Jules Verne dataset...")
	}

	data, err := os.ReadFile(filePath)
	if err != nil {
		panic("Error reading Jules Verne dataset...")
	}

	text := string(data)
	fmt.Printf("Length of text: %d characters\n", len(text))
	fmt.Printf("First 100 characters: %s\n", text[:100])

	AddCharactersToVocabulary(text)
	AddSubwordTokensToVocabulary("tokens.json")

	fmt.Printf("Vocabulary: %s\n", string(chars[:min(100, len(chars))]))

	encodedText := Encode(text)

	return encodedText, VocabSize()
}

func Encode(s string) *variable.Variable {
	result := make([]float64, 0)
	i := 0

	for _, ch := range s {
		matched := false

		if len(sortedTokens) > 0 {
			for _, token := range sortedTokens {
				if i+len(token) <= len(s) && s[i:i+len(token)] == token {
					if tokenID, ok := tokenStoi[token]; ok {
						result = append(result, float64(tokenID))
						i += len(token)
						matched = true
						break
					}
				}
			}
		}

		if !matched {
			if idx, ok := stoi[ch]; ok {
				result = append(result, float64(idx))
			} else {
				msg := fmt.Sprintf("Warning: Character '%c' (code %d) not in vocabulary\n", ch, ch)
				panic(msg)
			}
			i++
		}
	}

	return variable.New(result...)
}

func Decode(indices ...float64) string {
	var result strings.Builder

	for _, idx := range indices {
		id := int(idx)

		if token, ok := tokenItos[id]; ok {
			result.WriteString(token)
		} else if ch, ok := itos[id]; ok {
			result.WriteRune(ch)
		}
	}

	return result.String()
}

func VocabSize() int {
	return len(chars) + len(tokenStoi)
}

func Sample(data []float64, blockSize int) (*variable.Variable, *variable.Variable) {
	dataLen := len(data) - blockSize
	if dataLen <= 0 {
		panic("Not enough Data for the given block size")
	}

	idx := rand.Intn(dataLen)

	x := make([]float64, blockSize)
	y := make([]float64, blockSize)

	for j := 0; j < blockSize; j++ {
		x[j] = data[idx+j]
		y[j] = data[idx+j+1]
	}

	return variable.New(x...), variable.New(y...)
}

func AddCharactersToVocabulary(text string) {
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

func AddSubwordTokensToVocabulary(filepath string) {
	tokenStoi = make(map[string]int)
	tokenItos = make(map[int]string)
	sortedTokens = make([]string, 0)

	data, err := os.ReadFile(filepath)
	if err != nil {
		fmt.Printf("Warning: Could not read tokens file: %v\n", err)
		return
	}

	// Split file contents by newlines to get tokens
	tokensArray := strings.Split(strings.TrimSpace(string(data)), "\n")

	// Filter out any empty lines
	var filteredTokens []string
	for _, token := range tokensArray {
		if token != "" {
			filteredTokens = append(filteredTokens, token)
		}
	}
	tokensArray = filteredTokens

	// Apply vocabulary size limit if needed
	tokensArray = tokensArray[:min(limitVocab, len(tokensArray))]

	baseVocabSize := len(chars)

	for i, token := range tokensArray {
		tokenID := baseVocabSize + i
		tokenStoi[token] = tokenID
		tokenItos[tokenID] = token
	}

	sortedTokens = make([]string, len(tokensArray))
	copy(sortedTokens, tokensArray)
	sort.Slice(sortedTokens, func(i, j int) bool {
		return len(sortedTokens[i]) > len(sortedTokens[j])
	})
}
