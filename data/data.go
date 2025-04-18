package data

import (
	_ "embed"
	"fmt"
	"math/rand"
	"sort"
	"strings"

	"github.com/itsubaki/autograd/variable"
)

var (
	// On top of per-character token we can load pretrained tokens
	numSubwordTokens int

	chars []rune
	stoi  map[rune]int
	itos  map[int]rune

	tokenStoi     map[string]int
	tokenItos     map[int]string
	longestTokens []string

	//go:embed fairy_tales.txt
	data string
	//go:embed fairy_tokens.txt
	tokens string
)

func Data(subwordTokens int) ([]float64, int) {
	numSubwordTokens = subwordTokens
	fmt.Printf("Length of text: %d characters\n", len(data))
	fmt.Printf("First 100 characters:\n%s\n", strings.TrimSpace(data[:100]))

	AddCharactersToVocabulary(data)
	AddSubwordTokensToVocabulary(tokens)

	fmt.Printf("Vocabulary: %s\n", string(chars[:min(100, len(chars))]))

	encodedText := Encode(data)

	return encodedText, VocabSize()
}

func Encode(s string) []float64 {
	encoded := make([]float64, 0)
	i := 0

	runes := []rune(s)
	for i < len(runes) {
		matched := false

		for _, token := range longestTokens {
			tokenRunes := []rune(token)
			tokenRuneLength := len(tokenRunes)

			// Check if we have enough runes left and if the slices match
			if i+tokenRuneLength <= len(runes) {
				matchCandidate := runes[i : i+tokenRuneLength]

				if string(matchCandidate) == token {
					if tokenID, ok := tokenStoi[token]; ok {
						encoded = append(encoded, float64(tokenID))
						i += len(token)
						matched = true
						break
					}
				}
			}
		}

		// If no token matches, fall back to character encoding
		if !matched {
			ch := runes[i]
			if idx, ok := stoi[ch]; ok {
				encoded = append(encoded, float64(idx))
			} else {
				msg := fmt.Sprintf("Warning: Character '%c' (code %d) not in vocabulary\n", ch, ch)
				panic(msg)
			}
			i++
		}
	}

	return encoded
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

func AddSubwordTokensToVocabulary(subwordTokens string) {
	tokenStoi = make(map[string]int)
	tokenItos = make(map[int]string)
	longestTokens = make([]string, 0)

	splitTokens := strings.Split(strings.TrimSpace(subwordTokens), "\n")
	var filteredTokens []string
	for _, token := range splitTokens {
		if token != "" {
			filteredTokens = append(filteredTokens, token)
		}
	}
	filteredTokens = filteredTokens[:min(numSubwordTokens, len(filteredTokens))]

	baseVocabSize := len(chars)
	for i, token := range filteredTokens {
		tokenID := baseVocabSize + i
		tokenStoi[token] = tokenID
		tokenItos[tokenID] = token
	}

	longestTokens = make([]string, len(filteredTokens))
	copy(longestTokens, filteredTokens)
	sort.Slice(longestTokens, func(i, j int) bool {
		return len(longestTokens[i]) > len(longestTokens[j])
	})
}
