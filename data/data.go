package data

import (
	_ "embed"
	"fmt"
	"math/rand"
	"sort"
	"strings"

	"github.com/itsubaki/autograd/variable"
)

const (
	subwordTokensLimit = 0 // On top of per-character token we can load pretrained tokens
)

var (
	chars []rune
	stoi  map[rune]int
	itos  map[int]rune

	tokenStoi    map[string]int
	tokenItos    map[int]string
	sortedTokens []string

	//go:embed jules_verne.txt
	data string

	//go:embed tokens.txt
	subwordTokens string
)

func Data() (*variable.Variable, int) {
	rand.Seed(42)

	fmt.Printf("Length of text: %d characters\n", len(data))
	fmt.Printf("First 100 characters: %s\n", data[:100])

	AddCharactersToVocabulary(data)
	AddSubwordTokensToVocabulary("tokens.json")

	fmt.Printf("Vocabulary: %s\n", string(chars[:min(100, len(chars))]))

	encodedText := Encode(data)

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

	tokens := strings.Split(strings.TrimSpace(subwordTokens), "\n")

	var filteredTokens []string
	for _, token := range tokens {
		if token != "" {
			filteredTokens = append(filteredTokens, token)
		}
	}
	tokens = filteredTokens

	tokens = tokens[:min(subwordTokensLimit, len(tokens))]

	baseVocabSize := len(chars)
	for i, token := range tokens {
		tokenID := baseVocabSize + i
		tokenStoi[token] = tokenID
		tokenItos[tokenID] = token
	}

	sortedTokens = make([]string, len(tokens))
	copy(sortedTokens, tokens)
	sort.Slice(sortedTokens, func(i, j int) bool {
		return len(sortedTokens[i]) > len(sortedTokens[j])
	})
}
