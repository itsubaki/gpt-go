package data

import (
	_ "embed"
	"fmt"
	"math/rand"
	"slices"
	"sort"
	"strings"

	"github.com/itsubaki/autograd/variable"
)

var (
	tokenToID     map[string]int
	idToToken     map[int]string
	longestTokens []string

	//go:embed fairy_tales.txt
	text string
	//go:embed fairy_tokens.txt
	pretrainedTokens string
)

func Data(numPretrainedTokens int) ([]float64, int) {
	tokenToID = make(map[string]int)
	idToToken = make(map[int]string)

	addTokensFromText(text)
	addPretrainedTokens(pretrainedTokens, numPretrainedTokens)

	return Encode(text), VocabSize()
}

func Encode(s string) []float64 {
	encoded := make([]float64, 0)

	runes := []rune(s)
	for len(runes) > 0 {
		found := false
		for _, token := range longestTokens {
			tokenRunes := []rune(token)
			canTokenFit := len(runes) >= len(tokenRunes)
			if !canTokenFit {
				continue
			}

			if slices.Equal(runes[:len(tokenRunes)], tokenRunes) {
				encoded = append(encoded, float64(tokenToID[token]))
				runes = runes[len([]rune(token)):]
				found = true
				break
			}
		}

		if !found {
			msg := fmt.Sprintf("Can't encode token=%s", string(runes[0]))
			panic(msg)
		}
	}

	return encoded
}

func Decode(indices ...float64) string {
	var result strings.Builder

	for _, idx := range indices {
		id := int(idx)
		if token, ok := idToToken[id]; ok {
			result.WriteString(token)
		} else {
			msg := fmt.Sprintf("Uknown token ID=%d", id)
			panic(msg)
		}
	}

	return result.String()
}

func VocabSize() int {
	return len(tokenToID)
}

func Sample(data []float64, blockSize int) (*variable.Variable, *variable.Variable) {
	dataLen := len(data) - (blockSize + 1)
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

func Characters() string {
	var result strings.Builder
	for _, token := range longestTokens {
		if len(token) == 1 {
			result.WriteString(token)
		}
	}

	return result.String()
}

func addTokensFromText(text string) {
	var chars []string
	for _, ch := range text {
		chars = append(chars, string(ch))
	}
	addTokens(chars)
}

func addPretrainedTokens(tokens string, numPretrainedTokens int) {
	splitTokens := strings.Split(strings.TrimSpace(tokens), "\n")
	splitTokens = splitTokens[:min(numPretrainedTokens, len(splitTokens))]

	addTokens(splitTokens)
}

func addTokens(tokens []string) {
	for _, token := range tokens {
		if _, ok := tokenToID[token]; ok {
			continue
		}

		tokenID := len(tokenToID)
		tokenToID[token] = tokenID
		idToToken[tokenID] = token
		longestTokens = append(longestTokens, token)
	}

	sort.Slice(longestTokens, func(i, j int) bool {
		if len(longestTokens[i]) == len(longestTokens[j]) {
			return longestTokens[i] < longestTokens[j]
		}

		return len(longestTokens[i]) > len(longestTokens[j])
	})
}
