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
	tokenToID     map[string]int
	idToToken     map[int]string
	longestTokens []string
	// On top of per-character token we can load pretrained tokens
	pretrainedTokens int

	//go:embed fairy_tales.txt
	data string
	//go:embed fairy_tokens.txt
	tokens string
)

func Data(numPretrainedTokens int) ([]float64, int) {
	pretrainedTokens = numPretrainedTokens
	tokenToID = make(map[string]int)
	idToToken = make(map[int]string)

	addTokensFromText(data)
	addPretrainedTokens(tokens)

	return Encode(data), VocabSize()
}

func Encode(s string) []float64 {
	encoded := make([]float64, 0)

	runes := []rune(s)
	for len(runes) > 0 {
		found := false
		for _, token := range longestTokens {
			if strings.HasPrefix(string(runes), token) {
				encoded = append(encoded, float64(tokenToID[token]))
				runes = runes[len([]rune(token)):]
				found = true
				break
			}
		}

		if !found {
			panic("can't encode token")
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

func addTokensFromText(text string) {
	for _, ch := range text {
		addTokens(string(ch))
	}
}

func addPretrainedTokens(tokens string) {
	splitTokens := strings.Split(strings.TrimSpace(tokens), "\n")
	splitTokens = splitTokens[:min(pretrainedTokens, len(splitTokens))]
	fmt.Println(splitTokens, tokens)

	addTokens(splitTokens...)
}

func addTokens(tokens ...string) {
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

func Characters() string {
	var result strings.Builder
	for _, token := range longestTokens {
		if len(token) == 1 {
			result.WriteString(token)
		}
	}

	return result.String()
}
