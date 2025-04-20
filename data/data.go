package data

import (
	_ "embed"
	"fmt"
	"math/rand"
	"regexp"
	"strings"

	"github.com/itsubaki/autograd/variable"
)

var (
	tokenToID  map[string]int
	idToToken  map[int]string
	mergeRules map[int64]int
	rulesOrder []int64

	//go:embed fairy_tales.txt
	datasetFile string
	//go:embed vocab
	vocabFile string

	randInt = rand.Intn
)

var dataset = func() string {
	return datasetFile
}

var vocab = func() string {
	return vocabFile
}

func Tokenize(numMerges int) ([]float64, int) {
	tokenToID = make(map[string]int)
	idToToken = make(map[int]string)
	mergeRules = make(map[int64]int)
	rulesOrder = nil

	normDataset := normNewLines(dataset())
	addCharsToVocab(normDataset)
	createMergeRules(vocab(), numMerges)

	return Encode(normDataset), VocabSize()
}

func Encode(s string) []float64 {
	var tokens []float64
	for _, ch := range s {
		tok, ok := tokenToID[string(ch)]
		if !ok {
			panic(fmt.Sprintf("Char '%s' is missing from vocabulary", string(ch)))
		}
		tokens = append(tokens, float64(tok))
	}

	for _, rule := range rulesOrder {
		var newTokens []float64
		tok1, tok2 := unzip(rule)
		// Try to apply rule on every token pair
		for i := 0; i < len(tokens); {
			hasNextToken := i+1 < len(tokens)
			shouldMerge := hasNextToken && int(tokens[i]) == tok1 && int(tokens[i+1]) == tok2
			if shouldMerge {
				newTokens = append(newTokens, float64(mergeRules[rule]))
				i += 2 // eat two tokens
			} else {
				newTokens = append(newTokens, tokens[i])
				i++ // eat one token
			}
		}
		tokens = newTokens
	}

	return tokens
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

	offset := randInt(dataLen)

	x := make([]float64, blockSize)
	y := make([]float64, blockSize)

	for i := 0; i < blockSize; i++ {
		x[i] = data[i+offset]
		y[i] = data[i+offset+1]
	}

	return variable.New(x...), variable.New(y...)
}

func Characters() string {
	var result strings.Builder
	for token := range tokenToID {
		if len(token) == 1 {
			result.WriteString(token)
		}
	}

	return result.String()
}

func addCharsToVocab(text string) {
	var chars []string
	for _, ch := range text {
		chars = append(chars, string(ch))
	}
	addTokensToVocab(chars...)
}

func createMergeRules(rules string, numMerges int) {
	rules = strings.TrimSpace(rules)
	if len(rules) == 0 {
		return
	}

	merges := strings.Split(rules, "\n")
	merges = merges[:min(numMerges, len(merges))]

	// Mint new rules, create merging mergingRules
	for _, m := range merges {
		re := regexp.MustCompile(`\[(.*?)\]\[(.*?)\] -> \[(.*?)\]`)
		matches := re.FindStringSubmatch(m)
		if len(matches) != 4 {
			panic(fmt.Sprintf("Invalid vocab format: %s", m))
		}

		addTokensToVocab(matches[3])
		addRule(tokenToID[matches[1]], tokenToID[matches[2]], tokenToID[matches[3]])
	}
}

func addTokensToVocab(tokens ...string) {
	for _, token := range tokens {
		if _, ok := tokenToID[token]; ok {
			continue
		}

		tokenID := len(tokenToID)
		tokenToID[token] = tokenID
		idToToken[tokenID] = token
	}
}

func addRule(tok1, tok2, mergedTok int) {
	key := zip(tok1, tok2)
	mergeRules[key] = mergedTok
	rulesOrder = append(rulesOrder, key)
}

func zip(tok1, tok2 int) int64 {
	return int64(tok1)<<32 | int64(tok2&0xFFFFFFFF)
}

func unzip(tok int64) (int, int) {
	tok1 := int(tok >> 32)
	tok2 := int(tok & 0xFFFFFFFF)
	return tok1, tok2
}

func normNewLines(text string) string {
	text = strings.Replace(text, "\r\n", "\n", -1) // replace Windows line endings
	return strings.Replace(text, "\r", "\n", -1)   // replace remaining Mac line endings
}
