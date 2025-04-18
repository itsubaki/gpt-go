package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strings"

	"github.com/itsubaki/autograd/variable"
)

const (
	limitVocab = 2800
)

var (
	chars []rune
	stoi  map[rune]int
	itos  map[int]rune

	tokenStoi    map[string]int
	tokenItos    map[int]string
	sortedTokens []string // Tokens sorted by length for efficient longest-match-first processing
	nextTokenID  int      // Next available ID for tokens
)

func Data() (*variable.Variable, int) {
	rand.Seed(42) // TODO remove

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

	// Initialize the basic character-level tokenizer
	InitTextProcessor(text)

	// Load and add the token vocabulary from JSON
	LoadTokensFromJSON("tokens.json")

	// Print total vocabulary size after loading tokens
	fmt.Printf("Vocabulary size: %d\n", VocabSize())

	encodedText := Encode(text)

	return encodedText, VocabSize()
}

// Encode converts a string to a sequence of token IDs, preferring the longest matching token
func Encode(s string) *variable.Variable {
	result := make([]float64, 0)
	i := 0

	for i < len(s) {
		// Try to match the longest token first
		matched := false

		// Only try token matching if we have tokens
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

		// If no token matches, fall back to character encoding
		if !matched {
			ch := rune(s[i])
			if idx, ok := stoi[ch]; ok {
				result = append(result, float64(idx))
			} else {
				// If character is not in our vocabulary, log a warning
				fmt.Printf("Warning: Character '%c' (code %d) not in vocabulary\n", ch, ch)
			}
			i++
		}
	}

	return variable.New(result...)
}

// Decode converts a sequence of token IDs back to a string
func Decode(indices ...float64) string {
	var result strings.Builder

	for _, idx := range indices {
		id := int(idx)

		// First check if it's a token ID
		if token, ok := tokenItos[id]; ok {
			result.WriteString(token)
		} else if ch, ok := itos[id]; ok {
			// Fall back to character decoding
			result.WriteRune(ch)
		}
	}

	return result.String()
}

func VocabSize() int {
	// Return the total number of vocabulary items - both characters and tokens
	return len(chars) + len(tokenStoi)
}

func SampleVocabulary(count int) map[string]int {
	result := make(map[string]int)

	// Add some character tokens
	i := 0
	for ch, id := range stoi {
		if i >= count/2 {
			break
		}
		result[string(ch)] = id
		i++
	}

	// Add some multi-character tokens
	i = 0
	for token, id := range tokenStoi {
		if i >= count/2 {
			break
		}
		result[token] = id
		i++
	}

	return result
}

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
		y[j] = data[idx+j+1] // target is the next token
	}

	// Convert to tensors
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

	// Initialize token maps
	tokenStoi = make(map[string]int)
	tokenItos = make(map[int]string)
	sortedTokens = make([]string, 0)

	// Set the next token ID to start after the character IDs
	nextTokenID = len(chars)
}

// LoadTokensFromJSON loads tokens from a JSON file and adds them to the vocabulary
func LoadTokensFromJSON(filepath string) {
	// Read the JSON file
	data, err := os.ReadFile(filepath)
	if err != nil {
		fmt.Printf("Warning: Could not read tokens file: %v\n", err)
		return
	}

	// Parse the JSON as an array of strings
	var tokensArray []string
	err = json.Unmarshal(data, &tokensArray)
	if err != nil {
		fmt.Printf("Warning: Could not parse tokens JSON: %v\n", err)
		return
	}
	tokensArray = tokensArray[0:min(limitVocab, len(tokensArray))]

	// Make sure nextTokenID is set correctly to start after character tokens
	if nextTokenID < len(chars) {
		nextTokenID = len(chars)
		fmt.Printf("Setting nextTokenID to %d (after character tokens)\n", nextTokenID)
	}

	// Add tokens to our maps - preserve the order from the JSON array
	tokensAdded := 0
	for i, token := range tokensArray {
		// Calculate token ID by adding the array index to nextTokenID
		tokenID := nextTokenID + i

		tokenStoi[token] = tokenID
		tokenItos[tokenID] = token
		tokensAdded++
	}

	// Sort tokens by length (descending) for efficient encoding
	sortedTokens = make([]string, len(tokensArray))
	copy(sortedTokens, tokensArray)
	sort.Slice(sortedTokens, func(i, j int) bool {
		return len(sortedTokens[i]) > len(sortedTokens[j])
	})

	// Update nextTokenID for any future additions
	nextTokenID = nextTokenID + len(tokensArray)
}
