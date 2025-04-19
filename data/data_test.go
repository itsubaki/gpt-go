package data

import (
	"fmt"
	"math/rand"

	"gptgo/pkg"
)

func Example_addTokensFromText() {
	// Reset global state
	idToToken = make(map[int]string)
	tokenToID = make(map[string]int)
	longestTokens = nil

	testText := "hello world"
	addTokensFromText(testText)

	contains := true
	for _, token := range []string{"h", "e", "l", "o", " ", "w", "r", "d"} {
		if _, exists := tokenToID[token]; !exists {
			contains = false
			break
		}
	}

	fmt.Println(len(tokenToID))
	fmt.Println(contains)

	// Output:
	// 8
	// true
}

func Example_addPretrainedTokens() {
	// Reset global state
	idToToken = make(map[int]string)
	tokenToID = make(map[string]int)
	longestTokens = nil
	addTokensFromText("abc")

	addPretrainedTokens("test\ntoken\nlongest_token", 3)

	fmt.Println("number of tokens:", len(tokenToID))
	fmt.Println("test ID:", tokenToID["test"])
	fmt.Println("token ID:", tokenToID["token"])
	fmt.Println("longest_token ID:", tokenToID["longest_token"])
	fmt.Println("longest token:", longestTokens[0])

	// Output:
	// number of tokens: 6
	// test ID: 3
	// token ID: 4
	// longest_token ID: 5
	// longest token: longest_token
}

func ExampleEncode() {
	// Setup vocabulary
	longestTokens = []string{"hello", "world", "h", "e", "l", "o", " ", "w", "r", "d"}
	tokenToID = map[string]int{
		"h": 0, "e": 1, "l": 2, "o": 3, " ": 4, "w": 5, "r": 6, "d": 7, "hello": 8, "world": 9,
	}
	idToToken = map[int]string{
		0: "h", 1: "e", 2: "l", 3: "o", 4: " ", 5: "w", 6: "r", 7: "d", 8: "hello", 9: "world",
	}

	// Test encoding with subword tokens
	encoded := Encode("hello world")
	fmt.Println(encoded)

	// Test encoding with single characters
	encoded = Encode("he world")
	fmt.Println(encoded)

	// Output:
	// [8 4 9]
	// [0 1 4 9]
}

func ExampleDecode() {
	// Setup vocabulary
	longestTokens = []string{"hello", "world", "h", "e", "l", "o", " ", "w", "r", "d"}
	tokenToID = map[string]int{
		"h": 0, "e": 1, "l": 2, "o": 3, " ": 4, "w": 5, "r": 6, "d": 7, "hello": 8, "world": 9,
	}
	idToToken = map[int]string{
		0: "h", 1: "e", 2: "l", 3: "o", 4: " ", 5: "w", 6: "r", 7: "d", 8: "hello", 9: "world",
	}

	// Test decoding with subword tokens and characters
	decoded := Decode(8.0, 4.0, 9.0)
	fmt.Println(decoded)

	// Test decoding with only characters
	decoded = Decode(0.0, 1.0, 2.0, 2.0, 3.0)
	fmt.Println(decoded)

	// Output:
	// hello world
	// hello
}

func ExampleVocabSize() {
	// Setup vocabulary
	longestTokens = []string{"hello", "world", "h", "e", "l", "o", " ", "w", "r", "d"}
	tokenToID = map[string]int{
		"h": 0, "e": 1, "l": 2, "o": 3, " ": 4, "w": 5, "r": 6, "d": 7, "hello": 8, "world": 9,
	}
	idToToken = map[int]string{
		0: "h", 1: "e", 2: "l", 3: "o", 4: " ", 5: "w", 6: "r", 7: "d", 8: "hello", 9: "world",
	}

	size := VocabSize()
	fmt.Println(size)

	// Output: 10
}

func ExampleSample() {
	// Setup data
	testData := pkg.V{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	blockSize := 3

	randInt = func(_ int) int { return 0 }
	defer func() {
		randInt = rand.Intn
	}()

	x, y := Sample(testData, blockSize)
	fmt.Println(len(x.Data), len(x.Data[0]))
	fmt.Println(len(y.Data), len(y.Data[0]))
	fmt.Println(x.Data)
	fmt.Println(y.Data)

	// Output:
	// 1 3
	// 1 3
	// [[0 1 2]]
	// [[1 2 3]]
}
