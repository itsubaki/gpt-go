package data

import (
	"fmt"
	"testing"
)

func TestTokenize(t *testing.T) {
	dataset = func() string {
		return "abc"
	}
	vocab = func() string {
		return "[a][b] -> [z]"
	}
	encoded, vocabSize := Tokenize(1)

	areSlicesEqual(t, []float64{3, 2}, encoded)
	areEqual(t, 4, vocabSize)
}

func TestEncode(t *testing.T) {
	dataset = func() string {
		return "abcd"
	}
	vocab = func() string {
		return "[a][a] -> [Z]\n[a][b] -> [Y]\n[Z][Y] -> [X]"
	}

	Tokenize(3)

	// aaabdaaabac
	// ZabdZabac
	// ZYdZYac
	// XdXac
	encoded := Encode("aaabdaaabac")
	areSlicesEqual(t, []float64{6, 3, 6, 0, 2}, encoded)
}

func TestZipUnzip(t *testing.T) {
	zipped := zip(1, 2)
	expected := int64(4294967298)
	areEqual(t, zipped, expected)
	x, y := unzip(expected)
	areEqual(t, 1, x)
	areEqual(t, 2, y)
}

func Example_addTokensFromText() {
	idToToken = make(map[int]string)
	tokenToID = make(map[string]int)

	testText := "hello world"
	addCharsToVocab(testText)

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

//
//func ExampleEncode() {
//	// Setup vocabulary
//	longestTokens = []string{"hello", "world", "h", "e", "l", "o", " ", "w", "r", "d"}
//	tokenToID = map[string]int{
//		"h": 0, "e": 1, "l": 2, "o": 3, " ": 4, "w": 5, "r": 6, "d": 7, "hello": 8, "world": 9,
//	}
//	idToToken = map[int]string{
//		0: "h", 1: "e", 2: "l", 3: "o", 4: " ", 5: "w", 6: "r", 7: "d", 8: "hello", 9: "world",
//	}
//
//	// Test encoding with subword tokens
//	encoded := Encode("hello world")
//	fmt.Println(encoded)
//
//	// Test encoding with single characters
//	encoded = Encode("he world")
//	fmt.Println(encoded)
//
//	// Output:
//	// [8 4 9]
//	// [0 1 4 9]
//}
//
//func ExampleDecode() {
//	// Setup vocabulary
//	longestTokens = []string{"hello", "world", "h", "e", "l", "o", " ", "w", "r", "d"}
//	tokenToID = map[string]int{
//		"h": 0, "e": 1, "l": 2, "o": 3, " ": 4, "w": 5, "r": 6, "d": 7, "hello": 8, "world": 9,
//	}
//	idToToken = map[int]string{
//		0: "h", 1: "e", 2: "l", 3: "o", 4: " ", 5: "w", 6: "r", 7: "d", 8: "hello", 9: "world",
//	}
//
//	// Test decoding with subword tokens and characters
//	decoded := Decode(8.0, 4.0, 9.0)
//	fmt.Println(decoded)
//
//	// Test decoding with only characters
//	decoded = Decode(0.0, 1.0, 2.0, 2.0, 3.0)
//	fmt.Println(decoded)
//
//	// Output:
//	// hello world
//	// hello
//}
//
//func ExampleVocabSize() {
//	// Setup vocabulary
//	longestTokens = []string{"hello", "world", "h", "e", "l", "o", " ", "w", "r", "d"}
//	tokenToID = map[string]int{
//		"h": 0, "e": 1, "l": 2, "o": 3, " ": 4, "w": 5, "r": 6, "d": 7, "hello": 8, "world": 9,
//	}
//	idToToken = map[int]string{
//		0: "h", 1: "e", 2: "l", 3: "o", 4: " ", 5: "w", 6: "r", 7: "d", 8: "hello", 9: "world",
//	}
//
//	size := VocabSize()
//	fmt.Println(size)
//
//	// Output: 10
//}
//
//func ExampleSample() {
//	// Setup data
//	testData := pkg.V{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
//	blockSize := 3
//
//	randInt = func(_ int) int { return 0 }
//	defer func() {
//		randInt = rand.Intn
//	}()
//
//	x, y := Sample(testData, blockSize)
//	fmt.Println(len(x.Data), len(x.Data[0]))
//	fmt.Println(len(y.Data), len(y.Data[0]))
//	fmt.Println(x.Data)
//	fmt.Println(y.Data)
//
//	// Output:
//	// 1 3
//	// 1 3
//	// [[0 1 2]]
//	// [[1 2 3]]
//}

func areEqual[V comparable](t *testing.T, want, got V) {
	t.Helper()
	if want != got {
		t.Errorf("want: %v, got: %v", want, got)
	}
}

func areSlicesEqual[T comparable](t *testing.T, want, got []T) {
	t.Helper()

	if len(want) != len(got) {
		t.Errorf("length mismatch: want %d elements, got %d elements", len(want), len(got))
		t.Errorf("want: %v, got: %v", want, got)
		return
	}

	for i := range want {
		if want[i] != got[i] {
			t.Errorf("want: %v, got: %v", want, got)
			return
		}
	}
}
