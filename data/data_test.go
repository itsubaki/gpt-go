package data

import (
	"fmt"
	"math/rand"
	"testing"

	"gptgo/pkg"
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

func TestDecode(t *testing.T) {
	dataset = func() string {
		return "abcd"
	}
	vocab = func() string {
		return "[a][a] -> [aa]\n[a][b] -> [ab]\n[aa][ab] -> [aaab]"
	}

	Tokenize(3)

	decoded := Decode([]float64{6, 3, 6, 0, 2}...)

	areEqual(t, "aaabdaaabac", decoded)
}

func TestEncodeDecodeNewLines(t *testing.T) {
	dataset = func() string {
		return "a\nb\r\nc"
	}
	vocab = func() string {
		return ""
	}

	encoded, _ := Tokenize(3)
	areSlicesEqual(t, []float64{0, 1, 2, 1, 3}, encoded)

	decoded := Decode([]float64{0, 1, 2, 1, 3}...)
	areEqual(t, "a\nb\nc", decoded)
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

func TestNormNewLinesEmptyString(t *testing.T) {
	input := ""
	expected := ""
	result := normNewLines(input)
	areEqual(t, expected, result)
}

func TestNormNewLinesSingleLine(t *testing.T) {
	input := "hello world"
	expected := "hello world"
	result := normNewLines(input)
	areEqual(t, expected, result)
}

func TestNormNewLinesLinuxLineEndings(t *testing.T) {
	input := "line1\nline2\nline3"
	expected := "line1\nline2\nline3"
	result := normNewLines(input)
	areEqual(t, expected, result)
}

func TestNormNewLinesMixedContent(t *testing.T) {
	input := "title\n\nsome content\nmore content\n\nfooter"
	expected := "title\n\nsome content\nmore content\n\nfooter"
	result := normNewLines(input)
	areEqual(t, expected, result)
}

func TestNormNewLinesWindowsLineEndings(t *testing.T) {
	input := "line1\r\nline2\r\nline3"
	expected := "line1\nline2\nline3"
	result := normNewLines(input)
	areEqual(t, expected, result)
}

func TestNormNewLinesOldMacLineEndings(t *testing.T) {
	input := "line1\rline2\rline3"
	expected := "line1\nline2\nline3"
	result := normNewLines(input)
	areEqual(t, expected, result)
}

func TestNormNewLinesMixedLineEndings(t *testing.T) {
	input := "line1\nline2\r\nline3\rline4"
	expected := "line1\nline2\nline3\nline4"
	result := normNewLines(input)
	areEqual(t, expected, result)
}

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
