package data

import (
	"math"
	"math/rand"
	"testing"

	"github.com/itsubaki/autograd/variable"
)

func TestTokenize(t *testing.T) {
	Dataset = func() string {
		return "abc"
	}
	Vocab = func() string {
		return "[a][b] -> [z]"
	}
	encoded, vocabSize := Tokenize(1)

	areSlicesEqual(t, []float64{3, 2}, encoded)
	areEqual(t, 4, vocabSize)
}

func TestEncode(t *testing.T) {
	Dataset = func() string {
		return "abcd"
	}
	Vocab = func() string {
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
	Dataset = func() string {
		return "abcd"
	}
	Vocab = func() string {
		return "[a][a] -> [aa]\n[a][b] -> [ab]\n[aa][ab] -> [aaab]"
	}

	Tokenize(3)

	decoded := Decode([]float64{6, 3, 6, 0, 2}...)

	areEqual(t, "aaabdaaabac", decoded)
}

func TestEncodeDecodeDifferentNewLines(t *testing.T) {
	Dataset = func() string {
		return "a\nb\r\nc"
	}
	Vocab = func() string {
		return ""
	}

	encoded, _ := Tokenize(3)
	areSlicesEqual(t, []float64{0, 1, 2, 1, 3}, encoded)

	decoded := Decode([]float64{0, 1, 2, 1, 3}...)
	areEqual(t, "a\nb\nc", decoded)
}

func TestEncodeDecodeTokenizedNewLines(t *testing.T) {
	Dataset = func() string {
		return "a\nb\n\nc"
	}
	Vocab = func() string {
		return "[\\n][\\n] -> [\\n\\n]"
	}

	encoded, _ := Tokenize(1)
	areSlicesEqual(t, []float64{0, 1, 2, 4, 3}, encoded)

	decoded := Decode([]float64{0, 1, 2, 4, 3}...)
	areEqual(t, "a\nb\n\nc", decoded)
}

func TestZipUnzip(t *testing.T) {
	zipped := zip(1, 2)
	expected := int64(4294967298)
	areEqual(t, zipped, expected)
	x, y := unzip(expected)
	areEqual(t, 1, x)
	areEqual(t, 2, y)
}

func TestAddTokensFromText(t *testing.T) {
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

	areEqual(t, 8, len(tokenToID))
	areEqual(t, true, contains)
}

func TestSample(t *testing.T) {
	// Setup data
	testData := V{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	blockSize := 3

	RandInt = func(_ int) int { return 0 }
	defer func() {
		RandInt = rand.Intn
	}()

	x, y := Sample(testData, blockSize)
	areMatricesEqual(t, M{
		{0, 1, 2},
	}, x)
	areMatricesEqual(t, M{
		{1, 2, 3},
	}, y)
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

func areMatricesEqual(t *testing.T, want M, got *variable.Variable) {
	t.Helper()
	if len(want) != len(got.Data) {
		t.Errorf("matrix length mismatch: want length=%d, got length=%d", len(want), len(got.Data))
		return
	}

	for i := range want {
		if len(want[i]) != len(got.Data[i]) {
			t.Errorf("matrix row length mismatch at row %d: want length=%d, got length=%d", i, len(want[i]), len(got.Data[i]))
			return
		}
	}

	for i := range want {
		for j := range want[i] {
			if math.Abs(want[i][j]-got.Data[i][j]) > 1e-9 {
				t.Errorf("matrix mismatch at row %d, column %d: want %v, got %v", i, j, want[i][j], got.Data[i][j])
			}
		}
	}
}

type V []float64

func (v V) Var() *variable.Variable {
	return variable.NewOf(v)
}

type M [][]float64

func (m M) Var() *variable.Variable {
	return variable.NewOf(m...)
}
