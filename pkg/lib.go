package pkg

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
	"gonum.org/v1/gonum/stat/distuv"
)

// Just a small wrapper over autograd's params
type Params struct {
	params layer.Parameters
	count  int
}

func NewParams() *Params {
	return &Params{params: layer.Parameters{}}
}

func (p *Params) Add(params ...layer.Parameter) {
	for _, param := range params {
		p.params.Add(fmt.Sprintf("%d#params", p.count), param)
		p.count++
	}
}

func (p *Params) Params() layer.Parameters {
	return p.params
}

func (p *Params) String() string {
	numParams := 0
	for _, param := range p.params {
		numParams += len(param.Data) * len(param.Data[0])
	}

	return fmt.Sprintf("%.3fM parameters\n", float64(numParams)/1e6)
}

func (p *Params) ZeroGrad() {
	p.params.Cleargrads()
}

var (
	Add     = variable.Add
	Sub     = variable.Sub
	Mul     = variable.Mul
	Div     = variable.Div
	Zeros   = variable.Zero
	OneLike = variable.OneLike
	Pow     = variable.Pow
)

func Sample(probs *variable.Variable) float64 {
	r := rand.Float64()

	// Find the first index where cumulative probability exceeds r
	cumulativeProb := 0.0
	for i, p := range probs.Data[0] {
		cumulativeProb += p
		if r < cumulativeProb {
			return float64(i)
		}
	}

	// Fallback (should rarely happen due to floating point precision)
	return float64(len(probs.Data)) - 1
}

func Rows(x *variable.Variable, indexes ...float64) *variable.Variable {
	var intIndexes []int
	for _, index := range indexes {
		intIndexes = append(intIndexes, int(index))
	}

	return (&variable.Function{Forwarder: &variable.GetItemT{Slices: intIndexes}}).First(x)
}

// Add tests
func RandKaiming(dims ...int) *variable.Variable {
	sigma := math.Sqrt(2.0 / float64(dims[1]))
	dist := distuv.Normal{Mu: 0, Sigma: sigma}
	result := matrix.F(matrix.Zero(dims[0], dims[1]), func(_ float64) float64 { return dist.Rand() })

	return variable.NewOf(result...)
}

// Only works with 2D tensors
func Tril(m *variable.Variable) *variable.Variable {
	result := variable.ZeroLike(m)
	for i := 0; i < len(m.Data); i++ {
		for j := 0; j < len(m.Data[i]); j++ {
			if j <= i {
				result.Data[i][j] = m.Data[i][j]
			}
		}
	}

	return result
}

// The result would be added to computation graph and tied to m
func MaskedInfFill(m, mask *variable.Variable) *variable.Variable {
	negInfMaskedData := matrix.F2(m.Data, mask.Data, func(a, b float64) float64 {
		if b == 0 {
			return math.Inf(-1)
		}

		return a
	})
	mMasked := Add(variable.Mul(m, mask), variable.NewOf(negInfMaskedData...))

	return mMasked
}

func PrintShape(v *variable.Variable) {
	fmt.Printf("(%d, %d)\n", len(v.Data), len(v.Data[0]))
}
