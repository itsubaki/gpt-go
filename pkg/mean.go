package pkg

import (
	"github.com/itsubaki/autograd/variable"
)

func Mean(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{Forwarder: &MeanT{}}).First(x...)
}

type MeanT struct {
	x *variable.Variable
	n int
}

// Mean alongside rows
func (m *MeanT) Forward(x ...*variable.Variable) []*variable.Variable {
	m.x = x[0]
	m.n = len(x[0].Data[0])

	means := variable.Zero(len(x[0].Data), 1)
	for i := range x[0].Data {
		means.Data[i][0] = mean(x[0].Data[i])
	}

	return []*variable.Variable{
		means,
	}
}

// Derivative of mean(x1, x2) by xn = 1/n
func (m *MeanT) Backward(gy ...*variable.Variable) []*variable.Variable {
	g := variable.ZeroLike(m.x)
	for i := range g.Data {
		for j := range g.Data[i] {
			g.Data[i][j] = gy[0].Data[i][0] / float64(m.n)
		}
	}

	return []*variable.Variable{
		g,
	}
}

func mean(values []float64) float64 {
	sum := 0.0
	for _, v := range values {
		sum += v
	}

	return sum / float64(len(values))
}
