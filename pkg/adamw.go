package pkg

import (
	"math"

	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/autograd/variable"
)

type AdamW struct {
	Alpha       float64 // Learning rate
	Beta1       float64 // Exponential decay rate for first moment
	Beta2       float64 // Exponential decay rate for second moment
	WeightDecay float64 // Weight decay coefficient
	Hook        []optimizer.Hook
	iter        int
	ms, vs      map[*variable.Variable]matrix.Matrix
}

func (o *AdamW) Update(model Model) {
	params := optimizer.Params(model, o.Hook)

	if len(o.ms) == 0 {
		o.ms = make(map[*variable.Variable]matrix.Matrix)
		o.vs = make(map[*variable.Variable]matrix.Matrix)
	}

	o.iter++
	fix1 := 1.0 - math.Pow(o.Beta1, float64(o.iter))
	fix2 := 1.0 - math.Pow(o.Beta2, float64(o.iter))
	lr := o.Alpha * math.Sqrt(fix2) / fix1

	for _, p := range params {
		if _, ok := o.ms[p]; !ok {
			o.ms[p] = matrix.ZeroLike(p.Data)
			o.vs[p] = matrix.ZeroLike(p.Data)
		}

		// Update biased first moment estimate
		o.ms[p] = matrix.F2(o.ms[p], p.Grad.Data, func(m, grad float64) float64 {
			return m + ((1 - o.Beta1) * (grad - m))
		})

		// Update biased second raw moment estimate
		o.vs[p] = matrix.F2(o.vs[p], p.Grad.Data, func(v, grad float64) float64 {
			return v + ((1 - o.Beta2) * (grad*grad - v))
		})

		// The key difference for AdamW: apply weight decay directly to the weights
		// instead of incorporating it into the gradient

		// First compute the standard Adam update
		adamUpdate := matrix.F2(o.ms[p], o.vs[p], func(m, v float64) float64 {
			return lr * m / (math.Sqrt(v) + 1e-8)
		})

		// Then apply weight decay separately
		weightDecayUpdate := matrix.MulC(lr*o.WeightDecay, p.Data)

		// Update parameters: param = param - adamUpdate - weightDecayUpdate
		p.Data = matrix.Sub(p.Data, adamUpdate)
		p.Data = matrix.Sub(p.Data, weightDecayUpdate)
	}
}
