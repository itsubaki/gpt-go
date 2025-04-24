// Just a small wrapper over existing library's params
package pkg

import (
	"fmt"

	"github.com/itsubaki/autograd/layer"
)

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

	return fmt.Sprintf("Model size: %.3fM parameters\n", float64(numParams)/1e6)
}

func (p *Params) ZeroGrad() {
	p.params.Cleargrads()
}
