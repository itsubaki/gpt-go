// Just a small wrapper over existing library's params
package pkg

import (
	"encoding/binary"
	"fmt"
	"os"

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
		p.params.Add(fmt.Sprintf("%d", p.count), param)
		p.count++
	}
}

func (p *Params) Params() layer.Parameters {
	return p.params
}

func (p *Params) String() string {

	return fmt.Sprintf("%.3fM parameters\n", float64(p.Count())/1e6)
}

func (p *Params) Count() int {
	numParams := 0
	for _, param := range p.params {
		numParams += len(param.Data) * len(param.Data[0])
	}

	return numParams
}

func (p *Params) ZeroGrad() {
	p.params.Cleargrads()
}

// Currently we only differentiate the models by total count.
// Two models having different dimensions in weights but same
// total count are considered same, which is wrong.
func (p *Params) Save() {
	filename := fmt.Sprintf("model-%.3fM", float64(p.Count())/1e6)
	file, err := os.Create(filename)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	// Save map of params in ordered fashion
	for i := 0; i < len(p.params); i++ {
		key := fmt.Sprintf("%d", i)
		for _, row := range p.params[key].Data {
			if err := binary.Write(file, binary.LittleEndian, row); err != nil {
				panic(err)
			}
		}
	}
}

func (p *Params) Load() {
	filename := fmt.Sprintf("model-%.3fM", float64(p.Count())/1e6)
	file, err := os.Open(filename)
	if err != nil {
		return
	}
	defer file.Close()

	// Load map of params in ordered fashion
	for i := 0; i < len(p.params); i++ {
		key := fmt.Sprintf("%d", i)
		for j := range p.params[key].Data {
			if err := binary.Read(file, binary.LittleEndian, &p.params[key].Data[j]); err != nil {
				panic(err)
			}
		}
	}
}
