// Just a small wrapper over existing library's params
package pkg

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
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

func (p *Params) Save() {
	file, err := os.Create(p.filename())
	if err != nil {
		panic(err)
	}
	defer file.Close()

	// Save map of params in ordered fashion.
	hash := crc32.NewIEEE()
	for i := 0; i < len(p.params); i++ {
		key := fmt.Sprintf("%d", i)
		for _, row := range p.params[key].Data {
			if err := binary.Write(file, binary.LittleEndian, row); err != nil {
				panic(err)
			}
		}
		shape := fmt.Sprintf("%d:%d×%d", i, len(p.params[key].Data), len(p.params[key].Data[0]))
		hash.Write([]byte(shape))
	}

	// Write checksum at the end of the file.
	checksum := hash.Sum32()
	if err := binary.Write(file, binary.LittleEndian, checksum); err != nil {
		panic(err)
	}
}

func (p *Params) LoadPretrainedIfExists() {
	file, err := os.Open(p.filename())
	if err != nil {
		return
	}
	defer file.Close()

	// Load map of params in ordered fashion.
	hash := crc32.NewIEEE()
	for i := 0; i < len(p.params); i++ {
		key := fmt.Sprintf("%d", i)
		for j := range p.params[key].Data {
			if err := binary.Read(file, binary.LittleEndian, &p.params[key].Data[j]); err != nil {
				panic(err)
			}
		}
		shape := fmt.Sprintf("%d:%d×%d", i, len(p.params[key].Data), len(p.params[key].Data[0]))
		hash.Write([]byte(shape))
	}

	var savedChecksum uint32
	if err := binary.Read(file, binary.LittleEndian, &savedChecksum); err != nil {
		panic(fmt.Errorf("failed to read shapes checksum: %v", err))
	}
	if savedChecksum != hash.Sum32() {
		panic("model shapes mismatch, remove model-* files")
	}

	fmt.Printf("Loaded pretrained model: %s\n", p.filename())
}

func (p *Params) filename() string {
	return fmt.Sprintf("model-%.3fM", float64(p.Count())/1e6)
}
