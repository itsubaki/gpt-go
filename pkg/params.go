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
}

func NewParams() *Params {
	return &Params{params: layer.Parameters{}}
}

func (p *Params) Add(params ...layer.Parameter) {
	for _, param := range params {
		p.params.Add(fmt.Sprintf("%d", len(p.params)), param)
	}
}

func (p *Params) Params() layer.Parameters {
	return p.params
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
	for i := range len(p.params) {
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

func (p *Params) TryLoadPretrained() {
	file, err := os.Open(p.filename())
	if err != nil {
		return
	}
	defer file.Close()

	// Load map of params in ordered fashion.
	hash := crc32.NewIEEE()
	for i := range len(p.params) {
		key := fmt.Sprintf("%d", i)
		for j := range p.params[key].Data {
			if err := binary.Read(file, binary.LittleEndian, &p.params[key].Data[j]); err != nil {
				panic(fmt.Sprintf("model shapes mismatch, remove '%s' file", p.filename()))
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
		panic(fmt.Sprintf("model shapes mismatch, remove '%s' file", p.filename()))
	}

	fmt.Printf("Loaded pretrained params: %s\n", p.filename())
}

func (p *Params) filename() string {
	return fmt.Sprintf("model-%.3fM", Millions(p.Count()))
}
