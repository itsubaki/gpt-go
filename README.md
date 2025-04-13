### TODO
- add NewTensor() everywhere instead Tensor{}. We have to keep invariants
- options-like pattern tensor building? T2, T1 etc
- replace initialization with Kaiming distribution (currently normal which may cause issues, Building makemore Part 3: Activations & Gradients, BatchNorm)
- calc grad for forward
- implement more advanced optimizer
- float32 is faster on CPUs?
- be careful, any .At() child tensor's data reuse parent's data

// We need better way for copying
grad := gradOutput.Mul(layer.Weight.T())
embedGrad := embedsGrad.At(int(input))
grad = embedGrad.Add(grad)
for j := 0; j < len(embedGrad.Data); j++ {
    embedGrad.Data[j] += grad.At(j).First()
}


### ADR
- I decided to go without batches. I have Apple M3, not much profit using batches in terms of CPU speed. I'll give up complexity of batches for the sake of better understanding. Transformer's architecture allows me to do so, layer norm works fine without batches.