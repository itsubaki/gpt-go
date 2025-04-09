### TODO
- replace initialization with Kaiming distribution (currently normal which may cause issues, Building makemore Part 3: Activations & Gradients, BatchNorm)
- calc grad for forward
- implement more advanced optimizer
- float32 is faster on CPUs?
- be careful, any .At() child tensor's data reuse parent's data

### ADR
- I decided to go without batches. I have Apple M3, not much profit using batches in terms of CPU speed. I'll give up complexity of batches for the sake of better understanding. Transformer's architecture allows me to do so, layer norm works fine without batches.