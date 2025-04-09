### TODO
- replace initialization with Kaiming distribution (currently normal which may cause issues, Building makemore Part 3: Activations & Gradients, BatchNorm)
- calc grad for forward
- implement more advanced optimizer
- float32 is faster on CPUs?
- be careful, any .At() child tensor's data reuse parent's data