<img src="https://raw.githubusercontent.com/MindsMD/minds.md/refs/heads/main/header.svg" alt="gptgo" title="gptgo" align="right" height="60" />

# gpt-go
Simple GPT implementation in pure Go. Trained on favourite Jules Verne books.  

What kind of response you can expect from the model:  
```
Mysterious Island.
Well.
My days must follow
```

Or this:
```
Captain Nemo, in two hundred thousand feet weary in
the existence of the world.
```

## How to run
```shell
$ go run .
```

It takes about 40 minutes to train on MacBook Air M3. You can train on your own dataset by pointing the `data.dataset` variable to your text corpus.  

To run in chat-only mode once the training is done:  
```shell
$ go run . -chat
```

## How to understand
You can use this repository as a companion to the [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) course. Use `git checkout <tag>` to see how the model has evolved over time: `naive`, `bigram`, `multihead`, `block`, `residual`, `full`.  

In [main_test.go](https://github.com/zakirullin/gpt-go/blob/main/main_test.go) you can find step-by-step explanations. Starting from basic neuron example:
```go
// Our neuron has 2 inputs and 1 output (number of columns in weight matrix).
// Its goal is to predict next number in the sequence.
input := V{1, 2} // {x1, x2}
weight := M{
    {2}, // how much x1 contributes to the output
    {3}, // how much x2 contributes to the output
}
```

All the way to self-attention mechanism:
```go
// To calculate the sum of all previous tokens, we can multiply by this triangular matrix:
tril := M{
    {1, 0, 0, 0}, // first token attends only at itself ("cat"), it can't look into the future
    {1, 1, 0, 0}, // second token attends at itself and the previous token ( "cat" + ", ")
    {1, 1, 1, 0}, // third token attends at itself and the two previous tokens ("cat" + ", " + "dog")
    {1, 1, 1, 1}, // fourth token attends at itself and all the previous tokens ("cat" + ", " + "dog" + " and")
}.Var()
// So, at this point each embedding is enriched with the information from all the previous tokens.
// That's the crux of self-attention.
enrichedEmbeds := MatMul(tril, inputEmbeds)
```

## Design choices
No batches.  
I've given up the complexity of the batch dimension for the sake of better understanding. It's far easier to build intuition with 2D matrices, rather than with 3D tensors. Besides, batches aren't inherent to the transformer architecture.  

Removed `gonum`.  
The `gonum.matmul` gave us ~30% performance boost, but it brought additional dependency. We're not striving for maximum efficiency here, rather for radical simplicity. Current matmul implementation is quite effective, and it's only 40 lines of plain readable code.  

## Papers
You don't need to read them to understand the code :)  

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
[Deep Residual Learning](https://arxiv.org/abs/1512.03385)  
[DeepMind WaveNet](https://arxiv.org/abs/1609.03499)  
[Batch Normalization](https://arxiv.org/abs/1502.03167)  
[Deep NN + huge data = breakthrough performance](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)  
[OpenAI GPT-3 paper](https://arxiv.org/abs/2005.14165)  
[Analyzing the Structure of Attention](https://arxiv.org/abs/1906.04284)  

## Credits
Many thanks to [Andrej Karpathy](https://github.com/karpathy) for his brilliant [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) course.

Thanks to [@itsubaki](https://github.com/itsubaki) for his elegant [autograd](https://github.com/itsubaki/autograd) package.