<img src="https://raw.githubusercontent.com/MindsMD/minds.md/refs/heads/main/header.svg" alt="gptgo" title="gptgo" align="right" height="60" />

# gpt-go
Simple GPT implementation in pure Go. Trained on favourite Jules Verne books.  

What kind of response you can expect from the model:  
```
Mysterious Island.
Well.
My days must follow
```

Params and loss:
```
bs=32, es=64, lr=0.0010, ls=1.0, vs=3000, steps=20000
...
step: 18000, loss: 5.04248
step: 19000, loss: 4.97543
step: 20000, loss: 4.86982
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

In [main_test.go](https://github.com/zakirullin/gpt-go/blob/main/main_test.go) you can find step-by-step explanations:    
```go
// Our neuron has 2 inputs and 1 output (number of columns in weight matrix).
// Its goal is to predict next number in the sequence.
input := V{1, 2} // {x1, x2}
weight := M{
    {2}, // how much x1 contributes to the output
    {3}, // how much x2 contributes to the output
}
```

It goes from basic stuff all the way to self-attention and transformer architecture.  

## Design choices
No batches.  
I've given up the complexity of the batch dimension for the sake of better understanding. It's far easier to build intuition with 2D matrices, rather than with 3D tensors. Besides, batches aren't inherent to transformer architecture.  

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
Many thanks to Andrej Karpathy for his outstanding [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) course.

Thanks to [@itsubaki](https://github.com/itsubaki) for his elegant [autograd](https://github.com/itsubaki/autograd) package.