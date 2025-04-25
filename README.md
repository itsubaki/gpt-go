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
bs=32, es=64, lr=0.0010, ls=1.0, vs=3000, epochs=20000
...
epoch: 18000, loss: 5.04248
epoch: 19000, loss: 4.97543
epoch: 20000, loss: 4.86982
```

## How to run
```shell
$ go run .
```

You can train on your own dataset by pointing `data.dataset` variable to your text corpus.  
Trained params will be saved to `model-*` file.  

To chat with the model:  
```shell
$ go run . -chat
```

## How to understand
You can use this repository as a companion to the [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) course.  

Use `git checkout <tag>` to see how the model has evolved over time: `naive`, `bigram`, `multihead`, `block`, `residual`, `full`. Refer to [main_test.go](https://github.com/zakirullin/gpt-go/blob/main/main_test.go) for a step-by-step explanation.  

## Design choices
No batches.  
I've given up the complexity of the batch dimension for the sake of better understanding. Batches aren't inherent to transformer architecture.

Removed `gonum` dependency.  
The `gonum.matmul` gave us ~30% performance boost, but it brought additional complexity. We're not striving for maximum efficiency here, rather for radical simplicity. Current matmul implementation is quite effective, and it's only 40 lines of plain readable code.  

## Credits
Many thanks to Andrej Karpathy for his outstanding [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) course.  

Thanks to [@itsubaki](https://github.com/itsubaki) for his elegant [autograd](https://github.com/itsubaki/autograd) package.  