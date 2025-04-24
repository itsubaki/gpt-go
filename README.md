<img src="https://raw.githubusercontent.com/MindsMD/minds.md/refs/heads/main/header.svg" alt="gptgo" title="gptgo" align="right" height="60" />

# gptgo
Simple GPT cpu-only implementation in pure Go. Trained on favourite Jules Verne books.  

What kind of output you can expect from the model:  
```
Mysterious Island.
Well.
My days must follow
```

## How to run
```shell
$ go run .
```

You can train on your own dataset by pointing `data.datasetFile` to your txt file.  

## How it works
You can use this repo as a companion to the [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) course.  
Use `git tag` to see how the model has evolved over time: `naive`, `bigram`, `multihead`, `block`, `residual`, `full`.  
Refer to `main_test.go` for a step-by-step explanation.  

## WHYs
Decided to go without batches.  
I've given up the complexity of the batch dimension for the sake of better understanding. Batches aren't inherent to transformer architecture.

Removed gonum dependency.  
The gonum.matmul gave us ~30% performance boost, but it brought additional complexity. We're not striving for maximum efficiency here, rather for radical simplicity. Current matmul implementation is quite effective, and it's only 40 lines of readable plain code.  

## Credits
Many thanks to Andrej Karpathy for his outstanding [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) course.  
Thanks to [@itsubaki](https://github.com/itsubaki) for his elegant [autograd](https://github.com/itsubaki/autograd) library.  