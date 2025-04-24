<img src="https://raw.githubusercontent.com/MindsMD/minds.md/refs/heads/main/header.svg" alt="gptgo" title="gptgo" align="right" height="60" />

# gptgo
Simple GPT implementation in pure Go. Trained on favourite Jules Verne books.  

What kind of output you can expect from the model:  
```
Mysterious Island.
Well.
My days must follow
```

## how to run
```shell
$ go run .
```

You can train on your own dataset by point `data.datasetFile` to your txt file.  

## how it works
You can use this repo as a companion to the [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) course.  
Use `git tag` to see how the model has evolved over time: `naive`, `bigram`, `multihead`, `block`, `residual`, `full`.  
Refer to `main_test.go` for a step-by-step explanation.  

## whys
Decided to go without batches.  
I have an  Apple M3, not much profit using batches in terms of CPU speed. I'll give up complexity of batches for the sake of better understanding. Transformer's architecture allows me to do so, layer norm works fine without batches.  

Removed gonum dependency.  
The gonum.matmul gave us ~30% performance boost, but it brought additional complexity. We're not striving for maximum efficiency here, rather for radical simplicity. Current matmul implementation is quite effective, and it's only 40 lines of readable plain code.  

## credits
Many thanks to Andrej Karpathy for his outstanding [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) course.  
Thanks to @itsubaki for his elegant [autograd](https://github.com/itsubaki/autograd) library.  