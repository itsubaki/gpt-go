package main

import (
	"fmt"
	"sync"
)

const (
	batchSize    = 16
	learningRate = 0.01
	embedSize    = 32
	numWorkers   = 10
)

func main() {
	data, vocabSize := Data()

	embeds := RandKaiming(vocabSize, embedSize)
	embedsGrad := Zeros(vocabSize, embedSize)
	layer := NewLinear(embedSize, vocabSize)

	inputs, targets := Batch(data.Data, 1, len(data.Data)-1)
	inputs, targets = inputs.At(0), targets.At(0)

	// Channel for workers to send results
	type Result struct {
		loss         float64
		layerGrad    *Linear
		embedGrads   map[int]Tensor
		inputIndices []int
	}
	resultChan := make(chan Result, numWorkers)

	// Main training loop
	lossSum := 0.0
	for i := 0; i < len(targets.Data); i += batchSize {
		// Create a wait group to synchronize goroutines
		var wg sync.WaitGroup

		// Process multiple examples in parallel
		endIdx := i + batchSize
		if endIdx > len(targets.Data) {
			endIdx = len(targets.Data)
		}

		// Create smaller micro-batches for workers
		microBatchSize := (endIdx - i + numWorkers - 1) / numWorkers

		// Launch worker goroutines
		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			startIdx := i + w*microBatchSize
			workerEndIdx := startIdx + microBatchSize
			if workerEndIdx > endIdx {
				workerEndIdx = endIdx
			}
			if startIdx >= endIdx {
				wg.Done()
				continue
			}

			go func(start, end int) {
				defer wg.Done()

				// Local gradients for this worker
				workerLoss := 0.0
				workerLayerGrad := layer.Clone()
				workerLayerGrad.ZeroGrad()
				workerEmbedGrads := make(map[int]Tensor)
				inputIndices := make([]int, 0, end-start)

				// Process examples assigned to this worker
				for j := start; j < end; j++ {
					// Forward pass
					input := int(inputs.At(j).First())
					target := targets.At(j).First()
					inputIndices = append(inputIndices, input)

					embed := embeds.At(input)
					logits := workerLayerGrad.Forward(embed)

					// Backward pass
					probs := Softmax(logits)
					grads := make([]float64, vocabSize)
					for k := 0; k < vocabSize; k++ {
						oneHot := 0.0
						if target == float64(k) {
							oneHot = 1.0
						}
						grads[k] = probs.At(k).First() - oneHot
					}
					gradOutput := Tensor1D(grads...)
					workerLayerGrad.Backward(embed, gradOutput)

					// Calculate gradient for embed
					grad := gradOutput.Mul(workerLayerGrad.Weight.T())

					// Store embedding gradient for this input
					if existingGrad, ok := workerEmbedGrads[input]; ok {
						workerEmbedGrads[input] = *existingGrad.Add(grad)
					} else {
						workerEmbedGrads[input] = *grad
					}

					// Loss calculation
					workerLoss += CrossEntropyLoss(logits, target)
				}

				// Send results back
				resultChan <- Result{
					loss:         workerLoss,
					layerGrad:    workerLayerGrad,
					embedGrads:   workerEmbedGrads,
					inputIndices: inputIndices,
				}
			}(startIdx, workerEndIdx)
		}

		// Wait for all workers to finish
		go func() {
			wg.Wait()
			close(resultChan)
		}()

		// Collect and aggregate results
		for result := range resultChan {
			lossSum += result.loss

			// Accumulate layer gradients
			for j := 0; j < len(layer.WeightGrad.Data); j++ {
				layer.WeightGrad.Data[j] += result.layerGrad.WeightGrad.Data[j]
			}

			for j := 0; j < len(layer.BiasGrad.Data); j++ {
				layer.BiasGrad.Data[j] += result.layerGrad.BiasGrad.Data[j]
			}

			// Accumulate embedding gradients
			for inputIdx, grad := range result.embedGrads {
				embedGrad := embedsGrad.At(inputIdx)
				for j := 0; j < len(embedGrad.Data); j++ {
					embedGrad.Data[j] += grad.At(j).First()
				}
			}
		}

		// Parallel weight updates
		var updateWg sync.WaitGroup

		// Update weights in parallel
		updateWg.Add(1)
		go func() {
			defer updateWg.Done()
			for j := 0; j < len(layer.Weight.Data); j++ {
				layer.Weight.Data[j] -= learningRate * layer.WeightGrad.Data[j]
			}
		}()

		// Update bias in parallel
		updateWg.Add(1)
		go func() {
			defer updateWg.Done()
			for j := 0; j < len(layer.Bias.Data); j++ {
				layer.Bias.Data[j] -= learningRate * layer.BiasGrad.Data[j]
			}
		}()

		// Update embeddings in parallel
		updateWg.Add(1)
		go func() {
			defer updateWg.Done()
			// Split embedding updates across multiple workers
			updateChunkSize := len(embeds.Data) / numWorkers
			for w := 0; w < numWorkers; w++ {
				wg.Add(1)
				start := w * updateChunkSize
				end := (w + 1) * updateChunkSize
				if w == numWorkers-1 {
					end = len(embeds.Data)
				}

				go func(start, end int) {
					defer wg.Done()
					for j := start; j < end; j++ {
						embeds.Data[j] -= learningRate * embedsGrad.Data[j]
					}
				}(start, end)
			}
		}()

		// Wait for all updates to complete
		updateWg.Wait()

		if (i % (batchSize * 1000)) == 0 {
			fmt.Printf("Loss: %f\n", lossSum/float64(batchSize))
		}

		lossSum = 0.0
		layer.ZeroGrad()
		embedsGrad = Zeros(vocabSize, embedSize)

		// Create new result channel for next batch
		resultChan = make(chan Result, numWorkers)
	}

	// Generate text
	context := "A"
	maxTokens := 500
	token := int(Encode(context).First())
	fmt.Println("\nGenerated text after training:")
	for i := 0; i < maxTokens; i++ {
		embed := embeds.At(token)
		output := layer.Forward(embed)
		probs := Softmax(output)
		token = Sample(probs)
		decodedToken := Decode([]int{token})
		fmt.Printf(decodedToken)
	}
}

// Linear layer with a Clone method for parallel processing
func (l *Linear) Clone() *Linear {
	clone := NewLinear(l.In, l.Out)

	// Copy weights and biases
	for i := 0; i < len(l.Weight.Data); i++ {
		clone.Weight.Data[i] = l.Weight.Data[i]
	}

	for i := 0; i < len(l.Bias.Data); i++ {
		clone.Bias.Data[i] = l.Bias.Data[i]
	}

	return clone
}
