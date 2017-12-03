---
title: "Neural Networks and Deep Learning, Week 4"
author: "Alexander Wong"
tags: ["Machine Learning", "deeplearning.ai"]
date: 2017-12-02T15:21:37-06:00
draft: false
---

Taking the [Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning), **Neural Networks and Deep Learning** course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/). See [deeplearning.ai](https://www.deeplearning.ai/) for more details.

Assumes you have knowledge of [Week 3]({{% relref "neural-networks-and-deep-learning-week3.md" %}}).

{{% toc %}}

# Deep Neural Networks

## Deep Neural Network

### Deep L-layer neural network

![deep_neural_networks](/img/deeplearning-ai/deep_neural_networks.png)

![deep_neural_network_notation](/img/deeplearning-ai/deep_neural_network_notation.png)

Capital $L$ denotes the number of layers in the network. $ L = 4 $

We use $n^{[l]}$ to denote number of units in layer $l$.

$$ n^{[0]} = n\_x = 3, n^{[1]} = 5, n^{[2]} = 5, n^{[3]} = 3, n^{[4]} = 1, n^{[5]} = 1 $$

### Forward Propagation in a Deep Network

![deep_neural_network_forward_propagation](/img/deeplearning-ai/deep_neural_network_forward_propagation.png)


$$ Z^{[l]} = W^{[l]} a ^{[l-1]} + b^{[l]} $$
$$ a^{[l]} = g^{[l]}(Z^{[l]}) $$

Vectorized:

$$ X = A^{[0]} $$
$$ Z^{[1]} = W^{[1]} X + b^{[l]} $$
$$ A^{[1]} = g^{[1]}(Z^{[1]}) $$

### Getting your matrix dimensions right

![parameters_wl_and_bl](/img/deeplearning-ai/parameters_wl_and_bl.png)

$$ W^{[1]} : (n^{[1]}, n^{[0]}) $$

$$ W^{[l]} : (n^{[l]}, n^{[l-1]}) $$

The shape of $b$ should be $b^{[l]} : (n^{[l]}, 1) $.

![vectorized_matrix_dimensions](/img/deeplearning-ai/vectorized_matrix_dimensions.png)

### Why deep representations?

![intuition_about_deep_representation](/img/deeplearning-ai/intuition_about_deep_representation.png)

Composing functions of increasing complexity, ie consider a face classifier
- detect edges -> detect eyes, or noses -> detect groupings of eyes and noses

Circuit theory and deep learning:

Informally: There are functions you can compute with a "small" L-layer deep neural network that shallower networks require exponentially more hidden units to compute.

### Building blocks of deep neural networks

![forwards_and_backwards_functions](/img/deeplearning-ai/forwards_and_backwards_functions.png)
Z is cached and used in both forward and back propagation.

![building_blocks_of_deep_neural_networks](/img/deeplearning-ai/building_blocks_of_deep_neural_networks.png)

### Forward and Backward Propagation

Forward propagation

* input $a^{[l-1]}$
* output $a^{[l]}$, cache $(z^{[l]})$

$$ z^{[l]} = w^{[l]} z^{[l-1]} + b^{[l]} $$
$$ a^{[l]} = g^{[l]}(z^{[l]}) $$

Vectorized

$$ Z^{[l]} = W^{[l]} A^{[l-1]} = b^{[l]} $$
$$ A^{[l]} = g^{[l]} (Z^{[l]}) $$

Back propagation

* input $da^{[l]}$
* output $da^{[l-1]}, dW^{[l]}, db^{[l]}$

$$ dz^{[l]} = da^{[l]} \times g^{[l]}'(z^{[l]}) $$
$$ dW^{[l]} = dz^{[l]} \times a^{[l-1]} $$
$$ db^{[l]} = dz^{[l]} $$
$$ dz^{[l-1]} = w^{[l]T} \times dz^{[l]} $$

![backpropagation_summary](/img/deeplearning-ai/backpropagation_summary.png)

$$ da^{[l]} = -\dfrac{y}{a} + \dfrac{(1-y)}{(1-a)} $$ 

### Parameters vs Hyperparameters

Parameters $ W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}, \dots $

Hyperparameters:

* learning rate $\alpha$
* number of iterations
* number of hidden layers L
* number of hidden units per layer
* choice of activation function per layer

Later hyperparameters

* momentum
* minibatch size
* regularizations

Applied deep learning is a very empirical process.

```text
Idea -> Code -> Experiment
<- Repeat <-
```
### What does this have to do with the brain?

![forward_and_backpropagation](/img/deeplearning-ai/forward_and_backpropagation.png)

Less like brain, more like universal function approximator.
