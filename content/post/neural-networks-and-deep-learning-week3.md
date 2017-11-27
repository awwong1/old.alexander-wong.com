---
title: "Neural Networks and Deep Learning, Week 3"
author: "Alexander Wong"
tags: ["Machine Learning", "deeplearning.ai"]
date: 2017-11-22T15:21:37-06:00
draft: false
---

Taking the [Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning), **Neural Networks and Deep Learning** course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/). See [deeplearning.ai](https://www.deeplearning.ai/) for more details.

Assumes you have knowledge of [Week 2]({{% relref "neural-networks-and-deep-learning-week2.md" %}}).

{{% toc %}}

# Shallow Neural Networks

## Shallow Neural Network

### Neural Networks Overview

Recall that a neural network is very similar in the logistic regression problem defined last week. A Neural network is a stack of logistic regression calls chained together.

![neural_networks_overview](/img/deeplearning-ai/neural_networks_overview.png)

### Neural Network Representation

![two_layer_neural_network_diagram](/img/deeplearning-ai/two_layer_neural_network_diagram.png)

### Computing a Neural Network's Output

In logistic regression, the output looks like this:

![logistic_regression_node](/img/deeplearning-ai/logistic_regression_node.png)

$$ z = w^Tx + b $$
$$ a = \sigma(z) $$

For a neural network, each layer is broken out into its respective nodes.

![neural_network_node](/img/deeplearning-ai/neural_network_node.png)

$$ z^{[1]}\_1 = w^{[1]T}\_1x + b^{[1]}\_1 $$
$$ a\_1^{[1]} = \sigma(x^{[1]}\_1) $$

$$ a^{[1] \leftarrow \text{Layer} }\_{i \leftarrow \text{Node in layer}} $$
$$ w^{[1]}\_1 \leftarrow \text{is a vector} $$
$$ (w^{[1]})^T = w^{[1]T} \leftarrow \text{is a vector transposed} $$

![neural_network_calculation](/img/deeplearning-ai/neural_network_calculation.png)

![neural_network_calculation_2](/img/deeplearning-ai/neural_network_calculation_2.png)

### Vectorizing Across Multiple Examples

![square_vs_round_bracket_notation](/img/deeplearning-ai/square_vs_round_bracket_notation.png)

![vectorized_approach](/img/deeplearning-ai/vectorized_approach.png)

### Explanation for Vectorized Implementation

![justifcation_of_vectorized_approach](/img/deeplearning-ai/justifcation_of_vectorized_approach.png)

![recap_of_vectorized_approach](/img/deeplearning-ai/recap_of_vectorized_approach.png)

### Activation Functions

Tanh function may be a better activation function than sigmoid. Pretty the tanh function is almost always superior, except for the output layer.

If $ y \in \{0, 1\} $ the sigmoid function might be better for the output layer.
For all other units, ReLU (rectified linear unit) is best, tanh function is better, sigmoid is worst.

![activation_functions](/img/deeplearning-ai/activation_functions.png)

![summary_activation_functions](/img/deeplearning-ai/summary_activation_functions.png)

Leaky ReLU might be better than ReLU for neural nets.

### Why do you need non-linear activation functions?

If you do not have non-linear activation functions, the calculation of $x \rightarrow \hat{y}$ is linear.

Linear activation functions eliminate the benefit of hidden layers, as the composite of two linear functions is a linear function.

![linear_activation_function](/img/deeplearning-ai/linear_activation_function.png)

### Derivatives of Activation Functions

![derivative_sigmoid_activation_function](/img/deeplearning-ai/derivative_sigmoid_activation_function.png)

![derivative_tanh_activation_function](/img/deeplearning-ai/derivative_tanh_activation_function.png)

![derivative_relu_activation_function](/img/deeplearning-ai/derivative_relu_activation_function.png)

### Gradient Descent for Neural Networks

Formula for computing derivatives in Neural Networks

![neural_network_computing_derivatives](/img/deeplearning-ai/neural_network_computing_derivatives.png)

### Backpropagation Intuition (Optional)

- didn't watch

### Random Initialization

If you initialize all your weights to zero, your neural network won't work because your hidden layer will effectively become a hidden node.

![reason_for_initialized_weights_to_zero](/img/deeplearning-ai/reason_for_initialized_weights_to_zero.png)

`W_layer1 = np.random.randn((2, 2)) * 0.01`
`b_layer1 = np.zero((2, 1))`

![reason_for_initialized_weights_to_rand](/img/deeplearning-ai/reason_for_initialized_weights_to_rand.png)
