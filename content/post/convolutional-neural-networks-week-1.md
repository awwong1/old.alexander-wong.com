---
title: "Convolutional Neural Networks, Week 1"
author: "Alexander Wong"
tags: ["Machine Learning", "deeplearning.ai"]
date: 2018-01-20T12:21:37-06:00
draft: false
---

Taking the [Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning), **Convolutional Neural Networks** course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/). See [deeplearning.ai](https://www.deeplearning.ai/) for more details.

{{% toc %}}

# Foundations of Convolutional Neural Networks

## Convolutional Neural Networks

### Computer Vision

- image classification
- object detection in images
- neural network style transfer

### Edge Detection Example

![convolution_edge_detection](/img/deeplearning-ai/convolution_edge_detection.png)

Convolution is when you 'map' a kernel or filter matrix over your original matrix. Starting from the top left, element multiply the filter with the original matrix. Add all of these new elements.

![vertical_edge_detection](/img/deeplearning-ai/vertical_edge_detection.png)

### More Edge Detection

![vertical_and_horizontal_edge_detection](/img/deeplearning-ai/vertical_and_horizontal_edge_detection.png)

Can use other types of filters.

![learning_edge_detection_filters](/img/deeplearning-ai/learning_edge_detection_filters.png)

You can make the neural network learn about the filter through backpropagation by treating the filter as a bunch of parameters to be learned.

### Padding

- padding solves shrinking output and underutiliziation of edge and corner pixels
- basically add a border to your images.

![padding](/img/deeplearning-ai/padding.png)

- typically pad values are 0
- In above example, padding $p=1$.

$$ n + 2p - f + 1 \text{ by } n + 2p - f + 1 $$

How much to pad? **Valid and Same convolutions.**

* "Valid": $ (n * n) \text{ convlution } (f * f) \rightarrow (n - f + 1 * n - f + 1) $
* "Same" Pad so that the output size is the same as the input size.

$$ (n + 2p - f + 1 * n + 2p - f + 1) $$
$ p = \dfrac{f-1}{2} $
$f$ is usually odd.


### Strided Convolutions

- striding is the act of skipping over a number of cells during convolution.
- default case is stride of 1, where you move the filter one cell at a time.

![stride](/img/deeplearning-ai/stride.png)

- in the case that the stride puts the filter such that it hangs off of the original dimensions, convention is we simply don't use apply it. (round down)

![padding_and_stride_summary](/img/deeplearning-ai/padding_and_stride_summary.png)

### Convolutions Over Volume

- Same operation as a single layer convolution, except both the filter and the input  now have multiple channels.

![convolution_by_volume](/img/deeplearning-ai/convolution_by_volume.png)

- each cell of filter, multiply by each cell of input. output is the sum of all these values.
  - That is how a 6x6x3 * 3x3x3 becomes a 4x4x1.

![detailed_volume_convolution](/img/deeplearning-ai/detailed_volume_convolution.png)

- To handle multiple filters, you simply stack the results together.

![volume_convolution_multiple_filters](/img/deeplearning-ai/volume_convolution_multiple_filters.png)

### One Layer of a Convolutional Network

![convolution_layer_example](/img/deeplearning-ai/convolution_layer_example.png)

- bias for convolutional layer is always a real number

![how_many_parameters_convolution](/img/deeplearning-ai/how_many_parameters_convolution.png)

![convolution_notation](/img/deeplearning-ai/convolution_notation.png)

### Simple Convolutional Network Example

![example_conv_net](/img/deeplearning-ai/example_conv_net.png)

### Pooling Layers

![max_pooling](/img/deeplearning-ai/max_pooling.png)

- can also do averages
- for multiple layers, simply apply the same operation on the said layer
- nothing to learn (no parameters)

### CNN Example

- example inspired by [LeNet5](http://yann.lecun.com/exdb/lenet/)

![cnn_example](/img/deeplearning-ai/cnn_example.png)
![cnn_example_details](/img/deeplearning-ai/cnn_example_details.png)

## Why Convolutions?

- Convolutions allow you to reduce the number of parameters to train
![parameter_sharing_and_sparsity_of_connections](/img/deeplearning-ai/parameter_sharing_and_sparsity_of_connections.png)
- Parameter Sharing
  - parameters are shared across the entire input
- Sparsity of Connections
  - each output value depend on a small number of input values
