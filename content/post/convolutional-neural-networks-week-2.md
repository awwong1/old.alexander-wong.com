---
title: "Convolutional Neural Networks, Week 2"
author: "Alexander Wong"
tags: ["Machine Learning", "deeplearning.ai"]
date: 2018-02-11T12:21:37-06:00
draft: false
---

Taking the [Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning), **Convolutional Neural Networks** course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/). See [deeplearning.ai](https://www.deeplearning.ai/) for more details.

{{% toc %}}

# Deep Convolutional Models: Case Studies

## Learning Objectives

- Understand foundational papers of Convolutional Neural Networks (CNN)
- Analyze dymensionality reduction of a volume in a very deep network
- Understand and implement a residual network
- Build a deep neural network using Keras
- Implement skip-connection in your network
- Clone a repository from Github and use transfer learning

## Case Studies

### Why look at case studies

Good way to gain intuition about convolutional neural networks is to read existing architectures that utilize CNNs

**Classic Networks:**
- LeNet-5
- AlexNet
- VGG

**Modern Networks:**
- ResNet (152 layers)
- Inception Neural Network

### Classic Networks

**LeNet-5**

![lenet_5_slide](/img/deeplearning-ai/lenet_5_slide.png)

Goal was to recognize hand written images.

1. Inputs were 32x32x1 (greyscale images.)
2. Convolutional layer, 6 5x5 filters with stride of 1. 
3. Average Pooling with filter width 2, stride of 2.
4. Convolutional Layer, 16 5x5 filters with a stride of 1.
5. Average Pooling with filter width 2, stride of 2.
6. Fully connected layer (120 nodes)
7. Fully connected layer (84 nodes)
8. Softmax layer (10 nodes)

**AlexNet**

![alex_net_slide](/img/deeplearning-ai/alex_net_slide.png)

1. Inputs were 227x227x3
2. 96 11x11 filters with stride of 4.
3. Max pooling with 3x3 filter, stride of 2
4. 5x5 same convolution
5. Max pooling with 3x3filter, stride of 2.
6. 3x3 same convolution
7. 3x3 same convolution
8. 3x3 same convolution
9. Max Pooling with 3x3 filter, stride of 2.
10. FC layer (9215 nodes)
11. FC layer (4096 nodes)
12. FC layer (4096 nodes)
13. Softmax (1000 nodes)

**VGG-16**

Conv = 3x3filter, s=1, same
Max-Pool = 2x2filter, s=2

![vgg_16_slide](/img/deeplearning-ai/vgg_16_slide.png)

1. Inputs are 224x224x3
2. Conv 64 x2
3. Max-Pool
4. Conv 128 x 2
5. Max-Pool
6. Conv 256 x 3
7. Max-Pool
8. Conv 512 x 3
9. Max-Pool
10. Conv 512 x 3
11. Max-Pool
12. FC layer (4096)
13. FC layer (4096)
14. Softmax (1000 nodes)

### Residual Networks (ResNets)

![residual_block](/img/deeplearning-ai/residual_block.png)

Allow activation layers from earlier in the network to skip additional layers.

Using residual blocks allow you to train much deeper networks.

![residual_network_layers](/img/deeplearning-ai/residual_network_layers.png)

### Why ResNets Work

If you make a network deeper, in a plain neural network you can hurt your ability to train your neural network. This is why residual blocks were invented.

Residual networks usually default to the identity function, so it doesn't make the result worse. (usually can only get better)

![why_res_nets_work](/img/deeplearning-ai/why_res_nets_work.png)

Residual block usually have the same dimensions for shortcutting. Otherwise, a $W
_s$ matrix needs to be applied.
![res_net_example](/img/deeplearning-ai/res_net_example.png)

### Networks in Networks and 1x1 Convolutions

![one_by_one_convolution](/img/deeplearning-ai/one_by_one_convolution.png)

![using_one_by_one_conv](/img/deeplearning-ai/using_one_by_one_conv.png)

Useful in adding non-linearity to your neural network without utilizing a FC layer (more computing). 

### Inception Network Motivation

![inception_network_motivation](/img/deeplearning-ai/inception_network_motivation.png)

This is computationally expensive.

![conv_5x5](/img/deeplearning-ai/conv_5x5.png)

Computational complexity can be reduced by utilizing a 1x1 convolution

![bottleneck_conv_5x5](/img/deeplearning-ai/bottleneck_conv_5x5.png)

### Inception Network

Inception module takes the previous activation, then applies many convolution and pooling layers on it.

![inception_module](/img/deeplearning-ai/inception_module.png)

![inception_network](/img/deeplearning-ai/inception_network.png)

- Allows you to use the intermediate values in the network to make predictions (seems to have a regularization effect)

## Practical Advices for using ConvNets

### Using Open-Source Implementation

A lot of these neural networks are difficult to implement. Good thing there's open source software!

Basically clone the git repo and follow the author's instructions.

### Transfer Learning

Download weights that someone else has already trained and retrain it using your own dataset.

![transfer_learning](/img/deeplearning-ai/transfer_learning.png)

You can freeze earlier layers and only train the last few layers depending on your data set size.

- If your dataset is small, only train thefinal softmax layer
- If your dataset is medium, train the last few conv/fc layers
- If your dataset is large, unfreeze all layers, using them as initialization, train all layers

### Data Augmentation

![common_augmentations](/img/deeplearning-ai/common_augmentations.png)

1. Common augmentation method is **mirroring your dataset**. Preserves whatever you're still trying to recognize in the picture.
2. **Random cropping** so long as you crop the thing you're looking for
3. Rotation
4. Shearing
5. Local Warping
6. Color shifting

### State of Computer Vision

![data_vs_hand_engineering](/img/deeplearning-ai/data_vs_hand_engineering.png)

![computer_vision_tips](/img/deeplearning-ai/computer_vision_tips.png)
- Ensembling and 10-crop are not usually used for a practical system, but for competitions/benchmarking

Use Open Source Code! Contribute to open source as well.
