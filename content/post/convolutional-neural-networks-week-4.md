---
title: "Convolutional Neural Networks, Week 4"
author: "Alexander Wong"
tags: ["Machine Learning", "deeplearning.ai"]
date: 2018-02-26T12:21:37-06:00
draft: false
---

Taking the [Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning), **Convolutional Neural Networks** course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/). See [deeplearning.ai](https://www.deeplearning.ai/) for more details.

{{% toc %}}

# Special Applications: Face Recognition & Neural Style Transfer

- Discover how CNNs can be applied to multiple fields, like art generation and face recognition.
- Implement own algorithms to do both.

## Face Recognition

### What is face recognition?

- Training a neural network to detect a face from an input image/video

**Verification**
- Input image, name/ID
- Output whether the input image is that of the claimed person

**Recognition**
- Has a database of $K$ persons
- Get an input image
- Output ID if the image is any of the $K$ persons (or not recognized)

### One Shot Learning

You need to be able to recognize a person given just one example of an individual's face.
Training samples are low, you may only have one picture of the faces you need to recognize.

- Learning from one example to recognize the person again.

Instead, need to learn a similarity function.

![similarity_function](/img/deeplearning-ai/similarity_function.png)

### Siamese Network

![siamese_network_encoding](/img/deeplearning-ai/siamese_network_encoding.png)

![siamese_network_goal](/img/deeplearning-ai/siamese_network_goal.png)

### Triplet Loss

Define and apply gradient descent on the triplet loss function.

Must compare pairs of pictures. In the terminology of triplet loss, there's an `Anchor` image, `Positive` for match, `Negative` for mismatch.

![siamese_learning_objective](/img/deeplearning-ai/siamese_learning_objective.png)

The alpha is added so the trivial output of all zeros is punished.

$$ \mathscr{L}(A, P, N) = \max(||f(A)-f(P)||^2 - ||f(A)-f(N)||^2 + \alpha, 0) $$
$$ J = \sum\limits^m\_{i=1} \mathscr{L}(A^{(i)}, P^{(i)}, N^{(i)}) $$

![triplet_loss_function](/img/deeplearning-ai/triplet_loss_function.png)

Choosing the triplets A,P,N should be difficult to distinguish to more effectively train the neural network.

![triplet_choice](/img/deeplearning-ai/triplet_choice.png)

### Face Verification and Binary Classification

Instead of using triplet loss, you can use binary classification.

Compare pairs of pictures. Output is `1` if the pairs are of the same person, and output is `0` if the pairs are of different people.

![similarity_function_face_verification](/img/deeplearning-ai/similarity_function_face_verification.png)

In the siamese network, anchor faces can be pre-computed and stored rather than being computed from the image at runtime.

## Neural Style Transfer

### What is neural style transfer?

Neural style transfer is taking an image and applying the styles of other image onto it.

![neural_style_transfer](/img/deeplearning-ai/neural_style_transfer.png)

### What are deep ConvNets learning?

Look at what is 'activated' by different layers in your neural network.

![what_is_a_deep_network_learning](/img/deeplearning-ai/what_is_a_deep_network_learning.png)

Earlier layers see less, but deeper layers see larger image patches.

![visualizing_deep_layers](/img/deeplearning-ai/visualizing_deep_layers.png)

### Cost Function

Content image $C$, Style image $S$, goal is to generate a new image $G$

Cost function $J(G)$ needs to be defined. Need to check content and style.

$$ J(G) = \alpha J\_{\text{Content}}(C, G) + \beta J\_{\text{Style}}(S, G) $$

1. Initiate the generated G randomly.
2. Use gradient descent to minimize $J(G)$.

![style_transfer_cost_function](/img/deeplearning-ai/style_transfer_cost_function.png)

### Content Cost Function

![content_cost_function](/img/deeplearning-ai/content_cost_function.png)

### Style Cost Function

![how_correlated_are_the_channels](/img/deeplearning-ai/how_correlated_are_the_channels.png)

Correlation tells us which high level texture components occur together (or not together) in the image.

![style_matrix](/img/deeplearning-ai/style_matrix.png)

![style_cost_function](/img/deeplearning-ai/style_cost_function.png)

### 1D and 3D Generalizations

![convolutions_in_2D_and_1D](/img/deeplearning-ai/convolutions_in_2D_and_1D.png)

![convolutions_in_3D](/img/deeplearning-ai/convolutions_in_3D.png)
