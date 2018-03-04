---
title: "Sequence Models, Week 1"
author: "Alexander Wong"
tags: ["Machine Learning", "deeplearning.ai"]
date: 2018-03-03T14:27:35-07:00
draft: false
---

Taking the [Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning), **Sequence Models** course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/). See [deeplearning.ai](https://www.deeplearning.ai/) for more details.

{{% toc %}}

# Recurrent Neural Networks

- Learn about recurrent neural networks.
- This type of model has been proven to perform well on temporal data (data involving time)
- Several variants:
  - LSTM
  - GRU
  - Bidirectional RNN

## Recurrent Neural Networks

### Why sequence models

![examples_of_sequence_data](/img/deeplearning-ai/examples_of_sequence_data.png)

- In speech recognition, you are given an audio clip and asked to output a sentence.
- In music generation, you are trying to output a sequence of notes
- In sentiment classification, you are given a sentence and trying to determine rating, or analysis of the phrase (happy/sad, etc.)
- DNA sequence analysis; your DNA is represented by AGCT and you can use ML to label whether or not this sequence represents a protein
- Machine translation; one sentence to another sentence
- etc.

### Notation

Motivating example. Named entity recognition.

x: `Harry Potter and Herminone Granger invented a new spell.`

y: `[1, 1, 0, 1, 1, 0, 0, 0, 0] #is a word part of a person's name?`

Index into the input/output positions is angle brackets. Index starting by 1.

$$ x^{<1>} = \text{Harry} $$

$$ y^{<3>} = 0 $$

Length of the input sequence is denoted by $T\_x$. Length of the output sequence is denoted by $T\_{y}$. These don't have to be the same.

$$ T\_{x} = 9 $$
$$ T\_{y} = 9 $$

For multiple examples, use superscript round brackets.

For instance, the second training example, 3rd word would be represented by $x^{(2)<3>}$.

Might be useful to represent the words as a value.

![representing_words](/img/deeplearning-ai/representing_words.png)

Create a vocabulary dictionary where each word is laid out in an array from A to Z. (Dictionary sizes of up to 100,000 is not uncommon.) The word 'a' could be value 1. The word 'and' could be 367. This allows us to convert our sentence into a matrix of numbers.

Words are a one-hot array. One-hot means only one value is set, everything else is 0.

### Recurrent Neural Network Model

Why not a standard neural network?
- Input and outputs can be different lengths in different examples.
- Naive neural networks do not share features learned across different positions of text.
  - in a convolutional neural network, features are shared throughout the image, but this is less useful when ordering is important (ie: time)

Recurrent neural networks are networks where the activations calculated from the first word/sequence item are passed onto the second word/sequence item.

![recurrent_neural_network](/img/deeplearning-ai/recurrent_neural_network.png)

One weakness of this model (one directional recurrent neural network) is it doesn't use the future sequence items to calculate the initial sequence item's meaning.

![rnn_forward_propagation](/img/deeplearning-ai/rnn_forward_propagation.png)

Forward propagation steps:

$$ a^{\<t>} = g(W\_{aa}a^{\<t-1>} + W\_{ax}x^{\<t>} + b\_a) $$
$$ \hat{y}^{\<t>} = g(W\_{ya}a^{\<t>} + b\_y) $$

this can be simplified to:

$$ a^{\<t>} = g(W\_{a} [a^{\<t-1>}, x^{\<t>}] + b\_a ) $$
$$ \hat{y}^{\<t>} = g(W\_{y}a^{\<t>} + b\_{y}) $$

![rnn_simplified_notation](/img/deeplearning-ai/rnn_simplified_notation.png)

### Backpropagation through time

Forward propagation recall.
![forward_propgation_rnn_graph](/img/deeplearning-ai/forward_propgation_rnn_graph.png)

Loss function for a particular element in the sequence:

$$ \mathcal{L}^{\<t>}(\hat{y}^{\<t>}, y^{\<t>}) = -y^{\<t>} \log{\hat{y}^{\<t>}} - (1 - y^{\<t>}) \log{(1-\hat{y}^{\<t>})} $$

Loss function for the entire sequence.

$$ \mathcal{L}(\hat{y}, y) = \sum\limits\_{t=1}^{T\_y} \mathcal{L}^{\<t>}(\hat{y}^{\<t>}, y^{\<t>}) $$

Name for this is called Backpropagation through time.
![backpropagation_through_time](/img/deeplearning-ai/backpropagation_through_time.png)

### Different types of RNNs

So far, the example shown had $T\_x == T\_y$. This is not always the case.

Inspired by: [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)

![examples_of_rnn_architectures](/img/deeplearning-ai/examples_of_rnn_architectures.png)

* Many to Many, where the input length and output length are the same length.
* Many to One, where for instance you are trying to determine the rating of a length of text (whether a sentence is happy or not for instance)
* One to one, a standard neural net (not really recurrant)

![examples_of_rnn_architectures_2](/img/deeplearning-ai/examples_of_rnn_architectures_2.png)

* One to Many, such as seeding a music generation neural network
* Many to Many, where the input length and output length are different. Network is broken into two parts, an encoder and a decoder.

![summary_of_rnns](/img/deeplearning-ai/summary_of_rnns.png)

### Language model and sequence generation

What is language modelling? How does a machine tell the difference between `The apple and pair salad`, and `The apple and pear salad`?

Language modeler estimates the probability of a sequence of words. Training set requires a large corpus of english text.

Turn a sentence into a token. Turn a sentence into 'one hot vectors'. Another common thing to do is to model the end of sentences. `<EOS> token`
![language_modelling_with_rnn](/img/deeplearning-ai/language_modelling_with_rnn.png)

The RNN model is trying to determine the next item in the sequence given all of the items provided in the sequence earlier.

![rnn_language_model_sequence_generation](/img/deeplearning-ai/rnn_language_model_sequence_generation.png)

### Sampling novel sequences

**Word level language model**
![generation_of_random_sentence](/img/deeplearning-ai/generation_of_random_sentence.png)

**Character level language model**
![generation_of_random_sentence_char](/img/deeplearning-ai/generation_of_random_sentence_char.png)
- no unknown word token
- more computationally expensive and more difficult to capture longer term patterms

### Vanishing gradients with RNNs

One of the problems of the basic RNN is vanishing gradient problem.

Consider the following sentences:
- The cat, which already ate ten apples and three pears, was full.
- The cats, which already ate ten apples and three pears, were full.

How do you capture the long term dependency of `cat -> was` and `cats -> were`? The stuff in the middle can be arbitraily long. Difficult for an item in the sequence to be influenced by values much earlier/later in the sequence.

![vanishing_gradients_rnn](/img/deeplearning-ai/vanishing_gradients_rnn.png)
Apply gradient clipping if your gradient starts to explode. [Gradient Clipping](https://hackernoon.com/gradient-clipping-57f04f0adae)

### Gated Recurrent Unit (GRU)

Improvement to RNN to help capture long term dependencies. For reference, this is the basic recurrent neural network unit.

![basic_rnn_unit](/img/deeplearning-ai/basic_rnn_unit.png)

GRU (simplified) has a memory cell. 

$$ c = \text{memory cell} $$
$$ c^{\<t>} = a^{\<t>} $$
$$ \tilde{c}^{\<t>} = \tanh({W\_c [c^{\<t-1>}, x^{\<t>}] + b\_c}) $$
- The candidate new memory cell value
$$ \Gamma\_u = \sigma({W\_u [c^{\<t-1>}, x^{\<t>}] + b\_u}) $$
- Determine if this should be updated or not? The 'u' stands for update. Capital Gamma stands for Gate.
$$ c^{\<t>} = \Gamma\_u * \tilde{c}^{\<t>} + (1 - \Gamma\_u) * c^{\<t-1>} $$
- The new memory cell value

![simplified_gru](/img/deeplearning-ai/simplified_gru.png)
- Helps the neural network learn very long term dependencies because Gamma is either close to 0 or close to 1.

There is an additional 'gate', which takes the relevance into question to determine whether or not to update the memory cell.

![full_gru](/img/deeplearning-ai/full_gru.png)

### Long Short Term Memory (LSTM)

![LSTM_versus_GRU](/img/deeplearning-ai/LSTM_versus_GRU.png)

**LSTM Functions**

$$ \tilde{c}^{\<t>} = \tanh{(w\_c [a^{\<t-1>}, x^{\<t>}] + b\_c)} $$
$$ \Gamma\_u = \sigma(w\_u [a^{\<t-1>}, x^{\<t>}] + b\_u)$$
$$ \Gamma\_f = \sigma(w\_f [a^{\<t-1>}, x^{\<t>}] + b\_f)$$
$$ \Gamma\_o = \sigma(w\_o [a^{\<t-1>}, x^{\<t>}] + b\_o)$$
$$ c^{\<t>} = \Gamma\_u * \tilde{c}^{\<t>} + \Gamma\_f * c^{\<t-1>} $$
$$ a^{\<t>} = \Gamma\_o * c^{\<t>} $$


The LSTM is similar to GRU, but there are a few notable differences.

- LSTM has three gates. An Update Gate, a Forget Gate, and an Output Gate.
- LSTM does not equate $a^{\<t>} == c^{\<t>}$.

![LSTM_in_pictures](/img/deeplearning-ai/LSTM_in_pictures.png)

There isn't a widespread consensus as to when to use a GRU and when to use an LSTM. Neither algorithm is universally superior. GRU is computationally simplier. LSTM is more powerful and flexible.

### Bidirectional RNN

Bidirectional RNNS allow you to take information from both earlier and later in the sequence.

![getting_information_from_the_future](/img/deeplearning-ai/getting_information_from_the_future.png)

Forward propagation is run once from the sequence starting from the beginning to end. Simultaneously, forward propagation is run once from the sequence starting from the end going to the beginning.

The activation function $g$ is applied on the two blocks at each sequence item.

![bidirectional_rnn](/img/deeplearning-ai/bidirectional_rnn.png)

Disadvantage of this is the computation is now doubled. Also need to calculate the entire sequence before you can make predictions. When you are doing speech processing, you have to wait until the person stops talking before you can make a prediction.

### Deep RNNs

Added notation, square bracket superscript represents layer number.

![deep_rnn_visualization](/img/deeplearning-ai/deep_rnn_visualization.png)

Recurrent Neural Networks can be stacked on top of one another. Three layers is usually plenty enough.

