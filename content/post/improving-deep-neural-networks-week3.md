---
title: "Improving Deep Neural Networks, Week 3"
author: "Alexander Wong"
tags: ["Machine Learning", "deeplearning.ai"]
date: 2017-12-20T10:21:37-06:00
draft: false
---

Taking the [Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning), **Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization** course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/). See [deeplearning.ai](https://www.deeplearning.ai/) for more details.

Assumes you have knowledge of [Improving Deep Neural Networks, Week 2]({{% relref "improving-deep-neural-networks-week2.md" %}}).

{{% toc %}}

# Hyperparameter Tuning, Batch Normalization, and Programming Frameworks

## Hyperparameter Tuning

### Tuning Process

There are a lot of hyperparameters needed to train your neural network. How do you tune these hyperparameters?

* Learning rate $\alpha$
* Momentum term $\beta$
* Hyperparameters for ADAM $\beta\_1, \beta\_2, \epsilon$
* Number of layers to use
* Number of hidden units per layer
* Learning Rate Decay
* Mini-Batch Size


**Order of Importance**

1. Alpha is important to tune.  **must tune**
2. Momentum is also important $~0.9$.  **should tune** 
3. Mini-Batch Size is important as well  **should tune**
4. # of hidden units.  **should tune**
5. # Layers  **not as important to tune**
6. Learning Rate Decay  **not as important to tune**
7. Hyperparameters for ADAM are pretty much always left at defaults $\beta\_1 = 0.9, \beta\_2 = 0.999, \epsilon = 10^{-8}$  **not as important to tune**

Suggested to try random values. Don't use a grid. When using random values, you are trying out more unique values for your parameters rather than a systematic grid approach.

It is difficult to know in advance which hyperparameters are going to be important to your problem.

Another common practice is 'coarse to fine'. Cast a broad range, narrow range for hyper parameters once a more performant region is found.

### Using an appropriate scale to pick hyperparameters

How do you pick the scale for random sampling? (0-100, 0-1, etc.)

Lets say you're trying to find the number of hidden units in a layer.

$$ n^{[l]} = 50, \dots, 100 $$

Lets say you're trying to find the number of layers in your neural network.

$$ \text{# layers} \leftarrow 2 \text{ to } 4 $$

Say you are searching for the hyperparameter alpha.

$$ \alpha = 0.0001, \dots, 1 $$

This is a case where you want to sample uniformly at random on a logarithmic scale.

```python
r = 4 * np.random.rand() # r between [-4, 0]
alpha = 10 ** r  # values of alpha are between 10^-4 and 10^0
```

![hyperparameter_tuning](/img/deeplearning-ai/hyperparameter_tuning.png)

Say you are trying to sample values for hyperparameter $\beta$ for exponentially weighted averages.

$$ \beta = 0.9 \dots 0.999 $$
Recall that 0.9 is like the average of last 10, and 0.999 is the average of last 1000.

This is similar to the log method of sampling.

$$ 1- \beta = 0.1 \dots 0.001 $$
Sample $r \in [-3, -1]$. Set $ 1 - \beta = 10^r$. Then, $\beta = 1-10^r $

### Hyperparameters tuning in practice: Pandas vs Caviar

Remember to re-test your hyperparameters occasionally. Intuitions do get stale. Re-evaluate occasionally.

Two schools of thought:

1. **Panda approach**: Babysit one model. Tune hyperparameters over training. Usually when you're constrained by compute (don't have the capacity to train many models)
2. **Caviar approach**: Train many models in parallel. Tune hyperparameters for various models? This allows you to try a lot of different hyperparameter settings

![panda_vs_caviar](/img/deeplearning-ai/panda_vs_caviar.png)

If you have enough compute resources, do caviar.

## Batch Normalization

### Normalizing activations in a network

Recall that normalizing your features can speed up learning. This turns the controus of your learning problem from something that is very elongated into something that is more bowl shaped.

In a deeper model, you have many activation layers. How do you normalize the intermediate layers to help train the hidden layers? Can we normalize the values of something like $a^{[2]}$ so as to train $w^{[3]}, b^{[3]}$ faster?

![normalize_activation_values](/img/deeplearning-ai/normalize_activation_values.png)

Given some intermediate values in your neural net (say $z^{(1)} \dots z^{(m)}$)

Compute the mean as follows $\mu = \dfrac{1}{m} \sum\limits\_{i}z^{(i)}$

Compute variance $\sigma^2 = \dfrac{1}{m} \sum\limits\_{i}(z\_{i} - \mu)^2$

Norm $z\_{\text{norm}}^{(i)} = \dfrac{z^{(i)}-\mu}{\sqrt{\sigma^2} + \epsilon} $

![implementing_batch_norm](/img/deeplearning-ai/implementing_batch_norm.png)

**takeaway**

Applying normalization in the hidden layers may allow for faster training. Mean and variance are learnable and unlike the input features, may not be centered around 0. It simply ensures that your hidden units have standardized mean and variance.

### Fitting Batch Normalization into a neural network

Batch norm is applied to Z. The batch normalization is governed by $\beta \text{ and } \gamma$.

The intuition is that the normalized value of Z ($ \tilde{Z} $) performs better than the un normalized value of Z ($Z$).

![batch_norm_in_neural_networks](/img/deeplearning-ai/batch_norm_in_neural_networks.png)

No relation between the beta here and ADAM algorithm betas.

In practice, batch normalization is applied on mini-batches.

![mini_batch_batch_norm](/img/deeplearning-ai/mini_batch_batch_norm.png)

### Why does Batch Normalization Work?

Batch Normalization reduces covariate shift. No matter how the values of Z change, the mean and variance of these values will remain the same.

This limits the amount in which updating the parameters of the earlier layers will effect the deeper layer. 

![neural_network_batch_norm_intuition](/img/deeplearning-ai/neural_network_batch_norm_intuition.png)

* Each minibatch is scaled by the mean/variance computed on just that mini batch.
* This adds some noise to the value $z^{[l]}$ within that minibatch. Similar to dropout, it adds some noise to each hidden layer's activations.
* This has a slight regulariation effect.

### Batch Normalization at test time

Recall Batch Normalization Functions

$$ \mu = \dfrac{1}{m} \sum\limits\_i z^{(i)} $$
$$ \sigma ^2 = \dfrac{1}{m} \sum\limits\_i (z^{(i)}-\mu)^2 $$
$$ z\_{\text{norm}}^{(i)} = \dfrac{z^{(i)} - \mu}{\sqrt{\sigma^2 + \epsilon}} $$
$$ \tilde{z}^{(i)} = \gamma z\_{\text{norm}}^{(i)} + \beta $$

Estimate $\mu$ and $\sigma^2$ using exponentially weighted average (across mini-batches).

![batch_norm_during_test_time](/img/deeplearning-ai/batch_norm_during_test_time.png)

**Takeaway**

When using batch normalization during test time, it doesn't make sense to calculate mean and variance on a single example.

Therefore, mean and variance are calculated from your training examples.

In practice, exponentially weighted averages are used during training, use these values to perform your tests.

## Multi-Class Classification 

### Softmax Regression

$$ C = \text{ # classes you are trying to classify} $$

ie) Image classifier to detect cats, dogs, and birds.

Rather than previous examples of binary classification, the neural network output layer will have $C$ outputs cooresponding the probability that it is one of your classes.

In this example, $C = 4$.

![dogs_cats_and_birds](/img/deeplearning-ai/dogs_cats_and_birds.png)

Softmax activation function:

$$ t = e^{z^{[l]}} $$
$$ t \in (4, 1) \leftarrow \text{shape} $$
$$ a^{[l]} = \dfrac{e^{z^{[l]}}}{\sum\limits^{4}\_{j=1}t\_i} $$
$$ a^{[l]}\_i = \dfrac{t\_i}{\sum\limits\_{j=1}^4 t\_i} $$

![softmax_layer](/img/deeplearning-ai/softmax_layer.png)

### Training a softmax classifier

![understanding_softmax](/img/deeplearning-ai/understanding_softmax.png)

If $C=2$, then softmax reduces to logistic regression.

How to train a neural network with softmax? What is the loss function?

$$ y = \begin{bmatrix} 0 \newline 1 \newline 0 \newline 0 \end{bmatrix} \leftarrow \text{cat} $$ 
$$ a^{[l]} = \hat{y} = \begin{bmatrix} 0.3 \newline 0.2 \newline 0.1 \newline 0.4 \end{bmatrix} \leftarrow \text{cat, but our NN isn't doing well}$$

$$ \mathcal{L}(\hat{y}, y) = -\sum\limits^{4}\_{j=1} y\_j \log{\hat{y\_j}} $$

![softmax_loss_function](/img/deeplearning-ai/softmax_loss_function.png)

![softmax_backpropagation](/img/deeplearning-ai/softmax_backpropagation.png)

## Introduction to Programming Frameworks

### Deep learning frameworks

Many Frameworks!

* Caffe/Caffe2
* CNTK
* DL4J
* Keras
* Lasagne
* mxnet
* PaddlePaddle
* TensorFlow
* Theano
* Torch

Choosing deep learning frameworks

- Ease of programming (development and deployment)
- Running speed
- Truly open (open source with good governance)

### TensorFlow

Motivating problem:

$$ J(w) = w^2 - 10w + 25 $$
This is our cost function. We want to minimize $w$.

```python
import numpy as np
import tensorflow as tf

w = tf.Variable(0, dtype=tf.float32)
# because w is overloaded, you could also do
# cost = w**2 - 10*w + 25
cost = tf.add(tf.add(w**2, tf.multiply(-10., w)), 25)
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# idiomatic
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print(sess.run(w))

# idiomatic because the above three lines can be written as
# with tf.Session() as session:
#     session.run(init)
#     print(session.run(w))
```

At this point, `0.0` is printed out.
```python
session.run(train)
print(session.run(w))
```

At this point, `0.1` is printed out.

```python
for i in range(1000):
    session.run(train)
print(session.run(w))
```

At this point `4.99999` is printed out.

How do you get training data into a tensor flow program? The following code is a modification of the above, allowing you to have dynamic coefficients.

```python
import numpy as np
import tensorflow as tf

coefficients = np.array([[1.], [-10.], [25.]])

w = tf.Variable(0, dtype=tf.float32)
x = tf.placeholder(tf.float32, [3,1])
cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print(session.run(w))
```

`0.0`

```python
session.run(train, feed_dict={x:coefficients}) # this is how you do it
print(session.run(w))
```

`0.1`

```python
for i in range(1000):
    session.run(train, feed_dict={x:coefficients})
print(session.run(w))
```

`4.99999`
