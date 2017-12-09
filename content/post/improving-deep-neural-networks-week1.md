---
title: "Improving Deep Neural Networks, Week 1"
author: "Alexander Wong"
tags: ["Machine Learning", "deeplearning.ai"]
date: 2017-12-08T15:21:37-06:00
draft: false
---

Taking the [Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning), **Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization** course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/). See [deeplearning.ai](https://www.deeplearning.ai/) for more details.

Assumes you have knowledge of [Neural Networks and Deep Learning]({{% relref "neural-networks-and-deep-learning-week1.md" %}}).

{{% toc %}}

# Practical Aspects of Deep Learning

## Setting Up Your Machine Learning Application

### Train/Dev/Test Sets

Recall:

* **Training set** is used to teach your model how to accomplish tasks
* **Dev set** (Cross Validation Set) is used to decide which algorithms and hyperparameters to use in your neural network model.
* **Test set** is used to evaluate your model's performance.

Classic old training set divisions were among the range of 60% / 20% / 20%. This may have been fine when there were less than a million training examples

Modern machine learning divisions are much more skewed. Given 1 million training examples, might only allocate 10,000 (1%) to dev and 10,000 (1%) to test. It is not uncommon to use less than a single percent for dev and test given large datasets.

_Make sure_ that your development and test sets come from the same distribution!

It might be okay to only have a train and dev set. Test say may be ignored.

### Bias/Variance

![bias_and_variance](/img/deeplearning-ai/bias_and_variance.png)

* High Bias. Underfitting. Does not closely match the training data.

* High Variance. Overfitting. Extremely close match to the training data. 

### Basic Recipe for Machine Learning

After having training your model, evaluate whether or not your algoirthm has high bias. (Observe the training data performance.)

* If it does have high bias, perhaps make a bigger network, or train longer. (Maybe change your neural network architecture?).
* Increasing the network size pretty much always reduces your bias. (This does not effect your variance.)

Once the high bias propblem is solved, check if you have high variance (evaluate your dev set performance.)

* If it does have high variance, perhaps get more data, or perform regularization.
* Getting more data will pretty much always lower your variance. (This does not negatively effect your bias, usually)

## Regularizing your Neural Network

### Regularization

If you have a high variance problem (your model is overfitting and performing really well on your training data, but not on your dev set), regularization is one way to help.

In logistic regression, recall:

$$ J(w, b) = \dfrac{1}{m} \sum\limits^{m}\_{i=1} \mathcal{L}(\hat{y}^{(i)}, y^{(i)}) + \dfrac{\lambda}{2m} ||w||^2\_2 $$

**L2 Regularization**

$$ ||w||^2\_2 = \sum\limits^{n\_x}\_{j=1} w\_j^2 = w^Tw$$

**L1 Regularization**

$$ \dfrac{\lambda}{m} \sum\limits^{n\_x}\_{j=1}|w\_j| = \dfrac{\lambda}{m}||w||\_1 $$

In L1 regularization, $w$ will be sparse.

The Lambda $\lambda$ is known as the regularization parameter. This is another hyperparameter that one needs to tune.

For Neural Networks, regularization looks more like:

$$ J(w^{[1]}, b^{[1]}, \dots, w^{[L]}, b^{[L]}) = \dfrac{1}{m} \sum\limits^{m}\_{i=1} \mathcal{L}(\hat{y}^{(i)}, y^{(i)}) + \dfrac{\lambda}{2m} \sum\limits^{L}\_{l=1} ||w^{[l]}||^2\_F $$

$$ ||w^{[l]}||^2\_F = \sum\limits^{n^{[l-1]}}\_{i=1} \sum\limits^{n^{[l]}}\_{j=1} (w^{[l]}\_{ij})^2 $$

Recall the shape of w is $(n^{[l]}, n^{[l-1]})$. This matrix norm is called the *Forbenius Norm*.

$$ || \cdot || ^2\_2 \rightarrow || \cdot ||^2\_F$$

This is also known as *weight decay*.

$$ W^{[l]} = W^{[l]} - \alpha[(\text{From Backprop}) + \dfrac{\lambda}{m}W^{[l]}] $$
$$ (1-\dfrac{\alpha\lambda}{m})W^{[l]} = W^{[l]} - \dfrac{\alpha\lambda}{m} W^{[l]} - \alpha(\text{From Backprop}) $$

### Why regularization reduces overfitting?

Regularization reduces the impact of your weights in your neural network.

If your activation function is $g(z) = tanh(z)$, regularizing puts your values close to zero, allowing it to be effected by the linear portion of the tanh function.

![regularization_param](/img/deeplearning-ai/regularization_param.png)

Make sure you plot the correct value of J with the regularization parameter.

### Dropout Regularization

Dropout regularization is going through each of the layers in a neural network and for each node randomly remove a node. (For instance, each node has a 50% chance of being removed.)

![dropout_regularization](/img/deeplearning-ai/dropout_regularization.png)

Implementing dropout for layer `l=3`

```python
keep_prob = 0.8
d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob
a3 = np.multiply(a3, d3)
a3 /= keep_prob
```

If you had 50 units, you probably have around 10 units shut off.

$$ Z^{[4]} = W^{[4]}a^{[3]}+b^{[4]} $$
$$ a^{[3]} \leftarrow \text{ has been reduced by } 20% $$
the `/= 0.8` increases the weight back.

On each iteration of gradient descent, you zero out different patterns of your hidden units.

At Test Time, do not use dropout. Dropoout is primarily most effective during training. When testing, you don't want your output to be random.

### Understanding Dropout

Drop out works because you can't rely on any one feature, therefore you should spread out your weights. This serves to shrink the weights in your neural network.

You do not have to drop out each layer. Drop out can be layer specific.

Drop out is really effective for computer vision problems

One caveat is that this makes the cost function less defined as your nodes are being randomly killed each iteration. 

### Other regularization methods

What you can do is augment your training set. For instance, if your training set has a bunch of photos, it might be valid to double your training set by flipping it horizontally. Might also be good to take random distortions (rotations, cropping), to make more 'fake' training samples.

Early Stopping might also be another way to minimize the effects of high variance. 

![early_stopping](/img/deeplearning-ai/early_stopping.png)

The downside of early stopping is you're not performing the two step process of optimizing the cost function J and second step of not overfitting.

L2 Regularization might be better than early stopping, at the price of more computation.

## Setting up your optimization problem

### Normalizing Inputs

Normalizing your inputs corresponds to two steps

1. subtract out the mean
2. normalize the variances

![normalize_training_set](/img/deeplearning-ai/normalize_training_set.png)

You should also use the same values for $\mu$ and $\sigma$ on the dev/test set.

![why_normalize_inputs](/img/deeplearning-ai/why_normalize_inputs.png)

When the scale is more uniform, gradient descent perfroms better and your learning rate does not have to be extremely small.

### Vanishing/Exploding Gradients

Given a very deep network, it is possible to have slightly greater than one weights make the activtions explode to be very high.

It is also possible to have slightly less than one weights make the activations shrink to be some extremely small value.

To combat this, a partial solution is one must carefully initialize the weights.

### Weight Initialization for Deep Networks

One reasonable thing to do is to set the variance of $w\_i$ to be equal to $\dfrac{1}{n}$.

```python
w_l = np.random.randn(shape) * np.sqrt(2/n_last_l)
```

This is perfectly fine for ReLU.

Might also use Xavier initialization, for tanh. Look up research for weight initialization.

### Numerical Approximation of Gradients

Given your point, add an epsilon and subtract an epsilon. Calculate the triangle given these two points, and compare with the derivative computed from the point afterwards.

![checking_derivative_calculation](/img/deeplearning-ai/checking_derivative_calculation.png)

### Gradient Checking

Gradient checking is a technique to verify that your implementation of backpropagation is correct.

Take all of your parameters $W^{[1]}, b^{[1]}, \dots, W^{[L]}, b^{[L]} $ and reshape them into a big vector $\theta$

Take all of your parameter derivatives $dW^{[1]}, db^{[1]}, \dots, dW^{[L]}, db^{[L]} $ and reshape them into a big vector $d\theta$

Gradient Checking (Grad Check)
```python
for each i:
    dThetaApprox [i] = J(theta1, theta2, theta3, theta[i + e]) - 
      J(theta1 theta2, theat[i - e])
```

In practice, it might he useful to set epsilon to be $10^{-7}$.

### Gradient Checking Implementation Notes

* Don't use gradient checking in training, only to debug.
* If an algorithm fails grad check, look at components to try to identify bug.
* Remember to use your regularization terms.
* Gradient checking implementation does not work with drop out.
* Run at random intiailization; then run it again after some training.
