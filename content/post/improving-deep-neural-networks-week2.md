---
title: "Improving Deep Neural Networks, Week 2"
author: "Alexander Wong"
tags: ["Machine Learning", "deeplearning.ai"]
date: 2017-12-17T15:21:37-06:00
draft: false
---

Taking the [Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning), **Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization** course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/). See [deeplearning.ai](https://www.deeplearning.ai/) for more details.

Assumes you have knowledge of [Improving Deep Neural Networks, Week 1]({{% relref "improving-deep-neural-networks-week1.md" %}}).

{{% toc %}}

# Optimization Algorithms

## Mini-Batch Gradient Descent

Rather than training on your entire training set during each step of gradient descent, break out your examples into groups.

For instance, if you had 5,000,000 training examples, it might be useful to do 5000 batches of 1000 examples each.

New notation for each training batch should use curly brace super scripts.

$$ X^{\\{t\\}}, Y^{\\{t\\}} $$

![mini_batch_gradient_descent](/img/deeplearning-ai/mini_batch_gradient_descent.png)

Using Mini-Batch Gradient Descent (5000 batches of 1000 examples each)

For each batch, perform forward prop on $ X^{\\{t\\}} $.

Then compute the cost:

$$ J^{\\{t\\}} = \dfrac{1}{1000} \sum\limits^{l}\_{i=1}\mathcal{L}(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2 * 1000} \sum ||W^{[l]}||^2\_F $$

Backprop to compute the gradient with respect to $J^{\\{t\\}} $ (using $X^{\\{t\\}}, Y^{\\{t\\}}$)

One epoch means running through your entire training set.

## Understanding Mini-batch Gradient Descent

Training using mini-batch gradient descent does not show a smooth curve with constantly decreasing cost. It is a little bit more jumpy. It should show a slightly nosier trend downward.

![training_with_mini_batch_gradient_descent](/img/deeplearning-ai/training_with_mini_batch_gradient_descent.png)

If your mini-batch size is $m$, then you're just doing Batch gradient descent. This has the problem of taking too long per iteration.

If your mini-batch size is 1, then you're doing Stochastic gradient descent. Every example is its own mini-batch. Stochastic gradient descent never converges. This has the problem of losing all of the speedup from vectorization.

In practice, the mini-batch size used is somewhere inbetween 1 and m. This should give you the fastest learning. (Bonuses of Vectorization while making progress without needing to wait until the entire training set is processed.)

**Takeaways**

If you have a small training set (m < 2000), just use batch gradient descent.

Typical mini-batch sizes are 64-1024 (make it a power of 2).

Make sure your mini-batch fits in CPU/GPU memory! ($ X^{\\{t\\}}, Y^{\\{t\\}} $)

## Exponentially Weighted Averages

![exponentially_weighted_averages](/img/deeplearning-ai/exponentially_weighted_averages.png)

$$ V\_t = \beta V\_{t-1} + (1-\beta)\theta\_t $$
$$ \beta = 0.9 \approx \text{ 10 days average temperature} $$

![weighted_averages_high_beta](/img/deeplearning-ai/weighted_averages_high_beta.png)

$$ \beta = 0.98 \approx \text{ 50 days average temperature} $$

![weighted_averages_low_beta](/img/deeplearning-ai/weighted_averages_low_beta.png)

$$ \beta = 0.5 \approx \text{ 2 days average temperature} $$

## Understanding Exponentially Weighted Averages

Exponentially weighted averages are a way to sum the averages of all previous values.

![understanding_weighted_averages](/img/deeplearning-ai/understanding_weighted_averages.png)

$$ V\_{100} = 0.1\theta\_{100} + 0.1 \cdot 0.9 \cdot \theta\_{99} + 0.1 \cdot 0.9^{2} \cdot \theta\_{98} + 0.1 \cdot 0.9^{3} \cdot \theta\_{97} + \dots $$

## Bias Correction in Exponentially Weighted Averages

Slight bias occurs because $V\_0 = 0$. Therefore, you get a curve resembling the purple line.

![bias_correction](/img/deeplearning-ai/bias_correction.png)

Bias Correction by using $\dfrac{V\_t}{1-\beta^{t}}$.

## Gradient Descent with Momentum

Combine the weighted averages with gradient descent.

![gradient_descent_with_momentum](/img/deeplearning-ai/gradient_descent_with_momentum.png)

**Implementation details**

On iteration $t$:

Compute $dW, db$ on the current mini-batch

$$ v\_{dW} = \beta v\_{dW} + (1-\beta)dW $$
$$ v\_{db} = \beta v\_{db} + (1-\beta)db $$
$$ W = W - \alpha v\_{dW}, b = b - \alpha v\_{db} $$

## RMSprop

Root Means Squared prop.

![rmsprop](/img/deeplearning-ai/rmsprop.png)

## Adam Optimization Algorithm

Initialize $V\_{dW}=0, S\_{dW}=0, V\_{db}=0, S\_{db}=0$.

On iteration t:

Compute dW, db using mini-batch

**Momentum $\beta\_1$**
$$ V\_{dW} = \beta\_{1}V\_{dW} + (1-\beta\_{1})dW $$
$$ \hspace{1em} V\_{db} =\beta\_{1}V\_{db} + (1-\beta\_1)db $$

$$ V^{\text{corrected}}\_{dW} = V\_{dW}/(1-\beta\_{1}^t) $$
$$ \hspace{1em} V^{\text{corrected}}\_{db} = V\_{db}/(1-\beta\_1^t) $$


**RMSprop $\beta\_2$**
$$ S\_{dW} = \beta\_{2}S\_{dW} + (1-\beta\_{2})dW^2 $$
$$ \hspace{1em} S\_{db} = \beta\_{2}S\_{db} + (1-\beta\_2)db $$

$$ S^{\text{corrected}}\_{dW} = S\_{dW}/(1-\beta\_2^t) $$
$$ S^{\text{corrected}}\_{db} = S\_{db}/(1-\beta\_2^t) $$

**Finally**

$$  W := W - \alpha \dfrac{V^{\text{corrected}}\_{dW}}{\sqrt{S^{\text{corrected}}\_{dW}}+\epsilon}$$

$$ b := b - \alpha \dfrac{V^{\text{corrected}}\_{db}}{\sqrt{S^{\text{corrected}}\_{db}}+\epsilon} $$

**Hyperparameters Choice:**

$$ \alpha : \text{ needs to be tuned} $$
$$ \beta\_1: 0.9  \leftarrow (dW) $$
$$ \beta\_2: 0.999 \leftarrow (dW^2) $$
$$ \epsilon: 10^{-8} $$

Adam: adaptive moment estimation.

## Learning Rate Decay

One thing that might help speed up the learning algorithm is to slowly reduce the learning rate $\alpha$ over time.

This allows for faster approach to convergence near the end of the algorithm.

Recall 1 epoch = 1 pass through your entire training set.

$$ \alpha = \dfrac{1}{1 + \text{decay-rate} * \text{epoch-num}} * \alpha\_0 $$

**Example**
$$ \alpha\_0 = 0.2 $$
$$ \text{decay-rate} = 1 $$

| Epoch | $\alpha$ |
|-------|----------|
| 1     | 0.1      |
| 2     | 0.67     |
| 3     | 0.5      |
| 4     | 0.4      |

Many different ways of learning rate decay, like exponential decay, discrete staircase, manual decay.

## The Problem of Local Optima

Low dimensional spaces do not transfer to high dimensional spaces. The problem is of plateaus.

![local_optimum](/img/deeplearning-ai/local_optimum.png)

You are pretty unlikely to get stuck in local optima

![plateau](/img/deeplearning-ai/plateau.png)
