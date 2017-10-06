---
title: "Machine Learning, Week 7"
author: "Alexander Wong"
tags: ["Machine Learning"]
date: 2017-10-04T15:38:02-06:00
draft: false
---

Taking the [Coursera Machine Learning](https://www.coursera.org/learn/machine-learning) course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/).

Assumes you have knowledge of [Week 6]({{% relref "coursera-machine-learning-week6.md" %}}).

{{% toc %}}

* Lecture notes:
  * [Lecture12](/docs/coursera-machine-learning-week7/Lecture12.pdf)

# Support Vector Machines

## Large Margin Classification

### Optimization Objective

We are simplifying the logistic regression cost function by converting the sigmoid function into two straight lines, as shown here:

![svm_cost1_cost0](/img/coursera-machine-learning-week7/svm_cost1_cost0.png)

The following are two cost functions for support vector machines:

$$ \min\limits\_{\theta} \dfrac{1}{m} [\sum\limits\_{i=1}^m y^{(i)} \text{cost}\_1(\theta^Tx^{(i)}) + (1-y^{(i)}) \text{cost}\_0(\theta^Tx^{(i)}) ] + \dfrac{\lambda}{2m} \sum\limits\_{j=1}^n \theta\_j^2 $$
$$ \min\limits\_{\theta} C[ \sum\limits\_{i=1}^m y^{(i)} \text{cost}\_1(\theta^Tx^{(i)}) + (1-y^{(i)}) \text{cost}\_0(\theta^Tx^{(i)}) ] + \dfrac{1}{2} \sum\limits\_{j=1}^{n} \theta^2\_j  $$

They both give the same value of $\theta$ if $C = \dfrac{1}{\lambda} $.

Hypothesis will predict:

$$ h\_\theta(x) = 1 \hspace{1em} \text{if} \hspace{1em} \theta^Tx \geq 0 $$
$$ h\_\theta(x) = 0 \hspace{1em} \text{otherwise} $$

### Large Margin Intuition

Support Vector Machines are also known as Large Margin Classifiers. This is because when plotting the positive and negative examples, a support vector machine will draw a decision boundary with large margins:

![large_margin_classifier](/img/coursera-machine-learning-week7/large_margin_classifier.png)

This is different than linear regression, where the decision boundary can be very close to the positive and negative examples (due to $\theta^Tx \approx 0$ in the $y=1 \text{ or } y=0$ cases. )

![svm_vs_linear_regression](/img/coursera-machine-learning-week7/svm_vs_linear_regression.png)

When the data is not linearly sepearable, one should take into consideration the regularization parameter $C$. 

![svm_outliers](/img/coursera-machine-learning-week7/svm_outliers.png)

* The magenta line is when the regularization parameter is not large and there is a the one small outlier in the bottom left corner.
* The black line is when the regularization parameter is large.
* The black line could also be when the regularization parameter is small and there are many datapoints (the drawn X's and O's) making the plot difficult to separate linearly.

## Kernels

In a non linear decision boundary, we can have many choices for high order polynomials. A Kernel is, given $x$, compute a new feature depending on proximity to landmarks $l^{(1)}, l^{(2)}, l^{(3)} $

Given the example $x$:
$$ f\_1 = \text{ similarity}(x, l^{(1)}) = \exp(-\dfrac{|| x - l^{(1)} ||^2}{2 \sigma ^2}) $$
$$ f\_2 = \text{ similarity}(x, l^{(2)}) = \exp(-\dfrac{|| x - l^{(2)} ||^2}{2 \sigma ^2}) $$
$$ || x - l^{(1)} ||^2 = \text{ square of the euclidian distance between x and l}^{(1)} $$

These functions are kernels, these specific ones are gaussian kernels. Consider them similarity functions.

If $x \approx l^{(1)}$ then $ f\_1 \approx \exp(-\dfrac{0^2}{2\sigma^2}) \approx 1 $.

If $x$ is far from $l^{(1)}$ then $ f\_1 = \exp(-\dfrac{\text{large number}^2}{2\sigma^2}) \approx 0 $

The smaller the sigma, the feature falls more rapidly.

![kernel_sigma](/img/coursera-machine-learning-week7/kernel_sigma.png)

How do we choose the landmarks $l$?

Given $m$ training examples, set $l$ to be each one of your training examples.

![kernel_landmarks](/img/coursera-machine-learning-week7/kernel_landmarks.png)

The following is how you would train using kernels (similarity functions):

![kernel_training](/img/coursera-machine-learning-week7/kernel_training.png)

When using an SVM, one of the choices that need to be made is $C$. Also, one must consider the choice of $\sigma^2$. The following is the bias/variance tradeoff diagrams.

![kernel_bias_variance_tradeoff](/img/coursera-machine-learning-week7/kernel_bias_variance_tradeoff.png)

## Source Vector Machines (in Practice)

When to choose between a linear kernel and a gaussian kernel?

![which_kernel_to_use](/img/coursera-machine-learning-week7/which_kernel_to_use.png)

Note: When using the Gaussian kernel, it is important to perform feature scaling beforehand.

The kernels that you choose must satisfy a technical condition called "Mercer's Theorem" to make sure SVM packages' optimizations run correctly and do not diverge.

- Polynomial kernel: $ k(x,l) \in { (x^Tl)^2, (x^Tl)^3, (x^Tl + 7)^7 } $
  - usually performs worse than the gaussian kernel
- String kernel, chi-square kernel, histogram intersection kernel, etc.

**Logistic regression vs Source Vector Machines**

* If $n$ is large relative to $m$ (n = 10,000, m=10-1000)
  * use logistic regression, or SVM without a kernel ("linear kernel")
* If $n$ is small and $m$ is intermediate (n = 1-1000, m = 10-10,000)
  * use SVM with Gaussian kernel
* If n is small and m is large (n = 1-1000, m = 50000+)
  * Create/add more features, then use logistic regression or SVM without a kernel.

Neural networks are likely to work well for msot of these settings, but may be slower to train.
