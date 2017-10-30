---
title: "Machine Learning, Week 10"
author: "Alexander Wong"
tags: ["Machine Learning"]
date: 2017-10-29T19:21:37-06:00
draft: false
---

Taking the [Coursera Machine Learning](https://www.coursera.org/learn/machine-learning) course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/).

Assumes you have knowledge of [Week 9]({{% relref "coursera-machine-learning-week9.md" %}}).

{{% toc %}}

* Lecture notes:
  * [Lecture17](/docs/coursera-machine-learning-week10/Lecture17.pdf)

# Large Scale Machine Learning

## Gradient Descent with Large Datasets

### Learning With Large Datasets

One of the best ways to get a high performance machine learning system is to supply a lot of data into a low bias (overfitting) learning algorithm. The gradient descent algorithm can run very slowly if the training set is large, because a summation needs to be performed across all training examples to perform one step of gradient descent.

$$ h\_\theta(x) = \sum\limits\_{j=0}^n \theta\_jx\_j $$
$$ J\_\text{train}(\theta) = \dfrac{1}{2m}\sum\limits\_{i=1}^m(h\_\theta(x^{(i)})-y^{(i)})^2 $$
$$ \theta\_j := \theta\_j - \alpha \dfrac{1}{m}\sum\limits\_{i=1}^m (h\_\theta(x^{(i)})-y^{(i)})x\_j^{(i)} $$

- What if $m$ were 100,000,000? $\dfrac{1}{m}\sum\limits\_{i=1}^m (h\_\theta(x^{(i)})-y^{(i)})x\_j^{(i)}$ becomes very expensive.
- Could we do a sanity check by running gradient descent with 1000 randomly selected examples?
  - One way to verify this is to plot a learning curve for a range of values of m (say 100, 10,000, 1,000,000) and verify that the algorithm has high variance (overfitting) when m is small.

### Stochastic Gradient Descent

Rather than running gradient descent on the entire training set, one can run gradient descent on one training set at a time. The following is the Stochastic Gradient Descent algorithm:

$$ \text{cost}(\theta, (x^{(i)}, y^{(i)})) = \dfrac{1}{2}(h\_\theta(x^{(i)})-y^{(i)})^2 $$
$$ J\_\text{train}(\theta) = \dfrac{1}{2m}\sum\limits\_{i=1}^m\text{cost}(\theta, (x^{(i)}, y^{(i)})) $$

1. Randomly shuffle or reorder the dataset.
2. Repeat {
  for i = 1, ..., m {
    $$ \theta\_j := \theta\_j - \alpha(h\_\theta(x^{(i)}) - y^{(i)})x\_j^{(i)} $$
    for j = 0, ..., n
  }
}
Rather than waiting to sum up all of the training sets before taking a step, we can take a step on a single training example.

![stochastic_gradient_descent](/img/coursera-machine-learning-week10/stochastic_gradient_descent.png)

* Repeat the outer loop somewhere between 1 and 10 times. The inner loop would require iterating through all of your training examples.

### Mini-Batch Gradient Descent

Mini-Batch Gradient Descent is a variation of Stochastic Gradient Descent except rather than using a single example in each iteration, it uses $b$ examples in each iteration where $b$ is the mini-batch size. A typical choice for $b$ is 10, where $b$ ranges between 2-100.

b = 10 example:

$$ (x^{(i)}, y^{(i)}), \dots, (x^{(i+9)}, y^{(i+9)}) $$
$$ \theta\_j := \theta\_j - \alpha\dfrac{1}{10} \sum\limits\_{k=1}^{i+9}(h\_\theta(x^{(k)})-y^{(k)})x\_j^{(k)}$$
Increment $i$ by 10 and repeat until all training examples are used

![mini_batch_gradient_descent](/img/coursera-machine-learning-week10/mini_batch_gradient_descent.png)

This algorithm becomes the same as normal batch gradient descent if $b = m$.

### Stochastic Gradient Descent Convergence

Stochastic gradient descent does not converge nicely like Batch gradient descent. In Batch gradient descent, the cost function would decrease as the number of iterations of gradient descent increased. In Stochastic gradient descent, this is not certain.

$$ \text{cost}(\theta, (x^{(i)}, y^{(i)})) = \dfrac{1}{2}(h\_\theta(x^{(i)} - y^{(i)}))^2 $$
During learning, compute $\text{cost}(\theta, (x^{(i)}, y^{(i)}))$ before updating $\theta$ using $(x^{(i)}, y^{(i)})$. Every 1000 iterations (approximately, depends on your use case), plot $\text{cost}(\theta, (x^{(i)}, y^{(i)}))$ averaged over the last 1000 examples processed by the algorithm.

![stochastic_convergence](/img/coursera-machine-learning-week10/stochastic_convergence.png)

The learning rate should be sufficiently small. Additionally, when the stochasti gradient descent nears a minima, one way to make it converge is to slowly make the learning rate $\alpha$ decrease over time.

One example of doing this is to make $\alpha = \dfrac{\text{const1}}{\text{iterationNumber} + \text{const2}}$. The constants are application dependent.

## Advanced Topics

### Online Learning

Online learning allows one to model problems where data is commin in as a continuous stream. Your training set is infinite.
One way to handle this is to update your models whenever the training data is given to you (one step of Stochastic Gradient Descent), and then use the resulting trained model.

### Map Reduce and Data Parallelism

Map Reduce and Data Parallelism are ways to break up multiple chunks of data into smaller, parallelizable parts.
Take the following use case of Batch Gradient Descent:

![map_reduce_example](/img/coursera-machine-learning-week10/map_reduce_example.png)

Whenever your learning algorithm can be expressed as computing sums of functions over the training set, map reduce may offer you some better optimization. There is also a benefit if your single machine has multiple cores, as each of the cores can perform parallelised computation.

---

Week 11 TBD
