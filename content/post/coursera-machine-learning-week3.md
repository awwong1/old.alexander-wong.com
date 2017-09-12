---
title: "Machine Learning, Week 3"
author: "Alexander Wong"
tags: ["Machine Learning"]
date: 2017-09-07T00:04:44-06:00
draft: false
---

Taking the [Coursera Machine Learning](https://www.coursera.org/learn/machine-learning) course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/).

Assumes you have knowledge of [Week 2]({{% relref "coursera-machine-learning-week2.md" %}}).

{{% toc %}}

* Lecture notes:
  * [Lecture6](/docs/coursera-machine-learning-week3/Lecture6.pdf)
  * [Lecture7](/docs/coursera-machine-learning-week3/Lecture7.pdf)

# Logistic Regression

## Classification and Representation

### Classification

Recall that classification involves a hypothesis function which returns a discontinuous output (common example was whether or not a tumor was benign or cancerous based on size).

To attempt classification, one option is to use linear regression and map all the predictions greater than or equal 0.5 as a 1, and all of the values less than 0.5 as a 0. However, this method does not work well because classification is not actually a linear function.

The classification problem is similar to the regression problem, except the values we now predict can only be a small number of discrete values.

The **binary classification problem** is when y can take on two values, 0 and 1.

Example: When trying to build a spam classifier, then $x^{(i)}$ may be features of a piece of email, and $y$ can be 1 if it's spam or $y$ can be zero otherwise. Hence, $y \in \\{0, 1\\}$. The negative class here is 0, and the positive class here is 1. Given $x^{(i)}$, the corresponding $y^{(i)}$ is also called the label for the training example.

### Hypothesis Representation

We want to change the model of our hypothesis function to fit the use case where answers can either be 0 or 1. One representation of this hypothesis function is the **Sigmoid Function** or the **Logistic Function**.

$$ h\_\theta(x) = g(\theta^Tx) $$
$$ z = \theta^Tx $$
$$ g(z) = \dfrac{1}{1 + e^{-z}} $$

This is what the above function looks like:

![sigmoid_function](/img/coursera-machine-learning-week3/sigmoid_function.png)

The function $g(z)$ maps any real number to the (0, 1) interval. This is useful in transforming an arbitrary valued function into a function suited for classification.

The hypothesis function $h\_\theta(x)$ gives us the probability that our output is 1. For example, $h\_\theta(x) = 0.7$ tells us that input x is 70% likely to output as a 1. The probability that the prediction is 0 is the compliment (in this case it is 30%).

$$ h\_\theta(x) = P(y = 1|x;\theta) = 1 - P(y = 0|x; \theta) $$
$$ P(y = 0|x; \theta) + P(y = 1|x; \theta) = 1 $$

### Decision Boundary

In order to classify discretely into 0 or 1, we translate the output of the hypothesis function as follows:

$$ h\_\theta(x) \geq 0.5 \rightarrow y = 1 $$
$$ h\_\theta(x) \lt 0.5 \rightarrow y = 0 $$

The way the logistic function $g$ behaves is that when its input is greater than or equal to zero, its output is greater than or equal to 0.5.

$$ g(z) \geq 0.5 $$
$$ \text{when } z \geq 0 $$

Recall, from the definition of the sigmoid function:

$$ z=0, e^0-1 \Rightarrow g(z)=1 / 2 $$
$$ z \rightarrow \infty, e^{-\infty} \rightarrow 0 \Rightarrow g(z) = 1 $$
$$ z \rightarrow -\infty, e^{\infty} \rightarrow \infty \Rightarrow g(z) = 0 $$

If the input to $g$ is $\theta^TX$ then:

$$ h\_\theta(x) = g(\theta^Tx) \geq 0.5 $$
$$ \text{when } \theta^Tx \geq 0 $$

Combining the two above statements, we can now say:

$$ \theta^Tx \geq 0 \Rightarrow y = 1 $$
$$ \theta^Tx \lt 0 \Rightarrow y = 0 $$

This function is our **decision boundary**. It separates the area where $y = 0$ and where $y = 1$. It is created by the hypothesis function.

Example:

$$ \theta = \begin{bmatrix} 5 \newline -1 \newline 0 \end{bmatrix} $$
$$ y = 1 \text{ if } 5 + (-1)x\_1 + 0x\_2 \geq 0 $$
$$ 5 - x\_1 \geq 0 $$
$$ -x\_1 \geq -5 $$
$$ x\_1 \leq 5 $$

In this example, the decision boundary is a straight vertical line placed on the graph where $x\_1 = 5$. Everything to the left of this line denotes $y = 1$ while everything to the right of this line denotes $y = 0$.

Note: The input to the sigmoid function $g(z)$ does not need to be linear. A valid input function could be $z = \theta\_0 + \theta\_1x\_1^2 + \theta\_2x\_2^2 $ which describes a circle. We can use any arbitrary function which fits our data.

## Logistic Regression Model

### Cost Function

We cannot use the same cost function that we use for linear regression because the logistic function will cause the output to be wavy, causing many local optima. We want our function to be convex.

For logistic regression, the cost function looks like this:

$$ J(\theta) = \dfrac{1}{m} \sum\limits\_{i=1}^m \text{ Cost}(h\_\theta(x^{(i)}), y^{(i)}) $$
$$ \text{Cost}(h\_\theta(x), y) = -\log(h\_\theta(x)) \hspace{1em} \text{if} \hspace{1em} y = 1 $$
$$ \text{Cost}(h\_\theta(x), y) = -\log(1 - h\_\theta(x)) \hspace{1em} \text{if} \hspace{1em} y = 0 $$

When y = 1, we get the following plot for $J(\theta)$ vs $h\_\theta(x)$:

![logistic_regression_cost_function_y_eq_1](/img/coursera-machine-learning-week3/logistic_regression_cost_function_y_eq_1.png)

When y = 0, we get the following plot for $J(\theta)$ vs $h\_\theta(x)$:

![logistic_regression_cost_function_y_eq_0](/img/coursera-machine-learning-week3/logistic_regression_cost_function_y_eq_0.png)

This is equivalent to the following:

$$ \text{Cost}(h\_\theta(x), y) = 0 \text{ if } h\_\theta(x) = y $$
$$ \text{Cost}(h\_\theta(x), y) \rightarrow \infty \text{ if } y = 0 \text{ and } h\_\theta(x) \rightarrow 1 $$
$$ \text{Cost}(h\_\theta(x), y) \rightarrow \infty \text{ if } y = 1 \text{ and } h\_\theta(x) \rightarrow 0 $$

If the correct answer 'y' is 0, then the cost function will be 0 if our hypothesis function also outputs a 0. If the hypothesis approaches 1, then the cost function will approach infinity.

If the correct answer 'y' is 1, then the cost function will be 0 if our hypothesis function also outputs a 1. If the hypothesis approaches 0, then the cost function will approach infinity.

Writing the cost function in this way guarantees that $J(\theta)$ is convex for logistic regression.

### Simplified Cost Function and Gradient Descent

The two conditional cases of the cost function can be compressed into one case:

$$ \text{Cost}(h\_\theta(x), y) \hspace{1em} = \hspace{1em} -y \log(h\_\theta(x)) \hspace{1em} - \hspace{1em} (1 - y)\log(1 - h\_\theta(x)) $$

Notice! When y is equal to 1, then the second term $(1 - y)\log(1 - h\_\theta(x))$ will equal zero and will not effect the result. If y is equal to 0 then the first term $-y\log(h\_\theta(x))$ will be zero and will not effect the result.

We can fully write this entire cost function as the following:

$$ J(\theta) = - \dfrac{1}{m} \sum\limits\_{i=1}^m[y^{(i)}\log(h\_\theta(x^{(i)})) + (1 - y^{(i)})\log(1 - h\_\theta(x^{(i)}))] $$

This is equivalent to the following vectorized form:

$$ h = g(X\theta) $$
$$ J(\theta) = \dfrac{1}{m} \cdot (-y^T \log(h) - (1 - y)^T \log(1-h))$$

Recall that in Gradient descent, the general form is the following:

_Repeat_
$$ \theta\_j := \theta\_j - \alpha \dfrac{\partial}{\partial\theta\_j}J(\theta) $$

Calculating the derivative, we get:

_Repeat_
$$ \theta\_j := \theta\_j - \dfrac{\alpha}{m} \sum\limits\_{i=1}^m(h\_\theta(x^{(i)}) - y^{(i)}) \cdot x^{(i)} $$

This is cosmetically the same algorithm we used in linear regression, however in linear regression the value $h\_\theta(x) = \theta^Tx$ but in logistic regression, the value $h\_\theta(x) = \dfrac{1}{1 + e^{-\theta^Tx}}$. We still must update all values in theta.

Vectorized approach is $\theta := \theta - \dfrac{\alpha}{m}X^T(g(X\theta) - \overrightarrow{y}) $

### Advanced Optimization

There exist many algorithms, like "Conjugate Gradient, BFGS, L-BFGS" that are more sophisticated, faster ways to optimize $\theta$ instead of gradient descent. (It's suggested not to write these algorithms yourself, but instead use the libraries provided by Octave.)

We first provide a function that evaluates the following two functions for a given input value $\theta$:

$$J(\theta)$$
$$\dfrac{\partial}{\partial\theta\_j}J(\theta)$$

We can then write a single function that returns both of these:

```octave
function [jVal, gradient] = costFunction(theta)
  jVal = [... code to compute J(theta)...];
  gradient = [...code to compute derivative of J(theta)...];
end
```

The we can use octave's `fminfunc()` optimizatino algorithm with the `optimset()` function that creates an object containing the options we want to send to `fminunc()`

```octave
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2, 1);
[optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```

The function `fminunc()` is given our cost function, initial theta values, and the options object we created beforehand.

## Multiclass Classification

### Multiclass Classification: One-vs-all

The logistic regression classifier is extended to work in the case with more than two categories. Instead of $y = { 0, 1 }$ we will expand our definition such that $y = { 0, 1, ..., n }$.

Since $y = { 0, 1, ..., n }$ we divide our problem into $n + 1$ (Add 1 because the index starts at 0) binary classification problems. In each one of these problems, predict the probability that $y$ is a member of one of the classes.

$$y \in { 0, 1, ..., n } $$
$$ h\_\theta^{(0)}(x) = P(y = 0|x;\theta) $$
$$ h\_\theta^{(1)}(x) = P(y = 1|x;\theta) $$
$$ \vdots $$
$$ h\_\theta^{(n)}(x) = P(y = n|x;\theta) $$
$$ \text{prediction } = \max\limits\_i(h\_\theta^{(i)}(x)) $$

We are choosing one class and lumping all others into a single second class. This is done repeatedly by applying binary logistic regression to each case, then use the hypothesis that returned the highest value as our prediction.

This image shows an example for how one could classify three classes:

![multiclass_classification_one_vs_all](/img/coursera-machine-learning-week3/multiclass_classification_one_vs_all.png)

**Summary:**
Train a logistic regression classifier $h\_\theta(x)$ for each class to predict the probability that $y = i$. To make a prediction on a new x, pick the class that maximizes $h\_\theta(x)$.

# Regularization

## Solving the Problem of Overfitting

### The Problem of Overfitting

Consider the problem of predicting $y \text{ from } x \in \mathbb{R}$.

![overfitting_example](/img/coursera-machine-learning-week3/overfitting_example.png)

* The left most figure shows the result of fitting the straight line. $y = \theta\_0 + \theta\_1x$ to a dataset. The data doesn't really lie on a straight line, we can see the fit is not good. This figure is **underfitting** because the data clearly shows structure not captured by the model.
* The middle figure looks like an accurate fit to our dataset.
* The right most figure shows the result of fitting a $5^{\text{th}}$ order polynomial $y = \sum\_{j=0}^{5} \theta\_jx^j$. The curve passes through the data perfectly, but we would not expect this to be a good predictor for housing prices (y) for different living areas (x). This figure is **overfitting**.

Underfitting (high bias) is when the hypothesis function h maps poorly to the trend of the data. It is caused by a function that is too simple or uses too few features. Overfitting (high variance) is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. This is caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.

This terminology is applied to both logistic and linear regression. There are two main options to address the issue of overfitting.

1) Reduce the number of features

* Manually select which features to keep
* Use a model selection algorithm (later in course)

2) Regularization

* Keep all of the features, but reduce the magnitude of parameters $\theta\_j$
* Regularization works well when we have many slightly useful features.

### Cost Function

We can reduce the weight that some of the terms in our function carry by increasing their cost in order to reduce overfitting from our hypothesis function.

Let's say we wanted to make the following function more quadratic:

$$ \theta\_0 + \theta\_1x + \theta\_2x^2 + \theta\_3x^3 + \theta\_4x^4 $$

We will want to eliminate the influence of $\theta\_3x^3$ and $\theta\_4x^4$. Without getting rid of these features or changing the form of our hypothesis, we can instead modify the _cost function_:

$$ \min\_\theta \dfrac{1}{2m} \sum\limits\_{i=1}^m(h\_\theta(x^{(i)}) - y^{(i)})^2 + 1000 \cdot \theta\_3^2 + 1000 \cdot \theta\_4^2 $$

We add two extra terms at the end of the cost function to inflate the cost of $\theta\_3$ and $\theta\_4$. Now, in order for the cost function to get close to zero, we will need to reduce the values of $\theta\_3$ and $\theta\_4$ to near zero. This in return reduces the values of $\theta\_3x^3$ and $\theta\_4x^4$ in the hypothesis function. As seen below, the new hypothesis denoted by the pink curve looks like a quadratic function and fits the data better due to the small terms $\theta\_3x^3$ and $\theta\_4x^4$.

![cost_function_regularization](/img/coursera-machine-learning-week3/cost_function_regularization.png)

We can also regularize all of our theta parameters in a single summation.

$$ \min\_\theta \dfrac{1}{2m} \sum\limits\_{i=1}^m(h\_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum\limits\_{j=1}^n\theta\_j^2 $$

Note, we don't regularize $\theta\_0$, although in practice it usually doesn't matter.

The lambda ($\lambda$) is the **regularization parameter**. It determines how much theta parameter costs are inflated. We can use this to smooth the output of the hypothesis function to reduce overfitting. However, if lambda is too large, it may smooth out the function too much and cause underfitting (make the hypothesis appear to be a straight horizontal line). If the lambda is too small, we will not penalize costs enough and the problem of overfitting remains.

### Regularized Linear Regression

We will modify the gradient descent function to separate out $\theta\_0$ from the rest of the parameters because we do not want to penalize $\theta\_0$.

_Repeat {_
$$ \theta\_0 := \theta\_0 - \alpha \dfrac{1}{m} \sum\limits\_{i=1}^m(h\_\theta(x^{(i)}) - y^{(i)}) \cdot x\_0^{(i)} $$
$$ \theta\_j := \theta\_j - \alpha[(\dfrac{1}{m} \sum\limits\_{i=1}^m(h\_\theta(x^{(i)})-y^{(i)}) \cdot x\_j^{(i)}) + \dfrac{\lambda}{m}\theta\_j] \hspace{2em} j \in { 1, 2, \dots, n }$$
_}_

The term $\dfrac{\lambda}{m}\theta\_j$ performs the regularization. With some manipulation, the update rule can also be represented as:

$$ \theta\_j := \theta\_j(1 - \alpha\dfrac{\lambda}{m}) - \alpha\dfrac{1}{m} \sum\limits\_{i=1}^m(h\_\theta(x^{(i)})-y^{(i)}) \cdot x\_j^{(i)} $$
The term in the above question $1 - \alpha\dfrac{\lambda}{m}$ will always be less than 1. Intuitively it is reducing the value of $\theta\_j$ by some amount every update. The second term remains unchanged.

**Normal Equation**

To approach regularization using the alternate method of the non-iterative normal equation, we add another term inside the parenthesis:

$$\theta = (X^TX + \lambda \cdot L)^{-1}X^Ty $$
$$\text{where } L = \begin{bmatrix} 0 & & & & \newline & 1 & & & \newline & & 1 & & \newline & & & \ddots & \newline & & & & 1 \end{bmatrix} $$

L is a matrix with a 0 at the top left and 1's down the diagonal with 0s everywhere else. It should have dimension $(n+1) \text{ x } (n+1)$. Intuitively this is the identity matrix, excluding $x\_0$, multiplied by a single real number $\lambda$.

Recall that if $m < n$ then $X^TX$ is non invertible. However, when we add the term $\lambda \cdot L$ then $X^TX + \lambda \cdot L$ becomes invertible.

### Regularized Logistic Regression

We can regularize logistic regression in a similar way to how we regularize linear regression. The following image shows how the regularized function (dentoted by the pink line) is less likely to overfit than the non-regularized function represented by the blue line.

![regularized_logistic_regression](/img/coursera-machine-learning-week3/regularized_logistic_regression.png)

Recall that the cost function for logistic regression was:

$$ J(\theta) = - \dfrac{1}{m}\sum\limits\_{i=1}^m[y^{(i)} \log(h\_\theta(x^{(i)})) + (1-y^{(i)}) \log(1-h\_\theta(x^{(i)}))] $$

This can be regularized by adding a regularization term to the end:

$$ J(\theta) = - \dfrac{1}{m}\sum\limits\_{i=1}^m[y^{(i)} \log(h\_\theta(x^{(i)})) + (1-y^{(i)}) \log(1-h\_\theta(x^{(i)}))] + \dfrac{\lambda}{2m} \sum\limits\_{j=1}^n\theta\_j^2 $$

This second sum $\sum\limits\_{j=1}^n\theta\_j^2$ **explicitly excludes $\theta\_0$**, the bias term. The $\theta$ vector is indexed from 0 to n (length of n+1, $\theta\_0$ through to $\theta\_n$). Thus, when computing the equation, we should continuously update the two following equations: 

![gradient_descent_regularized_logistic_regression](/img/coursera-machine-learning-week3/gradient_descent_regularized_logistic_regression.png)


---

Move on to [Week 4]({{% relref "coursera-machine-learning-week4.md" %}}).
