---
title: "Neural Networks and Deep Learning, Week 2"
author: "Alexander Wong"
tags: ["Machine Learning", "deeplearning.ai"]
date: 2017-11-18T15:21:37-06:00
draft: false
---

Taking the [Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning), **Neural Networks and Deep Learning** course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/). See [deeplearning.ai](https://www.deeplearning.ai/) for more details.

Assumes you have knowledge of [Week 1]({{% relref "neural-networks-and-deep-learning-week1.md" %}}).

{{% toc %}}

# Neural Networks Basics

## Logistic Regression as a Neural Network

### Binary Classification

Binary classification is basically answering a yes or no question. For example: Is this an image of a cat? (1: Yes, 0: No).

Let's say you have an image of a cat that is 64 by 64 pixels. You have labeled training data indicating whether or not each image is a cat (`y=1`) or not a cat (`y=0`).

**Notation**

Let's say each picture can be represented as a single vector of size $n\_x$ combined by joining three vectors (64 * 64 red pixel values) + (64 * 64 green pixel values) + (64 * 64 blue pixel values).

$$ n\_x = \text{ unrolled image vector size } = 12288 $$
$$ m \text{ training examples } = \\{ (x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \dots,  (x^{(m)}, y^{(m)}) \\} $$
$$ x \in \mathbb{R}^{n\_x}, y \in \\{ 0, 1 \\} $$
$$ X \in \mathbb{R}^{n\_x \times m} $$
$$ Y \in \mathbb{R}^{1 \times m} $$
$$ X = \begin{bmatrix} \vdots & \vdots & \vdots & \vdots \newline x^{(1)} & x^{(2)} & \dots & x^{(m)} \newline \vdots & \vdots & \vdots & \vdots \end{bmatrix}  $$
$$ Y = \begin{bmatrix} y^{(1)} & y^{(2)} & \dots y^{(m)} \end{bmatrix} $$


### Logistic Regression

Logistic regression is when you want to have an answer in a continuous output. For instance, with the image of a cat problem, rather than having whether or not the image is of a cat or not, one could ask "What is the probability that this is a cat?"

**Notation**

Given $x$, want $\hat{y} = P(y=1 | x)$,
$$ x \in \mathbb{R}^{n\_x} $$
$$ 0 \leq \hat{y} \leq 1 $$
$$ \text{Parameters: } w \in \mathbb{R}^{n\_x}, b \in \mathbb{R} $$
$$ \text{Output: } \hat{y} = \sigma(w^Tx + b) $$
$$ z = w^Tx + b $$
$$ \sigma(z) = \dfrac{1}{1 + e^{-z}}  $$

If $z$ is a large positive number then $\sigma(z) = \dfrac{1}{1 + 0} \approx 1 $.

If $z$ is a large negative number then $\sigma(z) = \dfrac{1}{1 + \inf} \approx 0 $.

### Logistic Regression Cost Function

A loss function is applied to a single training example. For logistic regression, typical loss function used is:

$$ \mathcal{L}(\hat{y}, y) = -(y\log{\hat{y}} + (1-y)\log{(1-\hat{y})}) $$

* If $y = 1$; $ \mathcal{L}(\hat{y}, y) = -\log{\hat{y}} $
  * Want $\log{\hat{y}}$ to be large, we want $\hat{y}$ to be large.
* If $y = 0$; $ \mathcal{L}(\hat{y}, y) = -\log{(1-\hat{y})}$
  * Want $\log{(1-\hat{y})}$ to be large, we want $\hat{y}$ to be small.

A cost function is applied to the entire training set, it evaluates the parameters of your algorithm. (Cost of your parameters).

$$ J(w, b) = \dfrac{1}{m} \sum\limits^{m}\_{i=1} \mathcal{L}(\hat{y}^{(i)}, y^{(i)}) $$
$$ J(w, b) = -[\dfrac{1}{m} \sum\limits^{m}\_{i=1} y^{(i)} \log{\hat{y}^{(i)}} + (1-y^{(i)})\log{(1-\hat{y}^{(i)})}] $$

### Gradient Descent

The cost function measures how well $w, b$ measure the training set. We want to find the $w, b$ that minimize $J(w, b)$.

Repeat {
  $$ w := w - \alpha \dfrac{\partial J(w, b)}{\partial w} $$
  $$ b := b - \alpha \dfrac{\partial J(w, b)}{\partial b} $$
}

Typically, in code, the derivative term is written as `dw`. Example: `w = w - alpha * dw`.

### Derivatives

You don't need a lot of calculus to understand neural networks. This is a basic example of the derivative of a straight line $f(a) = 3a$.

![intuition_about_derivatives](/img/deeplearning-ai/intuition_about_derivatives.png)

### More Derivatives Examples

This is another example of the derivative of $f(a) = a^2$.

![intuition_about_derivatives_2](/img/deeplearning-ai/intuition_about_derivatives_2.png)

Here are three examples: $f(a) = a^2$, $f(a) = a^3$, and $f(a) = \log{a}$.

![intuition_about_derivatives_3](/img/deeplearning-ai/intuition_about_derivatives_3.png)

Take home:
- Derivative just means the slope of the line.
- You want to find slope? Look at calculus textbook.

### Computation Graph

Computation graph is a left to right pass visualization of the math behind your algorithm.

![computation_graph_example](/img/deeplearning-ai/computation_graph_example.png)

### Derivatives with a Computation Graph

Recall calculus, chain rule.

![derivatives_computation_graph](/img/deeplearning-ai/derivatives_computation_graph.png))

### Logistic Regression Gradient Descent

Recall the follwing logistic regression formula defined above.

![logistic_regression_formula_recap](/img/deeplearning-ai/logistic_regression_formula_recap.png)
![logistic_regression_formula_derivatives](/img/deeplearning-ai/logistic_regression_formula_derivatives.png)

### Gradient Descent on m Examples

Recall the cost function:

$$ J(w, b)  = \dfrac{1}{m} \sum\limits^m\_{i=1} \mathcal{L}(a^{(i)},y^{(i)}) $$
$$ a^{(i)} = \hat{y}^{(i)} = \sigma(z^{(i)}) = \sigma(w^Tx^{(i)} + b) $$

This is the naive formula for a single step of logistic regression on $m$ examples with $n = 2$ (two features) using gradient descent.

`begin single step of gradient descent`

$ J = 0; dw\_1 = 0; dw\_2 = 0; db = 0 $
`// define accumulator values`

For $i = 1 \text{ to } m$ do {
  $$ z^{(i)} = w^Tx^{(i)} + b $$
  $$ a^{(i)} = \sigma(z^{(i)}) $$
  $$ J = - [ y^{(i)} \log(a^{(i)}) + (1-y^{(i)})\log(1-a^{(i)}) ] $$
  $$ dz^{(i)} = a^{(i)} - y^{(i)} $$
  $$ dw\_1 = dw\_1 + x\_1^{(i)} \times dz^{(i)} $$
  $$ dw\_2 = dw\_2 + x\_2^{(i)} \times dz^{(i)} $$
  `// if n were greater than two, continue to do this for dw_3, etc`
  $$ db = db + dz^{(i)} $$
}

$ J = \dfrac{J}{m} $;
$ dw\_1 = \dfrac{dw\_1}{m} $;
$ dw\_2 = \dfrac{dw\_2}{m} $;
$ db = \dfrac{db}{m} $;

`end single step of gradient descent`

For each step of gradient descent, you need to do effectively two for loops:

1. for your $m$ number of training examples
2. for your $n$ number of example features.

This is why vectorization is important in deep learning.

## Python and Vectorization

### Vectorization

Vectorization is the art of getting rid of explicit for loops in code.

Example: $ z = w^Tx + b $ where $ w \in \mathbb{R}^{n\_x} $ and $ x \in \mathbb{R}^{n\_x} $ 

```python
// non vectorized
z = 0
for i in range(n-x):
  z += w[i] * x[i]
z += b

//vectorized
import numpy as np
z = np.dot(w,x) + b
```

The following is a vectorization demo.

```python
import numpy as np
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()
print("vectorized version: " + str(1000 * (toc-tic)) + "ms")
# vectorized version: 14.4419670105ms
c = 0
tic = time.time()
for i in range(1000000):
    c += a[i]*b[i]

toc = time.time()
print("non-vectorized version: " + str(1000 * (toc-tic)) + "ms")
# non-vectorized version: 428.48610878ms
```

Vectorization increases performance by allowing the program to take advantage of parallelization. Wherever possible, avoid for loops.

### More Vectorization Examples

Whenever possible, avoid explicit for-loops.

```python
# matrix times a vector, vectorized
A = np.dot(A, v)

# apply exponential operation on every element of a matrix/vector
u = np.zeros((n, 1))
for i in range(n):
    u[i] = math.exp(v[i])
# or vectorized
u = np.exp(v)

np.log(v) # element wise log
np.abs(v) # elementwise abs
np.maximum(v, 0) # ReLU
```

### Vectorizing Logistic Regression

We want to calculate:

for i in range of 1 to m {
  $$ z^{(i)} = w^Tx^{(i)} + b $$
  $$ a^{(i)} = \sigma(z^{(i)}) $$
}

Recall that $X$ is in the shape of $(n\_x, m)$, making it an $\mathbb{R}^{n\_x \times m}$ sized matrix

$$ X = \begin{bmatrix} \vdots & \vdots & \vdots & \vdots \newline x^{(1)} & x^{(2)} & \dots & x^{(m)} \newline \vdots & \vdots & \vdots & \vdots \end{bmatrix}  $$

$$ Z = w^TX+b $$

```python
Z = np.dot(w.T, X) + b
# Z is a row vector of size m
A = sigmoid(Z)
```

### Vectorizing Logistic Regression's Gradient Output

```python
db = 1 / m * (np.sum(dZ))
```

### Broadcasting in Python

```python
import numpy as np

A = np.array([[56.0, 0.0, 4.4, 68.0],
              [1.2, 104.0, 52.0, 8.0],
              [1.8, 135.0, 99.0, 0.9]])
cal = A.sum(axis=0)
print(cal)
# [59.  239.  155.4  76.9]

percentage = 100*A/cal.reshape(1,4)
print(percentage)
#[[ 94.91525424   0.           2.83140283  88.42652796]
# [  2.03389831  43.51464435  33.46203346  10.40312094]
# [  3.05084746  56.48535565  63.70656371   1.17035111]]
```

Python does some magic in broadcasting for matrix/array operations:

![python_broadcasting](/img/deeplearning-ai/python_broadcasting.png)

### Note on Python/NumPy Vectors

Broadcasting may introduce subtle bugs in code, as column/row mismatch no longer is thrown

```
import numpy as np

a = np.random.randn(5) # avoid rank 1 arrays, explicitly define your column vector (5, 1) or row vector (1, 5)
print(a)
# [ 1.2, 2.3, 3.4, 4.5, 5.6 ]
print(a.shape)
# (5,)
print(a.T)
# [ 1.2, 2.3, 3.4, 4.5, 5.6 ]
a = np.random.randn(5, 1)
print(a)
# [[1.2]
#  [2.3]
#  [3.4]
#  [4.5]
#  [5.6]]
```

Occastionally assert your shape when you're not sure `assert(a.shape == (5, 1))`.

---

Move on to [Week 3]({{% relref "neural-networks-and-deep-learning-week3.md" %}}).
