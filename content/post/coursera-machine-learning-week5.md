---
title: "Machine Learning, Week 5"
author: "Alexander Wong"
tags: ["Machine Learning"]
date: 2017-09-18T15:38:02-06:00
draft: false
---

Taking the [Coursera Machine Learning](https://www.coursera.org/learn/machine-learning) course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/).

Assumes you have knowledge of [Week 4]({{% relref "coursera-machine-learning-week4.md" %}}).

{{% toc %}}

* Lecture notes:
  * [Lecture9](/docs/coursera-machine-learning-week5/Lecture9.pdf)

# Neural Networks: Learning

## Cost Function and Backpropagation

### Cost Function

Let's define a few variables that we will need to use.

* $L$ = the total number of layers in the network
* $s\_l$ = number of units (not counting bias unit) in layer $l$
* $K$ = number of output units or classes

Recall that in neural networks, we may have many output nodes. We denote $h\_\Theta(x)\_k$ as being a hypothesis that results in the $k^{\text{th}}$ output. Our cost function for neural networks is going to be a generalization of the logistic regression cost function. Recall the regularized logistic regression cost function:

$$ J(\theta) = -\dfrac{1}{m} \sum\limits\_{i=1}^{m} [y^{(i)} \log(h\_\theta(x^{(i)})) + (1-y^{(i)}) \log(1 - h\_\theta(x^{(i)})) ] + \dfrac{\lambda}{2m} \sum\limits\_{j=1}^{n} \theta\_j^2 $$

For neural networks, our cost function is the following:

$$ J(\Theta) = -\dfrac{1}{m} \sum\limits\_{i=1}^{m} \sum\limits\_{k=1}^{K} [y\_k^{(i)} \log((h\_\{\Theta}(x^{(i)}))\_k) + (1 - y\_k^{(i)}) \log(1-(h\_\Theta(x^{(i)}))\_k)] + \dfrac{\lambda}{2m} \sum\limits\_{l=1}^{L-1} \sum\limits\_{i=1}^{s\_l} \sum\limits\_{j=1}^{s\_{l+1}} (\Theta\_{j,i}^{(l)})^2 $$

Multiple output nodes are accounted for by the nested summations. In the first part of the equation, before the square brackets, an additional nested summation loops through the number of output nodes. In the regularization term after the square brackets, we account for multiple theta matrices. The number of columns in the current theta matrix is equal to the number of nodes in the current layer (including the bias unit). The number of rows in the current theta matrix is equal to the number of nodes in the next layer (excluding the bias unit). As before with logistic regression, we square every term.

Note:

* The double sum simply adds up the logistic regression costs calculated for each cell in the output layer.
* The triple sum simply adds up the squares of all the individual $\Theta$s in the entire network.
* The $i$ in the triple sum does not refer to training example $i$

### Backpropagation Algorithm

"Backpropagation" is neural network terminology for minimizing our cost function. This is similar to what we did with gradient descent in logistic and linear regression. Our goal is to compute:

$$ \min\_\Theta J(\Theta) $$

That is, we want to minimize our cost function $J$ using an optimal set of parameters in theta. In this section, we will look at the equations we used to compute the partial derivative of $J(\Theta)$:

$$ \dfrac{\delta}{\delta\Theta\_{i,j}^{(l)}} J(\Theta) $$

To do so, we use the following algorithm:

![neural_network_backpropagation](/img/coursera-machine-learning-week5/neural_network_backpropagation.png)

Given training set $\{ (x^{(1)}, y^{(1)}), \dots, (x^{(m)}, y^{(m)}) \}$, set $ \Delta\_{i,j}^{(l)} := 0 $ for all $ (l, i, j) $. This gives us a matrix full of zeros.

For training example $t = 1$ to $m$:

1. Set $ a^{(l)} := x^{(t)} $
2. Perform forward propagation to compute $ a^{(l)} $ for $ l = 2, 3, \dots, L $
    ![neural_network_gradient_computation](/img/coursera-machine-learning-week5/neural_network_gradient_computation.png)
3. Using $y^{(t)}$, compute $ \delta^{(L)} = a^{(L)} = y^{(t)} $

    Where $L$ is our total number of layers and $a^{(L)}$ is the vector of outputs of the activation units for the last layer. Our "error values" for the last layer are simply the differences of our actual results in the last layer and the correct outputs in y. To get the delta values of the layers before the last layer, we can use an equation that steps us back from right to left:

4. Compute $ \delta^{(L-1)}, \delta^{(L-2)}, \dots, \delta^{(2)} $ using $ \delta^{(l)} = ((\Theta^{(l)})^T \delta^{(l+1)}) .* a^{(l)} .* (1 - a^{(l)}) $

    The delta values of layer $l$ are calculated by multiplying the delta values in the next layer with the theta matrix of layer $l$. We then element-wise multiply that with a function called $g'$ (g-prime), which is the derivative of the activation function $g$ evaluated with the input values given by $z^{(l)}$.

    The g-prime derivative terms can also be written out as:

    $$ g'(z^{(l)}) = a^{(l)} .* (1-a^{(l)}) $$

5. $ \Delta\_{i,j}^{(l)} := \Delta\_{i,j}^{(l)} + a\_j^{(l)} \delta\_i^{(l+1)} $ or, the vectorized approach, $ \Delta^{(l)} := \Delta^{(l)} + \delta^{(l+1)}(a^{(l)})^T $

    We update our new $\Delta$ matrix.

    * $ D\_{i,j}^{(l)} := \dfrac{1}{m}(\Delta\_{i,j}^{(l)} + \lambda\Theta\_{i,j}^{(l)} ) \hspace{2em} \text{ if } j \neq 0 $
    * $ D\_{i,j}^{(l)} := \dfrac{1}{m} \Delta\_{i,j}^{(l)} \hspace{2em} \text{ if } j = 0 $

The capital-delta matrix $D$ is used as an "accumulator" to add up our values as we go along and eventual compute our partial derivative. Thus, we get $ \dfrac{\delta}{\delta\Theta\_{i,j}^{(l)}} J(\Theta) = D\_{i,j}^{(l)} $

### Backpropagation Intuition

Recall the cost function for a neural network:

$$ J(\Theta) = - \dfrac{1}{m} \sum\limits\_{t=1}^{m} \sum\limits\_{k=1}^{K} [y\_k^{(t)} \log(h\_\Theta (x^{(t)}))\_k + (1-y\_k^{(t)}) \log(1-h\_\Theta(x^{(t)})\_k) ] + \dfrac{\lambda}{2m} \sum\limits\_{l=1}^{L-1} \sum\limits\_{i=1}^{s\_l} \sum\limits\_{j=1}^{s\_l+1} (\Theta\_{j,i}^{(l)})^2 $$

Considering the simple non-multiclass classification $ (k = 1) $ and disregarding regularization, the cost is computed with:

$$ \text{cost}(t) = y^{(t)} \log(h\_\Theta(x^{(t)})) + (1-y^{(t)}) \log(1-h\_\Theta(x^{(t)})) $$

Intuitively, $\delta\_j^{(l)}$ is the 'error' for $a\_j^{(l)}$ (unit j in layer l). More formally, the delta values are the derivative of the cost function:

$$ \delta\_j^{(l)} = \dfrac{\partial}{\partial z\_j^{(l)}} \text{cost}(t) $$

Recall the derivative is the slope of a line tangent to the cost function. The steeper the slope, the more incorrect we are. Consider the following neural network below and see how we could calculate $ \delta\_j^{(l)}$:

![neural_network_forward_and_back_propagation](/img/coursera-machine-learning-week5/neural_network_forward_and_back_propagation.png)

In the image above, to calculate $\delta\_2^{(2)}$ we multiply the weights $\Theta\_{12}^{(2)}$ and $\Theta\_{22}^{(2)}$ by their respective $\delta$ values found to the right of each edge. This gives us $ \delta\_2^{(2)} = \Theta\_{12}^{(2)} * \delta\_1^{(3)} + \Theta\_{22}^{(2)} * \delta\_2^{(3)} $. To calculate every single possible $ \delta\_j^{(l)} $ we need to start from the right of our diagram. We can think of the edges as our $ \Theta_{ij} $.Going from right to left, to calculate the value of $ \delta\_j^{(l)} $ we can take over all sums of each weight times the $\delta$ it is coming from. Another example here is $ \delta\_2^{(3)} = \Theta\_{12}^{(3)}*\delta\_1^{(4)} $.

## Backpropagation in Practice

### Implementation Note: Unrolling Parameters

With neural networks, we are utilizing sets of matrices:

$$ \Theta^{(1)}, \Theta^{(2)}, \Theta^{(3)}, \dots $$
$$ D^{(1)}, D^{(2)}, D^{(3)}, \dots $$

In order to use optimizing functions such as `fminunc()`, we want to unroll all elements and put them into one long vector.

```octave
thetaVector = [ Theta1(:); Theta2(:); Theta3(:); ];
deltaVector = [ D1(:); D2(:); D3(:); ];
```

If the dimensions of Theta1 is 10 by 11, Theta2 is 10 by 11, and Theta3 is 1 by 11, we can get back our original matrices from the unrolled versions as follows:

```octave
Theta1 = reshape(thetaVector(1:110), 10, 11);
Theta2 = reshape(thetaVector(111:220), 10, 11);
Theta3 = reshape(thetaVector(221:231), 1, 11);
```

To summarize:

![unrolling_parameters](/img/coursera-machine-learning-week5/unrolling_parameters.png)

### Gradient Checking

Gradient checking will assure that backpropagation works as intended. We approximate the derivative of our cost function with:

$$ \dfrac{\partial}{\partial\Theta} J(\Theta) \approx \dfrac{ J(\Theta + \epsilon) - J(\Theta - \epsilon) }{2\epsilon} $$

With multiple theta matrices, we can approximate the derivative **with respect to $\Theta\_j$** as follows:

$$ \dfrac{\partial}{\partial\Theta\_j} J(\Theta) \approx \dfrac{ J(\Theta\_1, \dots, \Theta\_j + \epsilon, \dots, \Theta\_n) - J(\Theta\_1, \dots, \Theta\_j - \epsilon, \dots, \Theta\_n) }{2\epsilon} $$

A small value for $\epsilon$ (epsilon) such as $\epsilon = 10^{-4}$ guarantees that the math works out properly. If the value for $\epsilon$ is too small or too large, we can end up with numerical problems.

Hence, we are only adding or subtracting epsilon to the $\Theta\_j$ matrix. In octave, we can do that as follows:

```octave
epsilon = 1e-4;
for i = 1:n,
    thetaPlus = theta;
    thetaPlus(i) += epsilon;
    thetaMinus = theta;
    thetaMinus(i) -= epsilon;
    gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon)
end
```

We previously saw how to calculate the delta vector. Once we compute our gradApprox vector, we can check that $ \text{gradApprox} \approx \text{deltaVector} $. Once backpropagation is verified to be correct, you do not need to compute gradApprox again. Code to compute gradApprox can be very slow.

### Random Initialization

Initializing all theta weights to zero does not work with neural networks. When backpropagation occurs, all nodes will update to the same value repeatedly. Instead, we should randomly initialize our weights for our $\Theta$ matrices using the following method:

![symmetry_breaking](/img/coursera-machine-learning-week5/symmetry_breaking.png)

We initialize each $\Theta\_{ij}^{(l)}$ to a random value between $[-\epsilon,\epsilon]$. Using the above formula guarantees that we will get the desired bound. The same procedure applies to all of the $\Theta$'s. Below is some working code for experimentation:

```octave
INIT_EPSILON = 1e-2;
% If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11.

Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
```

The `rand(x,y)` function will initialize a matrix of random real numbers between 0 and 1.
The `INIT_EPSILON` value is unrelated to the epsilon from Gradient Checking.

### Putting it Together

First, pick a network architecture. Choose the layout of your neural network, including how many hidden units in each layer and how many layers in total you want to have.

* Number of input units = dimension of features $x^{(i)}$
* Number of output units = number of classes
* Number of hidden units per layer = usually more is better (but computation increases)
* Defaults: 1 hidden layer. If more than 1 hidden layer, it is recommended that you have the same number of units in every hidden layer.

**Training a Neural Network**

1. Randomly initialize the weights
2. Implement forward propagation to get $h\_Theta(x^{(i)})$ for any $x^{(i)}$
3. Implement the cost function
4. Implement backpropagation to compute partial derivatives
5. Use gradient checking to confirm backpropagation works. Disable gradient checking afterwards
6. Use gradient descent or a built-in optimization function to minimize the cost function with the weights in theta.

When we perform forward and back propagation, we loop on every training example:

```octave
for i = 1:m,
   % Perform forward propagation and backpropagation using example (x(i),y(i))
   % (Get activations a(l) and delta terms d(l) for l = 2,...,L)
```

The following image gives us an intuition of what is happening as we implement the neural network:

![neural_network_gradient_descent](/img/coursera-machine-learning-week5/neural_network_gradient_descent.png)

Ideally, you want $ h\_\Theta(x^{(i)}) \approx y^{(i)} $. This will minimize our cost function. However, keep in mind that $J(\Theta)$ is not convex and we could end up in a local minimum instead.


## Application of Neural Networks

### Autonomous Driving

See Dean Pomerleau (Carnagie Mellon) autonomous driving neural network. (This was done already in 1992).

---

Week 6 tbd
