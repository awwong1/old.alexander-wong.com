---
title: "Machine Learning, Week 4"
author: "Alexander Wong"
tags: ["Machine Learning"]
date: 2017-09-12T12:47:44-06:00
draft: false
---

Taking the [Coursera Machine Learning](https://www.coursera.org/learn/machine-learning) course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/).

Assumes you have knowledge of [Week 3]({{% relref "coursera-machine-learning-week3.md" %}}).

{{% toc %}}

* Lecture notes:
  * [Lecture8](/docs/coursera-machine-learning-week4/Lecture8.pdf)

# Neural Networks: Representation

## Motivations

### Non-linear Hypothesis

Neural networks are another learning algorithm that exist in addition to linear regression and logistic regression. They are designed to mimic the way the human brain works.

In a vision problem, it is very difficult to perform logistic regression from pixel data input, as the number of possible features grows exponentially for higher order polynomials. Neural Networks are used to tackle these types of problems when **n** is very large.

### Neurons and the Brain

* Origins: Algorithms that try to mimic the brain. Was very widely used in the 80s and early 90s, popularity diminished in late 90s.
* Recent resurgence: State-of-the-art technique for many applications.

**The "one learning algorithm" hypothesis**

- Neuroscience experiment cut the neurons between an animal's ear and the auditory cortext, and instead attached the neurons for the animal's eye
  - Result was the auditory cortex learned to see
  - Neuro rewiring experiments

Many experiments have been done by connecting sensors to brains and measuring how the brain adapts to use these sensors.

## Neural Networks

### Model Representation I

Neurons are basically computational units that take inputs (**dendrites**) as electrical inputs (called "spikes") that are channeled to outputs (**axons**). In our model, the dendrites are like the input features $x\_1 \dots x\_n$ and the output is the result of the hypothesis function. In this model our $x\_0$ node is sometimes called the _bias unit_. It is always equal to 1. In neural networks, we use the same logistic function as in classification, $\dfrac{1}{1 + e^{-\theta^T}}$. This is sometimes called a sigmoid (logistic) **activation** function.  In this situation, the "theta" parameters are sometimes called "weights".

A visually simplistic representation looks like:
$$ \begin{bmatrix} x\_0 \newline x\_1 \newline x\_2 \end{bmatrix} \rightarrow \begin{bmatrix} & \end{bmatrix} \rightarrow h\_\theta(x) $$

Our input notes (layer 1), also known as the "input layer", go to another node (layer 2), which finally outputs the hypothesis function, known as the "output layer".

We can have intermediate layers of nodes between the input and output layers called "hidden layers".

In this example, we label these intermediate or "hidden" layer nodes $a\_0^2 \dots a\_n^2$ and call them "activation units".

$$ a\_i^{(j)} = \text{ "activation" of unit } i \text{ in layer } j $$
$$ \Theta^{(j)} = \text{ matrix of weights controlling function mapping from layer } j \text{ to layer } j + 1 $$

If one hidden layer exists, it may look like:

$$ \begin{bmatrix} x\_0 \newline x\_1 \newline x\_2 \newline x\_3 \end{bmatrix} \rightarrow \begin{bmatrix} a\_1^{(2)} \newline a\_2^{(2)} \newline a\_3^{(2)} \end{bmatrix} \rightarrow h\_\theta(x) $$

The values for each of the "activation" nodes is obtained by the following:

$$ a\_1^{(2)} = g(\Theta\_{10}^{(1)}x\_0 + \Theta\_{11}^{(1)}x\_1 + \Theta\_{12}^{(1)}x\_2 + \Theta\_{13}^{(1)}x\_3) $$
$$ a\_2^{(2)} = g(\Theta\_{20}^{(1)}x\_0 + \Theta\_{21}^{(1)}x\_1 + \Theta\_{22}^{(1)}x\_2 + \Theta\_{23}^{(1)}x\_3) $$
$$ a\_3^{(2)} = g(\Theta\_{30}^{(1)}x\_0 + \Theta\_{31}^{(1)}x\_1 + \Theta\_{32}^{(1)}x\_2 + \Theta\_{33}^{(1)}x\_3) $$
$$ h\_\Theta(x) = a\_1^{(3)} = g(\Theta\_{10}^{(2)}a\_0^{(2)} + \Theta\_{11}^{(2)}a\_1^{(2)} + \Theta\_{12}^{(2)}a\_2^{(2)} + \Theta\_{13}^{(2)}a\_3^{(2)}) $$

We compute the activation nodes by using a 3 x 4 matrix of parameters. We apply each row of the parameters to the inputs to obtain the value of one activation node. The hypothesis output is the logistic function applied to the sum of the values of the activation nodes, which have been multiplied by another parameter matrix $\Theta^{(2)}$ containing weights for our second layer of nodes.

Each layer gets its own matrix of weights, $\Theta^{(j)}$.

$$ \text{If network has } s\_j \text{ units in layer }j \text{ and } s\_{j+1} \text{ units in layer } j + 1 \text{, then } \Theta^{(j)} \text{ will be of dimension } s\_{j+1} \times (s\_j + 1) \text{.} $$

The $+1$ comes from the addition in $\Theta^{(j)}$ of the "bias nodes", $x\_0$ and $\Theta\_0^{(j)}$. The output nodes will not include the bias nodes while the inputs will. The following images summarizes the model representation:

![neural_network_representation](/img/coursera-machine-learning-week4/neural_network_representation.png)

Example: If layer 1 has 2 input nodes and layer 2 has 4 activation nodes, dimension of $\Theta^{(1)}$ is going to be $4\times3$ where $s\_j = 2$ and $s\_{j + 1} = 4$ so $s\_{j + 1} \times (s\_j + 1) = 4 \times 3$.

### Model Representation II

To reiterate, the following is an example of one layer of a sample neural network:

$$ a\_1^{(2)} = g(\Theta\_{10}^{(1)}x\_0 + \Theta\_{11}^{(1)}x\_1 + \Theta\_{12}^{(1)}x\_2 + \Theta\_{13}^{(1)}x\_3) $$
$$ a\_2^{(2)} = g(\Theta\_{20}^{(1)}x\_0 + \Theta\_{21}^{(1)}x\_1 + \Theta\_{22}^{(1)}x\_2 + \Theta\_{23}^{(1)}x\_3) $$
$$ a\_3^{(2)} = g(\Theta\_{30}^{(1)}x\_0 + \Theta\_{31}^{(1)}x\_1 + \Theta\_{32}^{(1)}x\_2 + \Theta\_{33}^{(1)}x\_3) $$
$$ h\_{\Theta}(x) = a\_1^{(3)} = g(\Theta\_{10}^{(2)}a\_0^{(2)} + \Theta\_{11}^{(2)}a\_1^{(2)} + \Theta\_{12}^{(2)}a\_2^{(2)} + \Theta\_{13}^{(2)}a\_3^{(2)}) $$

In this section, we'll do a vectorized implementation of the above functions. Define a new variable $z\_k^{(j)}$ that encompasses the parameters inside the $g$ function. If we perform the replacement in our above function, we get the following:

$$ a\_1^{(2)} = g(z\_1^{(2)}) $$
$$ a\_2^{(2)} = g(z\_2^{(2)}) $$
$$ a\_3^{(2)} = g(z\_3^{(2)}) $$

In other words, for layer $j = 2$ and node $k$, the variable $z$ will be:

$$ z\_k^{(2)} = \Theta\_{k,0}^{(1)}x\_0 + \Theta\_{k,1}^{(1)}x\_1 + \dots + \Theta\_{k,n}^{(1)}x\_n $$

The vector representation of $x$ and $z^j$ is:

$$ x = \begin{bmatrix} x\_0 \newline x\_1 \newline \vdots \newline x\_n \end{bmatrix} \hspace{1em} z^{(j)} = \begin{bmatrix} z\_1^{(j)} \newline z\_2^{(j)} \newline \vdots \newline z\_n^{(j)} \end{bmatrix} $$

Setting $x = a^{(1)} $, the equation can be rewritten as:

$$ z^{(j)} = \Theta^{(j - 1)}a^{(j - 1)} $$

We're multiplying the matrix $\Theta^{(j-1)}$ with dimensions $s\_j \times (n + 1)$ (where $s\_j$ is the number of our activation nodes) by our vector $a^{(j-1)}$ with height $(n+1)$. This gives us our vector $z^{(j)}$ with height $s\_j$. Now, we can get a vector of our activation nodes for layer j as the following:

$$ a^{(j)} = g(z^{(j)}) $$

The function g can be applied element-wise to the vector $z^{(j)}$.

We can add a bais unit, equal to 1, to layer $j$ after $a^{(j)}$ is computed. This will be element $a\_0^{(j)}$ and will be equal to 1. To compute the final hypothesis, let's compute another z vector:

$$ z^{(j+1)} = \Theta^{(j)}a^{(j)} $$

We obtain this final z vector by multiplying the next theta matrix after $\Theta^{(j-1)}$ with the values of all the activation nodes we just got. This last theta matrix $\Theta^{(j)}$ will have only **one row** which is multiplied by one column $a^{(j)}$ so that our result is a single number. The final result is calculated with:

$$ h\_\Theta(x) = a^{(j+1)} = g(z^{(j+1)}) $$

Note that in the last step, between layer $j$ and layer $j+1$, we are doing the same thing we did in logistic regression. Adding all of the intermediate layers in the neural networks allows us to more elegantly produce interesting and more complex non-linear hypothesis.

## Applications

### Examples and Intuitions I

This is how we would use neural networks to compute simple binary operations, like AND & OR. The graph of our functions will look like:

$$ \begin{bmatrix} x\_0 \newline x\_1 \newline x\_2 \end{bmatrix} \rightarrow \begin{bmatrix} g(x^{(2)}) \end{bmatrix} \rightarrow h\_\Theta(x) $$

Recall that $x\_0$ is our bias variable and is always equal to 1.

Let's set our first theta matrix as:

$$ \Theta^{(1)} = \begin{bmatrix} -30 \hspace{1em} 20 \hspace{1em} 20 \end{bmatrix} $$

This causes the output of our hypothesis to be positive only if both $x\_1$ and $x\_2$ are 1. In other words:

$$ h\_\Theta(x) = g(-30 + 20x\_1 + 20x\_2) $$
$$ x\_1 = 0 \text{ and } x\_2 = 0 \text{ then } g(-30) \approx 0 $$
$$ x\_1 = 0 \text{ and } x\_2 = 1 \text{ then } g(-10) \approx 0 $$
$$ x\_1 = 1 \text{ and } x\_2 = 0 \text{ then } g(-10) \approx 0 $$
$$ x\_1 = 1 \text{ and } x\_2 = 1 \text{ then } g(10) \approx 1 $$

We have constructed one of the fundamental operations in computers by using a small neural network rather than using an actual AND gate. Neural networks can be used to simulate all other logical gates. The following example is for logical OR, meaning either $x\_1$ is true or $x\_2$ is true, or both:

![neural_network_or_example](/img/coursera-machine-learning-week4/neural_network_or_example.png)

Recall that $g(z)$ is defined as the following:

![g_of_z_plot](/img/coursera-machine-learning-week4/g_of_z_plot.png)

### Examples and Intuitions II

The $\Theta^{(1)}$ matrices for AND, NOR, and OR are:

$$ \text{AND: } \Theta^{(1)} = \begin{bmatrix} -30 & 20 & 20 \end{bmatrix} $$
$$ \text{NOR: } \Theta^{(1)} = \begin{bmatrix} -10 & -20 & -20 \end{bmatrix} $$
$$ \text{OR: } \Theta^{(1)} = \begin{bmatrix} -10 & 20 & 20 \end{bmatrix} $$

We can combine these to get the XNOR logical operator (which gives us 1 if $x\_1$ and $x\_2$ are both 0 or both 1).

$$ \begin{bmatrix} x\_0 \newline x\_1 \newline x\_2 \end{bmatrix} \rightarrow \begin{bmatrix} a\_1^{(2)} \newline a\_2^{(2)} \end{bmatrix} \rightarrow \begin{bmatrix} a^{(3)} \end{bmatrix} \rightarrow h\_\Theta(x) $$

For the transition between the first and second layer, we will use a $\Theta^{(1)}$ matrix hat combines the values for AND and NOR:

$$ \Theta^{(1)} = \begin{bmatrix} -30 & 20 & 20 \newline 10 & -20 & -20 \end{bmatrix} $$

For the transition between the second and third layer, we will use $\Theta^{(2)}$ matrix that uses the value for OR:

$$ \Theta^{(2)} = \begin{bmatrix} -10 & 20 & 20 \end{bmatrix} $$

Let's write out the values for all our nodes:

$$ a^{(2)} = g(\Theta^{(1)} \cdot x) $$
$$ a^{(3)} = g(\Theta^{(2)} \cdot a^{(2)}) $$
$$ h\_\Theta(x) = a^{(3)} $$

This is the XNOR operator using a hidden layer with two nodes! The following image summarizes the above algorithm:

![neural_network_xnor_example](/img/coursera-machine-learning-week4/neural_network_xnor_example.png)

### Multiclass Classification

To classify data into multiple classes, the hypothesis should return a vector of values. For example, let's say we wanted to classify our data into one of four categories. We use the following example to see how this classification is done. The algorithm takes an image as input and classifies it accordingly:

![neural_network_multiclass_classification](/img/coursera-machine-learning-week4/neural_network_multiclass_classification.png)

We can define our set of resulting classes as $y$:

$$ y^{(i)} = \begin{bmatrix} 1 \newline 0 \newline 0 \newline 0 \end{bmatrix}, \begin{bmatrix} 0 \newline 1 \newline 0 \newline 0 \end{bmatrix}, \begin{bmatrix} 0 \newline 0 \newline 1 \newline 0 \end{bmatrix}, \begin{bmatrix} 0 \newline 0 \newline 0 \newline 1 \end{bmatrix} $$

Each $y^{(i)}$ represents a different image corresponding to either a pedestrian, car, motorcycle, or truck. The inner layers each provide us with some new information which leads to our final hypothesis function. The setup looks like:

$$ \begin{bmatrix} x\_0 \newline x\_1 \newline x\_2 \newline \vdots \newline x\_n \end{bmatrix} \rightarrow \begin{bmatrix} a\_0^{(2)} \newline a\_1^{(2)} \newline a\_2^{(2)} \newline \vdots \newline a\_n^{(2)} \end{bmatrix} \rightarrow \begin{bmatrix} a\_0^{(3)} \newline a\_1^{(3)} \newline a\_2^{(3)} \newline \vdots \newline a\_n^{(3)} \end{bmatrix} \rightarrow \dots \rightarrow \begin{bmatrix} h\_\Theta(x)\_1 \newline h\_\Theta(x)\_2 \newline h\_\Theta(x)\_3 \newline h\_\Theta(x)\_4 \end{bmatrix} $$

Our resulting hypothesis for one set of inputs may look like:

$$ h\_\Theta(x) = \begin{bmatrix} 0 \newline 0 \newline 1 \newline 0 \end{bmatrix} $$

In this case, our resulting class is $h\_\Theta(x)\_3 $, which represents the motorcycle.

---
 Week 5 tbd.
