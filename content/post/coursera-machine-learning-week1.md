---
title: "Machine Learning, Week 1"
date: 2017-08-31T10:25:51-06:00
tags: ["Machine Learning"]
draft: false
---

Taking the [Coursera Machine Learning](https://www.coursera.org/learn/machine-learning) course. Will post condensed notes every week as part of the review process. All material originates from the free coursera course, taught by [Andrew Ng](http://www.andrewng.org/).

{{% toc %}}

* Lecture notes:
  * [Lecture1](/docs/coursera-machine-learning-week1/Lecture1.pdf)
  * [Lecture2](/docs/coursera-machine-learning-week1/Lecture2.pdf)

# Introduction

## What is Machine Learning

* **Arthur Samuel (1959)**: The field of study that gives computers the ability to
learn without explicitly programmed.
* **Tom Mitchell (1998)**: Well-posed Learning Problem; A computer program is said to _learn_ from experience **E** with respect to some task **T** and some performance measure **P** if its performance on **T**, as measured by **P**, improves with experience **E**.

Example: playing checkers.

* **E** = The experience of playing many games of checkers.
* **T** = The task of playing checkers.
* **P** = The probability that the program will win the next game.

In general, any machine learning problem can be assigned to one of two broad classifications, Supervised learning and Unsupervised learning.

## Supervised Learning

In supervised learning, we have a data set and we already know what the correct output should look like. There is an idea that a relationship exists between the input and output.

Supervised learning is categorized into **regression** and **classification** problems.

* **Regression**: Results are within a continuous output. We are trying to map input variables to some continuous function.
* **Classification**: Results are discrete. We are trying to map input variables into separate categories.

Example 1:

Given data about the sizes of houses on the real estate market, attempt to predict price. Price as a function of size is a continuous output, so this is a _regression_ problem.

Example 2:

Given data about a patient with a tumor, predict whether or not the tumor is malignant or benign. The function does not produce a continuous output, only two categories are given, therefore this is a _classification_ problem.

Examples:

* Given email labeled as spam/not spam, learn a spam filter
* Given a dataset of patients diagnosed as either having diabetes or not, learn to classify new patients as having diabetes or not.

## Unsupervised Learning

Unsupervised learning allows appraoches to problems with little or no idea what the results should look like. Structure is derived from data where we do not know the effect of the variables. This can be done by clustering the data based on relationships or variables within the data.

With unsupervised learning, there is no feedback based on the preduction results.

Examples:

* **Clustering**: Take a collection fo 1,000,000 differenge genes and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, etc.
* **Non-Clustering**: The "Cocktail Party Algorithm" allows you to find structure in a chaotic environment (such as identifying individual voices and music from a mesh of sounds at a cocktail party).

* Given a set of news articles found on the web, group them into sets of articles about the same stories.
* Given a database of customer data, automatically discover market segments and group customers into different market segments.

# Linear Regression with One Variable

## Model Representation

This is the notation we will use moving forward.

* Input variables (features) are denoted as $x^{(i)}$.
* Output variables are denoted as $y^{(i)}$.
* A pair $(x^{(i)}, y^{(i)})$ is called a training example.
* The dataset we'll be using to learn is a list of training examples is called a training set and is denoted as $(x^{(i)}, y^{(i)}); i = 1, ..., m$
* The value $X$ is used to denote the space of input values and $Y$ is used to denote the space of output values.
  * $X = Y = \mathbb{R}$
* Note: The _(i)_ is not exponentiation, but identifying.

Given a training set, learn a function $h:X \rightarrow Y$ such that $h(x)$ is a _good_ predictor for the corresponding value of $y$. For historical reasons, the function $h$ is called a hypothesis.

![hypothesis](/img/coursera-machine-learning-week1/hypothesis.png)

## Cost Function & Intuitions

We measure the accuracy of a hypothesis functions by using a **cost function**. This takes an average difference of all the results of the hypothesis with inputs from X and the outputs Y.

The cost function we will be using for now is the **Squared Error Function**, also known as **Mean Squared Error**.

$$ J(\theta\_{0}, \theta\_{1}) = \dfrac{1}{2m} \sum\_{i=1}^m (\hat{y}\_{i} - y\_{i})^{2} = \dfrac{1}{2m} \sum\_{i=1}^m (h\_{\theta}(x\_{i}) - y\_{i})^{2}  $$

Thinking about this in visual terms, training data set is scattered on the x,y plane. We are trying to make a straight line pass through these scattered points. We want the best possible line such that the average squared vertical distances of the scattered points from the line will be the least.

![cost_function_1](/img/coursera-machine-learning-week1/cost_function_1.png)
![cost_function_2](/img/coursera-machine-learning-week1/cost_function_2.png)
![cost_function_3](/img/coursera-machine-learning-week1/cost_function_3.png)
![cost_function_4](/img/coursera-machine-learning-week1/cost_function_4.png)

## Gradient Descent

Gradient descent is a method of estimating the parameters in the hypothesis function using the cost function. Imagine that we graph the hypothesis function based on its fields $\theta\_0, \theta\_1$. We put these variables on the x and y axis and we plot the cost function on the vertical z axis. The points on the graph will be the result of the cost function using the hypothesis with those specific theta parameters.

![gradient_descent_1](/img/coursera-machine-learning-week1/gradient_descent_1.png)

We need to minimize our cost function by "stepping" down from the top to the bottom points of this graph. The red arrows show local minimums in the graph.

This is done by taking the derivative (the tangential line to a function) of our cost function. The slope of the tangent is the derivative at that point and it will give us a direction to move towards. We make steps down the cost function in the direction with the steepest descent. The size of each step is determined by the parameter (alpha), which is the learning rate.

The gradient descent algorithm is:

_repeat until convergence:_
$$\theta\_j := \theta\_j - \alpha \dfrac{\partial}{\partial\theta\_j} J(\theta\_0,\theta\_1)$$
_where:_ $j = 0,1$ represents the feature index number.

At each iteration j, one should simultaneously update the parameters $\theta\_1, \theta\_2, \dots, \theta\_n$. Updating a specific parameter prior to calculating another one on the $j^{(th)}$ iteration leads to a wrong implementation.

![gradient_descent_2](/img/coursera-machine-learning-week1/gradient_descent_2.png)

Regardless of the slope's sign for the derivative, $\theta\_1$ eventually converges to its minimum value. The following figure shows that when the slope is negative, the value of $\theta\_1$ increases. When the slope is positive, the value of $\theta\_1$ decreases.

![gradient_descent_3](/img/coursera-machine-learning-week1/gradient_descent_3.png)

We must adjust the parameter alpha to ensure that the gradient descent algorithm converges in reasonable time. Failure to converge or too much time to obtain the minimum value implies that the step size is wrong.

![gradient_descent_4](/img/coursera-machine-learning-week1/gradient_descent_4.png)

Even with a fixed step size, gradient descent can converge. The reason is because as we approach the bottom of the convex function, the derivative approaches zero.

![gradient_descent_5](/img/coursera-machine-learning-week1/gradient_descent_5.png)

## Gradient Descent for Linear Regression

Applying the gradient descent algorithm to the cost functions defined earlier, we must calculate the necessary derivatives.

![gradient_descent_linear_regression_1](/img/coursera-machine-learning-week1/gradient_descent_linear_regression_1.png)

This gives us the new gradient descent algorithm:

_repeat until convergence:_
$$\theta\_0 := \theta\_0 - \alpha \frac{1}{m} \sum\limits\_{i=1}^{m}(h\_\theta(x\_{i}) - y\_{i})$$
$$\theta\_1 := \theta\_1 - \alpha \frac{1}{m} \sum\limits\_{i=1}^{m}((h\_\theta(x\_{i}) - y\_{i}) x\_{i}$$

If we start with a guess for our hypothesis function and we repeatedly apply the gradient descent equations, our hypothesis will become more and more accurate.

This is simply gradient descent on the original cost function J. This method looks at every example in the entire training set at every step, therefore this is called **batch gradient descent**. Note: while gradient descent can be susceptible to local minima in general, the optimization problem we have posed here for linear regression has only one global, and no other local, optima. Thus, gradient descent here always converges (assuming alpha isn't too large) to the global minimum. J is a convex quadratic function.

![gradient_descent_linear_regression_2](/img/coursera-machine-learning-week1/gradient_descent_linear_regression_2.png)

The ellipses shown above are the contours of a quadratic function. Also shown is the trajectory taken by gradient descent, which was initialized at (48, 30). The x's in the figure represent each step in gradient descent as it converged to its minimum.
