---
title: "Machine Learning, Week 6"
author: "Alexander Wong"
tags: ["Machine Learning"]
date: 2017-09-20T15:38:02-06:00
draft: false
---

Taking the [Coursera Machine Learning](https://www.coursera.org/learn/machine-learning) course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/).

Assumes you have knowledge of [Week 5]({{% relref "coursera-machine-learning-week5.md" %}}).

{{% toc %}}

* Lecture notes:
  * [Lecture10](/docs/coursera-machine-learning-week6/Lecture10.pdf)
  * [Lecture11](/docs/coursera-machine-learning-week6/Lecture11.pdf)

# Advice for Applying Machine Learning

## Evaluating a Learning Algorithm

### Evaluating a Hypothesis

Once we have done some trouble shooting for errors in our predictions by:

* Getting more training examples
* Trying out smaller sets of features
* Trying additional features
* Trying polynomial features
* Increasing or decreasing $\lambda$

We can move on to evaluating our new hypothesis.

A hypothesis may have a low error for the training data but still be inaccurate, due to overfitting. One way to evaluate a hypothesis, given a dataset of training examples, is to split the data up into two sets: a **training set** and a **test set**.
Typically, the training set consists of 70% of the data and the test set is the remaining 30%. The data should be randomized so the 70% and 30% do not reflect any sort of ordering.

The new procedure using the two sets is:

1. Learn $\Theta$ and minimize $J\_\text{train}(\Theta)$ using the training set
2. Compute the test set error $J\_\text{test}(\Theta)$.

The test set error for linear regression is:

$$ J\_\text{test}(\Theta) = \dfrac{1}{2m\_\text{test}} \sum\limits\_{i=1}^{m\_\text{test}} (h\_\Theta(x\_\text{test}^{(i)}) - y\_\text{test}^{(i)})^2 $$

For classification, we use the following to determine misclassification error:

$$ \text{err}(h\_\Theta(x),y) = \begin{matrix} 1 & \mbox{if } h\_\Theta(x) \geq 0.5\ \text{and}\ y = 0\ \text{or}\ h\_\Theta(x) < 0.5\ \text{and}\ y = 1 \newline 0 & \mbox{otherwise} \end{matrix} $$

This gives us a binary 0 or 1 error result based on a misclassification. The average test error for the test set is:

$$ \text{Test Error} = \dfrac{1}{m\_\text{test}} \sum\limits\_{i=1}^{m\_\text{test}} \text{err}(h\_\Theta(x\_\text{test}^{(i)}), y\_\text{test}^{(i)}) $$

That tells us the proportion of the test data that was misclassified.

### Model Selection and Train/Validation/Test Sets

Simply because a learning algorithm fits the training set well, does not mean that it is a good hypothesis. It could over fit and as a result your predictions on the test set would be poor. The error of your hypothesis as measured on the data set with which you trained the parameters will be lower than the error on any other data set/

Given many models with different polynomial degrees, we can use a systematic approach to identify the 'best' function. In order to choose the model of your hypothesis, you can test each degree of polynomial and look at the error result.

One way to partition our dataset into three sets is:

* Training Set : 60%
* Cross Validation Set: 20%
* Test Set: 20%

We can now calculate three separate error values for the three sets using the following method:

1. Optimize the parameters in $\Theta$ using the training set for each polynomial degree.
2. Find the polynomial degree _d_ with the least error using the cross validation set.
3. Estimate the generalization error using the test set with $ J\_\text{test}(\Theta^{(d)}) $, where d is the theta from the polynomial with the lowest error.

This way, the degree of the polynomial d has not been trained using the test set.

### Diagnosing Bias versus Variance

In this section, we examine the relationship between the degree of the polynomial _d_ and the underfitting or the overfitting of our hypothesis.

* We need to distinguish whether **bias** or **variance** is the problem contributing to bad predictions.
* High bias is underfitting and high variance is overfitting. We should find a value that minimizes both.

The training error tends to **decrease** as we increase the degree _d_ of the polynomial.

At the same time, the cross validation error will tend to **decrease** as we increase _d_ up to a certain point, then it will **increase** as _d_ increased, forming a convex curve.

**High bias (underfitting)** Both $J\_\text{train}(\Theta)$ and $J\_\text{CV}(\Theta)$ will be high. Also, $J\_\text{CV}(\Theta) \approx J\_\text{train}(\Theta)$.

**High variance (overfitting)**: $J\_\text{train}(\Theta)$ will be low and $J\_\text{CV}(\Theta)$ will be much greater than $J\_\text{train}(\Theta)$.

This is the summarized figure:

![underfitting_overfitting_costs](/img/coursera-machine-learning-week6/underfitting_overfitting_costs.png)

### Regularization and Bias/Variance

Given the following equations, we get the following plots:

$$ h\_\theta(x) = \theta\_0 + \theta\_1x + \theta\_2x^2 + \theta\_3x^3 + \theta\_4x^4 $$
$$ J(\theta) = \dfrac{1}{2m} \sum\limits\_{i=1}^{m} (h\_\theta(x^{(i)}) - y^{(i)})^2 + \dfrac{\lambda}{2m} \sum\limits\_{j=1}^{n} \theta\_j^2 $$
![regularization_in_bias_and_variance](/img/coursera-machine-learning-week6/regularization_in_bias_and_variance.png)

In the figure above, as $\lambda$ increases, our fit becomes more rigid. On the otherhand, as $\lambda$ approaches 0, we tend to overfit the data. How do we choose our parameter $\lambda$ to get it 'just right'? In order to choose the model and the regularization term $\lambda$, we need to:

1. Create a list of lambdas, for example increment by powers of 2 from 0.01:
    $$ \lambda \in \\{0, 0.01, 0.02, 0.04, 0.08, \dots, 10.24 \\} $$
2. Create a set of models with different degrees or any other variants.
3. Iterate through the $\lambda$s and for each $\lambda$ go through all the models to learn some $\Theta$.
4. Compute the cross validation error using the learned $\Theta$ (computed with $\lambda$) on the $J\_\text{CV}(\Theta)$ **without** regularization.
5. Select the best combo that produces the lowest error on the cross validation set.
6. Using the best combo $\Theta$ and $\lambda$, apply it on $J\_\text{test}(\Theta)$ to see if it has a good generalization of the problem.

### Learning Curves

Training an algorithm on a very few number of data points will easily have 0 errors because we can always find a quadratic curve that touches exactly those number of points. Therefore:

* As the training set gets larger, the error for a quadratic function increases.
* The error value will plateau out after a certain m, or training set size.

**Experiencing High Bias:**

* Small training set size causes $J\_\text{train}(\Theta)$ to be low and $J\_\text{CV}(\Theta)$ to be high.
* Large training set size causes $J\_\text{train}(\Theta)$ and $J\_\text{CV}(\Theta)$ to be high with $J\_\text{train}(\Theta) \approx J\_\text{CV}(\Theta)$.

If a learning algorithm is suffering from high bias, getting more training data will not (by itself) help much.

![high_bias](/img/coursera-machine-learning-week6/high_bias.png)

**Experiencing High Variance:**

* Small training set size will have $J\_\text{train}(\Theta)$ be low and $J\_\text{CV}(\Theta)$ be high.
* Large training set size will have $J\_\text{train}(\Theta)$ increase with training set size and $J\_\text{CV}(\Theta)$ continue to decrease without leveling off. $J\_\text{train}(\Theta) \lt J\_\text{CV}(\Theta)$ but the difference between the two remains significant.

If a learning algorithm is suffering from high variance, getting more training data is likely to help.

![high_variance](/img/coursera-machine-learning-week6/high_variance.png)

### Deciding What to Do Next

Our decision process can be broken down as follows:

| Action | Fix |
|---|---|
| Getting more training examples | Fixes high variance |
| Trying smaller sets of features | Fixes high variance |
| Adding features | Fixes high bias |
| Adding polynomial features | Fixes high bias |
| Decreasing $\lambda$ | Fixes high bias |
| Increasing $\lambda$ | Fixes high variance |

**Diagnosing Neural Networks**

* A neural network with fewer parameters is **prone to underfitting**. It is also computationally cheaper.
* A large neural network with more parameters is **prone to overfitting**. It is also computationally expensive. In this case, you can use regularization (increase $\lambda$) to address the overfitting.

Using a single hidden layer is a good starting default. You can train the neural network on a number of hidden layers using your cross validation set. You can then select the one that performs best.

**Model Complexity Effects:**

* Lower-order polynomials (lower model complexity) have high bias and low variance. In this case, the model fits poorly consistently.
* Higher-order polynomials (high model complexity) fit the training data extremely well and the test data extremely poorly. These have low bias but high variance.
* In reality, we want to choose a model in between that generalizes well and also fits the data reasonably well.

# Machine Learning System Design

## Building a Spam Classifier

### Prioritizing What to Work On

**System Design Example:**

Given a data set of emails, one could construct a vector for each email. An entry in this vector represents a word. The vector contains 10,000-50,000 entries gathered by finding the most frequently used words in the dataset. If a word is found in the email, we assign its entry as a 1, otherwise the entry would be 0. Once we have our X vectors ready, we train the algorithm and use it to classify if an email is spam or not.

![spam_classifier](/img/coursera-machine-learning-week6/spam_classifier.png)

There are many ways to improve the accuracy of the this classifier.

* Collect lots of data (ex: honeypot project, but this doesn't always work)
* Develop more sophisticated features (ex: using email header data in spam emails)
* Develop algorithms to process input in different ways (recognize mispellings in spam)

It is difficult to tell which of the options will be most helpful.

### Error Analysis

The recommended approach to solving a machine learning problem is to:

* Start with a simple algorithm that you can implement quickly and test it early on your cross validation data.
* Plot learning curves to decide if more data, more features, etc. are likely to help.
* Manually examine the examples (in cross validation set) that your algorithm made errors on. See if you can spot systematic trends in what type of examples it is making errors on.

For example, assume that we have 500 emails and the algorithm misclassifies 100 of them. Manually analyze the 100 emails and categorize them based on what type of emails they are. Then one could try to come up with new cues and features that woudl help classify these 100 emails correctly. For example, if most of the misclassified emails are those which try to steal passwords, we could find some features that are particular to those emails and add them to our model. We can also see how classifying each word according to its root changes our error rate.

![numerical_evaluation](/img/coursera-machine-learning-week6/numerical_evaluation.png)

It is important to get error results as a single, numerical value. Otherwise, it is difficult to assess the algorithm's performance.

## Machine Learning Practical Tips

### How to Handle Skewed Data

**Precision/Recall:**

The following metric is useful for datasets with very skewed data:

![precision_recall](/img/coursera-machine-learning-week6/precision_recall.png)

Trade off precision and recall depending on the use case of your classifier. You can compare various precision/recall numbers using a value called an $\text{F}\_1$ score. 

![f_score](/img/coursera-machine-learning-week6/f_score.png)

### When to Utilize Large Data Sets

Large data rationale; assume features $x \in \mathbb{R}^{n+1}$ has sufficient information to predict $y$ accurately.

* Useful test: Given the input $x$, can a human expert confidently predict $y$?

Use a learning algorithm with many parameters (eg logistic/linear regression with many features; neural network with many hidden units). Low bias algorithms

$$ J\_{\text{train}}(\Theta) \text{will be small.} $$

Use a very large training set (unlikely to overfit).

$$ J\_{\text{train}}(\Theta) \approx J\_{\text{test}}(\Theta) $$

