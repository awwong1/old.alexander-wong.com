---
title: "Machine Learning, Week 9"
author: "Alexander Wong"
tags: ["Machine Learning"]
date: 2017-10-22T14:21:37-06:00
draft: false
---

Taking the [Coursera Machine Learning](https://www.coursera.org/learn/machine-learning) course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/).

Assumes you have knowledge of [Week 8]({{% relref "coursera-machine-learning-week8.md" %}}).

{{% toc %}}

* Lecture notes:
  * [Lecture15](/docs/coursera-machine-learning-week9/Lecture15.pdf)
  * [Lecture16](/docs/coursera-machine-learning-week9/Lecture16.pdf)

# Anomoly Detection

## Density Estimation

### Problem Motivation

Imagine being a manufacturor of aircraft engines. Measure heat generated, vibration intensity, etc. You know have a dataset of these features which you can plot. Anomoly detection would be determining if a new aircraft engine $x\_{\text{test}}$ is anomolous in relation to the previously measured engine features.

Given a dataset $x^{(1)}, x^{(2)}, \dots, x^{(m)}$, is $x\_{\text{test}}$ anomalous? Check if $p(x) \gt \epsilon$ for given test set.

Some useful cases include fraud detection, manufacturing, monitoring computers in a data center, etc.

![anomoly_detection](/img/coursera-machine-learning-week9/anomoly_detection.png)

Suppose the anomaly detection system flags $x$ as anomalous whenever $p(x) \leq \epsilon$. It is flagging too many things as anomalous that are not actually so. The corrective step in this case would be to decrease the value of $\epsilon$.

### Gaussian Distribution

The Gaussian Distribution (Normal Distribution), where $ x \in \mathbb{R}$, has mean $\mu$ and variance $\sigma^2$:

$$ x \thicksim \mathcal{N}(\mu, \sigma^2) $$ (The tilde means "distributed as", Script 'N' stands for normal distribution

$$ p(x;\mu, \sigma^2) = \dfrac{1}{\sqrt{2\pi}\sigma} \times \exp{(-\dfrac{(x-\mu)^2}{2\sigma^2})} $$

![gaussian_distribution](/img/coursera-machine-learning-week9/gaussian_distribution.png)

![gaussian_distribution_examples](/img/coursera-machine-learning-week9/gaussian_distribution_examples.png)

Red shaded area is equal to 1.

To calculate the average $\mu$ and variance $\sigma^2$ we use the following formulae:

$$ \mu = \dfrac{1}{m} \sum\limits\_{i=1}^{m}x^{(i)} $$
$$ \sigma^2 = \dfrac{1}{m} \sum\limits\_{i=1}^{m}(x^{(i)} - \mu)^2 $$

![average_variance_formulae](/img/coursera-machine-learning-week9/average_variance_formulae.png)

### Algorithm

Algorithm for density estimation. Given a training set {$x^{(1)}, \dots, x^{(m)} $} where each example is $x \in \mathbb{R}^n$

$$p(x) = p(x\_1;\mu\_1,\sigma^2\_1)p(x\_2;\mu\_2,\sigma^2\_2)p(x\_3;\mu\_3,\sigma^2\_3)\dots p(x\_n;\mu\_n,\sigma^2\_n)$$

where $ x\_1 \thicksim \mathcal{N}(\mu\_1, \sigma^2\_1) $,$ x\_2 \thicksim \mathcal{N}(\mu\_2, \sigma^2\_2) $, and so on.

More conscicely, this algorithm can be written as:

$$ p(x) = \prod\limits\_{j=1}^n p(x\_j;\mu\_j, \sigma^2\_j) $$

The capital $\Pi$ is the product symbol, it is similar to the $\sum$ function except rather than adding, it performs multiplication.

1. Choose features $x\_i$ that you think might be indicative of anomalous examples.

2. Fit parameters $\mu\_1, \mu\_2, \dots, \mu\_n; \sigma\_1^2, \sigma\_2^2, \dots, \sigma\_n^2$

3. Given new example $x$, compute $p(x)$. The example is an anomaly if $p(x) < \epsilon $.

![anomaly_detection_example](/img/coursera-machine-learning-week9/anomaly_detection_example.png)

## Building an Anomaly Detection System

### Developing and Evaluating an Anomaly Detection System

The importance of real-number evaluation, when developing a learning algorithm, learning decisions is much easier if we have a way of evaluating our learning algorithm. Assume we have some labeled data of anomalous and non-anomalous examples. ($y = 0$ if normal and $y = 1$ as anomalous).

Training set: $x^{(1)}, x^{(2)}, \dots, x^{(m)} $ (assume that the training set is normal and not anomalous)

Cross validation set: $ (x\_{\text{cv}}^{(1)}, y\_{\text{cv}}^{(1)}), \dots, (x\_{\text{cv}}^{(m\_{\text{cv}})}, y\_{\text{cv}}^{(m\_{\text{cv}})}) $

Test set: $(x\_{\text{test}}^{(1)}, y\_{\text{test}}^{(1)}), \dots, (x\_{\text{test}}^{(m\_{\text{test}})}, y\_{\text{test}}^{(m\_{\text{test}})})$

The following would be a reccomended split of training sets and cross validation sets for an aircraft engine monitoring example:

![aircraft_engine_example](/img/coursera-machine-learning-week9/aircraft_engine_example.png)

One can evaluate the algoirthm by using precision and recall, or the $F\_1$ score.

![evaluate_anomaly_detection](/img/coursera-machine-learning-week9/evaluate_anomaly_detection.png)

### Anomaly Detection vs. Supervised Learning

**Anomaly detection** can be used if there are a very small number of positive examples ($y=1$, a range between zero and twenty is common). Anomaly detection should also have a large number of negative ($y=0$) examples. Anomalies should have many types, it's hard for any algorithm to learn what anomalies look like, as future anomalies may look nothing like what we have seen so far.

* Fraud Detection
* Manufacturing (eg aircraft engines)
* Monitoring machines in a data center

**Supervised learning** should be used when there are a large number of positive and negative ezamples. There are enough positive examples for the algorithm to get a sense of what positive examples are like. Future positive examples are likely to be similar to the ones in the training set.

* Email/Spam classification
* Weather prediction (sunny/rainy/etc)
* Cancer classification

### Choosing What Features to Use

One thing that we have done was plotting the features to see if the features fall into a normal (gaussian) distribution. Given a dataset that does not look gaussian, it might be useful to transform the features to look more gaussian. There are multiple different functions one can play with to make the data look more gaussian.

![transform_data_to_gaussian](/img/coursera-machine-learning-week9/transform_data_to_gaussian.png)

Error analysis for anomaly detection- we want $p(x)$ to be large for normal examples and $p(x)$ to be small for anomalous examples.

How do we fix the common problem where $p(x)$ is comparable for both normal and anomalous examples? This is still a manual process, look at the anomalous examples and distinguish features that make the irregular example anomalous.

Choose features that might take on unusually large or small values in the event of an anomaly.

## Multivariate Gaussian Distribution

### Algorithm

Multivariate Gaussian Distribution can be useful if there are correlation between the features that need to be accounted for when determining anomalies.

It may be useful to not model $p(x\_1), p(x\_2)$ separately. Model $p(x)$ all in one go. Parameters are $\mu \in \mathbb{R}^n$, $\Sigma \in \mathbb{R}^{n\times n}$.

![multivariate_gaussian_formula](/img/coursera-machine-learning-week9/multivariate_gaussian_formula.png)
![multivariate_gaussian_distribution](/img/coursera-machine-learning-week9/multivariate_gaussian_distribution.png)

Here are some examples of multivariate gaussian examples with varying sigma:

![multivariate_gaussian_examples](/img/coursera-machine-learning-week9/multivariate_gaussian_examples.png)
![multivariate_gaussian_examples_2](/img/coursera-machine-learning-week9/multivariate_gaussian_examples_2.png)

To perform parameter fitting, plug in the follwing formula:

![multivariate_gaussian_parameter_fitting](/img/coursera-machine-learning-week9/multivariate_gaussian_parameter_fitting.png)

# Reccomender Systems

## Predicting Movie Ratings

### Problem Forumulation

Imagine you are a company that sells or rents out movies. You allow users to rate different movies from zero to five stars. You have a matrix of users, their sparsely populated rated movies, and you want to find out what they would also rate similarly according to existing data.

![reccomender_system_problem_formulation](/img/coursera-machine-learning-week9/reccomender_system_problem_formulation.png)

### Content Based Recommendations

If the movies have features associated to the movie, they can be represented with a feature vector.

![content_based_recommendations](/img/coursera-machine-learning-week9/content_based_recommendations.png)

To learn $\theta^{(j)}$, refer to the following problem formulation:

$$ \theta^{(j)} = \min\limits\_{\theta^{(j)}} \dfrac{1}{2} \sum\limits\_{i:r(i,j)=1} ((\theta^{(j)})^Tx^{(i)} - y^{(i,j)})^2 + \dfrac{\lambda}{2} \sum\limits\_{k=1}^n (\theta\_k^{(j)})^2 $$

To learn $\theta^{(1)}, \theta^{(2)}, \dots, \theta^{(n\_u)} $:

$$ J(\theta^{(1)}, \dots, \theta^{(n\_u)}) = \min\limits\_{\theta^{(1)}, \dots, \theta^{(n\_u)}} \dfrac{1}{2} \sum\limits\_{j=1}^{n\_u} \sum\limits\_{i:r(i,j)=1} ((\theta^{(j)})^Tx^{(i)} - y^{(i,j)})^2 + \dfrac{\lambda}{2} \sum\limits\_{j=1}^{n\_u} \sum\limits\_{k=1}^n (\theta\_k^{(j)})^2 $$

Gradient Descent:

$$ \theta\_k^{(j)} := \theta\_k^{(j)} - \alpha \sum\limits\_{i:r(i,j)=1} ((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})x\_k^{(i)} \hspace{1em} \text{for } k=0 $$
$$ \theta\_k^{(j)} := \theta\_k^{(j)} - \alpha (\sum\limits\_{i:r(i,j)=1} ((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})x\_k^{(i)} + \lambda\theta\_k^{(j)} ) \hspace{1em} \text{for } k \neq 0 $$

## Collaborative Filtering

### Collaborative Filtering Algorithm

This algorithm learns what features to use for an existing data set. It can be very difficult to determine how 'romantic' a movie is, or how much 'action' a movie has. Suppose we have a data set where we do not know the features of our movies.

The assumption here is that the users pre-specify what genres of movies they like.

![collaborative_filtering](/img/coursera-machine-learning-week9/collaborative_filtering.png)

Given $\theta^{(1)}, \dots, \theta^{(n\_u)}$ to learn $x^{(i)}$:

$$ x^{(i)} = \min\limits\_{x^{(i)}} \dfrac{1}{2} \sum\limits\_{j:r(i,j)=1} ((\theta^{(j)})^Tx^{(i)} - y^{(i,j)})^2 + \dfrac{\lambda}{2}\sum\limits\_{k=1}^{n}(x\_k^{(i)})^2 $$

Given $\theta^{(1)}, \dots, \theta^{(n\_u)}$ to learn $x^{(1)}, \dots, x^{(n\_m)}$:

$$ \min\limits\_{x^{(i)}} \dfrac{1}{2} \sum\limits\_{i=1}^{n\_m} \sum\limits\_{j:r(i,j)=1} ((\theta^{(j)})^Tx^{(i)} - y^{(i,j)})^2 + \dfrac{\lambda}{2} \sum\limits\_{i=1}^{n\_m} \sum\limits\_{k=1}^{n}(x\_k^{(i)})^2 $$

1. Initialize $x^{(1)}, \dots, x^{(n\_m)}$ and $\theta^{(1)}, \dots, \theta^{(n\_u)}$ to random small values.

2. Minimize $J(x^{(1)}, \dots, x^{(n\_m)}, \theta^{(1)}, \dots, \theta^{(n\_u)})$ using gradient descent (or an advanced optimization algorithm).

3. For a user with parameters $\theta$ and a movie with (learned) features $x$ predict a star rating of $\theta^Tx$.


## Low Rank Matrix Factorization

### Vectorization: Low Rank Matrix Factorization

The vectorized implementation for the reccomender system can be visualized as the following:

![low_rank_matrix_factorization](/img/coursera-machine-learning-week9/low_rank_matrix_factorization.png)

Movies can be related if the feature vectors between the movies are small.

### Implementational Detail: Mean Normalization

For users who have not rated any movies, the only term that effects the user is the regularization term.

![before_mean_normalization](/img/coursera-machine-learning-week9/before_mean_normalization.png)

This does not help us perform reccomendations as the value of all predicted stars will be 0 for the user who has not rated any movies. One way to address this is to apply mean normalization to the input data and pretend the normalized data is the new data to perform prediction with.

This allows us to perform reccomendations even though a user has not rated any movies, because we have the average rating of a movie based on all users.

![after_mean_normalization](/img/coursera-machine-learning-week9/after_mean_normalization.png)

---

Move on to [Week 10]({{% relref "coursera-machine-learning-week10.md" %}}).
