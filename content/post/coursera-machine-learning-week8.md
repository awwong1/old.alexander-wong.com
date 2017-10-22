---
title: "Machine Learning, Week 8"
author: "Alexander Wong"
tags: ["Machine Learning"]
date: 2017-10-14T09:21:37-06:00
draft: false
---


Taking the [Coursera Machine Learning](https://www.coursera.org/learn/machine-learning) course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/).

Assumes you have knowledge of [Week 7]({{% relref "coursera-machine-learning-week7.md" %}}).

{{% toc %}}

* Lecture notes:
  * [Lecture13](/docs/coursera-machine-learning-week8/Lecture13.pdf)
  * [Lecture14](/docs/coursera-machine-learning-week8/Lecture14.pdf)

# Unsupervised Learning

## Clustering

### Introduction

Unsupervised learning is the class of problem solving where when given a set of data with no labels, find structure in the dataset.

![unsupervised_learning](/img/coursera-machine-learning-week8/unsupervised_learning.png)

Clustering is good for problems like:

* Market segmentation (Create groups for your potential customers)
* Social Network analysis (analyze groups of friends)
* Organize computing clusters (arrange servers in idea locations to one another)
* Astronomical data analysis (Understand groupings of stars)

### K-Means Algorithm

This is a clustering algorithm.

1. Randomly initialize your cluster centroids.
2. Cluster assignment step: Assign each example to a cluster centroid based on distance.
3. Move centroid step: Take the centroids, move them to the average of all the assigned examples.
4. Iterate through step 2 and 3 until convergence.

![k_means_1](/img/coursera-machine-learning-week8/k_means_1.png)
![k_means_2](/img/coursera-machine-learning-week8/k_means_2.png)
![k_means_3](/img/coursera-machine-learning-week8/k_means_3.png)
![k_means_4](/img/coursera-machine-learning-week8/k_means_4.png)
![k_means_5](/img/coursera-machine-learning-week8/k_means_5.png)

**Formal Definition**

Input:

* K (number of clusters)
* Training set $ \\{ x^{(1)}, x^{(2)}, \dots, x^{(m)} \\}$

$ x^{(i)} \in \mathbb{R}^n $ (drop $x\_0 = 1$ convention)

Randomly initialize $K$ cluster centroids $\mu\_1, \mu\_2, \dots, \mu\_k \in \mathbb{R}^{n}$

Repeat {

* Cluster Assignment Step
  * for $i = 1$ to $m$, $c^{(i)} :=$ index (from $1$ to $K$) of cluster centroid closest to $x^{(i)}$
* Move Centroid Step
  * for $k = 1 \text{to} K$, $\mu\_k :=$ average (mean) of points assigned to cluster $k$

}

If there are clusters with no points assigned to it, it is common practice to remove that cluster. Alternatively, one may reinitialize the algorithm with new cluster centroids.

### Optimization Objective

The K-Means cost function (optimization objective) is defined here:

$$ c^{(i)} = \text{ index of cluster (1, 2,}\dots, K\text{) to which example } x^{(i)} \text{ is currently assigned} $$
$$ \mu\_k = \text{ cluster centroid } k \hspace{1em} (\mu\_k \in \mathbb{R}^{n}) $$
$$ \mu\_{c^{(i)}} = \text{ cluster centroid of cluster to which example } x^{(i)} \text{ has been assigned} $$

$$ J(c^{(1)}, \dots, c^{(m)}, \mu\_1, \dots, \mu\_K) = \dfrac{1}{m} \sum\limits\_{i=1}^{m} || x^{(i)} - \mu\_{c^{(i)}} || ^2 $$
$$ \min\limits\_{c^{(1)}, \dots, c^{(m)}, \mu\_1, \dots, \mu\_K} J(c^{(1)}, \dots, c^{(m)}, \mu\_1, \dots, \mu\_K) $$

### Random Initialization

How do you initialize the cluster centroids?

* Pick a number of clusters less than the number of examples you have
  * should have $K \lt m$
* Randomly pick $K$ training examples
* Set $\mu\_1, \dots, \mu\_K$ equal to these $K$ examples

![k_means_local_optima](/img/coursera-machine-learning-week8/k_means_local_optima.png)

The K-Means algorithm may end up in local optima. The way to get around this is to run K-Means multiple times with multiple random initializations. A typical number of times to run K-Means is 50 - 1000 times. Compute the cost function J and pick the clustering that gives the lowest cost.

### Choosing the Number of Clusters

Choosing the number of clusters $K$ in K-means is a non trivial problem as clusters may or may not be intuitive. Usually, it is still a manual step where an individual picks the number of clusters by looking at a plot.

![num_clusters](/img/coursera-machine-learning-week8/num_clusters.png)

* are there 2, 3, or 4 clusters? It's ambiguous

**Elbow Method**

* Run K-Means with varying number of clusters. (1 cluster, then 2 clusters, then 3... so on and so on)
* Ends up with a curve showing how distortion decreases as the number of clusters increases

![elbow_method](/img/coursera-machine-learning-week8/elbow_method.png)

* usually the 'elbow' is not clearly defined

Usually K-Means is downstream purpose specific. For example, when calculating clusters for market segmentation, if we are selling T-Shirts, perhaps it is more useful to have pre-defined clusters "Small, Medium, Large" or "Extra Small, Medium, Large, Extra Large" sizes.

# Dimensionality Reduction

## Motivation

### Data Compression

There are two primary reasons to perform dimensionality reduction. One of them is data compression, and the other is that dimensionality reduction can increase performance of our learning algorithms.

Given a two features like length in inches and centemeters, with slight roundoff error, there is a lot of redundancy. It would be useful to convert a 2D plot into a 1D vector.

![data_compression](/img/coursera-machine-learning-week8/data_compression.png)

Before, we needed two numbers to represent an example. After compression, only one number is necessary to represent the example.

The typical example of dimensionality reduction is from 1000D to 100D.

![3D_data_compression](/img/coursera-machine-learning-week8/3D_data_compression.png)

### Visualization

Dimensionality Reduction also helps us visualize the data better. Suppose we have the following dataset:

![visualization_dataset](/img/coursera-machine-learning-week8/visualization_dataset.png)

We want to reduce the features to a two or three dimensional vector in order to better understand the data, rather than attempt to plot a 50 dimension table.

![visualization_analysis](/img/coursera-machine-learning-week8/visualization_analysis.png)

## Principal Component Analysis

### Principal Component Analysis Problem Formulation

PCA is the most popular algorithm to perform dimensionality reduction.

* Find a lower dimensional surface such that the sum of squares error (projection error) is minimized.
* Standard practice is to perform feature scalining and mean normalization before scaling the data.

![pca_problem_formulation](/img/coursera-machine-learning-week8/pca_problem_formulation.png)

**PCA is not linear regression**

![pca_not_linear_regression](/img/coursera-machine-learning-week8/pca_not_linear_regression.png)

* In linear regression, we are minimizing the point and the value predicted b the hypothesis
* In PCA, we are minimizing the distance between the point and the line

### Principal Component Analysis Algorithm

1. Data preprocessing step:
  * Given your training set $x^{(1)}, x^{(2)}, \dots, x^{(m)}$
  * Perform feature scaling and mean normalization
  $$ \mu\_j = \dfrac{1}{m} \sum\limits\_{i=1}^m x\_j^{(i)} $$
  * Replace each $x\_j^{(i)}$ with $ x\_j - \mu\_j $.
  * If different features on different scales, (e.g. size of house, number of bedrooms) scale features to have comparable range of values.
  $$ x\_j^{(i)} \leftarrow \dfrac{x\_j^{(i)} - \mu\_j}{s\_j} $$

2. PCA algorithm
  * Reduce data from $n$-dimensions to $k$-dimensions
  * Compute the "covariance matrix":
    * (it is unfortunate that the Sigma value is used, do not confuse with summation)
  $$ \Sigma = \dfrac{1}{m} \sum\limits\_{i=1}^{n} (x^{(i)})(x^{(i)})^T $$
  * Compute the 'eigenvectors' of matrix $\Sigma$

  ```octave
  [ U, S, V ] = svd(Sigma)
  % svd stands for singular value decomposition
  % another function that does this is the eig(Sigma) function
  % svd(Sigma) returns an n * n matrix
  ```
  ![eigenvectors_to_pca](/img/coursera-machine-learning-week8/eigenvectors_to_pca.png)

![pca_algorithm](/img/coursera-machine-learning-week8/pca_algorithm.png)

## Applying PCA

### Reconstruction from Compressed Representation

After compression, how do we go back to the higher dimensional state?

![pca_reconstruction](/img/coursera-machine-learning-week8/pca_reconstruction.png)

### Choosing the Number of Principal Components

In the PCA algorithm, how do we choose the value for $k$? How do we choose the number of principal components?

Recall that PCA tries to minimize the average squared projection error

$$ \dfrac{1}{m} \sum\_{i=1}^{m} || x^{(i)} - x\_{\text{approx}}^{(i)} ||^2 $$

Also, the total variation in the data is defined as

$$ \dfrac{1}{m} \sum\_{i=1}^{m} || x^{(i)} || ^2 $$

Typically choose $k$ to be the smallest value such that

$$ \dfrac{ \dfrac{1}{m} \sum\_{i=1}^{m} || x^{(i)} - x\_{\text{approx}}^{(i)} || ^ 2 }{ \dfrac{1}{m} \sum\_{i=1}^{m} || x^{(i)} || ^ 2 } \leq 0.01 $$
$$ \text{This means that 99% of variance is retained.} $$

Vary the percentage between 95-99 percent, depending on your application and requirements.

This is how to calculate the variance using the `svd(Sigma)` function return values:

![calculate_pca_variance](/img/coursera-machine-learning-week8/calculate_pca_variance.png)

### Advice for Applying PCA

PCA can be applied to reduce your training set dimensions before feeding the resulting training set to a learning algorithm.

Only run PCA on the training set. Do not run PCA on the cross validation and test sets.

One bad use of PCA is to prevent overfitting. Use regularization instead. PCA throws away some information without knowing what the corresponding values of y are.

Do not unnecessarily run PCA. It is valid to run your learning algorithms using the raw data $x^{(i)}$ and only when that fails, implement PCA.

---

Move on to [Week 9]({{% relref "coursera-machine-learning-week9.md" %}}).
