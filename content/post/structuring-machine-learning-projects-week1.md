---
title: "Structuring Machine Learning Projects, Week 1"
author: "Alexander Wong"
tags: ["Machine Learning", "deeplearning.ai"]
date: 2018-01-01T12:21:37-06:00
draft: false
---

Taking the [Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning), **Structuring Machine Learning Projects** course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/). See [deeplearning.ai](https://www.deeplearning.ai/) for more details.

{{% toc %}}

# ML Strategy

## Introduction to ML Strategy

### Why ML Strategy

Important to know how to structure your machine learning project. This will prevent time wasted pursuing ineffective optimization.

Ideas:

* Collect more data
* Collect more diverse training set
* Train algorithm longer with gradient descent
* Try Adam instead of Gradient Descent
* Try a bigger/smaller network
* Try Dropout
* Add $L\_2$ regularization
* Network Architecture
  * Activation functions
  * # hidden units
  * etc...

These are ways of analyzing your problem to determine what idea to try.

### Orthogonalization

There are so many things to try and change! Orthogonalization is the understanding of what tuning changes what effect. 

![orthogonalization_analogy](/img/deeplearning-ai/orthogonalization_analogy.png)

Chain of Assumptions in Machine Learning

1. Fit Training Set Well on Cost Function
  * Bigger Network
  * ADAM instead of Gradient Descent
2. Fit Dev Set Well on Cost Function
  * Regularization
  * Getting a bigger training set
3. Fit Test Set Well on Cost Function
  * Getting a bigger dev set
4. Performs Well in Real World
  * Change dev set or cost function

## Setting Up Your Goal

### Single Number Evaluation Metric

Machine Learning is an emperical process. It consistantly iterates between `Idea > Code > Experiment`.

**Dog Classification Example**

| Precision                                                | Recall                                          |
|----------------------------------------------------------|-------------------------------------------------|
| Of examples recognized as dog, what % are actually dogs? | What % of actual dogs are correctly recognized? |

Use an $F\_1$ Score. "Average" of Precision P and Recall R.

$$ \dfrac{2}{(1/p) + (1/r)} \leftarrow \text{"Harmonic Mean"} $$

| Classifier | Precision | Recall | F1 Score |
|:---|:---|:---|:---|
| A | 95% | 90% | 92.4% |
| B | 98% | 85% | 91.0% |

### Satisficing and Optimizing Metric

Another classification example. Accuracy may be an F1 Score.

| Classifier | Accuracy | Running Time |
|:--|:--|:--|
| A | 90% | 80ms |
| B | 92% | 95ms |
| C | 95% | 1500ms |

$$ \text{Cost} = \text{Accuracy} - 0.5 * \text{Running Time} $$

Maximize accuracy, subject to running time $\leq$ 100 ms.

In this example, Accuracy is the optimizing metric. Running Time is the satisficing metric.

Given N metrics: Pick 1 optimizing metric, pick N-1 satisficing metrics.

### Train/Dev/Test Distributions

Cat classification example development (hold out cross validation set) and test sets.

Example: Regions

* US
* UK
* Other Europe
* South America
* India
* China
* Other Asia
* Australia

Make sure your dev and test sets come from the same distribution.

Rather than making dev sets certain regions and test set certain regions, take all of the regions and randomly shuffle the data into the dev/test set.

**Takeaway**

Chose a dev set and test set to reflect data you expect to get in the future and consider important to do well on.

### Size of the Dev and Test Sets

Old way of splitting data is approximately 70% Train, 30% Test. (or 60% Train, 20% Dev, 20% Test). Dataset size is around 100, 1000, 10000.

Given a million training examples, might be reasonable to have (98% Train, 1% Dev, 1% Test). Once again, ensure distribution is random.

Set your test set to be big enough to give high condifence in the overall performance of your system.

### When to Change Dev/Test Sets and Metrics

Classification Example.

Metric: Classification Error

* Algorithm A: 3% error
  * lets through a lot of pornographic images (classifies boobs as a cat, for instance)
  * Even though it's 2% better, it's a worse algorithm
* Alforithm B: 5% error

Metric + Dev prefer A. You and Users prefer B.

$$ \text{Error} \rightarrow \dfrac{1}{\text{M}\_{\text{dev}}} \sum \limits ^{\text{M}\_{\text{dev}}} \_ {i=1} \mathcal{L} \\{ y^{(i)}\_{\text{pred}} \neq y^{(i)} \\} $$

* Predicted Value is (0/1)

May add a weight value:

$$ \text{Error} \rightarrow \dfrac{1}{\text{M}\_{\text{dev}}} \sum \limits ^{\text{M}\_{\text{dev}}} \_ {i=1} w^{(i)} \mathcal{L} \\{ y^{(i)}\_{\text{pred}} \neq y^{(i)} \\} $$

$w^{(i)}$ is 1 if $x^{(i)}$ is non-porn or set $w^{(i)}$ is 10 if $x^{(i)}$ is porn.

1. So far we've only discussed ho to define a metric to evaluate classifiers. (Place the target)
2. Worry separately about how to do well on this metric afterwards. (Hit the target.)

## Comparing to Human-Level Performance

### Why Human-level Performance?

Progress tends to be relatively rapid towards human level performance, then after surpassing human level performance the accuracy gains tend to plateau.

The hope is this achieves theoretical optimal performance (Bayes optimal error). Bayes Optimal Error is the best possible error such that there is no way a function mapping $x \rightarrow y$ exists that can perform better.

Human level performance, for many cases, is usually very close to Bayes optimal error.

Also, below human level performance, you can perform the following:

- Get labeled data from humans.
- Gain insight from manual error analysis: Why did a person get this right?
- Better analysis of bias/variance.

### Avoidable Bias

Classification Example:

| Metric | Example A | Example B |
|:--|:--|:--|
| Humans | 1% | 7.5% |
| Training Error | 8% | 8% |
| Dev Error | 10% | 10% |
| Reccomendation | Focus on Bias | Focus on Variance |

Example A, learning algorithm isn't even fitting the training data well.

In Example B, learning algorithm performs very close to human performance. Might be a variance problem.

Think of Human-Level error as a proxy, or estimate, for Bayes error.

- Difference between Humans (Bayes) and Training Error is _Avoidable Bias_.
  - If this value is large, maybe use a bigger neural network?
- Difference between Training and Error is _Variance_.
  - If this value is large, you're overfitting your training data.

### Understanding Human-level Performance

**Metical Image classification Example**

Suppose:

1. Typical Human: 3% error
2. Typical Doctor: 1% error
3. Experienced doctor: 0.7% error
4. Team of experienced doctors: 0.5% error

What is "human-level" error? Recall that the "human-error" is a proxy for Bayes error. It is the team of experienced doctors. $\text{Bayes error} \leq 0.5%$

Use the appropraite metric for your application requirements. Maybe it is sufficent to beat the typical doctor. Maybe you must optimize for the team of experienced doctors.

### Surpassing Human-level Performance



| Metric | Example A | Example B |
|:--|:--|:--|
| Team of Humans | 0.5% | 0.5% |
| Humans | 1% | 1% |
| Training Error | 0.6% | 0.3% |
| Dev Error | 0.8% | 0.4% |

In Example B, you don't have enough information to tell if you are overfitting or if you're not fitting enough. No longer have objective metric to determine Bayes error.

**Problems where ML significantly surpasses human-level performance**

- Online advertising
- Product recommendations
- Logistics (predicting transit time)
- Loan approvals

All of these examples learn from structured data. These are not natural perception problems.
There are some natural perception problems that machines have solved.

- Speech recognition
- Some image recognition
- Medical (ECG, Skin Cancer)

### Improving your Model Performance

**The Two Fundamental Assumptions of Supervised Learning**

- You can fit the training set pretty well. You acheive low avoidable bias.
- The training set performance generalizes pretty well to the dev/test set. (Variance)

![improving_model_performance](/img/deeplearning-ai/improving_model_performance.png)
