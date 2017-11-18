---
title: "Neural Networks and Deep Learning, Week 1"
author: "Alexander Wong"
tags: ["Machine Learning", "deeplearning.ai"]
date: 2017-11-11T15:21:37-06:00
draft: false
---

Taking the [Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning), **Neural Networks and Deep Learning** course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/). See [deeplearning.ai](https://www.deeplearning.ai/) for more details.

{{% toc %}}

# Introduction to Deep Learning

There are five courses in the Coursera Deep Learning Specialization. (This is course 1.)

1. Neural Networks and Deep Learning
2. Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization
3. Structuring your Machine Learning Project
4. Convolutional Neural Networks
5. Natural Language Processing: Building Sequence Models.

## What is a Neural Network

A neural network is a learning algorithm comprised of many stacked neurons. THey are chained together to create representations of functions based on input data.

![neural_network_example](/img/deeplearning-ai/neural_network_example.png)

## Supervised Learning with Neural Networks

Supervised learning is when you are given specific inputs and outputs. For instance, in the housing pricing example you can have `x: number of bedrooms, size of house, ..., house features` map to `y: price of house`. These types of algorithms can calculate a given `y` with input `x`. Typically the machine is trained on data that has both `x` and `y` values.

## Why is Deep Learning Taking Off?

Primarily 3 reasons:

1. Massive increase in collected data
2. Increase in computational power
3. Algorithm modifications (usually increases computation speed)


## About this Course

This course is broken out into four weeks.

* Week 1: Introduction
* Week 2: Basics of Neural Network Programming
* Week 3: One Hidden Layer Neural Networks
* Week 4: Deep Neural Networks

## Optional: Heroes of Deep Learning (Geoffrey Hinton)

Interview between Andrew Ng and Geoffrey Hinton.

{{< youtube -eyhCTvrEtE >}}

---

Move on to [Week 2]({{% relref "neural-networks-and-deep-learning-week2.md" %}}).
