---
title: "Machine Learning, Week 11"
author: "Alexander Wong"
tags: ["Machine Learning"]
date: 2017-11-03T11:21:37-06:00
draft: false
---

Taking the [Coursera Machine Learning](https://www.coursera.org/learn/machine-learning) course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/).

Assumes you have knowledge of [Week 10]({{% relref "coursera-machine-learning-week10.md" %}}).

{{% toc %}}

* Lecture notes:
  * [Lecture18](/docs/coursera-machine-learning-week11/Lecture18.pdf)

# Application Example: Photo OCR

## Photo OCR

### Problem Description and Pipeline

Photo OCR (Object Character Recognition) is the task of trying to recognize objects, characters (words and digits) given an image. This can be done as a type of machine learning pipeline where the different stages of the image recognition algorithm are broken out into various learning algorithm steps.

For example, a pipeline could be:

1. Detect bounding boxes of text in an image
2. Perform character segmentation within the bounding box of text
3. Classify the individual character into some ASCII value
4. Perform spelling correction on the cummulative classified word

### Sliding Windows

Example problem; pedestrial detection in an image.

Given an image, create a small window (for instace, your image is 800x300, and you use a window size of 80x30). Starting at the top left corner of the image, run that patch through your classifier to determine if the object is human or not. Then, move the window to the right by some step size (stride) and run that patch through the classifier again. When the window reaches the right of the screen, start again from the left side of the screen with the window moved slightly lower than before.

Once the whole image is completed, perform the same thing with a larger image patch. eg. 100x40

### Getting Lots of Data and Artificial Data

How do you get a lot of data to train your OCR algorithm?

For text recognition, you can generate input data by using a variety of fonts and backgrounds. Create an application that outputs many images with text, random backgrounds, random fonts. Another way would be to use existing images of text and apply distortions (blur, warping).

For something like audio, you could modify your original audio clip to have some background noises (beeps, noisy crowds) in order to create more input data.

The distortions you use should be similar to some examples you want to classify. If you have random distortions that are not relevant to your classification, it would be meaningless noise and would be less likely to be useful.

1. Make sure you have a low bias classifier before expanding your data set. Keep increasing the number of features/number of hidden units in your neural network until you have a low bias classifier.
2. Ask: "How much work would it be to get 10x the amount of data as we currently have?"
  * artificial data synthesis
  * distort existing examples
  * collect data, label it yourself
  * crowd source (eg. "Amazon Mechanical Turk")

### Ceiling Analysis: What Part of the Pipeline to Work on Next

What part of the pipeline should we work on improving?

![ceiling_analysis](/img/coursera-machine-learning-week11/ceiling_analysis.png)

At each part of the pipeline, take that individual step and maximize that portion of the pipeline manually. See how much the overall performance of the system improves. This can give insight into which portion of the pipeline can give the highest improvement.

---

Coursera Machine Learning course finished.
