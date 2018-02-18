---
title: "Convolutional Neural Networks, Week 3"
author: "Alexander Wong"
tags: ["Machine Learning", "deeplearning.ai"]
date: 2018-02-18T12:21:37-06:00
draft: false
---

Taking the [Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning), **Convolutional Neural Networks** course. Will post condensed notes every week as part of the review process. All material originates from the free Coursera course, taught by [Andrew Ng](http://www.andrewng.org/). See [deeplearning.ai](https://www.deeplearning.ai/) for more details.

{{% toc %}}

# Object Detection

## Learning Objectives

- Understand the challenges of Object Localization, Object Detection, Landmark Finding
- Understand and implement non-max suppression
- Understand and implement intersection over union
- Understand how to label a dataset for an object detection application
- Remember the vocabulary of object detection (landmark, anchor, bounding box, grid)

## Detection Algorithms

### Object Localization

![object_localization](/img/deeplearning-ai/object_localization.png)

Image classification: One object (Is cat or no cat)

Classification with Localization: One object (is cat or not cat), bounding box over the object

Detection: Multiple objects, multiple bounding boxes.

![classification_with_localization](/img/deeplearning-ai/classification_with_localization.png)

![defining_target_label_y](/img/deeplearning-ai/defining_target_label_y.png)

In practice, you don't have to use squared error. You can use different loss functions for different output values. (mean squared error for bounding box, logistic regression loss for $P\_c$)

### Landmark Detection

Detection of certain points on your image.

Define the landmarks you want to detect in your training set, then set the output parameters in the neural network.

![landmark_detection](/img/deeplearning-ai/landmark_detection.png)

### Object Detection

Start with really closely cropped images. Given this labeled training set, train a convnet to return $y=\\{0,1\\}$.

![conv_net_example](/img/deeplearning-ai/conv_net_example.png)

Perform a sliding window detection with bounding boxes of increasing sizes.

![sliding_windows_detection](/img/deeplearning-ai/sliding_windows_detection.png)

This is extremely computationally expensive. Granularity, box size, computational cost, all needs to be taken into account. We can implement this better.

### Convolutional Implementation of Sliding Windows

![fc_to_convolutional_layers](/img/deeplearning-ai/fc_to_convolutional_layers.png)

![convolution_implementation_of_sliding_windows](/img/deeplearning-ai/convolution_implementation_of_sliding_windows.png)

You can implement sliding windows convolutionally. This algorithm has a weakness- bounding box predictions aren't too accurate.

### Bounding Box Predictions

![yolo_algorithm](/img/deeplearning-ai/yolo_algorithm.png)

![specify_the_bounding_boxes](/img/deeplearning-ai/specify_the_bounding_boxes.png)

### Intersection Over Union

![intersection_over_union](/img/deeplearning-ai/intersection_over_union.png)

The higher the IoU is, the more 'correct' the bounding box. 0.5 is a human chosen convention.

### Non-max Suppression

One problem of object detection is the algorithm might detect a single object more than once.

![non_max_suppression](/img/deeplearning-ai/non_max_suppression.png)

Take the highest probability box from all the overlaps, then suppress the overlapped box with lower probability.

![non_max_suppression_algorithm](/img/deeplearning-ai/non_max_suppression_algorithm.png)

### Anchor Boxes

What if a grid cell wants to detect multiple objects?

![overlapping_objects](/img/deeplearning-ai/overlapping_objects.png)

![anchor_box_algorithm](/img/deeplearning-ai/anchor_box_algorithm.png)

![anchor_box_example](/img/deeplearning-ai/anchor_box_example.png)

Example here has two anchor boxes.

- This does not handle three objects in the same grid cell.
- This does not handle two same anchor box sizes in the same grid cell.

### YOLO Algorithm

![training_yolo](/img/deeplearning-ai/training_yolo.png)

![making_predictions](/img/deeplearning-ai/making_predictions.png)

![outputting_non_max_suppressed_outputs](/img/deeplearning-ai/outputting_non_max_suppressed_outputs.png)

### (Optional) Region Proposals

R-CNN (Region Convolutional Neural Network). Run a segmentation algorithm first to determine what could be objects.

![r-cnn_high_level](/img/deeplearning-ai/r-cnn_high_level.png)

![faster_rcnn_algorithms](/img/deeplearning-ai/faster_rcnn_algorithms.png)
