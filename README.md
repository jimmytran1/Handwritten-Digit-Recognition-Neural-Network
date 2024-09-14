# Handwritten Digit Recognition with Neural Networks

This project demonstrates the use of a **neural network** built using **TensorFlow** and **Keras** to recognize handwritten digits from the popular **MNIST dataset**. The model achieves an accuracy of 97.77% on the test dataset and showcases how neural networks can be applied to image classification tasks.

## Introduction

This project leverages a sequential neural network architecture to classify images of handwritten digits (0-9) from the MNIST dataset. Using TensorFlow and Keras, the model processes input images, normalizes the pixel values, and trains a neural network to predict the correct digit. The model is saved and can be reused for future predictions or extended into more complex applications.

## What are Neural Networks?

Neural networks are a series of algorithms inspired by the human brain, designed to recognize patterns in data. They consist of layers of nodes (neurons), where each layer transforms the input data before passing it to the next layer. By training on large datasets, neural networks can learn complex representations, allowing them to make accurate predictions.

In this project, we use a **fully connected neural network** (also known as a multi-layer perceptron) with multiple hidden layers, activation functions, and an output layer using the **softmax** function to classify digits.

## What is the MNIST Dataset?

The **MNIST dataset** is a large database of handwritten digits commonly used for training various image processing systems. It contains 60,000 training images and 10,000 testing images, each consisting of a 28x28 pixel grayscale image of a digit from 0 to 9. The dataset is widely used for benchmarking image classification algorithms.

## What is TensorFlow?

**TensorFlow** is an open-source machine learning library developed by Google. It is widely used for building and deploying machine learning models, especially deep learning models like neural networks. TensorFlow provides high-level APIs like Keras, which simplify the process of building, training, and evaluating models.

## Project Overview

This project includes:
- Loading and preprocessing the MNIST dataset.
- Building a sequential neural network using Keras.
- Training the model for handwritten digit classification.
- Evaluating the model on test data.
- Saving the trained model for future use.

## Installation

To run this project, you'll need to have Python and the following libraries installed:

```bash
pip install tensorflow numpy matplotlib opencv-python
```

## Model Architecture
The neural network consists of:

**Input Layer:** 28x28 flattened input for each image.
**Two Hidden Layers:** Each with 128 neurons using the ReLU activation function to introduce non-linearity.
**Output Layer:** 10 neurons with the softmax activation function to classify digits 0-9.


## Results
The model achieves an accuracy of 97.77% on the test set. Below is a sample of the model’s performance:

Epoch	Loss	Accuracy
1	0.4676	86.63%
2	0.1174	96.38%
3	0.0730	97.77%
