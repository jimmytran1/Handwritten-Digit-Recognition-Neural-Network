import os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# loading MNIST data set of handwritten digits (0-9)
mnist = tf.keras.datasets.mnist
# Splitting the data into training data (x_train, y_train) and testing data (x_test, y_test)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizing the images so the values are between 0 and 1 instead of 0 and 255
# This makes it so the model can learn better
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Creating a model (Sequential means it will be built step by step, layer by layer)
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

model.save('handwritten_model.h5')