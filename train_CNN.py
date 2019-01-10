#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 22:24:33 2019

@author: louisgiron
"""
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_processing import (train_img_new, valid_img_new, test_img_new,
                             train_labels, valid_labels, test_labels)
from NeuralNetwork import CNN_model

# Input data
nb_images = train_img_new.shape[0]
nb_input = train_img_new.shape[1]
batch_size = 1000
nb_classes = 10
learning_rate = 0.005
nb_epoch = 5

X_train = train_img_new
y_train = train_labels
X_valid = valid_img_new
y_valid = valid_labels
X_test = test_img_new
y_test = test_labels

# declare the training data placeholders
X_tf = tf.placeholder(tf.float32, [None, 784])
y_tf = tf.placeholder(tf.float32, [None, 10])

# dynamically reshape the input
x_shaped = tf.reshape(X_tf, [-1, 28, 28, 1])
# now declare the output data placeholder - 10 digits

output, dense_layer2 = CNN_model(x_shaped)

#
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y_tf))

# add an optimiser
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y_tf, 1), tf.argmax(output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

avg_cost_v = []

with tf.Session() as sess:
    # initialise the variables
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    # Create the batches
    total_batch = int(X_train.shape[0]/batch_size)

    for epoch in range(nb_epoch):

        # Random permutation
        idx_shuffle = np.random.permutation(np.arange(X_train.shape[0]))
        X_train = X_train[idx_shuffle, :]
        y_train = y_train[idx_shuffle]

        for i in range(total_batch):
            avg_cost = 0.

            # Batch definition
            batch_X = X_train[i*batch_size:(i+1)*batch_size]
            batch_y = y_train[i*batch_size:(i+1)*batch_size]

            # Train process
            _, c, acc = sess.run([optimiser, cross_entropy, accuracy],
                                 feed_dict={X_tf: batch_X, y_tf: batch_y})
            avg_cost += c / total_batch

        # Average cost - train
        avg_cost_v.append(avg_cost)
        pred, acc_valid, cost_valid = sess.run([output, accuracy,
                                                cross_entropy],
                                               feed_dict={X_tf: X_valid,
                                                          y_tf: y_valid})
        print("Epoch:", (epoch + 1), "| Cost (train set):",
              np.round(avg_cost_v[epoch], 3), "| Accuracy (train set):", acc,
              "| Cost (valid set):", np.round(cost_valid, 3),
              "| Accuracy (valid set):", acc_valid)

    print("\nTraining complete!")
    acc_test = sess.run(accuracy, feed_dict={X_tf: X_test, y_tf: y_test})
    print('\nAccuracy on the test set:', acc_test)

    # Close the session
    sess.close()
