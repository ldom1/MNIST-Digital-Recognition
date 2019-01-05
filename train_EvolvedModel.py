#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 19:21:50 2019

@author: louisgiron
"""
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_processing import (train_img_new, valid_img_new, test_img_new,
                             train_labels, valid_labels, test_labels)
from NeuralNetwork import neural_net_model_evol


# Input data
nb_images = train_img_new.shape[0]
nb_input = train_img_new.shape[1]
batch_size = 1000
nb_hidden1 = 64
nb_hidden2 = 64
nb_classes = 10
nb_epoch = 20

# Drop out level
prob_1 = 0.7
prob_2 = 0.7


def get_batch_idx(i, batch_size, nb_images):
    # Input
    max_batch_size = int(nb_images/batch_size)
    j = i % max_batch_size
    return np.arange(j*batch_size, (j+1)*batch_size)


# Init
X = tf.placeholder(tf.float32, [None, nb_input])
Y = tf.placeholder(tf.float32, [None, nb_classes])
keep_prob_1 = tf.placeholder(tf.float32)
keep_prob_2 = tf.placeholder(tf.float32)

p = neural_net_model_evol(X, nb_input, nb_hidden1, nb_hidden2, nb_classes,
                          keep_prob_1, keep_prob_2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(p),
                                              reduction_indices=[1]))

correct_pred = tf.equal(tf.argmax(p, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

acc_train = []
loss_train = []
learn_rate = []

# Piecewise constant
min_lr = 0.1
max_lr = 0.5
nb_values = 20

global_step = tf.Variable(0, trainable=False)
boundaries = list(np.linspace(batch_size,
                              batch_size*nb_epoch, nb_values,
                              dtype=np.int32)[:-1])
values = list(np.round(np.linspace(max_lr, min_lr, nb_values), 2))
learning_rate_pc = tf.train.piecewise_constant(global_step, boundaries, values)

j = 0
# Passing global_step to minimize() will increment it at each step.
learning_step_pc = tf.train.GradientDescentOptimizer(learning_rate_pc).minimize(cross_entropy,
                                                                 global_step=global_step)

# Exponential decay of the learning rate
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.7
learning_rate_ed = tf.train.exponential_decay(starter_learning_rate,
                                              global_step, batch_size, 0.69,
                                              staircase=True)
j = 0
# Passing global_step to minimize() will increment it at each step.
learning_step_ed = tf.train.GradientDescentOptimizer(learning_rate_ed).minimize(cross_entropy,
                                                   global_step=global_step)
# Train the neural network
# initialize variables and session
with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    try:
        saver.restore(sess, 'mnist_predictions.ckpt')
    except Exception:
        pass

    # train the model mini batch with 100 elements, for 1K times
    for epoch in tqdm(range(nb_epoch)):

        for i in range(batch_size):
            # Update learning rate
            l_rate = sess.run(learning_rate_pc, {global_step: j})
            j += 1

            batch_index = get_batch_idx(i, batch_size, nb_images)
            batch_x, batch_y = (train_img_new[batch_index, :],
                                train_labels[batch_index])
            sess.run([learning_step_pc], feed_dict={X: batch_x, Y: batch_y,
                                                    keep_prob_1: prob_1,
                                                    keep_prob_2: prob_2})

        # Accuracy and cross entropy - Training phase
        loss, acc = sess.run([cross_entropy, accuracy],
                             feed_dict={X: valid_img_new, Y: valid_labels,
                                        keep_prob_1: 1.0,
                                        keep_prob_2: 1.0})
        loss_train.append(loss)
        acc_train.append(acc)
        learn_rate.append(l_rate)
        print("\nEpoch", str(epoch), "Loss :", str(loss),
              "Accuracy :", str(acc), "Learning rate:", l_rate)

    # Test
    # evaluate the accuracy of the model
    test_accuracy = sess.run(accuracy, feed_dict={X: test_img_new,
                                                  Y: test_labels,
                                                  keep_prob_1: 1.0,
                                                  keep_prob_2: 1.0})
    print("\nAccuracy on test set:", test_accuracy)

    # pred
    pred = sess.run(p, feed_dict={X: test_img_new, Y: test_labels,
                                  keep_prob_1: 1.0, keep_prob_2: 1.0})
    pred = np.array([np.argmax(pred[y]) for y in range(len(pred))])
    labels = np.array([np.argmax(test_labels[y])
                       for y in range(len(test_labels))])

    print("\nErrors (%):", np.sum(labels != pred)/len(pred)*100, '%')

    # Save the model
    '''if input('Save model ? [Y/N]') == 'Y':
        import os
        saver.save(sess, os.getcwd() +
                   '/nn_saved_sessions/mnist_predictions.ckpt')
        print('Model Saved')'''

    # Close the session
    sess.close()

# Visualisation of the accuracy and the loss
plt.figure(figsize=(10, 7))
plt.plot(np.arange(nb_epoch), acc_train, label='accuracy',
         color='blue')
plt.plot(np.arange(nb_epoch), loss_train, label='loss', color='red')
#plt.plot(np.arange(nb_epoch), learn_rate, label='learning_rate', color='green')
plt.title('Training phase')
plt.legend()
plt.show()
