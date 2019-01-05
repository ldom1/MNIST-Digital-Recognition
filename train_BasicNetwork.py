#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 16:30:25 2019

@author: louisgiron
"""
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_processing import (train_img_new, valid_img_new, test_img_new,
                             train_labels, valid_labels, test_labels)
from NeuralNetwork import neural_net_model


# Input data
nb_images = train_img_new.shape[0]
nb_input = train_img_new.shape[1]
batch_size = 1000
nb_hidden = 16
nb_classes = 10
learning_rate = 0.5
nb_epoch = 10


def get_batch_idx(i, batch_size, nb_images):
    # Input
    max_batch_size = int(nb_images/batch_size)
    j = i % max_batch_size
    return np.arange(j*batch_size, (j+1)*batch_size)


# Init
X = tf.placeholder(tf.float32, [None, nb_input])
Y = tf.placeholder(tf.float32, [None, nb_classes])

p = neural_net_model(X, nb_input, batch_size, nb_hidden, nb_classes,
                     learning_rate)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(p),
                                              reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(p, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

acc_train = []
loss_train = []

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
            batch_index = get_batch_idx(i, batch_size, nb_images)
            batch_x, batch_y = (train_img_new[batch_index, :],
                                train_labels[batch_index])
            sess.run([train_step], feed_dict={X: batch_x, Y: batch_y})

        # Accuracy and cross entropy - Training phase
        loss, acc = sess.run([cross_entropy, accuracy],
                             feed_dict={X: valid_img_new, Y: valid_labels})
        loss_train.append(loss)
        acc_train.append(acc)
        print("\nEpoch", str(epoch), "Loss :", str(loss),
              "Accuracy :", str(acc))

    # Test
    # evaluate the accuracy of the model
    test_accuracy = sess.run(accuracy, feed_dict={X: test_img_new,
                                                  Y: test_labels})
    print("\nAccuracy on test set:", test_accuracy)

    # pred
    pred = sess.run(p, feed_dict={X: test_img_new, Y: test_labels})
    pred = np.array([np.argmax(pred[y]) for y in range(len(pred))])
    labels = np.array([np.argmax(test_labels[y])
                       for y in range(len(test_labels))])

    print("\nErrors (%):", np.sum(labels != pred)/len(pred), '%')

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
plt.title('Training phase')
plt.legend()
plt.show()
