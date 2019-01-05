#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 16:20:05 2019

@author: louisgiron
"""
import tensorflow as tf


# Define the graph - Basic neural net
def neural_net_model(X, nb_input, batch_size, nb_hidden, nb_classes,
                     learning_rate=0.5):

    # Parameters
    weight_0 = tf.Variable(tf.random.normal((nb_input, nb_hidden),
                                            mean=0.0, stddev=1.0,
                                            dtype=tf.float32,))
    biase_0 = tf.Variable(tf.zeros([nb_hidden]))

    weight_1 = tf.Variable(tf.random.normal((nb_hidden, nb_classes),
                                            mean=0.0, stddev=1.0,
                                            dtype=tf.float32,))
    biase_1 = tf.Variable(tf.zeros([nb_classes]))

    # Graphs
    # First layer
    a = tf.nn.bias_add(tf.matmul(X, weight_0), biase_0)
    a = tf.math.sigmoid(a)

    # Second layer
    z = tf.nn.bias_add(tf.matmul(a, weight_1), biase_1)

    # Output of the graph
    return tf.nn.softmax(z)


# Define the graph - Evolved neural network
def neural_net_model_evol(X, nb_input, nb_hidden1, nb_hidden2, nb_classes,
                          keep_prob_1, keep_prob_2):
    # Parameters
    weight_0 = tf.Variable(tf.random.normal((nb_input, nb_hidden1),
                                            mean=0.0, stddev=1.0,
                                            dtype=tf.float32,))
    biase_0 = tf.Variable(tf.zeros([nb_hidden1]))

    weight_1 = tf.Variable(tf.random.normal((nb_hidden1, nb_hidden2),
                                            mean=0.0, stddev=1.0,
                                            dtype=tf.float32,))
    biase_1 = tf.Variable(tf.zeros([nb_hidden2]))

    weight_2 = tf.Variable(tf.random.normal((nb_hidden2, nb_classes),
                                            mean=0.0, stddev=1.0,
                                            dtype=tf.float32,))
    biase_2 = tf.Variable(tf.zeros([nb_classes]))

    # Graphs
    # First layer
    layer_1 = tf.nn.bias_add(tf.matmul(X, weight_0), biase_0)
    layer_1 = tf.math.sigmoid(layer_1)
    drop_out_1 = tf.nn.dropout(layer_1, keep_prob_1)  # DROP-OUT here

    # Second layer
    layer_2 = tf.nn.bias_add(tf.matmul(drop_out_1, weight_1), biase_1)
    layer_2 = tf.math.sigmoid(layer_2)
    drop_out_2 = tf.nn.dropout(layer_2, keep_prob_2)  # DROP-OUT here

    # Output layer
    layer_out = tf.nn.bias_add(tf.matmul(drop_out_2, weight_2), biase_2)
    return tf.nn.softmax(layer_out)
