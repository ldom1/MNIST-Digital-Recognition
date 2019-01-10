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


# Define the graph - CNN
def CNN_model(x_shaped):

    # Create a convolutionnal layer
    def create_new_conv_layer(input_data, num_input_channels, num_filters,
                              filter_shape, pool_shape, name):
        # setup the filter input shape for tf.nn.conv_2d
        conv_filt_shape = [filter_shape[0], filter_shape[1],
                           num_input_channels, num_filters]

        # initialise weights and bias for the filter
        weights = tf.Variable(tf.truncated_normal(conv_filt_shape,
                                                  stddev=0.03),
                              name=name+'_W')
        bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

        # setup the convolutional layer operation
        out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1],
                                 padding='SAME')

        # add the bias
        out_layer += bias

        # apply a ReLU non-linear activation
        out_layer = tf.nn.relu(out_layer)

        # now perform max pooling
        ksize = [1, pool_shape[0], pool_shape[1], 1]
        strides = [1, 2, 2, 1]
        out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides,
                                   padding='SAME')

        return out_layer

    # create some convolutional layers
    layer1 = create_new_conv_layer(x_shaped, 1, 32, [5, 5], [2, 2],
                                   name='layer1')
    layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2],
                                   name='layer2')

    flattened = tf.reshape(layer2, [-1, 7 * 7 * 64])

    # setup some weights and bias values for this layer, then activate with
    # ReLU
    wd1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1000], stddev=0.03),
                      name='wd1')
    bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
    dense_layer1 = tf.matmul(flattened, wd1) + bd1
    dense_layer1 = tf.nn.relu(dense_layer1)

    # another layer with softmax activations
    wd2 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.03), name='wd2')
    bd2 = tf.Variable(tf.truncated_normal([10], stddev=0.01), name='bd2')
    dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
    output = tf.nn.softmax(dense_layer2)
    return output, dense_layer2
