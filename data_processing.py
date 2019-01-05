#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 16:07:47 2019

@author: louisgiron
"""
import os
import numpy as np
import random as rd
import tools.tools as tls

# Import data
path = os.getcwd() + '/mnist/data/'

train_img = np.load(path+'training_images.npy')
train_labels = np.load(path+'training_labels.npy')

valid_img = np.load(path+'validation_images.npy')
valid_labels = np.load(path+'validation_labels.npy')

test_img = np.load(path+'test_images.npy')
test_labels = np.load(path+'test_labels.npy')

# img
train_img_new = train_img.reshape(train_img.shape[0],
                                  train_img.shape[1]**2).astype('uint64')
valid_img_new = valid_img.reshape(valid_img.shape[0],
                                  valid_img.shape[1]**2).astype('uint64')
test_img_new = test_img.reshape(test_img.shape[0],
                                test_img.shape[1]**2).astype('uint64')

# label
train_lab_new = np.array([np.argmax(train_labels[y])
                          for y in range(len(train_labels))])
valid_lab_new = np.array([np.argmax(valid_labels[y])
                          for y in range(len(valid_labels))])
test_lab_new = np.array([np.argmax(valid_labels[y])
                         for y in range(len(valid_labels))])

# Random permutation of the train set
permut_index = np.random.permutation(train_img_new.shape[0])
train_img = train_img[permut_index, :, :]
train_img_new = train_img_new[permut_index, :]
train_labels = train_labels[permut_index, :]
train_lab_new = train_lab_new[permut_index]


# Visualisation
def visualize(path, name, X, y, nb_img=36):
    """Allows to vizualise 36 img and the labels"""
    vector_viz = [rd.randint(0, len(X)-1) for y in np.arange(nb_img)]
    # Img
    nb_vertically = int(np.sqrt(nb_img))
    name_img_out = path+'visualize_img' + '_' + str(name) + '.png'
    tls.visualize_grayscale_images(X[vector_viz, :, :, :],
                                   nb_vertically, name_img_out)

    # Labels
    name_lab_out = path+'visualize_labels' + '_' + str(name)
    labels = train_lab_new[vector_viz]
    labels = labels.reshape((nb_vertically, nb_vertically))

    file = open(name_lab_out, "w")
    labels_str = ''
    for line in labels:
        for elt in line:
            labels_str = labels_str + str(elt) + ' '
        labels_str = labels_str + '\n'
    file.write(labels_str)
    file.close()
    print('Visualisation exported')


path = os.getcwd() + '/img_observation/'
X = train_img
y = train_lab_new
visualize(path, 'obs', X, y, nb_img=36)
