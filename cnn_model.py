# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 12:16:48 2017

@author: jpkak
"""

import tensorflow as tf
import numpy as np


def conv2d(x,W,s):
    return tf.nn.conv2d(x,W,strides = [1,s,s,1],padding='SAME')

def maxpool(x,f,s):
    return tf.nn.max_pool(x,ksize=[1,f,f,1],strides=[1,s,s,1],padding='SAME')

def convolution_neural_network(x):
    weights = {
            #
            'W_conv7':tf.Variable(tf.random_normal([7,7,1,64])),
            # 5x5 conv, 32 inputs, 64 outputs 
            'W_conv3': tf.Variable(tf.random_normal([3, 3, 64, 192])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'W_fc': tf.Variable(tf.random_normal([7*7*64, 1024])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, n_classes]))
            }
    biases = {
        'b_conv1': tf.Variable(tf.random_normal([32])),
        'b_conv2': tf.Variable(tf.random_normal([64])),
        'b_fc': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }