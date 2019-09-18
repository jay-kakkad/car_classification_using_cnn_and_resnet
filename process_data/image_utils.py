
## BASE_DIR = ospathdirname(ospathabspath(__file__))
## syspathappend(BASE_DIR)
import time
import math
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import process_data.dataset as dataset
import cv2

from sklearn.metrics import confusion_matrix
from datetime import timedelta

import inception_v4 as v4

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))
def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True
                   ,stride=1,padding = 'SAME'):  # Use 2x2 max-pooling.

    shape = [filter_size, filter_size, num_input_channels, num_filters]

    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, stride, stride, 1],
                         padding=padding)
    layer += biases
    logits = tf.matmul(tf_train_dataset, weights) + biases 
    # Original loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) )
    # Loss function using L2 Regularization
    regularizer = tf.nn.l2_loss(weights)
    loss = tf.reduce_mean(loss + beta * regularizer)
    
    
    layer = tf.nn.batch_normalization(layer,scale=False,)

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    layer = tf.nn.relu(layer)

    return layer, weights

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features
def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

    

num_channels = 3
img_size = 299  
img_size_flat = 229
img_shape = (img_size, img_size)
classes = ['audi', 'bmw']
num_classes = len(classes)
batch_size = 32

validation_size = .16   

train_path = 'C:/Innefu/Project_car/data/train/'
test_path = 'C:/Innefu/Project_car/data/test/'
checkpoint = 'C:/Innefu/Project_car/models/'

data = dataset.read_train_sets(train_path,img_size,classes,validation_size=validation_size)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Validation-set:\t{}".format(len(data.valid.labels)))

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

x_image = tf.reshape(x,[-1,img_size,img_size,num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

model = v4.create_model(num_classes=num_classes)
y_pred = tf.nn.softmax(model.predict(x_image))




net = new_conv_layer(input = x_image,num_input_channels=num_channels,filter_size=3,stride = 2)
net = conv2d_bn(input, 32, 3, 3, strides=(2,2), padding='valid')
    net = conv2d_bn(net, 32, 3, 3, padding='valid')
    net = conv2d_bn(net, 64, 3, 3)





    



    

    

