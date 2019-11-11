
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

####################################

import numpy as np
import tensorflow as tf
import keras

from bc_utils.init_tensor import init_filters
from bc_utils.init_tensor import init_matrix

from keras.layers import Input, Conv2D, BatchNormalization, AveragePooling2D, Dense, Flatten, ReLU, Softmax
from keras.models import Model, Sequential

####################################

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

assert(np.shape(x_train) == (50000, 32, 32, 3))
x_train = x_train - np.mean(x_train, axis=0, keepdims=True)
x_train = x_train / np.std(x_train, axis=0, keepdims=True)
y_train = keras.utils.to_categorical(y_train, 100)

assert(np.shape(x_test) == (10000, 32, 32, 3))
x_test = x_test - np.mean(x_test, axis=0, keepdims=True)
x_test = x_test / np.std(x_test, axis=0, keepdims=True)
y_test = keras.utils.to_categorical(y_test, 100)

####################################

x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 100])

####################################

def block(x, f, p):
    conv1 = tf.layers.conv2d(inputs=x, filters=f, kernel_size=[3, 3], padding='same')
    bn1   = tf.contrib.layers.group_norm(conv1)
    relu1 = tf.nn.relu(bn1)

    conv2 = tf.layers.conv2d(inputs=relu1, filters=f, kernel_size=[3, 3], padding='same')
    bn2   = tf.contrib.layers.group_norm(conv2)
    relu2 = tf.nn.relu(bn2)

    pool = tf.layers.max_pooling2d(inputs=relu2, pool_size=[p, p], strides=[p, p], padding='same')
    return pool

block1 = block(x,      128, 2) # 32 -> 16
block2 = block(block1, 256, 2) # 16 -> 8
block3 = block(block2, 512, 2) #  8 -> 4
block4 = block(block3, 512, 4) #  4 -> 1
flat = tf.contrib.layers.flatten(block4)
out = tf.layers.dense(inputs=flat, units=100)

####################################

predict = tf.argmax(out, axis=1)
correct = tf.equal(predict, tf.argmax(y, 1))
sum_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=out)
params = tf.trainable_variables()
grads = tf.gradients(loss, params)
grads_and_vars = zip(grads, params)
train = tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1.).apply_gradients(grads_and_vars)

####################################

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

####################################

for ii in range(args.epochs):
    for jj in range(0, 50000, args.batch_size):
        s = jj
        e = jj + args.batch_size
        xs = np.reshape(x_train[s:e], (args.batch_size, 32, 32, 3))
        ys = np.reshape(y_train[s:e], (args.batch_size, 100))
        sess.run([train], feed_dict={x: xs, y: ys})
        
    total_correct = 0

    for jj in range(0, 10000, args.batch_size):
        s = jj
        e = jj + args.batch_size
        xs = np.reshape(x_test[s:e], (args.batch_size, 32, 32, 3))
        ys = np.reshape(y_test[s:e], (args.batch_size, 100))
        _sum_correct = sess.run(sum_correct, feed_dict={x: xs, y: ys})
        total_correct += _sum_correct

    print ("acc: " + str(total_correct * 1.0 / 10000))

####################################






