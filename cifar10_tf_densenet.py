
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='4'

####################################

import numpy as np
import tensorflow as tf
import keras

from bc_utils.init_tensor import init_filters
from bc_utils.init_tensor import init_matrix

####################################

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

assert(np.shape(x_train) == (50000, 32, 32, 3))
x_train = x_train - np.mean(x_train, axis=0, keepdims=True)
x_train = x_train / np.std(x_train, axis=0, keepdims=True)
y_train = keras.utils.to_categorical(y_train, 10)

assert(np.shape(x_test) == (10000, 32, 32, 3))
x_test = x_test - np.mean(x_test, axis=0, keepdims=True)
x_test = x_test / np.std(x_test, axis=0, keepdims=True)
y_test = keras.utils.to_categorical(y_test, 10)

####################################

epochs = 10
batch_size = 50
x = tf.placeholder(tf.float32, [None, 32 , 32 , 3])
y = tf.placeholder(tf.float32, [None, 10])

####################################

def batch_norm(x, f, name, vars_dict):
    gamma = tf.Variable(np.ones(shape=f), dtype=tf.float32)
    beta = tf.Variable(np.zeros(shape=f), dtype=tf.float32)
    vars_dict[name + '_gamma'] = gamma
    vars_dict[name + '_beta'] = beta

    mean = tf.reduce_mean(x, axis=[0,1,2])
    _, var = tf.nn.moments(x - mean, axes=[0,1,2])
    bn = tf.nn.batch_normalization(x=x, mean=mean, variance=var, offset=beta, scale=gamma, variance_epsilon=1e-3)
    return bn

def conv_block(x, f, name, vars_dict):
    filters = tf.Variable(init_filters(size=f, init='alexnet'), dtype=tf.float32)
    vars_dict[name] = filters

    fh, fw, fc, fo = f

    conv = tf.nn.conv2d(x, filters, [1,1,1,1], 'SAME')
    bn   = batch_norm(conv, fo, name, vars_dict)
    relu = tf.nn.relu(bn)

    return relu

def fc_block(x, size):
    input_size, output_size = size
    w = tf.Variable(init_matrix(size=size, init='alexnet'), dtype=tf.float32)
    b  = tf.Variable(np.zeros(shape=output_size), dtype=tf.float32)
    fc = tf.matmul(x, w) + b
    return fc

####################################

def concat(blocks):
    return tf.concat(blocks, axis=3)

def dense_block(x, nfmap, k, name, vars_dict):
    conv1 = conv_block(x,     [1,1,nfmap,4*k], name + '_conv1x1', vars_dict)
    conv2 = conv_block(conv1, [3,3,4*k,k], name + '_conv3x3', vars_dict)
    return conv2

# checkout ssdfa/lib/DenseBlock for this one.
def dense_model(x, xshape, k, block_sizes, name):
    vars_dict = {}
    _,_,_,c = xshape

    blocks = [x]
    for ii in range(len(block_sizes)):
        block_size = block_sizes[ii]
        for jj in range(block_size):
            nfmap = c + k * (sum(block_sizes[0:ii]) + jj)
            block = dense_block(concat(blocks), nfmap, k, name + '_block%d' % (nfmap), vars_dict) # block_ii_jj -> ii can equal jj
            blocks.append(block)

        pool = tf.nn.avg_pool(concat(blocks), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        blocks = [pool]

    return blocks[-1], vars_dict

####################################

block_sizes = [4, 4, 4]
k = 32
nhidden = 3 + k * sum(block_sizes)

dense, vars_dict = dense_model(x, [batch_size, 32, 32, 3], k, block_sizes, 'dense')
pool  = tf.nn.avg_pool(dense, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')  # 4 -> 1
flat  = tf.reshape(pool, [batch_size, nhidden])
out   = fc_block(flat, [nhidden, 10])

####################################

predict = tf.argmax(out, axis=1)
correct = tf.equal(predict, tf.argmax(y, 1))
sum_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=out)
params = tf.trainable_variables()
grads = tf.gradients(loss, params)
grads_and_vars = zip(grads, params)
train = tf.train.AdamOptimizer(learning_rate=1e-2, epsilon=1.).apply_gradients(grads_and_vars)

####################################

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

####################################

for ii in range(epochs):
    for jj in range(0, 50000, batch_size):
        s = jj
        e = jj + batch_size
        xs = x_train[s:e]
        ys = y_train[s:e]
        sess.run([train], feed_dict={x: xs, y: ys})
        
    total_correct = 0

    for jj in range(0, 10000, batch_size):
        s = jj
        e = jj + batch_size
        xs = x_test[s:e]
        ys = y_test[s:e]
        _sum_correct = sess.run(sum_correct, feed_dict={x: xs, y: ys})
        total_correct += _sum_correct

    '''
    param = sess.run(vars_dict, feed_dict={})
    print (param.keys())
    '''
  
    print ("acc: " + str(total_correct * 1.0 / 10000))
        
        
####################################








