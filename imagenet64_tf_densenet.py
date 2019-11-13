
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--gpu', type=int, default=2)
parser.add_argument('--blocks', type=int, default=4)
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

exxact = 0
if exxact:
    val_path = '/home/bcrafton3/Data_SSD/64x64/tfrecord/val/'
    train_path = '/home/bcrafton3/Data_SSD/64x64/tfrecord/train/'
else:
    val_path = '/usr/scratch/bcrafton/64x64/tfrecord/val/'
    train_path = '/usr/scratch/bcrafton/64x64/tfrecord/train/'

##############################################

import keras
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=1000)

from bc_utils.init_tensor import init_filters
from bc_utils.init_tensor import init_matrix

MEAN = [122.77093945, 116.74601272, 104.09373519]

####################################

def parse_function(filename, label):
    conv = tf.read_file(filename)
    return conv, label

def get_val_filenames():
    val_filenames = []

    print ("building validation dataset")

    for subdir, dirs, files in os.walk(val_path):
        for file in files:
            val_filenames.append(os.path.join(val_path, file))

    np.random.shuffle(val_filenames)

    remainder = len(val_filenames) % args.batch_size
    val_filenames = val_filenames[:(-remainder)]

    return val_filenames

def get_train_filenames():
    train_filenames = []

    print ("building training dataset")

    for subdir, dirs, files in os.walk(train_path):
        for file in files:
            train_filenames.append(os.path.join(train_path, file))

    np.random.shuffle(train_filenames)

    remainder = len(train_filenames) % args.batch_size
    train_filenames = train_filenames[:(-remainder)]

    return train_filenames

def extract_fn(record):
    _feature={
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)
    }
    sample = tf.parse_single_example(record, _feature)
    image = tf.decode_raw(sample['image_raw'], tf.uint8)
    # this was tricky ... stored as uint8, not float32.
    image = tf.cast(image, dtype=tf.float32)
    image = tf.reshape(image, (1, 64, 64, 3))

    means = tf.reshape(tf.constant(MEAN), [1, 1, 1, 3])
    image = (image - means) / 255. * 2.

    label = sample['label']
    return [image, label]

####################################

train_filenames = get_train_filenames()
val_filenames = get_val_filenames()

filename = tf.placeholder(tf.string, shape=[None])

###############################################################

val_dataset = tf.data.TFRecordDataset(filename)
val_dataset = val_dataset.map(extract_fn, num_parallel_calls=4)
val_dataset = val_dataset.batch(args.batch_size)
val_dataset = val_dataset.repeat()
val_dataset = val_dataset.prefetch(8)

###############################################################

train_dataset = tf.data.TFRecordDataset(filename)
train_dataset = train_dataset.map(extract_fn, num_parallel_calls=4)
train_dataset = train_dataset.batch(args.batch_size)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.prefetch(8)

###############################################################

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
features, labels = iterator.get_next()
features = tf.reshape(features, (args.batch_size, 64, 64, 3))
labels = tf.one_hot(labels, depth=1000)

train_iterator = train_dataset.make_initializable_iterator()
val_iterator = val_dataset.make_initializable_iterator()

####################################

def batch_norm(x, f, name, vars_dict):

    gamma_name = name + '_gamma'
    beta_name = name + '_beta'

    if gamma_name in vars_dict.keys():
        gamma = tf.Variable(vars_dict[gamma_name], trainable=False, dtype=tf.float32)
    else:
        gamma = tf.Variable(np.ones(shape=f), dtype=tf.float32)

    if beta_name in vars_dict.keys():
        beta = tf.Variable(vars_dict[beta_name], trainable=False, dtype=tf.float32)
    else:
        beta = tf.Variable(np.zeros(shape=f), dtype=tf.float32)

    vars_dict[gamma_name] = gamma
    vars_dict[beta_name] = beta

    ########################################

    mean = tf.reduce_mean(x, axis=[0,1,2])
    _, var = tf.nn.moments(x - mean, axes=[0,1,2])
    bn = tf.nn.batch_normalization(x=x, mean=mean, variance=var, offset=beta, scale=gamma, variance_epsilon=1e-3)
    return bn

def conv_block(x, f, name, vars_dict):

    if name in vars_dict.keys():
        filters = tf.Variable(vars_dict[name], trainable=False, dtype=tf.float32)
    else:
        filters = tf.Variable(init_filters(size=f, init='alexnet'), dtype=tf.float32)

    vars_dict[name] = filters

    ##################################

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

def dense_model(x, xshape, k, block_sizes, name, vars_dict):
    _, _, _, c = xshape

    blocks = [x]
    for ii in range(len(block_sizes)):
        block_size = block_sizes[ii]
        for jj in range(block_size):
            nfmap = c + k * (sum(block_sizes[0:ii]) + jj)
            block = dense_block(concat(blocks), nfmap, k, name + '_block_%d_%d' % (ii, jj), vars_dict)
            blocks.append(block)

        pool = tf.nn.avg_pool(concat(blocks), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        blocks = [pool]

    return blocks[-1]

####################################

block_sizes = [0, 0, 0, 0, 0]
for ii in range(args.blocks):
    block_sizes[ii] = 8

k = 32
nhidden = 3 + k * sum(block_sizes)

try:
    vars_dict = np.load('cifar10_densenet.npy', allow_pickle=True).item()
    print ('loaded weights: %d' % (len(vars_dict.keys())))
except:
    vars_dict = {}
    print ('no weights found!')

dense = dense_model(features, [args.batch_size, 32, 32, 3], k, block_sizes, 'dense', vars_dict)
pool  = tf.nn.avg_pool(dense, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # 4 -> 1
flat  = tf.reshape(pool, [args.batch_size, nhidden])
out   = fc_block(flat, [nhidden, 1000])

####################################

predict = tf.argmax(out, axis=1)
tf_correct = tf.reduce_sum(tf.cast(tf.equal(predict, tf.argmax(labels, 1)), tf.float32))

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=out)
params = tf.trainable_variables()
grads = tf.gradients(loss, params)
grads_and_vars = zip(grads, params)
train = tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1.).apply_gradients(grads_and_vars)

####################################

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

train_handle = sess.run(train_iterator.string_handle())
val_handle = sess.run(val_iterator.string_handle())

###############################################################

for ii in range(0, args.epochs):

    print('epoch %d/%d' % (ii, args.epochs))

    ##################################################################

    sess.run(train_iterator.initializer, feed_dict={filename: train_filenames})

    train_total = 0.0
    train_correct = 0.0
    train_acc = 0.0

    for jj in range(0, len(train_filenames), args.batch_size):

        [np_correct, _] = sess.run([tf_correct, train], feed_dict={handle: train_handle})

        train_total += args.batch_size
        train_correct += np_correct
        train_acc = train_correct / train_total

        if (jj % (100 * args.batch_size) == 0):
            p = "train accuracy: %f" % (train_acc)
            print (p)

    ##################################################################

    sess.run(val_iterator.initializer, feed_dict={filename: val_filenames})

    val_total = 0.0
    val_correct = 0.0
    val_acc = 0.0

    for jj in range(0, len(val_filenames), args.batch_size):

        [np_correct] = sess.run([tf_correct], feed_dict={handle: val_handle})

        val_total += args.batch_size
        val_correct += np_correct
        val_acc = val_correct / val_total

        if (jj % (100 * args.batch_size) == 0):
            p = "val accuracy: %f" % (val_acc)
            print (p)

####################################
        
weights = sess.run(vars_dict, feed_dict={})
np.save('imagenet64_densenet', weights)
        
####################################








