
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=5e-2)
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

exxact = 0
if exxact:
    val_path = '/home/bcrafton3/Data_SSD/ILSVRC2012/val/'
    train_path = '/home/bcrafton3/Data_SSD/ILSVRC2012/train/'
else:
    val_path = '/usr/scratch/bcrafton/ILSVRC2012/val/'
    train_path = '/usr/scratch/bcrafton/ILSVRC2012/train/'

val_labels = './imagenet_labels/validation_labels.txt'
train_labels = './imagenet_labels/train_labels.txt'

##############################################

import keras
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=1000)

from keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization, AveragePooling2D, Dense, Flatten, ReLU, Softmax
from keras.models import Model

##############################################

IMAGENET_MEAN = [123.68, 116.78, 103.94]

##############################################

def in_top_k(x, y, k):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)

    _, topk = tf.nn.top_k(input=x, k=k)
    topk = tf.transpose(topk)
    correct = tf.equal(y, topk)
    correct = tf.cast(correct, dtype=tf.int32)
    correct = tf.reduce_sum(correct, axis=0)
    return correct

##############################################

# Preprocessing (for both training and validation):
# (1) Decode the image from jpg format
# (2) Resize the image so its smaller side is 256 pixels long
def parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
    image = tf.cast(image_decoded, tf.float32)

    smallest_side = 256.0
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    height = tf.to_float(height)
    width = tf.to_float(width)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)

    resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
    return resized_image, label

# Preprocessing (for training)
# (3) Take a random 224x224 crop to the scaled image
# (4) Horizontally flip the image with probability 1/2
# (5) Substract the per color mean `IMAGENET_MEAN`
# Note: we don't normalize the data here, as VGG was trained without normalization
def train_preprocess(image, label):
    crop_image = tf.random_crop(image, [224, 224, 3])                       # (3)
    flip_image = tf.image.random_flip_left_right(crop_image)                # (4)

    means = tf.reshape(tf.constant(IMAGENET_MEAN), [1, 1, 3])
    centered_image = flip_image - means                                     # (5)

    return centered_image, label
    

# Preprocessing (for validation)
# (3) Take a central 224x224 crop to the scaled image
# (4) Substract the per color mean `IMAGENET_MEAN`
# Note: we don't normalize the data here, as VGG was trained without normalization
def val_preprocess(image, label):
    crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)    # (3)

    means = tf.reshape(tf.constant(IMAGENET_MEAN), [1, 1, 3])
    centered_image = crop_image - means                                     # (4)

    return centered_image, label

##############################################

def get_validation_dataset():
    label_counter = 0
    validation_images = []
    validation_labels = []

    print ("building validation dataset")

    for subdir, dirs, files in os.walk(val_path):
        for file in files:
            validation_images.append(os.path.join(val_path, file))
    validation_images = sorted(validation_images)

    validation_labels_file = open(val_labels)
    lines = validation_labels_file.readlines()
    for ii in range(len(lines)):
        validation_labels.append(int(lines[ii]))

    remainder = len(validation_labels) % args.batch_size
    validation_images = validation_images[:(-remainder)]
    validation_labels = validation_labels[:(-remainder)]

    return validation_images, validation_labels
    
def get_train_dataset():

    label_counter = 0
    training_images = []
    training_labels = []

    f = open(train_labels, 'r')
    lines = f.readlines()

    labels = {}
    for line in lines:
        line = line.split(' ')
        labels[line[0]] = label_counter
        label_counter += 1

    f.close()

    print ("building train dataset")

    for subdir, dirs, files in os.walk(train_path):
        for folder in dirs:
            for folder_subdir, folder_dirs, folder_files in os.walk(os.path.join(subdir, folder)):
                for file in folder_files:
                    training_images.append(os.path.join(folder_subdir, file))
                    training_labels.append(labels[folder])

    remainder = len(training_labels) % args.batch_size
    training_images = training_images[:(-remainder)]
    training_labels = training_labels[:(-remainder)]

    return training_images, training_labels

###############################################################

filename = tf.placeholder(tf.string, shape=[None])
label = tf.placeholder(tf.int64, shape=[None])

###############################################################

val_imgs, val_labs = get_validation_dataset()

val_dataset = tf.data.Dataset.from_tensor_slices((filename, label))
val_dataset = val_dataset.shuffle(len(val_imgs))
val_dataset = val_dataset.map(parse_function, num_parallel_calls=4)
val_dataset = val_dataset.map(val_preprocess, num_parallel_calls=4)
val_dataset = val_dataset.batch(args.batch_size)
val_dataset = val_dataset.repeat()
val_dataset = val_dataset.prefetch(2)

###############################################################

train_imgs, train_labs = get_train_dataset()

train_dataset = tf.data.Dataset.from_tensor_slices((filename, label))
train_dataset = train_dataset.shuffle(len(train_imgs))
train_dataset = train_dataset.map(parse_function, num_parallel_calls=4)
train_dataset = train_dataset.map(train_preprocess, num_parallel_calls=4)
train_dataset = train_dataset.batch(args.batch_size)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.prefetch(2)

###############################################################

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
features, labels = iterator.get_next()
features = tf.reshape(features, (-1, 224, 224, 3))
labels = tf.one_hot(labels, depth=1000)

train_iterator = train_dataset.make_initializable_iterator()
val_iterator = val_dataset.make_initializable_iterator()

###############################################################

# keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, ...)
# keras.layers.DepthwiseConv2D(kernel_size, strides=(1, 1), padding='valid', depth_multiplier=1, data_format=None, activation=None, use_bias=True, ...)
# keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, ...)
# keras.layers.Dense(units, activation=None, use_bias=True, ...)
# keras.layers.Flatten(data_format=None)
# keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
# keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)
# keras.layers.Softmax(axis=-1)

def block(x, f, s):
    conv1 = Conv2D(filters=f, kernel_size=[3, 3], strides=[s, s], padding='same', use_bias=False)(x)
    bn1   = BatchNormalization()(conv1)
    relu1 = ReLU()(bn1)

    return relu1

def mobile_block(x, f1, f2, s):
    conv1 = DepthwiseConv2D(   kernel_size=[3, 3], strides=[s, s], padding='same', use_bias=False)(x)
    bn1   = BatchNormalization()(conv1)
    relu1 = ReLU()(bn1)

    conv2 = Conv2D(filters=f2, kernel_size=[1, 1], strides=[1, 1], padding='same', use_bias=False)(relu1)
    bn2   = BatchNormalization()(conv2)
    relu2 = ReLU()(bn2)

    return relu2

def create_model(input_shape, classes)
    inputs = Input(shape=input_shape)

    bn     = BatchNormalization()(inputs)          # 224
    block1 = block(bn,              32,         2) # 224

    block2 = mobile_block(block1,   32,   64,   1) # 112
    block3 = mobile_block(block2,   64,   128,  2) # 112

    block4 = mobile_block(block3,   128,  128,  1) # 56
    block5 = mobile_block(block4,   128,  256,  2) # 56

    block6 = mobile_block(block5,   256,  256,  1) # 28
    block7 = mobile_block(block6,   256,  512,  2) # 28

    block8  = mobile_block(block7,  512,  512,  1) # 14
    block9  = mobile_block(block8,  512,  512,  1) # 14
    block10 = mobile_block(block9,  512,  512,  1) # 14
    block11 = mobile_block(block10, 512,  512,  1) # 14
    block12 = mobile_block(block11, 512,  512,  1) # 14

    block13 = mobile_block(block12, 512,  1024, 2) # 14
    block14 = mobile_block(block13, 1024, 1024, 1) # 7

    pool    = AveragePooling2D(pool_size=[7,7], padding='same')(block14)
    flat    = Flatten()(pool)
    fc1     = Dense(units=classes)(flat)

    model = Model(inputs=inputs, outputs=fc1)
    return model

###############################################################

batch_size = tf.placeholder(tf.int32, shape=())
lr = tf.placeholder(tf.float32, shape=())

model = create_model(input_shape=[batch_size, 224, 224, 3], classes=1000)
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1.0), metrics=['accuracy'])

###############################################################

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc1, labels=labels))
correct = tf.equal(tf.argmax(predict, axis=1), tf.argmax(labels, 1))
total_correct = tf.reduce_sum(tf.cast(correct, tf.float32))
train = tf.train.AdamOptimizer(learning_rate=lr, epsilon=args.eps).minimize(loss)

params = tf.trainable_variables()

###############################################################

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

train_handle = sess.run(train_iterator.string_handle())
val_handle = sess.run(val_iterator.string_handle())

###############################################################


    
    
    

