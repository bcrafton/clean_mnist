
import numpy as np
import tensorflow as tf
import keras

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

f0 = 3
f1 = 64
f2 = 96
f3 = 128

w1 = tf.get_variable("w1", [3,3,f0,f1], dtype=tf.float32)
w2 = tf.get_variable("w2", [3,3,f1,f2], dtype=tf.float32)
w3 = tf.get_variable("w3", [3,3,f2,f3], dtype=tf.float32)

fc1 = tf.get_variable("fc1", [4*4*f3,10], dtype=tf.float32)
fc1_bias = tf.get_variable("fc1_bias", [10], dtype=tf.float32)

####################################

conv1 = tf.nn.conv2d(x, w1, [1,1,1,1], 'SAME')
bn1   = tf.layers.batch_normalization(inputs=conv1)
relu1 = tf.nn.relu(bn1)
pool1 = tf.nn.avg_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv2 = tf.nn.conv2d(pool1, w2, [1,1,1,1], 'SAME')
bn2   = tf.layers.batch_normalization(inputs=conv2)
relu2 = tf.nn.relu(bn2)
pool2 = tf.nn.avg_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv3 = tf.nn.conv2d(pool2, w3, [1,1,1,1], 'SAME')
bn3   = tf.layers.batch_normalization(inputs=conv3)
relu3 = tf.nn.relu(bn3)
pool3 = tf.nn.avg_pool(relu3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

flat = tf.reshape(pool3, [-1, 4*4*f3])
fc1 = tf.matmul(flat, fc1) + fc1_bias

####################################

predict = tf.argmax(fc1, axis=1)
correct = tf.equal(predict, tf.argmax(y, 1))
sum_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=fc1)
params = tf.trainable_variables()
grads = tf.gradients(loss, params)
grads_and_vars = zip(grads, params)
train = tf.train.AdamOptimizer(learning_rate=1e-2, epsilon=1.).apply_gradients(grads_and_vars)

####################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

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
            
    print ("acc: " + str(total_correct * 1.0 / 10000))
        
        
