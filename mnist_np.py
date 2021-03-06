
import numpy as np
import argparse
import keras

#######################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-2)
args = parser.parse_args()

LAYER1 = 784
LAYER2 = 100
LAYER3 = 10

TRAIN_EXAMPLES = 60000
TEST_EXAMPLES = 10000

#######################################

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def sigmoid(x):
  return 1. / (1. + np.exp(-x))
  
def dsigmoid(x):
  return x * (1. - x)

def relu(x):
  return x * (x > 0)
  
def drelu(x):
  # USE A NOT Z
  return 1.0 * (x > 0)

#######################################

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

y_train = keras.utils.to_categorical(y_train, 10)
x_train = x_train.reshape(TRAIN_EXAMPLES, 784)
x_train = x_train.astype('float32')
x_train = x_train / np.max(x_train)

y_test = keras.utils.to_categorical(y_test, 10)
x_test = x_test.reshape(TEST_EXAMPLES, 784)
x_test = x_test.astype('float32')
x_test = x_test / np.max(x_test)

#######################################

high = 1. / np.sqrt(LAYER1)
weights1 = np.random.uniform(low=-high, high=high, size=(LAYER1, LAYER2))

high = 1. / np.sqrt(LAYER2)
weights2 = np.random.uniform(low=-high, high=high, size=(LAYER2, LAYER3))

#######################################

for epoch in range(args.epochs):
    
    for ex in range(0, TRAIN_EXAMPLES, args.batch_size):
        start = ex 
        stop = ex + args.batch_size
    
        A1 = x_train[start:stop]
        Z2 = np.dot(A1, weights1)
        A2 = relu(Z2)
        Z3 = np.dot(A2, weights2)
        A3 = softmax(Z3)
        
        labels = y_train[start:stop]
        
        D3 = A3 - labels
        D2 = np.dot(D3, np.transpose(weights2)) * drelu(A2)
        
        DW2 = np.dot(np.transpose(A2), D3)
        DW1 = np.dot(np.transpose(A1), D2)
        
        weights2 = weights2 - args.lr * DW2
        weights1 = weights1 - args.lr * DW1
        
    correct = 0
    for ex in range(0, TEST_EXAMPLES, args.batch_size):
        start = ex 
        stop = ex + args.batch_size
    
        A1 = x_test[start:stop]
        Z2 = np.dot(A1, weights1)
        A2 = relu(Z2)
        Z3 = np.dot(A2, weights2)
        A3 = softmax(Z3)
        
        labels = y_test[start:stop]
        
        correct += np.sum(np.argmax(A3, axis=1) == np.argmax(labels, axis=1))
            
    test_acc = 1. * correct / TEST_EXAMPLES

    print ("epoch: %d/%d | test acc: %f" % (epoch, args.epochs, test_acc))

    
    
    
    
    
    
