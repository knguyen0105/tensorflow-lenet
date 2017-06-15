from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train           = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test             = mnist.test.images, mnist.test.labels

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))


import numpy as np

# Pad images with 0s
X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    
print("Updated Image Shape: {}".format(X_train[0].shape))

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)

import tensorflow as tf

EPOCHS = 10
BATCH_SIZE = 128

from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    filter_size_width = 5
    filter_size_height = 5
    weight = tf.Variable(tf.truncated_normal(
                [filter_size_height, filter_size_width, 1, 6],mean=mu,stddev=sigma))
    bias = tf.Variable(tf.truncated_normal([6],mean=mu,stddev=sigma))
    conv_layer = tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding="VALID")    
    conv_layer = tf.nn.bias_add(conv_layer, bias)
    
    # TODO: Activation.
    conv_layer = tf.nn.relu(conv_layer)    
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv_layer = tf.nn.max_pool(
                conv_layer,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME')
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    weight2 = tf.Variable(tf.truncated_normal(
                [filter_size_height, filter_size_width, 6, 16],mean=mu,stddev=sigma))
    bias2 = tf.Variable(tf.truncated_normal([16],mean=mu,stddev=sigma))
    conv_layer = tf.nn.conv2d(conv_layer, weight2, strides=[1, 1, 1, 1], padding='VALID')
    conv_layer = tf.nn.bias_add(conv_layer, bias2)    
    
    # TODO: Activation.
    conv_layer = tf.nn.relu(conv_layer)    
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv_layer = tf.nn.max_pool(
                conv_layer,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME')
    
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    flatten = tf.contrib.layers.flatten(conv_layer)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    W = tf.Variable(tf.truncated_normal([400, 120],mean=mu,stddev=sigma))
    b = tf.Variable(tf.truncated_normal([120],mean=mu,stddev=sigma))
    hidden_layer = tf.add(tf.matmul(flatten, W), b)
    # TODO: Activation.
    hidden_layer = tf.nn.relu(hidden_layer)
    
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    W1 = tf.Variable(tf.truncated_normal([120, 84],mean=mu,stddev=sigma))
    b1 = tf.Variable(tf.truncated_normal([84],mean=mu,stddev=sigma))
    hidden_layer = tf.add(tf.matmul(hidden_layer, W1), b1)    
    # TODO: Activation.
    hidden_layer = tf.nn.relu(hidden_layer)
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    W2 = tf.Variable(tf.truncated_normal([84, 10],mean=mu,stddev=sigma))
    b2 = tf.Variable(tf.truncated_normal([10],mean=mu,stddev=sigma))
    logits = tf.add(tf.matmul(hidden_layer, W2), b2)    
    
    return logits

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
    
    
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))