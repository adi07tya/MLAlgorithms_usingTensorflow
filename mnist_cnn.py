import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def get_data():
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    return (mnist.train.images, mnist.train.labels,
            mnist.test.images, mnist.test.labels)

def cnn():
    X_train, y_train, X_test, y_test = get_data()

    X_train = X_train.astype(np.float32)
    X_train = X_train.reshape(-1,28,28,1)
    y_train = y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    X_test = X_test.reshape(-1,28,28,1)
    y_test = y_test.astype(np.float32)

    N, H, W, C = X_train.shape
    batch_sz = 500
    n_batch = N // batch_sz

    X = tf.placeholder(tf.float32, shape=(None, H, W, C))
    T = tf.placeholder(tf.float32, shape=(None, 10))

    W1 = tf.Variable(tf.truncated_normal((6, 6, 1, 32), stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[W1.shape[3]]))
    W2 = tf.Variable(tf.truncated_normal((6, 6, 32, 64), stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[W2.shape[3]]))

    Z1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') + b1)
    Z1 = tf.nn.max_pool(Z1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #print(Z1.shape)
    Z2 = tf.nn.relu(tf.nn.conv2d(Z1, W2, strides=[1, 1, 1, 1], padding='SAME') + b2)
    Z2 = tf.nn.max_pool(Z2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #print(Z2.shape)
    Z3 = tf.reshape(Z2, [-1,7*7*64])
    W3 = tf.Variable(tf.truncated_normal((3136, 1024), stddev=0.1))
    b3 = tf.Variable(tf.constant(0.1, shape=[W3.shape[1]]))
    Y3 = tf.nn.relu(tf.matmul(Z3, W3) + b3)
    W4 = tf.Variable(tf.truncated_normal((1024, 10), stddev=0.1))
    b4 = tf.Variable(tf.constant(0.1, shape=[W4.shape[1]]))
    Y = tf.matmul(Y3, W4) + b4

    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=T,logits=Y))

    train_op = tf.train.RMSPropOptimizer(0.0001, decay=0.99, momentum=0.9).minimize(cost)

    predict = tf.argmax(Y, 1)

    LL = []
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(2):
            for j in range(n_batch):
                Xbatch = X_train[j*batch_sz:(j*batch_sz + batch_sz),]
                #print(Xbatch.shape)
                Ybatch = y_train[j*batch_sz:(j*batch_sz + batch_sz),]

                sess.run(train_op, feed_dict={X:Xbatch, T:Ybatch})
                if j % 10 == 0:
                    test_cost = sess.run(cost, feed_dict={X: X_test, T: y_test})
                    prediction = sess.run(predict, feed_dict={X: X_test})
                    print("Cost at iteration i=%d, j=%d: %.3f" % (i, j, test_cost))
                    LL.append(test_cost)

    plt.plot(LL)
    plt.show()

if __name__ == '__main__':
    cnn()
