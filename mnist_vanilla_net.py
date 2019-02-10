import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def get_data():
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    return (mnist.train.images, mnist.train.labels,
            mnist.test.images, mnist.test.labels)

def vanilla_net():
    X_train, y_train, X_test, y_test = get_data()

    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    N, D = X_train.shape
    batch_sz = 500
    n_batch = N // batch_sz

    X = tf.placeholder(tf.float32, shape=(None, D))
    T = tf.placeholder(tf.float32, shape=(None, 10))

    M1 = 1000
    M2 = 500
    K = 10

    W1 = tf.Variable((np.random.randn(D, M1) / np.sqrt(D + M1)).astype(np.float32))
    b1 = tf.Variable((np.zeros(M1)).astype(np.float32))
    W2 = tf.Variable((np.random.randn(M1, M2) / np.sqrt(M1 + M2)).astype(np.float32))
    b2 = tf.Variable((np.zeros(M2)).astype(np.float32))
    W3 = tf.Variable((np.random.randn(M2, K) / np.sqrt(M2 + K)).astype(np.float32))
    b3 = tf.Variable((np.zeros(K)).astype(np.float32))

    Z1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    Z2 = tf.nn.relu(tf.matmul(Z1, W2) + b2)
    Y = tf.matmul(Z2, W3) + b3

    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=T,logits=Y))

    train_op = tf.train.RMSPropOptimizer(0.0001, decay=0.99, momentum=0.9).minimize(cost)

    predict = tf.argmax(Y, 1)

    LL = []
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(20):
            for j in range(n_batch):
                Xbatch = X_train[j*batch_sz:(j*batch_sz + batch_sz),]
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
    vanilla_net()
