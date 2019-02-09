import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def get_data():
    x_data = np.linspace(0.0, 10.0, 10000)
    noise = np.random.randn(len(x_data))
    y_true = (0.5 * x_data) + 5 + noise
    return x_data, y_true

def linear_reg():
    X, y = get_data()

    X_train = X[:7000]
    y_train = y[:7000]
    X_test = X[7000:]
    y_test = y[7000:]

    X = tf.placeholder(tf.float32)
    T = tf.placeholder(tf.float32)

    m = tf.Variable(0.5)
    b = tf.Variable(1.0)

    y_hat = tf.add(tf.multiply(X,m), b)

    cost = tf.reduce_sum(tf.square(T - y_hat))

    train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

    LL = []
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(2000):
            rand_ind = np.random.randint(len(X_train),size=32)
            feed = {X:X_train[rand_ind],T:y_train[rand_ind]}

            sess.run(train, feed_dict=feed)
            if i%10 == 0:
                costing = sess.run(cost,feed_dict={X:X_test,T:y_test})
                LL.append(costing)
        print(sess.run([m,b]))
    plt.plot(LL)
    plt.show()

if __name__ == '__main__':
    linear_reg()
