import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

def get_data():
    x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
    y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
    return x_data.astype(np.float32), y_label.astype(np.float32)

def save_model():
    X, y = get_data()

    m = tf.Variable(0.39)
    b = tf.Variable(0.2)

    y_hat = tf.add(tf.multiply(X,m),b)

    cost = tf.reduce_sum(tf.square(y - y_hat))

    train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

    LL = []
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(2000):
            sess.run(train)
            costing = sess.run(cost)
            LL.append(costing)

        print(sess.run([m, b]))
        saver.save(sess,'models/simple_model.ckpt')

    plt.plot(LL)
    plt.show()

def load_model():
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,'models/simple_model.ckpt')

        print(sess.run([m,b]))

if __name__ == '__main__':
    save_model()
    load_model()
