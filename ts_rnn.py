import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def next_batch(batch_size, steps):
    xmin = 0
    xmax = 10
    resolution = (xmax - xmin)/250
    rand_start = np.random.rand(batch_size,1)
    ts_start = rand_start * (xmax- xmin - (steps*resolution))
    x_batch = ts_start + np.arange(0.0,steps+1) * resolution
    y_batch = np.sin(x_batch)
    return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)

def rnn():
    num_inputs = 1
    num_neurons = 100
    num_outputs = 1
    learning_rate = 0.0001
    batch_size = 1
    num_time_steps = 30

    X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
    y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

    #forward
    cell = tf.contrib.rnn.OutputProjectionWrapper(
        tf.contrib.rnn.BasicRNNCell(num_units=num_neurons, activation=tf.nn.relu),
        output_size=num_outputs)

    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    #compute loss
    loss = tf.reduce_mean(tf.square(outputs - y))

    #optimize
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(2000):
            X_batch, y_batch = next_batch(batch_size, num_time_steps)
            sess.run(train, feed_dict={X: X_batch, y: y_batch})

            if i % 100 == 0:
                mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                print(i, "\tMSE:", mse)

        X_new, y_new = next_batch(batch_size, num_time_steps)
        y_pred = sess.run(outputs, feed_dict={X: X_new})

        print(y_new.reshape(30))
        print(y_pred.reshape(30))


if __name__ == '__main__':
    rnn()
