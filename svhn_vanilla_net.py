import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.utils import shuffle

def get_data():
    train = loadmat('train_32x32.mat')
    test = loadmat('test_32x32.mat')

    return train, test

def one_hot(Y):
    N = len(Y)
    ind = np.zeros((N,10))
    for i in range(N):
        ind[i,Y[i]] = 1
    return ind

def error_rate(p, t):
    return np.mean(p != t)

def flatten(X):
    N = X.shape[-1]
    x = np.zeros((N, 3072))
    for i in range(N):
        x[i] = X[:,:,:,i].reshape(3072)
    return x

def vanilla_net():
    train, test = get_data()

    X_train = flatten(train['X'].astype(np.float32) / 255.)
    Y_train = train['y'].flatten() - 1
    X_train, Y_train = shuffle(X_train, Y_train)

    X_test = flatten(test['X'].astype(np.float32) / 255.)
    Y_test = test['y'].flatten() - 1

    #Y_train = one_hot(Y_train)
    #Y_test = one_hot(Y_test)

    N, D = X_train.shape
    batch_sz = 500
    n_batch = N // batch_sz

    X = tf.placeholder(tf.float32, shape=(None,D), name='X')
    T = tf.placeholder(tf.int32, shape=(None,), name='T')

    M1 = 1000
    M2 = 500
    K = 10

    W1 = tf.Variable((np.random.randn(D, M1) / np.sqrt(D + M1)).astype(np.float32))
    b1 = tf.Variable((np.zeros(M1)).astype(np.float32))
    W2 = tf.Variable((np.random.randn(M1, M2) / np.sqrt(M1 + M2)).astype(np.float32))
    b2 = tf.Variable((np.zeros(M2)).astype(np.float32))
    W3 = tf.Variable((np.random.randn(M2, K) / np.sqrt(M2 + K)).astype(np.float32))
    b3 = tf.Variable((np.zeros(K)).astype(np.float32))

    Z1 = tf.nn.relu(tf.matmul(X,W1) + b1)
    Z2 = tf.nn.relu(tf.matmul(Z1,W2) + b2)
    Y = tf.matmul(Z2,W3) + b3

    cost = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Y, labels=T))

    train_op = tf.train.RMSPropOptimizer(0.0001, decay=0.99, momentum=0.9).minimize(cost)

    predict = tf.argmax(Y, 1)

    LL = []
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(20):
            for j in range(n_batch):
                Xbatch = X_train[j*batch_sz:(j*batch_sz + batch_sz),]
                Ybatch = Y_train[j*batch_sz:(j*batch_sz + batch_sz),]

                sess.run(train_op, feed_dict={X:Xbatch, T:Ybatch})
                if j % 10 == 0:
                    test_cost = sess.run(cost, feed_dict={X: X_test, T: Y_test})
                    prediction = sess.run(predict, feed_dict={X: X_test})
                    err = error_rate(prediction, Y_test)
                    print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
                    LL.append(test_cost)

    plt.plot(LL)
    plt.show()

if __name__ == '__main__':
    vanilla_net()
