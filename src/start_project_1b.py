#
# Project 1, starter code part b
#

import tensorflow as tf
import numpy as np
import pylab as plt

from src.utils import *

NUM_FEATURES = 8

lr = 1e-7
beta = 1e-3
epochs = 5000
batch_size = 32
num_neuron = 30
seed = 7
np.random.seed(seed)

# read and divide data into test and train sets
cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
np.random.shuffle(cal_housing)

n_total = cal_housing.shape[0]
split = round(0.7*n_total)

data_train, data_test = cal_housing[:split, :], cal_housing[split:, :]
n_train = data_train.shape[0]
n_test = data_test.shape[0]

X, D = data_train[:, :-1], data_train[:, -1][:, None]
X_tst, D_tst = data_test[:, :-1], data_test[:, -1][:, None]
mean, std = np.mean(X, axis=0), np.std(X, axis=0)
X = (X - mean) / mean

# Create the model
x0 = tf.placeholder(tf.float32, [None, NUM_FEATURES])
d = tf.placeholder(tf.float32, [None, 1])

# Build the graph for the deep net
W1 = init_weights(num_neuron, 1)
b1 = init_bias(1)
W0 = init_weights(NUM_FEATURES, num_neuron)
b0 = init_bias(num_neuron)

# Build the graph for the deep net
# x0 --W0,b0--> u0 --f0--> x1 --W1,b1--> y
u0 = tf.matmul(x0, W0) + b0
x1 = tf.nn.relu(u0)  # f0
y = tf.matmul(x1, W1) + b1
reg = tf.nn.l2_loss(W0) + tf.nn.l2_loss(W1)
loss = tf.reduce_mean(tf.square(d - y) + beta * reg)

# Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(lr)
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_err = []
    for i in range(epochs):

        # randomly choose batch
        rand_index = np.random.choice(n_train, size=batch_size)
        x_batch = X[rand_index]
        d_batch = D[rand_index]

        train_op.run(feed_dict={x0: x_batch, d: d_batch})
        err = loss.eval(feed_dict={x0: X, d: D})
        train_err.append(err)

        if i % 500 == 0:
            print('iter %d: test error %g' % (i, train_err[i]))

# plot learning curves
plt.figure(1)
plt.plot(range(epochs), train_err)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Train Error')
plt.show()
