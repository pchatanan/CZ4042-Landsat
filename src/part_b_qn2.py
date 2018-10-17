#
# Project 1: Part B Question 2
#
import pylab as plt

from src.utils import *
import random

NUM_FEATURES = 8

lr_list = [0.5e-6, 1e-7, 0.5e-8, 1e-9, 1e-10]
beta = 1e-3
epochs = 20
batch_size = 32
num_neuron = 30
seed = 7
random.seed(7)
np.random.seed(seed)
tf.set_random_seed(seed)

# read and divide data into test and train sets
cal_housing = np.loadtxt('./raw/cal_housing.data', delimiter=',')
np.random.shuffle(cal_housing)

n_total = cal_housing.shape[0]
split = round(0.7 * n_total)

data_train, data_test = cal_housing[:split, :], cal_housing[split:, :]
n_train = data_train.shape[0]
n_test = data_test.shape[0]

X, D = data_train[:, :-1], data_train[:, -1][:, None]
X_tst, D_tst = data_test[:, :-1], data_test[:, -1][:, None]
mean, std = np.mean(X, axis=0), np.std(X, axis=0)
X = (X - mean) / std
X_tst = (X_tst - mean) / std

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
error = tf.reduce_mean(tf.square(d - y))
loss = error + beta * reg

five_fold_idx = k_fold(n_train, 5)
cv_err_mean = []
cv_err_std = []
for lr in lr_list:
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_op = optimizer.minimize(loss)
    fold_error = []
    for i, (train_idx, test_idx) in enumerate(five_fold_idx):
        print("\nLR: {} : {} Fold:".format(lr, i+1))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            idx = np.arange(split)
            for j in range(epochs):

                # shuffle data set and execute mini batch
                np.random.shuffle(idx)
                X, D = X[idx], D[idx]
                for start, end in zip(range(0, split, batch_size), range(batch_size, split, batch_size)):
                    x_batch, d_batch = X[start:end], D[start:end]
                    train_op.run(feed_dict={x0: x_batch, d: d_batch})

            # cross-validation error
            x_tst = X[test_idx]
            d_tst = D[test_idx]
            err = error.eval(feed_dict={x0: x_tst, d: d_tst})
            fold_error.append(err)

    # keep cross-validation error for lr
    fold_error = np.array(fold_error)
    fold_mean, fold_std = np.mean(fold_error), np.std(fold_error)
    cv_err_mean.append(fold_mean)
    cv_err_std.append(fold_std)

# plot learning curves
for i in range(len(lr_list)):
    print("Lr {}: Cross-Validation Error mean: {}, std: {}".format(lr_list[i], cv_err_mean[i], cv_err_std[i]))
plt.figure(1)
x = range(len(lr_list))
plt.errorbar(x, cv_err_mean, cv_err_std, linestyle='None', marker='.', capsize=3, c='b')
plt.xticks(x, lr_list, rotation='vertical')
plt.xlabel('Learning Rate')
plt.ylabel('Cross-Validation Error')
plt.savefig('partB_Qn2a.png')

epochs = 100
optimal_lr = 1e-07
optimizer = tf.train.GradientDescentOptimizer(optimal_lr)
train_op = optimizer.minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    test_err = []

    idx = np.arange(split)
    for i in range(epochs):

        np.random.shuffle(idx)
        X, D = X[idx], D[idx]
        for start, end in zip(range(0, split, batch_size), range(batch_size, split, batch_size)):
            x_batch, d_batch = X[start:end], D[start:end]
            train_op.run(feed_dict={x0: x_batch, d: d_batch})

        err = error.eval(feed_dict={x0: X_tst, d: D_tst})
        test_err.append(err)

        if i % (epochs // 10) == 0 or i == epochs - 1:
            print('epoch {0:5d}: Test Error: {1:g}'.format(i + 1, test_err[i]))

# plot learning curves
plt.figure(2)
plt.plot(range(epochs), test_err, 'b', label='Test Error')
plt.xlabel('Epochs')
plt.ylabel('Test Error')
plt.legend(loc='best')
plt.savefig('partB_Qn2b.png')

plt.show()
