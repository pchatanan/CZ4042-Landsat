#
# Project 1: Part B Question 3
#
import pylab as plt

from src.utils import *
import random

NUM_FEATURES = 8

lr = 0.5e-6
beta = 1e-3
epochs = 1000
batch_size = 32
num_neuron_list = [20, 40, 60, 80, 100]
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

five_fold_idx = k_fold(n_train, 5)
cv_err_mean = []
cv_err_std = []
for num_neuron in num_neuron_list:

    # Create the model
    x0 = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    d = tf.placeholder(tf.float32, [None, 1])

    # Initialize weights and bias
    W, b = init_weights_bias(NUM_FEATURES, [num_neuron], 1)

    # Build a computational graph
    y = build_graph(x0, W, b, [tf.nn.relu] * len([num_neuron]), None)

    reg = l2_reg(W)

    loss = tf.reduce_mean(tf.square(d - y) + beta * reg)
    error = tf.reduce_mean(tf.square(d - y))

    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_op = optimizer.minimize(loss)

    fold_error = []
    for i, (train_idx, test_idx) in enumerate(five_fold_idx):
        print("\nNo. of hidden neurons: {} : {} Fold:".format(num_neuron, i + 1))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for j in range(epochs):
                # randomly choose batch
                rand_index = random.sample(train_idx, batch_size)
                x_batch = X[rand_index]
                d_batch = D[rand_index]

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
for i in range(len(num_neuron_list)):
    print("No. of hidden neurons {}: Cross-Validation Error mean: {}, std: {}".format(num_neuron_list[i], cv_err_mean[i], cv_err_std[i]))
plt.figure(1)
x = range(len(num_neuron_list))
plt.errorbar(x, cv_err_mean, cv_err_std, linestyle='None', marker='.', capsize=3, c='b')
plt.xticks(x, num_neuron_list, rotation='vertical')
plt.xlabel('Number of hidden neurons')
plt.ylabel('Cross-Validation Error')
plt.savefig('partB_Qn3a.png')


optimal_num_neuron = 60

# Create the model
x0 = tf.placeholder(tf.float32, [None, NUM_FEATURES])
d = tf.placeholder(tf.float32, [None, 1])

# Build the graph for the deep net
W1 = init_weights(optimal_num_neuron, 1)
b1 = init_bias(1)
W0 = init_weights(NUM_FEATURES, optimal_num_neuron)
b0 = init_bias(optimal_num_neuron)

# Build the graph for the deep net
# x0 --W0,b0--> u0 --f0--> x1 --W1,b1--> y
u0 = tf.matmul(x0, W0) + b0
x1 = tf.nn.relu(u0)  # f0
y = tf.matmul(x1, W1) + b1

reg = tf.nn.l2_loss(W0) + tf.nn.l2_loss(W1)
loss = tf.reduce_mean(tf.square(d - y) + beta * reg)
error = tf.reduce_mean(tf.square(d - y))

optimizer = tf.train.GradientDescentOptimizer(lr)
train_op = optimizer.minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    test_err = []
    for i in range(epochs):
        # randomly choose batch
        rand_index = np.random.choice(n_train, size=batch_size)
        x_batch = X[rand_index]
        d_batch = D[rand_index]

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
plt.savefig('partB_Qn3b.png')

plt.show()
