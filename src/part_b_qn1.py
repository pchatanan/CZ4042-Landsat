#
# Project 1: Part B Question 1
#
import pylab as plt

from src.utils import *

NUM_FEATURES = 8

lr = 1e-7
beta = 1e-3
epochs = 10000
batch_size = 32
num_neuron = 30
seed = 7
np.random.seed(seed)
tf.set_random_seed(seed)

# read and divide data into test and train sets
cal_housing = np.loadtxt('./raw/cal_housing.data', delimiter=',')
np.random.shuffle(cal_housing)

n_total = cal_housing.shape[0]
split = round(0.7*n_total)

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
loss = tf.reduce_mean(tf.square(d - y) + beta * reg)

# Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(lr)
train_op = optimizer.minimize(loss)

error = tf.reduce_mean(tf.square(d - y))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_err = []
    for i in range(epochs):

        # randomly choose batch
        rand_index = np.random.choice(n_train, size=batch_size)
        x_batch = X[rand_index]
        d_batch = D[rand_index]

        train_op.run(feed_dict={x0: x_batch, d: d_batch})
        err = error.eval(feed_dict={x0: X, d: D})
        train_err.append(err)

        if i % (epochs // 10) == 0 or i == epochs - 1:
            print('epoch {0:5d}: Validation Error: {1:g}'.format(i + 1, train_err[i]))

    # randomly choose 50 samples
    sample_index = np.random.choice(n_test, size=50)
    x_tst_sample = X_tst[sample_index]
    d_tst_sample = D_tst[sample_index]
    y_tst_sample = y.eval(feed_dict={x0: x_tst_sample})

    # plot learning curves
    plt.figure(1)
    plt.plot(range(epochs), train_err, 'b', label='Validation Error')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Error')
    plt.legend(loc='upper right')
    plt.savefig('partB_Qn1a.png')


    plt.figure(2)
    max_value = max([d_tst_sample.max(), y_tst_sample.max()])
    plt.plot([0, max_value], [0, max_value], 'r', label='Zero Loss')
    plt.scatter(d_tst_sample, y_tst_sample, c='b')
    plt.xlabel('Target')
    plt.ylabel('Prediction')
    plt.legend(loc='best')
    plt.savefig('partB_Qn1b.png')

    plt.show()
