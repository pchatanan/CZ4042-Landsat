#
# Project 1: Part B Question 1
#
import pylab as plt

from src.utils import *

NUM_FEATURES = 8

lr = 1e-9
beta = 1e-3
epochs = 20000
batch_size = 32
num_neuron_list = [[60], [60, 20], [60, 20, 20]]
keep_prob_list = [None, 0.9]
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

figure = 1

for num_neuron in num_neuron_list:
    for keep_prob in keep_prob_list:
        print("\n{}-Layer: dropout: {}".format(len(num_neuron)+2, str(keep_prob)))
        # Build the graph for the deep net
        W, b = init_weights_bias(NUM_FEATURES, num_neuron, 1)
        # Build the graph for the deep net
        # x0 --W0,b0--> u0 --f0--> x1 --W1,b1--> u1 --f1--> x2 --W2,b2--> y
        y = build_graph(x0, W, b, [tf.nn.relu] * len(num_neuron), keep_prob)

        reg = l2_reg(W)
        loss = tf.reduce_mean(tf.square(d - y) + beta * reg)

        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train_op = optimizer.minimize(loss)

        error = tf.reduce_mean(tf.square(d - y))

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
            plt.figure(figure)
            figure += 1
            plt.plot(range(epochs), test_err, 'b', label='Test Error')
            plt.xlabel('Epochs')
            plt.ylabel('Test Error')
            plt.legend(loc='best')
            plt.savefig('partB_Qn4({}-Layer drop-{}).png'.format(len(num_neuron)+2, str(keep_prob)))

plt.show()
