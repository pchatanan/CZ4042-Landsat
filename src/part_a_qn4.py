#
# Project 1: Part A Question 4
#
import pylab as plt
from src.utils import *

lr = 0.01
epochs = 50000
batch_size = 16
# L2 weight decay (beta)
beta_list = [0, 1e-3, 1e-6, 1e-9, 1e-12]

seed = 7
np.random.seed(seed)
tf.set_random_seed(seed)

# read train data
X, K = load_data('./raw/sat_train.txt')
X_tst, K_tst = load_data('./raw/sat_test.txt')

num_features = X.shape[1]
num_classes = K.shape[1]
num_hidden = 10
num_data = X.shape[0]

time_list = []
test_set_acc = []
for i, beta in enumerate(beta_list, 1):
    # Create the model
    # input
    x = tf.placeholder(tf.float32, [None, num_features])
    # output
    k = tf.placeholder(tf.float32, [None, num_classes])

    # Define variables:
    V = init_weights(num_hidden, num_classes)
    c = init_bias(num_classes)
    W = init_weights(num_features, num_hidden)
    b = init_bias(num_hidden)

    # Build the graph for the deep net
    # x --W,b--> z --g--> h --V,c--> u --f--> p --> y
    z = tf.matmul(x, W) + b
    h = tf.nn.sigmoid(z)  # g
    u = tf.matmul(h, V) + c
    p = tf.nn.softmax(u)  # f
    y = tf.argmax(p, axis=1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=k, logits=u)
    loss_no_reg = tf.reduce_mean(cross_entropy)
    reg = tf.nn.l2_loss(V) + tf.nn.l2_loss(W)
    loss = tf.reduce_mean(loss_no_reg + beta * reg)

    # Create the gradient descent optimizer with the given learning rate
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_op = optimizer.minimize(loss)

    correct_prediction = tf.cast(tf.equal(tf.argmax(u, 1), tf.argmax(k, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    print("\nBeta: {}\n".format(beta))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_mini_err = []
        train_err = []
        test_acc = []
        for j in range(epochs):

            # randomly choose batch
            rand_index = np.random.choice(num_data, size=batch_size)
            x_batch = X[rand_index]
            k_batch = K[rand_index]

            train_op.run(feed_dict={x: x_batch, k: k_batch})
            train_mini_err.append(1 - accuracy.eval(feed_dict={x: x_batch, k: k_batch}))
            train_err.append(1 - accuracy.eval(feed_dict={x: X, k: K}))
            test_acc.append(accuracy.eval(feed_dict={x: X_tst, k: K_tst}))

            if j % (epochs//10) == 0 or j == epochs - 1:
                print('epoch {0:5d}: Train Err: {1:8.4f} Test Acc: {2:8.4f}'.format(j + 1, train_err[j], test_acc[j]))

        test_set_acc.append(test_acc[-1])

        # plot learning curves
        plt.figure(i)
        plt.plot(range(epochs), train_mini_err, 'g', label='Train(batch) Error')
        plt.plot(range(epochs), train_err, 'b', label='Train(all) Error')
        plt.plot(range(epochs), test_acc, 'r', label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Train Error and Test Accuracy')
        plt.legend(loc='center right')
        plt.savefig('partA_Qn4a_(beta{0}).png'.format(beta))

# Plot for Question 4b
fig = len(beta_list) + 1
x = range(1, len(beta_list) + 1)
plt.figure(fig)
plt.plot(x, test_set_acc, 'b')
plt.xticks(x, beta_list, rotation='vertical')
plt.xlabel('Weight Decay (beta)')
plt.ylabel('Test Accuracy')
plt.savefig('partA_Qn4b.png')

plt.show()
