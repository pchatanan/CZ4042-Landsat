#
# Project 1: Part A Question 2
#
import pylab as plt
import time
from src.utils import *

lr = 0.01
epochs = 50000
batch_size_list = [4, 8, 16, 32, 64]
# L2 weight decay (beta)
beta = 1e-6

seed = 7
np.random.seed(seed)
tf.set_random_seed(seed)

# read train data
X, K = load_data('./raw/sat_train.txt')
X_tst, K_tst = load_data('./raw/sat_train.txt')

num_features = X.shape[1]
num_classes = K.shape[1]
num_hidden = 10
num_data = X.shape[0]

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

time_list = []
for i, batch_size in enumerate(batch_size_list, 1):
    print("\nBatch size: {}\n".format(batch_size))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_err = []
        test_acc = []
        start_time = time.time()
        for j in range(epochs):

            # randomly choose batch
            rand_index = np.random.choice(num_data, size=batch_size)
            x_batch = X[rand_index]
            k_batch = K[rand_index]

            train_op.run(feed_dict={x: x_batch, k: k_batch})
            train_err.append(1 - accuracy.eval(feed_dict={x: x_batch, k: k_batch}))
            test_acc.append(accuracy.eval(feed_dict={x: X_tst, k: K_tst}))

            if j % (epochs//10) == 0 or j == epochs - 1:
                print('epoch {0:5d}: Train Err: {1:8.4f} Test Acc: {2:8.4f}'.format(j + 1, train_err[j], test_acc[j]))

        end_time = time.time()
        time_taken = (end_time-start_time)/epochs
        time_list.append(time_taken)
        print("Time taken per epoch(sec): {0:10.6f} sec".format(time_taken))

    # plot learning curves
    plt.figure(i)
    plt.plot(range(epochs), train_err, 'b', label='Train Error')
    plt.plot(range(epochs), test_acc, 'r', label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Train Error and Test Accuracy')
    plt.legend(loc='center right')
    plt.savefig('partA_Qn2a_(batch{0}).png'.format(batch_size))

# Plot for Question 2b
fig = len(batch_size_list) + 1
plt.figure(fig)
plt.plot(batch_size_list, time_list, 'b')
plt.xlabel('Batch Size')
plt.ylabel('Time taken per epoch (sec)')
plt.savefig('partA_Qn2b.png')

plt.show()
