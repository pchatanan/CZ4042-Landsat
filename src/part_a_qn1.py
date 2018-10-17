#
# Project 1: Part A Question 1
#
import pylab as plt
from src.utils import *

lr = 0.01
epochs = 10
batch_size = 32
# L2 weight decay (beta)
beta = 1e-6

seed = 7
np.random.seed(seed)
tf.set_random_seed(seed)

# read train and test data
X, K = load_data('./raw/sat_train.txt')
X_tst, K_tst = load_data('./raw/sat_test.txt')
min_feature = np.min(X, axis=0)
max_feature = np.max(X, axis=0)
X = scale(X, min_feature, max_feature)
X_tst = scale(X_tst, min_feature, max_feature)

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
loss = loss_no_reg + beta * reg

# Create the gradient descent optimizer with the given learning rate
optimizer = tf.train.GradientDescentOptimizer(lr)
train_op = optimizer.minimize(loss)

correct_prediction = tf.cast(tf.equal(tf.argmax(u, 1), tf.argmax(k, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # train_mini_acc = []
    train_err = []
    test_acc = []

    idx = np.arange(num_data)
    for i in range(epochs):

        # shuffle data set and execute mini batch
        np.random.shuffle(idx)
        X, K = X[idx], K[idx]
        for start, end in zip(range(0, num_data, batch_size), range(batch_size, num_data, batch_size)):
            x_batch, k_batch = X[start:end], K[start:end]
            train_op.run(feed_dict={x: x_batch, k: k_batch})

        # train_mini_acc.append(accuracy.eval(feed_dict={x: x_batch, k: k_batch}))
        train_err.append(1-accuracy.eval(feed_dict={x: X, k: K}))
        test_acc.append(accuracy.eval(feed_dict={x: X_tst, k: K_tst}))

        if i % (epochs//10) == 0 or i == epochs - 1:
            print('epoch {0:5d}: Train Err: {1:8.4f} Test Acc: {2:8.4f}'.format(i + 1, train_err[i], test_acc[i]))

# plot learning curves
plt.figure(1)
plt.plot(range(epochs), train_err, 'b', label='Train Error')
plt.plot(range(epochs), test_acc, 'r', label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig('partA_Qn1.png')
plt.show()
