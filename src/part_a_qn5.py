#
# Project 1: Part A Question 5
#
import pylab as plt
from src.utils import *

lr = 0.01
epochs = 200000
batch_size = 32
# L2 weight decay (beta)
beta = 1e-6

seed = 7
np.random.seed(seed)
tf.set_random_seed(seed)

# read train and test data
X, K = load_data('./raw/sat_train.txt')
X_tst, K_tst = load_data('./raw/sat_train.txt')

num_features = X.shape[1]
num_classes = K.shape[1]
num_hidden = [10, 10]
num_data = X.shape[0]

# Create the model
# input
x0 = tf.placeholder(tf.float32, [None, num_features])
# output
k = tf.placeholder(tf.float32, [None, num_classes])

# Define variables:
W2 = init_weights(num_hidden[1], num_classes)
b2 = init_bias(num_classes)
W1 = init_weights(num_hidden[0], num_hidden[1])
b1 = init_bias(num_hidden[1])
W0 = init_weights(num_features, num_hidden[0])
b0 = init_bias(num_hidden[0])

# Build the graph for the deep net
# x0 --W0,b0--> u0 --f0--> x1 --W1,b1--> u1 --f1--> x2 --W2,b2--> u2 --f2--> p --> y
u0 = tf.matmul(x0, W0) + b0
x1 = tf.nn.sigmoid(u0)  # f0
u1 = tf.matmul(x1, W1) + b1
x2 = tf.nn.sigmoid(u1)  # f1
u2 = tf.matmul(x2, W2) + b2
p = tf.nn.softmax(u2)  # f2
y = tf.argmax(p, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=k, logits=u2)
loss_no_reg = tf.reduce_mean(cross_entropy)
reg = tf.nn.l2_loss(W0) + tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
loss = tf.reduce_mean(loss_no_reg + beta * reg)

# Create the gradient descent optimizer with the given learning rate
optimizer = tf.train.GradientDescentOptimizer(lr)
train_op = optimizer.minimize(loss)

correct_prediction = tf.cast(tf.equal(tf.argmax(u2, 1), tf.argmax(k, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_acc = []
    test_acc = []
    for i in range(epochs):

        # randomly choose batch
        rand_index = np.random.choice(num_data, size=batch_size)
        x_batch = X[rand_index]
        k_batch = K[rand_index]

        train_op.run(feed_dict={x0: x_batch, k: k_batch})
        train_acc.append(accuracy.eval(feed_dict={x0: x_batch, k: k_batch}))
        test_acc.append(accuracy.eval(feed_dict={x0: X_tst, k: K_tst}))

        if i % (epochs//10) == 0 or i == epochs-1:
            print('epoch {0:5d}: accuracy Train: {1:8.4f} Test: {2:8.4f}'.format(i+1, train_acc[i], test_acc[i]))

# plot learning curves
plt.figure(1)
plt.plot(range(epochs), train_acc, 'b', label='Train Accuracy')
plt.plot(range(epochs), test_acc, 'r', label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig('partA_Qn5.png')
plt.show()
