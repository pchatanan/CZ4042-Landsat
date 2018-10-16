import math
import tensorflow as tf
import numpy as np


def scale(data, data_min, data_max):
    """
    This function scale matrix data to be between 0 and 1.
    :param data: numpy array
    :param data_min: minimum value possible in data
    :param data_max: maximum value possible in data
    :return: scaled numpy array
    """
    return (data - data_min) / (data_max - data_min)


def init_weights(n_in=1, n_out=1):
    """
    This function initialises weight matrix.
    The weight follows normal distribution with mean = 0 and standard deviation = 1/sqrt(n_in).
    :param n_in: number of node before this layer
    :param n_out: number of node of this layer
    :return: weight matrix
    """
    return tf.Variable(tf.truncated_normal([n_in, n_out], stddev=1.0 / math.sqrt(float(n_in))))


def init_bias(n):
    """
    This function initialises a bias vector to zeros.
    :param n: number of node of this layer
    :return: zero bias vector
    """
    return tf.Variable(np.zeros(n), dtype=tf.float32)


def load_data(path):
    """
    This function reads in a text file and clean up data.
    1. Read a text file into a numpy array
    2. Extract input features and output label
    3. Scale the input features and perform one-hot encoding for output label
    Take note that we have to convert 7 to 6 and (1,2,3,...,6) to (0,1,2,...,5)
    :param path: The file path.
    :return: a tuple (data_x, data_k)
        data_x: the scaled input feature
        data_k: the one-hot encoding for output label.
    """
    # read train data
    data_set = np.loadtxt(path, delimiter=' ')
    data_x, data_y = data_set[:, :-1], data_set[:, -1].astype(int)
    # scale X (8-bit image with value 0 to 255)
    data_x = scale(data_x, 0, 255)
    # clean Y
    data_y[data_y == 7] = 6
    data_y -= 1
    # one-hot encoding
    data_k = np.zeros((data_y.size, data_y.max() + 1))  # data_k = np.zeros(n,6)
    data_k[np.arange(data_y.size), data_y] = 1
    return data_x, data_k


def k_fold(n, k):
    """
    This function creates a list of indexes for k-fold
    For example: n = 10 k = 5
    return => [[7,0], [1,4], [2,3], [9,6], [5,9]]
    :param n: Total number of train sets
    :param k: k-fold
    :return: list of list of indexes
    """
    num_per_fold = n // k
    result = []
    index_set = set(range(n))
    for i in range(k):
        if i != k - 1:
            test_idx = range(i * num_per_fold, (i + 1) * num_per_fold)
        else:
            test_idx = range(i * num_per_fold, n)
        train_idx = list(index_set - set(test_idx))
        result.append((train_idx, test_idx))
    return result


def init_weights_bias(n_in, n_hidden_list, n_out):
    """
    This function initializes weights and biases
    :param n_in: Number of features input
    :param n_hidden_list:  a list of number of hidden neurons
    :param n_out: Number of output node
    :return: a tuple of weights and biases lists
    """
    W, b = [], []
    node_list = n_hidden_list
    node_list.insert(0, n_in)
    node_list.append(n_out)
    total_layer = len(node_list)
    for i in range(total_layer):
        if i == total_layer - 1:
            break
        W.append(init_weights(node_list[i], node_list[i + 1]))
        b.append(init_bias(node_list[i + 1]))
    del node_list[0]
    del node_list[-1]
    return W, b


def build_graph(x0, w, b, act_func_list, dropout_keep_prob):
    """
    This function builds a computational graph.
    :param x0: the placeholder for input
    :param w: weight list
    :param b: bias list
    :param act_func_list: a list of activation functions for hidden layers
    :param dropout_keep_prob: if None then no dropout, else the value should be between 0 to 1
    :return: the output of computational graph
    """
    x, u = [x0], []
    for i, act_func in enumerate(act_func_list):
        u.append(tf.matmul(x[i], w[i]) + b[i])
        if dropout_keep_prob is not None:
            x.append(tf.nn.dropout(act_func(u[i]), dropout_keep_prob))
        else:
            x.append(act_func(u[i]))
    y = tf.matmul(x[-1], w[-1]) + b[-1]
    return y


def l2_reg(w):
    """
    Compute L2 regularization from weights
    :param w: weights list
    :return: regularization factor
    """
    reg = tf.nn.l2_loss(w.pop(0))
    for weight in w:
        reg += tf.nn.l2_loss(weight)
    return reg
