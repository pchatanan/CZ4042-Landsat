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
