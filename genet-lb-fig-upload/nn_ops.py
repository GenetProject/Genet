import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def fc_nn_init(input_dim, hid_dims, output_dim, scope='default'):
    weights = []
    bias = []

    curr_in_dim = input_dim

    # hidden layers
    for hid_dim in hid_dims:
        weights.append(
            glorot([curr_in_dim, hid_dim], scope=scope))
        bias.append(
            zeros([hid_dim], scope=scope))
        curr_in_dim = hid_dim

    # output layer
    weights.append(glorot([curr_in_dim, output_dim], scope=scope))
    bias.append(zeros([output_dim], scope=scope))

    return weights, bias


def fc_nn(inputs, weights, bias, act_fn):
    # non-linear feed forward
    x = inputs

    for (w, b) in zip(weights[:-1], bias[:-1]):
        x = tf.matmul(x, w)
        x += b
        x = act_fn(x)

    # final linear output layer
    x = tf.matmul(x, weights[-1])
    x += bias[-1]

    return x


def glorot(shape, dtype=tf.float32, scope='default'):
    # Xavier Glorot & Yoshua Bengio (AISTATS 2010) initialization (Eqn 16)
    with tf.variable_scope(scope):
        init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
        init = tf.random_uniform(
            shape, minval=-init_range, maxval=init_range, dtype=dtype)
        return tf.Variable(init)


def leaky_relu(features, alpha=0.2, name=None):
  """Compute the Leaky ReLU activation function.
  "Rectifier Nonlinearities Improve Neural Network Acoustic Models"
  AL Maas, AY Hannun, AY Ng - Proc. ICML, 2013
  http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf
  Args:
    features: A `Tensor` representing preactivation values.
    alpha: Slope of the activation function at x < 0.
    name: A name for the operation (optional).
  Returns:
    The activation value.
  """
  with ops.name_scope(name, "LeakyRelu", [features, alpha]):
    features = ops.convert_to_tensor(features, name="features")
    alpha = ops.convert_to_tensor(alpha, name="alpha")
    return math_ops.maximum(alpha * features, features)


def normalize(inputs, activation, reuse, scope, norm):
    if norm == 'batch_norm':
        return tf.contrib.layers.batch_norm(
            inputs, activation_fn=activation, reuse=reuse, scope=scope)
    elif norm == 'layer_norm':
        return tf.contrib.layers.layer_norm(
            inputs, activation_fn=activation, reuse=reuse, scope=scope)
    elif norm == 'None':
        if activation is not None:
            return activation(inputs)
        else:
            return inputs


def ones(shape, dtype=tf.float32, scope='default'):
    with tf.variable_scope(scope):
        init = tf.ones(shape, dtype=dtype)
        return tf.Variable(init)


def zeros(shape, dtype=tf.float32, scope='default'):
    with tf.variable_scope(scope):
        init = tf.zeros(shape, dtype=dtype)
        return tf.Variable(init)