from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.platform import tf_logging as logging

from lib.utils import calculate_scaled_laplacian
# from lib import dcrnn_utils


class DCGRUCell(RNNCell):
    """Graph Convolution Gated Recurrent Unit cell.
    """

    def call(self, inputs, **kwargs):
        pass

    def _compute_output_shape(self, input_shape):
        pass

    def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes, input_size=None,
                 num_proj=None, drop_out=0.0, is_training=False,
                 activation=tf.nn.tanh, reuse=None, filter_type="laplacian"):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param input_size:
        :param num_proj:
        :param activation:
        :param reuse:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        """
        super(DCGRUCell, self).__init__(_reuse=reuse)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_proj = num_proj
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._drop_out = drop_out
        self._is_training = is_training
        self._supports = []
        supports = []
        # if filter_type == "laplacian":
        supports.append(calculate_scaled_laplacian(adj_mx, lambda_max=None))
        # elif filter_type == "random_walk":
        #     supports.append(calculate_random_walk_matrix(adj_mx).T)
        # elif filter_type == "dual_random_walk":
        #     supports.append(calculate_random_walk_matrix(adj_mx).T)
        #     supports.append(calculate_random_walk_matrix(adj_mx.T).T)
        # else:
        #     supports.append(calculate_scaled_laplacian(adj_mx))
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        return tf.sparse_reorder(L)

    @property
    def state_size(self):
        return self._num_nodes * self._num_units

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        if self._num_proj is not None:
            output_size = self._num_nodes * self._num_proj
        return output_size

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)

        :return
        - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
        - New state: Either a single `2-D` tensor, or a tuple of tensors matching
            the arity and shapes of `state`
        """
        with tf.variable_scope(scope or "dcgru_cell"):
            with tf.variable_scope("gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                value = tf.nn.sigmoid(
                    self._gconv(inputs, state, 2 * self._num_units, bias_start=1.0, scope=scope))
                value = tf.layers.dropout(value, rate=self._drop_out, training=self._is_training)
                r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
                # r, u = sigmoid(r), sigmoid(u)
            with tf.variable_scope("candidate"):
                c = self._gconv(inputs, r * state, self._num_units, scope=scope)
                c = tf.layers.dropout(c, rate=self._drop_out, training=self._is_training)
                if self._activation is not None:
                    c = self._activation(c)
            output = new_state = u * state + (1 - u) * c
            if self._num_proj is not None:
                with tf.variable_scope("projection"):
                    w = tf.get_variable('w', shape=(self._num_units, self._num_proj))
                    batch_size = inputs.get_shape()[0].value
                    output = tf.reshape(new_state, shape=(-1, self._num_units))
                    output = tf.reshape(tf.matmul(output, w), shape=(batch_size, self.output_size))
                    output = tf.layers.dropout(output, rate=self._drop_out, training=self._is_training)
        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    def _gconv(self, inputs, state, output_size, bias_start=0.0, scope=None):
        """Graph convolution between input and the graph matrix.

        :param args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        :param output_size:
        :param bias:
        :param bias_start:
        :param scope:
        :return:
        """
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = tf.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2].value
        dtype = inputs.dtype

        x = inputs_and_state
        x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, total_arg_size, batch_size)
        x0 = tf.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = tf.expand_dims(x0, axis=0)

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            if self._max_diffusion_step == 0:
                pass
            else:
                for support in self._supports:
                    x1 = tf.sparse_tensor_dense_matmul(support, x0)
                    x = self._concat(x, x1)

                    for k in range(2, self._max_diffusion_step + 1):
                        x2 = 2 * tf.sparse_tensor_dense_matmul(support, x1) - x0
                        x = self._concat(x, x2)
                        x1, x0 = x2, x1

            num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Adds for x itself.
            x = tf.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
            x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_size, order)
            x = tf.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

            weights = tf.get_variable(
                'weights', [input_size * num_matrices, output_size], dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer())
            x = tf.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

            biases = tf.get_variable(
                "biases", [output_size],
                dtype=dtype,
                initializer=tf.constant_initializer(bias_start, dtype=dtype))
            x = tf.nn.bias_add(x, biases)
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return tf.reshape(x, [batch_size, self._num_nodes * output_size])
