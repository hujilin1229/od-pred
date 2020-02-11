from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.platform import tf_logging as logging

from lib.utils import calculate_scaled_laplacian
from lib.graph import laplacian

class MGGRUCell(RNNCell):
    """Graph Convolution Gated Recurrent Unit cell.
    """

    def call(self, inputs, **kwargs):
        pass

    def _compute_output_shape(self, input_shape):
        pass

    def __init__(self, num_units, adj_mx1, adj_mx2, max_diffusion_step,
                 num_nodes, num_coarsed_nodes, num_proj=None, activation=tf.nn.sigmoid, reuse=None,
                 filter_type="laplacian", pooling_size=None, pool_type='_mpool'):
        """

        :param num_units:
        :param adj_mx1: np.array: apply to input gate
        :param adj_mx2: np.array apply to gate
        :param max_diffusion_step:
        :param num_nodes:
        :param num_proj:
        :param activation:
        :param reuse:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        """
        super(MGGRUCell, self).__init__(_reuse=reuse)

        self._activation = activation
        self._num_nodes = num_nodes
        self._num_coarsen_nodes = num_coarsed_nodes
        self._num_proj = num_proj
        self._num_units = num_units
        self._pool_size = pooling_size
        self._pool_func = getattr(self, pool_type)
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._supports2 = []
        supports = []
        supports2 = []
        if filter_type == "laplacian":
            supports.append(calculate_scaled_laplacian(adj_mx1, lambda_max=None))
            supports2.append(calculate_scaled_laplacian(adj_mx2, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx1).T)
            supports2.append(calculate_random_walk_matrix(adj_mx2).T)
        elif filter_type == "dual_random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx1).T)
            supports.append(calculate_random_walk_matrix(adj_mx1.T).T)
            supports2.append(calculate_random_walk_matrix(adj_mx2).T)
            supports2.append(calculate_random_walk_matrix(adj_mx2.T).T)
        else:
            supports.append(calculate_scaled_laplacian(adj_mx1))
            supports2.append(calculate_scaled_laplacian(adj_mx2))
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))
        for support in supports2:
            self._supports2.append(self._build_sparse_matrix(support))

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        return tf.sparse_reorder(L)

    @property
    def state_size(self):
        return int(self._num_nodes * self._num_coarsen_nodes * self._num_units / self._pool_size)

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_coarsen_nodes * self._num_units
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
        with tf.variable_scope(scope or "mggru_cell"):
            with tf.variable_scope("input_gate", reuse=tf.AUTO_REUSE):
                # make the shape of inputs same with shape of value
                inputs = self._gconv_pool(inputs, self._num_units, bias_start=1.0)
            with tf.variable_scope("gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                value = tf.nn.sigmoid(
                    self._lin_transform(inputs, state, 2 * self._num_units, bias_start=1.0, scope=scope))
                r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
            with tf.variable_scope("candidate"):
                c = self._lin_transform(inputs, r * state, self._num_units, scope=scope)
                if self._activation is not None:
                    c = self._activation(c)
            output = u * state + (1 - u) * c

            # [batch_size, coarsened_nodes * self._num_nodes * output_size]
            output = new_state = output
            if self._num_proj is not None:
                with tf.variable_scope("projection"):
                    batch_size = output.get_shape()[0].value
                    output = tf.reshape(output, shape=[batch_size, -1, self._num_nodes, self._num_units])
                    output = tf.transpose(output, perm=[0, 2, 1, 3])
                    output = tf.reshape(output, shape=[batch_size*self._num_nodes, -1])
                    new_state_feature = output.get_shape()[-1]
                    w = tf.get_variable('w', shape=(new_state_feature, self._num_proj))
                    # output = tf.reshape(new_state, shape=(-1, self._num_units))
                    output = tf.reshape(tf.matmul(output, w), shape=(batch_size, self.output_size))
                    output = self._activation(output)

        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    def _lin_transform(self, inputs, state, output_size, bias_start=0.0, scope=None):
        """Graph convolution between input and the graph matrix.

        :param inputs: a 2D Tensor or a list of 2D, batch x n, Tensors.
        :param output_size:
        :param bias:
        :param bias_start:
        :param scope:
        :return:
        """
        # Reshape input and state to (batch_size, c_num_nodes, o_num_nodes, input_dim/state_dim)
        batch_size = inputs.get_shape()[0].value
        coarsened_nodes = int(self._num_coarsen_nodes / self._pool_size)
        # input_channel = self._input_dim * 2
        # transform input into (batch_size, c_nodes, o_nodes * feature)
        inputs = tf.reshape(inputs, (batch_size, coarsened_nodes, self._num_nodes, self._num_units))
        state = tf.reshape(state, (batch_size, coarsened_nodes, self._num_nodes, self._num_units))
        # transpose the tensor dimension such that it follows the batch_size, num_nodes, coarsen_nodes, feat_dim
        # inputs = tf.transpose(inputs, perm=[0, 2, 1, 3])
        # state = tf.transpose(state, perm=[0, 2, 1, 3])
        inputs = tf.reshape(inputs, (batch_size * coarsened_nodes * self._num_nodes, self._num_units))
        state = tf.reshape(state, (batch_size * coarsened_nodes * self._num_nodes, self._num_units))
        inputs_and_state = tf.concat([inputs, state], axis=1)
        dtype = inputs.dtype
        if scope is None:
            scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            weights = tf.get_variable(
                'weights', [self._num_units * 2, output_size], dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer())
            inputs_and_state = tf.matmul(inputs_and_state, weights)  # (batch_size * self._num_nodes, output_size)

            biases = tf.get_variable(
                "biases", [output_size],
                dtype=dtype,
                initializer=tf.constant_initializer(bias_start, dtype=dtype))
            inputs_and_state = tf.nn.bias_add(inputs_and_state, biases)

        outputs_and_state = tf.reshape(
            inputs_and_state, (batch_size, self._num_nodes, coarsened_nodes, -1))
        outputs_and_state = tf.transpose(outputs_and_state, perm=[0, 2, 1, 3])
        # Reshape res back to 2D: (batch_size, num_node, o_node, state_dim) -> (batch_size, num_node * state_dim)
        outputs_and_state = tf.reshape(outputs_and_state, (batch_size, -1))

        # Reshape res back to 2D: (batch_size, num_node, o_node, state_dim) -> (batch_size, num_node * state_dim)
        return outputs_and_state

    def _gconv(self, inputs, state, output_size, bias_start=0.0, scope=None):
        """Graph convolution between input and the graph matrix.

        :param inputs: a 2D Tensor or a list of 2D, batch x n, Tensors.
        :param output_size:
        :param bias:
        :param bias_start:
        :param scope:
        :return:
        """
        # Reshape input and state to (batch_size, c_num_nodes, o_num_nodes, input_dim/state_dim)
        batch_size = inputs.get_shape()[0].value
        coarsened_nodes = int(self._num_coarsen_nodes / self._pool_size)
        # input_channel = self._input_dim * 2
        # transform input into (batch_size, c_nodes, o_nodes * feature)
        inputs = tf.reshape(inputs, (batch_size, coarsened_nodes, -1))
        state = tf.reshape(state, (batch_size, coarsened_nodes, -1))
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2].value # o_nodes * feature
        input_channel = int(input_size / self._num_nodes)
        dtype = inputs.dtype
        # print("Input shape ", inputs.get_shape())
        # print("State shape ", state.get_shape())
        # print("Inputs and States Shape ", inputs_and_state.get_shape())
        x = inputs_and_state
        x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, total_arg_size, batch_size)
        x0 = tf.reshape(x0, shape=[coarsened_nodes, input_size * batch_size])
        x = tf.expand_dims(x0, axis=0)

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            if self._max_diffusion_step == 0:
                pass
            else:
                for support in self._supports2:
                    x1 = tf.sparse_tensor_dense_matmul(support, x0)
                    x = self._concat(x, x1)

                    for k in range(2, self._max_diffusion_step + 1):
                        x2 = 2 * tf.sparse_tensor_dense_matmul(support, x1) - x0
                        x = self._concat(x, x2)
                        x1, x0 = x2, x1

            # #filter size
            num_matrices = len(self._supports2) * self._max_diffusion_step + 1  # Adds for x itself.
            # print("X shape is ", x.get_shape())
            x = tf.reshape(x, shape=[num_matrices, coarsened_nodes, self._num_nodes, input_channel, batch_size])
            x = tf.transpose(x, perm=[4, 1, 2, 3, 0])  # (batch_size, num_nodes, o_node, in_chan, order)
            x = tf.reshape(x, shape=[batch_size * coarsened_nodes * self._num_nodes, input_channel * num_matrices])

            weights = tf.get_variable(
                'weights', [input_channel * num_matrices, output_size], dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer())
            x = tf.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

            biases = tf.get_variable(
                "biases", [output_size],
                dtype=dtype,
                initializer=tf.constant_initializer(bias_start, dtype=dtype))
            x = tf.nn.bias_add(x, biases)
        # Reshape res back to 2D: (batch_size, num_node, o_node, state_dim) -> (batch_size, num_node * state_dim)
        return tf.reshape(x, [batch_size, coarsened_nodes * self._num_nodes * output_size])

    def _gconv_pool(self, inputs, output_size, bias_start=0.0):
        """Graph convolution between input and the graph matrix.

        :param inputs: a 2D Tensor or a list of 2D, batch x n, Tensors.
        :param output_size:
        :param bias_start:
        :param scope:
        :return:
        """

        # reshape input to  (batch_size, c_num_nodes, o_num_nodes, input_dim/state_dim)
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size, self._num_coarsen_nodes, self._num_nodes, -1))
        # Reshape input and state to (batch_size, c_num_nodes, o_num_nodes, input_dim/state_dim)
        num_coarsen_nodes = inputs.get_shape()[1].value
        # input_channel = self._input_dim * 2
        # transform input into (batch_size, c_nodes, o_nodes * feature)
        inputs = tf.reshape(inputs, (batch_size, num_coarsen_nodes, -1))
        input_size = inputs.get_shape()[2].value # o_nodes * feature
        input_channel = int(input_size / self._num_nodes)
        dtype = inputs.dtype
        # print("Input shape ", inputs.get_shape())
        # print("State shape ", state.get_shape())
        # print("Inputs and States Shape ", inputs_and_state.get_shape())
        x = inputs
        x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, total_arg_size, batch_size)
        x0 = tf.reshape(x0, shape=[num_coarsen_nodes, input_size * batch_size])
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

            # #filter size
            num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Adds for x itself.
            # print("X shape is ", x.get_shape())
            x = tf.reshape(x, shape=[num_matrices, num_coarsen_nodes, self._num_nodes, input_channel, batch_size])
            x = tf.transpose(x, perm=[4, 1, 2, 3, 0])  # (batch_size, num_nodes, o_node, in_chan, order)
            x = tf.reshape(x, shape=[batch_size * num_coarsen_nodes * self._num_nodes, input_channel * num_matrices])

            weights = tf.get_variable(
                'weights', [input_channel * num_matrices, output_size], dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer())
            x = tf.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

            biases = tf.get_variable(
                "biases", [output_size],
                dtype=dtype,
                initializer=tf.constant_initializer(bias_start, dtype=dtype))
            x = tf.nn.bias_add(x, biases)
            x = tf.nn.relu(x)
            # reshape output back to 4D: (batch_size, num_node * o_node * state_dim) -> (batch_size, num_node, o_node, state_dim)
            x = tf.reshape(x, shape=(batch_size, num_coarsen_nodes, -1, output_size))
            # Do Pooling first to reduce the feature map size
            x = self._pool_func(x, self._pool_size)

        # Reshape res back to 2D: (batch_size, num_node, o_node, state_dim) -> (batch_size, num_node * state_dim)
        return tf.reshape(x, [batch_size, -1])

    def _mpool(self, x, p):
        """
        Max pooling of size p.
        The size of the input x is [batch, len_feature, nb_kernels, nb_bins].

        x: [batch, height, width, channels]
        """

        if p > 1:
            x = tf.nn.max_pool(x, ksize=[1, p, 1, 1], strides=[
                1, p, 1, 1], padding='SAME')
            # tf.maximum
            return x  # N x M/p x F
        else:
            return x

    def _apool(self, x, p):
        """
        Average pooling of of size p.
        The size of the input x is [batch, len_feature, nb_kernels, nb_bins].

        """
        if p > 1:
            x = tf.nn.avg_pool(x, ksize=[1, p, 1, 1], strides=[
                1, p, 1, 1], padding='SAME')

            return x  # N x M/p x F x B
        else:
            return x  # N x M x F x B
