from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import legacy_seq2seq

from lib.metrics_weight import masked_mse_loss, masked_mae_loss, \
    masked_rmse_loss, masked_mape_loss, \
    masked_kl_loss, masked_l2_loss, masked_emd_loss
from model.tf_model import TFModel
import numpy as np

class FCRNNModel(TFModel):
    def __init__(self, is_training, config, scaler=None, adj_mxs=None, adj_origin=None):
        super(FCRNNModel, self).__init__(config, scaler=scaler)
        self._batch_size = int(config.get('batch_size'))
        max_diffusion_step = int(config.get('max_diffusion_step', 2))
        self._cl_decay_steps = int(config.get('cl_decay_steps', 1000))
        self._filter_type = config.get('filter_type', 'laplacian')
        self._horizon = int(config.get('horizon', 1))
        self._input_dim = int(config.get('input_dim', 1))
        self._output_dim = int(config.get('output_dim', 1))
        self._latent_dim = int(config.get('latent_dim', 1))
        self._dropout = float(config.get('drop_out', 0.0))
        loss_func = config.get('loss_func', 'MAPE')
        optim_name = config.get('optimizer', 'adam')
        max_grad_norm = float(config.get('max_grad_norm', 5.0))
        self._num_nodes = int(config.get('num_nodes', 1))
        self._mode = config.get('mode', 'avg')
        self._model_construction = config.get('fc_method', 'od')
        num_rnn_layers = int(config.get('num_rnn_layers', 1))
        # self._output_dim = int(config.get('output_dim', 1))
        self._seq_len = int(config.get('seq_len'))
        self._use_curriculum_learning = bool(config.get('use_curriculum_learning', False))

        self._conv_filter_size = config.get('c_filter_size', [max_diffusion_step] * num_rnn_layers)
        if self._model_construction == 'od':
            num_rnn_layers = 3
        else:
            num_rnn_layers = 2
        self._conv_filter_num = config.get('nb_c_filter', [num_rnn_layers] * num_rnn_layers)
        self._pool_size = config.get('pooling', [1] * num_rnn_layers)
        self._fc_size = config.get('nb_fc_filter', [])
        self._adj_mxs = adj_mxs
        self._adj_origin = adj_origin
        self._aux_dim = self._input_dim - self._output_dim
        trace_ratio = float(config.get('trace_ratio', 0.0))

        # assert input_dim == output_dim, 'input_dim: %d != output_dim: %d' % (input_dim, output_dim)
        # Input (batch_size, timesteps, num_sensor, input_dim)
        self._inputs = tf.placeholder(tf.float32,
                                      shape=(self._batch_size, self._seq_len, self._num_nodes,
                                             self._num_nodes, self._input_dim),
                                      name='inputs')

        # Labels: (batch_size, timesteps, num_nodes, num_nodes, input_dim),
        # same format with input except the temporal dimension.
        self._labels = tf.placeholder(tf.float32,
                                      shape=(self._batch_size, self._horizon, self._num_nodes,
                                             self._num_nodes, self._input_dim),
                                      name='labels')
        self._weight = tf.placeholder(tf.bool,
                                      shape=(self._batch_size, self._horizon, self._num_nodes,
                                             self._num_nodes),
                                      name='weight')
        self._num = tf.placeholder(tf.int8,
                                   shape=(self._batch_size, self._horizon, self._num_nodes,
                                          self._num_nodes),
                                   name='num')

        # print("RNN construction for O_inputs...")
        # Do input and output with latent space model
        if self._model_construction == 'od':
            o_inputs = tf.transpose(self._inputs, perm=(0, 1, 3, 2, 4))
            o_labels = tf.transpose(self._labels, perm=(0, 1, 3, 2, 4))

            o_output = self._construct_fc_gru_seq2seq_input(
                o_inputs, o_labels, 'O_inputs', is_training)
            d_output = self._construct_fc_gru_seq2seq_input(
                self._inputs, self._labels, 'D_inputs', is_training)
            self._outputs = self._matmul_seq_latent_spaces(o_output, d_output)
            self._trace_loss = self._get_norm_rank(o_output, d_output, self._adj_origin)
            #
            # o_inputs = tf.transpose(self._inputs, perm=(0, 1, 3, 2, 4))
            # o_labels = tf.transpose(self._labels, perm=(0, 1, 3, 2, 4))

            # one RNN encoder
            # rnn_encoded = self._construct_fc_gru_seq2seq_input_generic(
            #     self._inputs, self._labels, 'RNN_F', is_training)
            # o_output = self._construct_latent_space(
            #     rnn_encoded, 'O_inputs', is_training, tf.float32)
            # d_output = self._construct_latent_space(
            #     rnn_encoded, 'D_inputs', is_training, tf.float32)
            # self._outputs = self._matmul_seq_latent_spaces(o_output, d_output)
            # self._trace_loss = self._get_norm_rank(o_output, d_output, self._adj_origin)
        else:
            # Do FC layer directly...
            self._outputs = self._construct_fc_gru_together_seq2seq_input(
                self._inputs, self._labels, 'Input', is_training)
            self._trace_loss = 0.0

        preds = self._outputs
        labels = self._labels[..., :self._output_dim]

        null_val = config.get('null_val', 0.)

        if loss_func == 'MSE':
            self._loss = masked_mse_loss(self._scaler, null_val)(preds=preds, labels=labels, mask=self._num)
        elif loss_func == 'MAE':
            self._loss = masked_mae_loss(self._scaler, null_val)(preds=preds, labels=labels, mask=self._num)
        elif loss_func == 'RMSE':
            self._loss = masked_rmse_loss(self._scaler, null_val)(preds=preds, labels=labels, mask=self._num)
        elif loss_func == 'MAPE':
            self._loss = masked_mape_loss(self._scaler, null_val)(preds=preds, labels=labels, mask=self._num)
        elif loss_func == 'KL':
            self._loss = masked_kl_loss(preds=preds, labels=labels, mask=self._weight)
        elif loss_func == 'L2':
            self._loss = masked_l2_loss(preds=preds, labels=labels, mask=self._weight)
        elif loss_func == 'EMD':
            self._loss = masked_emd_loss(preds=preds, labels=labels, mask=self._weight)
        elif loss_func == 'L2Norm':
            self._loss = masked_l2_loss(preds=preds, labels=labels, mask=self._weight)+ \
                         trace_ratio * self._trace_loss/ \
                         (self._num_nodes*self._latent_dim*self._horizon)
        else:
            self._loss = masked_mse_loss(self._scaler, null_val)(preds=preds, labels=labels, mask=self._num)
        if is_training:
            self._train_op = self.back_propagation(optim_name, self._loss)

        print("MGRNN Model Constructed....")

        self._merged = tf.summary.merge_all()

    @property
    def model_construction(self):
        return self._model_construction

    def _get_norm_rank(self, o_output, d_output, adj_mat):
        """
        Calculate the rank loss for o_output and d_output

        :param o_output: latent tensor for O
        :param d_output: latent tensor for D
        :param adj_mat: adjacent matrix for OD
        :return: rank loss in sum
        """

        return tf.reduce_sum(tf.square(o_output)) + tf.reduce_sum(tf.square(d_output))

    def _back_propagation(self, optim_name, loss):
        """
        Do normal back_propagation

        :param optim_name: name of the optimizer
        :param loss: loss function
        :return: back_prop operations
        """
        global_step = tf.train.get_or_create_global_step()
        if optim_name == 'adam':
            optimizer = tf.train.AdamOptimizer(self._lr)
        elif optim_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
        elif optim_name == 'ada':
            optimizer = tf.train.AdagradOptimizer(self._lr)
        else:
            optimizer = tf.train.AdamOptimizer(self._lr)
        return optimizer.minimize(loss, global_step=global_step)

    def back_propagation(self, optim_name, loss):
        return self._back_propagation(optim_name, loss)

    def _construct_latent_space(self, output, scope, is_training=False, dtype=None):
        """
        Construct the latent space by giving an output

        :param output: 2D tensor: [batch_size, horizon, num_nodes, output_dim]
        :return: 4D tensor: [batch_size, orizon, num_nodes, latent_dim, output_dim]
        """

        with tf.variable_scope(scope):
            output = tf.reshape(output, shape=[self._batch_size, self._horizon, -1])
            cur_feature = output.get_shape()[-1]
            weights = tf.get_variable(
                'weights', [cur_feature, self._num_nodes * self._latent_dim * self._output_dim],
                dtype=dtype, initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable(
                "biases", [self._num_nodes * self._latent_dim * self._output_dim], dtype=dtype,
                initializer=tf.constant_initializer(1.0, dtype=dtype))
            feature_decodes = []
            for horizon_i  in tf.unstack(output, axis=1):
                horizon_i = tf.matmul(horizon_i, weights)  # (batch_size * self._num_nodes, output_size)
                horizon_i = tf.nn.bias_add(horizon_i, biases)
                horizon_i = tf.tanh(horizon_i)
                horizon_i = tf.layers.dropout(horizon_i, self._dropout, training=is_training)
                horizon_i = tf.reshape(horizon_i, shape=[
                    self._batch_size, self._num_nodes, self._latent_dim, self._output_dim])
                feature_decodes.append(horizon_i)
            output = tf.stack(feature_decodes, axis=1)

            return output

    def _construct_fc_gru_together_seq2seq_input(self, inputs, labels=None, scope_name=None, is_training=False):
        # new code for constructing GRUCell by applying 2D graph convolution and pooling,
        #  even possibly fc layer
        # Do GCNN on Origin side
        encoding_cells, decoding_cells = [], []
        num_filters = len(self._conv_filter_size)
        output_proj_dim = self._output_dim * self._num_nodes * self._num_nodes
        # print("Output Proj Dim is ", output_proj_dim)
        for i in range(num_filters):
            nb_filter = self._conv_filter_num[i]
            cell = tf.nn.rnn_cell.GRUCell(num_units=nb_filter)
            if is_training:
                keep_out = 1 - self._dropout
                cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell, input_keep_prob=keep_out,
                    output_keep_prob=keep_out,
                    state_keep_prob=keep_out)
            encoding_cells.append(cell)
            if i < num_filters - 1:
                decoding_cells.append(cell)
            else:
                cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=output_proj_dim)
                decoding_cells.append(cell)

        encoding_cells = tf.contrib.rnn.MultiRNNCell(encoding_cells, state_is_tuple=True)
        decoding_cells = tf.contrib.rnn.MultiRNNCell(decoding_cells, state_is_tuple=True)

        GO_SYMBOL = tf.zeros(shape=(self._batch_size, self._num_nodes * self._num_nodes * self._input_dim))

        # # get the list of tensors with number of list size
        # inputs = tf.unstack(inputs, axis=1)
        # for input_i in inputs:
        #     for i in range(num_filters):
        #         nb_filter = self._conv_filter_num[i]
        #         with tf.variable_scope(scope_name+'fc_{}'.format(i), reuse=True):
        #             input_i = self._fc(input_i, nb_filter)
        global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope(scope_name):
            # inputs and labels are lists
            print("Input shape is ", inputs.get_shape())
            inputs = tf.unstack(tf.reshape(inputs, (self._batch_size, self._seq_len,
                                                    self._num_nodes * self._num_nodes * self._input_dim)),
                                axis=1)
            labels_i = tf.unstack(
                tf.reshape(labels[..., :self._output_dim],
                           (self._batch_size, self._horizon, self._num_nodes *
                            self._num_nodes * self._output_dim)), axis=1)
            labels_i.insert(0, GO_SYMBOL)
            def loop_function(prev, i):
                """
                This function will be applied to the i-th output in order to generate the i+1-st input

                :param prev: 2D Tensor of shape [batch_size x output_size],
                :param i: integer, the step number (when advanced control is needed),
                :return: 2D Tensor of shape [batch_size x input_size].
                """
                if is_training:
                    # Return either the model's prediction or the previous ground truth in training.
                    if self._use_curriculum_learning:
                        c = tf.random_uniform((), minval=0, maxval=1.)
                        threshold = self._compute_sampling_threshold(global_step, self._cl_decay_steps)
                        result = tf.cond(tf.less(c, threshold), lambda: labels_i[i], lambda: prev)
                    else:
                        result = labels_i[i]
                else:
                    # Return the prediction of the model in testing.
                    result = prev
                # if self._aux_dim > 0:
                #     result = tf.reshape(result, (
                #     self._batch_size, self._num_coarsened_nodes, self._num_nodes, self._output_dim))
                #     result = tf.concat([result, aux_info[i]], axis=-1)
                #     result = tf.reshape(result, (
                #     self._batch_size, self._num_coarsened_nodes * self._num_nodes * self._input_dim))
                return result

            _, enc_state = tf.contrib.rnn.static_rnn(encoding_cells, inputs, dtype=tf.float32)
            outputs, final_state = legacy_seq2seq.rnn_decoder(labels_i, enc_state, decoding_cells,
                                                              loop_function=None)

            outputs_latent = []
            for h_i, output in enumerate(outputs[:-1]):
                output = tf.reshape(output, shape=[
                    self._batch_size, self._num_nodes, self._num_nodes, self._output_dim])
                outputs_latent.append(output)
            outputs = tf.stack(outputs_latent, axis=1)

        if self._mode == 'avg':
            outputs = tf.nn.sigmoid(outputs)
        else:
            # if output should be hist, do softmax to make it reasonable
            outputs = tf.nn.softmax(outputs, -1)

        return outputs

    def _construct_fc_gru_seq2seq_input(self, inputs, labels=None,
                                        scope_name=None, is_training=False):
        # new code for constructing GRUCell by applying 2D graph convolution and pooling,
        #  even possibly fc layer
        # Do GCNN on Origin side
        encoding_cells, decoding_cells = [], []
        num_filters = len(self._conv_filter_size)
        output_proj_dim = self._latent_dim * self._output_dim * self._num_nodes
        # print("Output Proj Dim is ", output_proj_dim)
        for i in range(num_filters):
            nb_filter = self._conv_filter_num[i]
            cell = tf.nn.rnn_cell.GRUCell(num_units=nb_filter)
            if is_training:
                keep_out = 1 - self._dropout
                cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell, input_keep_prob=keep_out,
                    output_keep_prob=keep_out,
                    state_keep_prob=keep_out)
            encoding_cells.append(cell)
            if i < num_filters - 1:
                decoding_cells.append(cell)
            else:
                cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=output_proj_dim)
                decoding_cells.append(cell)

        encoding_cells = tf.contrib.rnn.MultiRNNCell(encoding_cells, state_is_tuple=True)
        decoding_cells = tf.contrib.rnn.MultiRNNCell(decoding_cells, state_is_tuple=True)

        GO_SYMBOL = tf.zeros(shape=(self._batch_size, self._num_nodes * self._num_nodes * self._input_dim))

        # # get the list of tensors with number of list size
        # inputs = tf.unstack(inputs, axis=1)
        # for input_i in inputs:
        #     for i in range(num_filters):
        #         nb_filter = self._conv_filter_num[i]
        #         with tf.variable_scope(scope_name+'fc_{}'.format(i), reuse=True):
        #             input_i = self._fc(input_i, nb_filter)
        global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope(scope_name):
            # inputs and labels are lists
            print("Input shape is ", inputs.get_shape())
            inputs = tf.unstack(tf.reshape(inputs, (self._batch_size, self._seq_len,
                                                    self._num_nodes * self._num_nodes * self._input_dim)),
                                axis=1)
            labels_i = tf.unstack(
                tf.reshape(labels[..., :self._output_dim],
                           (self._batch_size, self._horizon, self._num_nodes *
                            self._num_nodes * self._output_dim)), axis=1)
            if self._aux_dim > 0:
                aux_info = tf.unstack(labels[..., self._output_dim:], axis=1)
                aux_info.insert(0, None)
            labels_i.insert(0, GO_SYMBOL)
            def loop_function(prev, i):
                """
                This function will be applied to the i-th output in order to generate the i+1-st input

                :param prev: 2D Tensor of shape [batch_size x output_size],
                :param i: integer, the step number (when advanced control is needed),
                :return: 2D Tensor of shape [batch_size x input_size].
                """
                if is_training:
                    # Return either the model's prediction or the previous ground truth in training.
                    if self._use_curriculum_learning:
                        c = tf.random_uniform((), minval=0, maxval=1.)
                        threshold = self._compute_sampling_threshold(global_step, self._cl_decay_steps)
                        result = tf.cond(tf.less(c, threshold), lambda: labels_i[i], lambda: prev)
                    else:
                        result = labels_i[i]
                else:
                    # Return the prediction of the model in testing.
                    result = prev
                # if self._aux_dim > 0:
                #     result = tf.reshape(result, (
                #     self._batch_size, self._num_coarsened_nodes, self._num_nodes, self._output_dim))
                #     result = tf.concat([result, aux_info[i]], axis=-1)
                #     result = tf.reshape(result, (
                #     self._batch_size, self._num_coarsened_nodes * self._num_nodes * self._input_dim))
                return prev

            _, enc_state = tf.contrib.rnn.static_rnn(encoding_cells, inputs, dtype=tf.float32)
            outputs, final_state = legacy_seq2seq.rnn_decoder(labels_i, enc_state, decoding_cells,
                                                              loop_function=None)

            outputs_latent = []
            for h_i, output in enumerate(outputs[:-1]):
                output = tf.reshape(output, shape=[
                    self._batch_size, self._num_nodes, self._latent_dim, self._output_dim])
                outputs_latent.append(output)
            outputs = tf.stack(outputs_latent, axis=1)

        return outputs

    def _construct_fc_gru_seq2seq_input_generic(self, inputs, labels=None,
                                                scope_name=None, is_training=False):
        # new code for constructing GRUCell by applying 2D graph convolution and pooling,
        #  even possibly fc layer
        # Do GCNN on Origin side
        encoding_cells, decoding_cells = [], []
        num_filters = len(self._conv_filter_size)
        for i in range(num_filters):
            nb_filter = self._conv_filter_num[i]
            cell = tf.nn.rnn_cell.GRUCell(num_units=nb_filter)
            if is_training:
                keep_out = 1 - self._dropout
                cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell, input_keep_prob=keep_out,
                    output_keep_prob=keep_out,
                    state_keep_prob=keep_out)
            encoding_cells.append(cell)
            decoding_cells.append(cell)

        encoding_cells = tf.contrib.rnn.MultiRNNCell(encoding_cells, state_is_tuple=True)
        decoding_cells = tf.contrib.rnn.MultiRNNCell(decoding_cells, state_is_tuple=True)

        GO_SYMBOL = tf.zeros(shape=(self._batch_size, self._num_nodes * self._num_nodes * self._input_dim))

        # # get the list of tensors with number of list size
        # inputs = tf.unstack(inputs, axis=1)
        # for input_i in inputs:
        #     for i in range(num_filters):
        #         nb_filter = self._conv_filter_num[i]
        #         with tf.variable_scope(scope_name+'fc_{}'.format(i), reuse=True):
        #             input_i = self._fc(input_i, nb_filter)

        with tf.variable_scope(scope_name):
            # inputs and labels are lists
            print("Input shape is ", inputs.get_shape())
            inputs = tf.unstack(tf.reshape(inputs, (self._batch_size, self._seq_len,
                                                    self._num_nodes * self._num_nodes * self._input_dim)),
                                axis=1)
            labels_i = tf.unstack(
                tf.reshape(labels[..., :self._output_dim],
                           (self._batch_size, self._horizon, self._num_nodes *
                            self._num_nodes * self._output_dim)), axis=1)
            if self._aux_dim > 0:
                aux_info = tf.unstack(labels[..., self._output_dim:], axis=1)
                aux_info.insert(0, None)
            labels_i.insert(0, GO_SYMBOL)

            _, enc_state = tf.contrib.rnn.static_rnn(encoding_cells, inputs, dtype=tf.float32)
            outputs, final_state = legacy_seq2seq.rnn_decoder(labels_i, enc_state, decoding_cells,
                                                              loop_function=None)

            outputs_latent = []
            for h_i, output in enumerate(outputs[:-1]):
                outputs_latent.append(output)
            outputs = tf.stack(outputs_latent, axis=1)

        return outputs

    # def _fc(self, inputs, output_size, bias_start=1.0, is_training=False):
    #     """Graph convolution between input and the graph matrix.
    #
    #     :param inputs: a Tensor: (batch_size, c_num_nodes, o_num_nodes, input_dim/state_dim)
    #     :param adj_mx1: adjacency matrix
    #     :param feat_size: the size of filters
    #     :param output_size: #filters
    #     :param pool_size: pooling size
    #     :param bias_start: bias
    #     :return:(batch_size, output_size)
    #     """
    #
    #     inputs = tf.reshape(inputs, (batch_size, -1))
    #     input_size = inputs.get_shape()[2].value # o_nodes * feature
    #     dtype = inputs.dtype
    #
    #     weights = tf.get_variable(
    #         'weights', [input_size , output_size], dtype=dtype,
    #         initializer=tf.contrib.layers.xavier_initializer())
    #     inputs = tf.matmul(inputs, weights)  # (batch_size * self._num_nodes, output_size)
    #
    #     biases = tf.get_variable("biases", [output_size], dtype=dtype,
    #         initializer=tf.constant_initializer(bias_start, dtype=dtype))
    #     inputs = tf.nn.bias_add(inputs, biases)
    #     inputs = tf.layers.dropout(inputs, self._dropout, training=is_training)
    #
    #     inputs = tf.nn.relu(inputs)
    #
    #     return inputs

    # def _project_latent_feature(self, x, latent_dim, bias_start=1.0):
    #     """
    #     Project the results from gcnn to latent space
    #
    #     :param input: 4D tensor with (batch_size, num_node * o_node * state_dim)
    #     :param latent_dim: latent dim for each output_dim
    #     :return:
    #     """
    #     x = tf.transpose(x, perm=[0, 2, 1, 3])
    #     x = tf.reshape(x, shape=[self._batch_size*self._num_nodes, -1])
    #     origin_feat_size = int(x.get_shape()[1])
    #     output_feat_size = latent_dim * self._output_dim
    #     weights = tf.get_variable(
    #         'weights', [origin_feat_size, output_feat_size], dtype=dtype,
    #         initializer=tf.contrib.layers.xavier_initializer())
    #     x = tf.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)
    #
    #     biases = tf.get_variable("biases", [output_feat_size], dtype=dtype,
    #                              initializer=tf.constant_initializer(bias_start, dtype=dtype))
    #     x = tf.nn.bias_add(x, biases)
    #     # reshape output back to 2D: (batch_size, o_node * latent_dim * output_dim)
    #     x = tf.reshape(x, shape=(batch_size, -1))
    #
    #     return x

    def _matmul_seq_latent_spaces(self, o_latents, d_latents):
        """

        :param o_latent: 4D tensor: [batch_size, horizon, num_nodes, latent_dim, output_dim]
        :param d_latent: 4D tensor: [batch_size, horizon, num_nodes, latent_dim, output_dim]
        :return:
        """
        o_latent_list = tf.unstack(o_latents, axis=0)
        d_latent_list = tf.unstack(d_latents, axis=0)
        od_matrix_batches = []
        for i in range(self._batch_size):
            o_latent_h_list = tf.unstack(o_latent_list[i], axis=0)
            d_latent_h_list = tf.unstack(d_latent_list[i], axis=0)
            od_matrix_h_list = []
            for k in range(self._horizon):
                o_latent_i_o = tf.unstack(o_latent_h_list[k], axis=-1)
                d_latent_i_o = tf.unstack(d_latent_h_list[k], axis=-1)
                od_matrix_list = []
                for j in range(self._output_dim):
                    od_matrix_j = tf.matmul(o_latent_i_o[j], tf.transpose(d_latent_i_o[j], perm=[1, 0]))
                    # od_matrix_j = tf.expand_dims(od_matrix_j, axis=-1)
                    od_matrix_list.append(od_matrix_j)
                od_matrix = tf.stack(od_matrix_list, axis=-1)
                # od_matrix = tf.expand_dims(od_matrix, axis=0)
                od_matrix_h_list.append(od_matrix)
            od_matrix_h = tf.stack(od_matrix_h_list, axis=0)

            od_matrix_batches.append(od_matrix_h)
        od_matrix_converted = tf.stack(od_matrix_batches, axis=0)
        if self._mode == 'avg':
            od_matrix_converted = tf.nn.sigmoid(od_matrix_converted)
        else:
            # if output should be hist, do softmax to make it reasonable
            od_matrix_converted = tf.nn.softmax(od_matrix_converted, -1)
        # od_matrix_converted = tf.concat(od_matrix_batches, axis=0)

        return od_matrix_converted

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return tf.cast(k / (k + tf.exp(global_step / k)), tf.float32)

    @staticmethod
    def run_epoch(sess, model, inputs, labels, weights, nums,
                  return_output=False, train_op=None, writer=None, shuffle=False):
        losses = []
        outputs = []
        trace_loss = []

        fetches = {
            'loss': model.loss,
            'global_step': tf.train.get_or_create_global_step()
        }

        if model.model_construction == 'od':
            fetches.update({'trace_loss': model.trace_loss})

        if train_op is not None:
            fetches.update({
                'train_op': train_op,
            })
            merged = model.merged
            if merged is not None:
                fetches.update({'merged': merged})

        if return_output:
            fetches.update({
                'outputs': model.outputs
            })

        if shuffle:
            inputs, labels, weights, nums = \
                FCRNNModel.shuffle_data(inputs, labels, weights, nums)

        for _, (x, y, w, num_i) in enumerate(zip(
                inputs, labels, weights, nums)):
            # train on one batch
            feed_dict = {
                model.inputs: x,
                model.labels: y,
                model.weight: w,
                model.num: num_i
            }
            vals = sess.run(fetches, feed_dict=feed_dict)

            losses.append(vals['loss'])
            if model.model_construction == 'od':
                trace_loss.append(vals['trace_loss'])
            if writer is not None and 'merged' in vals:
                writer.add_summary(vals['merged'], global_step=vals['global_step'])
            if return_output:
                outputs.append(vals['outputs'])

        results = {
            'loss': np.nanmean(losses),
            'trace_loss': np.nanmean(trace_loss)
        }
        if return_output:
            results['outputs'] = outputs

        return results

    @staticmethod
    def run_step(sess, model, inputs, labels, weights, nums,
                 return_output=False, train_op=None, writer=None):
        losses = []
        maes = []
        outputs = []

        fetches = {
            'mae': model.mae,
            'loss': model.loss,
            'global_step': tf.train.get_or_create_global_step()
        }
        if train_op is not None:
            fetches.update({
                'train_op': train_op,
            })
            merged = model.merged
            if merged is not None:
                fetches.update({'merged': merged})

        if return_output:
            fetches.update({
                'outputs': model.outputs
            })

        num_elements = len(inputs)
        rand_int = np.random.randint(num_elements)
        feed_dict = {
            model.inputs: inputs[rand_int],
            model.labels: labels[rand_int],
            model.weight: weights[rand_int],
            model.num: nums[rand_int]
        }

        print("Num shape is ", nums.shape)

        vals = sess.run(fetches, feed_dict=feed_dict)

        losses.append(vals['loss'])
        maes.append(vals['mae'])
        if writer is not None and 'merged' in vals:
            writer.add_summary(vals['merged'], global_step=vals['global_step'])
        if return_output:
            outputs.append(vals['outputs'])

        results = {
            'loss': vals['loss'],
            'mae': vals['mae'],
            'global_step': vals['global_step']
        }
        if return_output:
            results['outputs'] = outputs

        return results

    @staticmethod
    def shuffle_data(inputs, labels, weights, nums):
        num_elements = len(inputs)
        seq_ind = np.arange(num_elements, dtype=int)
        ran_ind = np.random.choice(seq_ind, num_elements, replace=False)
        inputs = inputs[ran_ind, ...]
        labels = labels[ran_ind, ...]
        weights = weights[ran_ind, ...]
        nums = nums[ran_ind, ...]

        return inputs, labels, weights, nums

    @property
    def weight(self):
        return self._weight

    @property
    def num(self):
        return self._num