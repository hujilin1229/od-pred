from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import legacy_seq2seq
from lib.metrics_weight import masked_mse_loss, masked_mae_loss, \
    masked_rmse_loss, masked_mape_loss, \
    masked_kl_loss, masked_l2_loss, masked_emd_loss
from model.tf_model import TFModel
from model.mgrnn_cell import MGGRUCell
from model.dcrnn_cell import DCGRUCell
import numpy as np
import scipy.sparse as sp
from lib.utils import calculate_scaled_laplacian, calculate_normalized_laplacian
import json

class MGRNNModel(TFModel):
    def __init__(self, is_training, config, scaler=None, adj_mxs=None, adj_origin=None):
        super(MGRNNModel, self).__init__(config, scaler=scaler)
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
        # define two num_nodes: un-coarsened and coarsened
        self._num_nodes = int(config.get('num_nodes', 1))
        self._mode = config.get('mode', 'avg')
        self._num_coarsened_nodes = int(config.get('num_coarsened_nodes', self._num_nodes))
        num_rnn_layers = int(config.get('num_rnn_layers', 1))
        # self._output_dim = int(config.get('output_dim', 1))
        self._seq_len = int(config.get('seq_len'))
        self._use_curriculum_learning = bool(config.get('use_curriculum_learning', False))

        self._conv_filter_size = config.get('c_filter_size', [max_diffusion_step] * num_rnn_layers)
        self._conv_filter_num = config.get('nb_c_filter', [num_rnn_layers] * num_rnn_layers)
        self._pool_size = config.get('pooling', [1] * num_rnn_layers)
        self._fc_size = config.get('nb_fc_filter', [])
        self._adj_mxs = adj_mxs
        self._adj_origin = adj_origin
        self._aux_dim = self._input_dim - self._output_dim

        # define activate function
        activate_fun = config.get('activate_func', 'tanh')
        if activate_fun == 'tanh':
            self.act_fun = tf.nn.tanh
        elif activate_fun == 'sigmoid':
            self.act_fun = tf.nn.sigmoid
        else:
            self.act_fun = tf.nn.relu
        # define the pooling function
        self._pool_type = config.get('pool_type', '_mpool')
        self.pool = getattr(self, self._pool_type)

        trace_ratio = float(config.get('trace_ratio', 0.0))
        # assert input_dim == output_dim, 'input_dim: %d != output_dim: %d' % (input_dim, output_dim)
        # Input (batch_size, timesteps, num_sensor, input_dim)
        self._inputs = tf.placeholder(tf.float32,
                                      shape=(self._batch_size, self._seq_len, self._num_nodes,
                                             self._num_nodes, self._input_dim),
                                      name='inputs')

        self._o_inputs = tf.placeholder(tf.float32,
                                        shape=(self._batch_size, self._seq_len, self._num_nodes,
                                               self._num_coarsened_nodes, self._input_dim),
                                        name='inputs_O')
        self._d_inputs = tf.placeholder(tf.float32,
                                        shape=(self._batch_size, self._seq_len,
                                               self._num_coarsened_nodes,
                                               self._num_nodes, self._input_dim),
                                        name='inputs_D')
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
        # implement seq2seq
        self._o_labels = tf.placeholder(tf.float32,
                                        shape=(self._batch_size, self._horizon, self._num_nodes,
                                               self._num_coarsened_nodes, self._input_dim),
                                        name='labels_O')
        self._d_labels = tf.placeholder(tf.float32,
                                        shape=(self._batch_size, self._horizon,
                                               self._num_coarsened_nodes,
                                               self._num_nodes, self._input_dim),
                                        name='labels_D')

        o_inputs = tf.transpose(self._o_inputs, perm=(0, 1, 3, 2, 4))
        o_labels = tf.transpose(self._o_labels, perm=(0, 1, 3, 2, 4))

        # # # use self defined gru cell
        # o_output = self._construct_gru_seq2seq_input_row(o_inputs, o_labels, is_training, 'O_inputs')
        # d_output = self._construct_gru_seq2seq_input_row(self._d_inputs, self._d_labels, is_training, 'D_inputs')

        # use DCRNN cell
        with tf.variable_scope('O_inputs'):
            o_output = self._construct_gcnn_gru_seq2seq_input(o_inputs, o_labels, is_training, 'O_inputs')
        with tf.variable_scope('D_inputs'):
            d_output = self._construct_gcnn_gru_seq2seq_input(self._d_inputs, self._d_labels, is_training, 'D_inputs')

        # using the vila gru
        # o_output = self._construct_gcnn_vila_gru_seq2seq_input(o_inputs, o_labels, is_training, 'O_inputs')
        # d_output = self._construct_gcnn_vila_gru_seq2seq_input(self._d_inputs, self._d_labels, is_training, 'D_inputs')

        # print("D Output shape is ", d_output.get_shape())
        # print("Construction Seq to Seq model complete...")
        with tf.variable_scope('Outputs'):
            self._outputs = self._matmul_seq_latent_spaces(o_output, d_output)
        # preds = self._outputs[..., 0]
        preds = tf.identity(self._outputs, 'output')
        labels = self._labels[..., :self._output_dim]

        self._trace_loss = self._get_norm_rank(o_output, d_output, self._adj_origin)
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

    def _construct_gru_seq2seq_input(self, inputs, labels=None, is_training=False, scope_name=None):
        # new code for constructing GRUCell by applying 2D graph convolution and pooling,
        #  even possibly fc layer
        # Do GCNN on Origin side
        encoding_cells, decoding_cells = [], []
        num_filters = len(self._conv_filter_size)
        adj_index = 0
        output_proj_dim = self._num_coarsened_nodes * self._output_dim
        # output_proj_dim = self._latent_dim * self._output_dim
        # print("Output Proj Dim is ", output_proj_dim)
        for i in range(num_filters):
            nb_filter = self._conv_filter_num[i]
            fz_i = self._conv_filter_size[i]
            pool_size_i = self._pool_size[i]
            if type(self._adj_mxs[adj_index]) is np.ndarray:
                adj_i = self._adj_mxs[adj_index]
            else:
                adj_i = self._adj_mxs[adj_index].todense()
            coarsend_node = adj_i.shape[0]

            adj_index += int(np.log2(pool_size_i))
            if type(self._adj_mxs[adj_index]) is np.ndarray:
                adj_j = self._adj_mxs[adj_index]
            else:
                adj_j = self._adj_mxs[adj_index].todense()

            cell = MGGRUCell(num_units=nb_filter, adj_mx1=adj_i, adj_mx2=adj_j, max_diffusion_step=fz_i,
                             num_nodes=self._num_nodes, num_coarsed_nodes=coarsend_node,
                             filter_type=self._filter_type, pooling_size=pool_size_i)
            encoding_cells.append(cell)
            if i < num_filters - 1:
                decoding_cells.append(cell)
            else:
                cell = MGGRUCell(num_units=nb_filter, adj_mx1=adj_i, adj_mx2=adj_j, max_diffusion_step=fz_i,
                                 num_nodes=self._num_nodes, num_coarsed_nodes=coarsend_node,
                                 num_proj=output_proj_dim,
                                 filter_type=self._filter_type, pooling_size=pool_size_i)
                decoding_cells.append(cell)

        encoding_cells = tf.contrib.rnn.MultiRNNCell(encoding_cells, state_is_tuple=True)
        decoding_cells = tf.contrib.rnn.MultiRNNCell(decoding_cells, state_is_tuple=True)

        GO_SYMBOL = tf.zeros(shape=(self._batch_size, self._num_coarsened_nodes *
                                    self._num_nodes * self._input_dim))
        global_step = tf.train.get_or_create_global_step()
        # Outputs: (batch_size, timesteps, num_nodes, output_dim)
        with tf.variable_scope(scope_name):
            # inputs and labels are lists
            print("Input shape is ", inputs.get_shape())
            inputs = tf.unstack(tf.reshape(inputs, (self._batch_size, self._seq_len,
                                                    self._num_coarsened_nodes * self._num_nodes * self._input_dim)),
                                axis=1)
            labels_i = tf.unstack(
                tf.reshape(labels[..., :self._output_dim],
                           (self._batch_size, self._horizon, self._num_coarsened_nodes *
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
                if self._aux_dim > 0:
                    result = tf.reshape(result, (
                    self._batch_size, self._num_coarsened_nodes, self._num_nodes, self._output_dim))
                    result = tf.concat([result, aux_info[i]], axis=-1)
                    result = tf.reshape(result, (
                    self._batch_size, self._num_coarsened_nodes * self._num_nodes * self._input_dim))
                return result

            _, enc_state = tf.contrib.rnn.static_rnn(encoding_cells, inputs, dtype=tf.float32)
            outputs, final_state = legacy_seq2seq.rnn_decoder(labels_i, enc_state, decoding_cells,
                                                              loop_function=loop_function)

            outputs_latent = []
            for h_i, output in enumerate(outputs[:-1]):
                with tf.variable_scope(scope_name + '{}'.format(h_i)):
                    output = self._construct_latent_space(output, dtype=tf.float32)
                    output = tf.nn.leaky_relu(output)
                    outputs_latent.append(output)
            # Dim of outputs is [Batch_size, Horizon, #Node, Latent_Dim]
            outputs = tf.stack(outputs_latent, axis=1)

        return outputs

    def _construct_latent_space(self, output, dtype=None):
        """
        Construct the latent space by giving an output

        :param output: 2D tensor: [batch_size, num_nodes * oarsened_nodes * output_dim]
        :return: 4D tensor: [batch_size, num_nodes, latent_dim, output_dim]
        """

        with tf.variable_scope('output_latent_space'):
            output = tf.reshape(output, shape=[self._batch_size * self._num_nodes, -1])
            cur_feature = output.get_shape()[-1]
            weights = tf.get_variable(
                'weights', [cur_feature, self._latent_dim * self._output_dim], dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer())
            output = tf.matmul(output, weights)  # (batch_size * self._num_nodes, output_size)

            biases = tf.get_variable(
                "biases", [self._latent_dim * self._output_dim],
                dtype=dtype,
                initializer=tf.constant_initializer(1.0, dtype=dtype))
            output = tf.nn.bias_add(output, biases)
            output = tf.reshape(output, shape=[
                self._batch_size, self._num_nodes, self._latent_dim, self._output_dim])
            output = tf.nn.dropout(output, )

            return output

    def _construct_gru_seq2seq_input_row(self, inputs, labels=None, is_training=False,
                                         scope_name=None):
        """
        Do gcnn on 3rd dimension

        :param inputs: 5D tensor with shape (batch_size, seq_len, num_coarsen, num_nodes, input_dim)
        :param labels: 5D tensor with shape (batch_size, horizon, num_coarsen, num_nodes, input_dim)
        :param is_training: bool, training or not
        :param scope_name:
        :return: 5D tensor with shape (batch_size, horizon_len, num_nodes, latent_dim, output_dim)
        """

        # get the shape of inputs
        num_nodes = inputs.get_shape()[3].value

        encoding_cells, decoding_cells = [], []
        num_filters = len(self._conv_filter_size)
        adj_index = 0
        output_proj_dim = self._latent_dim * self._output_dim
        # print("Output Proj Dim is ", output_proj_dim)
        for i in range(num_filters):
            nb_filter = self._conv_filter_num[i]
            fz_i = self._conv_filter_size[i]
            pool_size_i = self._pool_size[i]
            if type(self._adj_mxs[adj_index]) is np.ndarray:
                adj_i = self._adj_mxs[adj_index]
            else:
                adj_i = self._adj_mxs[adj_index].todense()
            coarsend_node = adj_i.shape[0]

            adj_index += int(np.log2(pool_size_i))
            if type(self._adj_mxs[adj_index]) is np.ndarray:
                adj_j = self._adj_mxs[adj_index]
            else:
                adj_j = self._adj_mxs[adj_index].todense()

            cell = MGGRUCell(num_units=nb_filter, adj_mx1=adj_i, adj_mx2=adj_j, max_diffusion_step=fz_i,
                             num_nodes=num_nodes, num_coarsed_nodes=coarsend_node, activation=self.act_fun,
                             filter_type=self._filter_type, pooling_size=pool_size_i, pool_type=self._pool_type,
                             reuse=tf.AUTO_REUSE)
            encoding_cells.append(cell)
            if i < num_filters - 1:
                decoding_cells.append(cell)
            else:
                cell = MGGRUCell(num_units=nb_filter, adj_mx1=adj_i, adj_mx2=adj_j, max_diffusion_step=fz_i,
                                 num_nodes=num_nodes, num_coarsed_nodes=coarsend_node, activation=self.act_fun,
                                 num_proj=output_proj_dim, filter_type=self._filter_type, pooling_size=pool_size_i,
                                 pool_type=self._pool_type, reuse=tf.AUTO_REUSE)
                decoding_cells.append(cell)

        encoding_cells = tf.contrib.rnn.MultiRNNCell(encoding_cells, state_is_tuple=True)
        decoding_cells = tf.contrib.rnn.MultiRNNCell(decoding_cells, state_is_tuple=True)

        GO_SYMBOL = tf.zeros(shape=(self._batch_size, self._num_coarsened_nodes *
                                    num_nodes * self._input_dim))

        # Outputs: (batch_size, timesteps, num_nodes, output_dim)
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            # inputs and labels are lists
            inputs = tf.unstack(tf.reshape(inputs, (self._batch_size, self._seq_len,
                                                    self._num_coarsened_nodes * num_nodes * self._input_dim)),
                                axis=1)
            labels_i = tf.unstack(
                tf.reshape(labels[..., :self._output_dim],
                           (self._batch_size, self._horizon, self._num_coarsened_nodes *
                            num_nodes * self._output_dim)), axis=1)
            labels_i.insert(0, GO_SYMBOL)

            _, enc_state = tf.contrib.rnn.static_rnn(encoding_cells, inputs, dtype=tf.float32)
            outputs, final_state = legacy_seq2seq.rnn_decoder(
                labels_i, enc_state, decoding_cells, loop_function=None)

            outputs_latent = []
            for h_i, output in enumerate(outputs[:-1]):
                output = tf.reshape(output, shape=[
                    self._batch_size, num_nodes, self._latent_dim, self._output_dim])
                outputs_latent.append(output)
            outputs = tf.stack(outputs_latent, axis=1)

        return outputs

    def _construct_gcnn_gru_seq2seq_input(self, inputs, labels=None, is_training=False, scope_name=None):
        """
        Apply GCNN on the signal before feeding into RNN

        :param inputs:
        :param labels:
        :param scope_name:
        :return:
        """
        num_filters = len(self._conv_filter_size)
        final_coarsened_nodes = self._num_coarsened_nodes
        for i in range(num_filters):
            final_coarsened_nodes /= self._pool_size[i]

        # get the list of tensors with number of list size
        filters_num = self._conv_filter_num.copy()
        filters_range = self._conv_filter_size.copy()
        pool_size = self._pool_size.copy()
        filters_num = filters_num + [self._output_dim]
        filters_range = filters_range + [int(filters_range[-1]/pool_size[-1])]
        pool_size = pool_size + [1]
        gcnn_inputs = self._od_encoder(inputs, filters_num, filters_range, pool_size, scope_name)
        gcnn_labels = self._od_encoder(labels, filters_num, filters_range, pool_size, scope_name)
        gcnn_labels = [tf.reshape(label_i, [self._batch_size, -1]) for label_i in gcnn_labels]
        self._latent_dim = int(gcnn_inputs[0].get_shape()[-1].value / self._output_dim)
        GO_SYMBOL = tf.zeros(shape=(self._batch_size, self._num_nodes * self._output_dim *
                                    self._latent_dim))
        gcnn_labels.insert(0, GO_SYMBOL)

        # TODO: include the following parameter into config
        rnn_units = 32
        max_diffusion_step = 4
        num_rnn_layers = 2
        cell = DCGRUCell(rnn_units, self._adj_origin, max_diffusion_step=max_diffusion_step,
                         num_nodes=self._num_nodes, drop_out=self._dropout, is_training=is_training,
                         filter_type=self._filter_type)
        cell_with_projection = DCGRUCell(rnn_units, self._adj_origin, max_diffusion_step=max_diffusion_step,
                                         num_nodes=self._num_nodes, drop_out=self._dropout, is_training=is_training,
                                         num_proj=self._latent_dim*self._output_dim, filter_type=self._filter_type)
        encoding_cells = [cell] * num_rnn_layers
        # make sure the dim of output for decoding is num_nodes
        decoding_cells = [cell] * (num_rnn_layers - 1) + [cell_with_projection]
        encoding_cells = tf.contrib.rnn.MultiRNNCell(encoding_cells, state_is_tuple=True)
        decoding_cells = tf.contrib.rnn.MultiRNNCell(decoding_cells, state_is_tuple=True)

        global_step = tf.train.get_or_create_global_step()
        # Outputs: (batch_size, timesteps, num_nodes, output_dim)
        with tf.variable_scope('DCRNN_SEQ'):
            def loop_function(prev, i):
                if is_training:
                    # Return either the model's prediction or the previous ground truth in training.
                    if self._use_curriculum_learning:
                        c = tf.random_uniform((), minval=0, maxval=1.)
                        threshold = self._compute_sampling_threshold(global_step, self._cl_decay_steps)
                        result = tf.cond(tf.less(c, threshold), lambda: gcnn_labels[i], lambda: prev)
                    else:
                        result = gcnn_labels[i]
                else:
                    # Return the prediction of the model in testing.
                    result = prev
                return result

            _, enc_state = tf.contrib.rnn.static_rnn(encoding_cells, gcnn_inputs, dtype=tf.float32)
            outputs, final_state = legacy_seq2seq.rnn_decoder(gcnn_labels, enc_state, decoding_cells,
                                                              loop_function=loop_function)
            outputs_latent = []
            for h_i, output in enumerate(outputs[:-1]):
                output = tf.reshape(output, shape=[
                    self._batch_size, self._num_nodes, self._latent_dim, self._output_dim])
                outputs_latent.append(output)
            outputs = tf.stack(outputs_latent, axis=1)

        return outputs

    def _construct_gcnn_vila_gru_seq2seq_input(
            self, inputs, labels=None, is_training=False, scope_name=None):
        # new code for constructing GRUCell by applying 2D graph convolution and pooling,
        #  even possibly fc layer
        # Do GCNN on Origin side

        num_filters = len(self._conv_filter_size)
        final_coarsened_nodes = self._num_coarsened_nodes
        for i in range(num_filters):
            final_coarsened_nodes /= self._pool_size[i]

        # get the list of tensors with number of list size
        filters_num = self._conv_filter_num.copy()
        filters_range = self._conv_filter_size.copy()
        pool_size = self._pool_size.copy()
        filters_num = filters_num + [1]
        filters_range = filters_range + [int(filters_range[-1] / pool_size[-1])]
        pool_size = pool_size + [1]
        gcnn_inputs = self._od_encoder(inputs, filters_num, filters_range, pool_size, scope_name)
        gcnn_labels = self._od_encoder(labels, filters_num, filters_range, pool_size, scope_name)
        gcnn_labels = [tf.reshape(label_i, [self._batch_size, -1]) for label_i in gcnn_labels]
        self._latent_dim = int(gcnn_inputs[0].get_shape()[-1].value)
        gcnn_inputs = [tf.reshape(label_i, [self._batch_size, -1]) for label_i in gcnn_inputs]
        GO_SYMBOL = tf.zeros(shape=(self._batch_size, self._num_nodes * self._latent_dim))
        gcnn_labels.insert(0, GO_SYMBOL)

        encoding_cells, decoding_cells = [], []
        num_filters = len(self._conv_filter_size)
        output_proj_dim = self._latent_dim * self._output_dim * self._num_nodes
        # print("Output Proj Dim is ", output_proj_dim)
        for i in range(num_filters):
            nb_filter = self._num_nodes * self._latent_dim
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


        with tf.variable_scope(scope_name):
            _, enc_state = tf.contrib.rnn.static_rnn(encoding_cells, gcnn_inputs, dtype=tf.float32)
            outputs, final_state = legacy_seq2seq.rnn_decoder(
                gcnn_labels, enc_state, decoding_cells, loop_function=None)

            outputs_latent = []
            for h_i, output in enumerate(outputs[:-1]):
                output = tf.reshape(output, shape=[
                    self._batch_size, self._num_nodes, self._latent_dim, self._output_dim])
                outputs_latent.append(output)
            outputs = tf.stack(outputs_latent, axis=1)

        return outputs

    def _od_encoder(self, inputs, filters_num, filters_range, pool_size, scope_name):
        """
        Encoder the signal of OD matrix alone Coarsened Dimension

        :param inputs: 5D tensor, [batch_size, horizon, num_coarsened_nodes, num_nodes, input_dim]
        :param filters_num: list, number of filters
        :param filters_range: list, size of filters
        :param pool_size: list, size of pooling
        :param scope_name:
        :return: list of tensors, length is horizon;
                each tensor with [batch_size, num_nodes, latent * output_dim]
        """
        gcnn_inputs = []
        inputs = tf.unstack(inputs, axis=1)
        for input_i in inputs:
            adj_index = 0
            for i, nb_filter in enumerate(filters_num):
                fz_i = filters_range[i]
                pool_size_i = pool_size[i]
                if type(self._adj_mxs[adj_index]) is np.ndarray:
                    adj_i = self._adj_mxs[adj_index]
                else:
                    adj_i = self._adj_mxs[adj_index].todense()
                adj_index += int(np.log2(pool_size_i))
                with tf.variable_scope(scope_name+'gcnn_{}'.format(i), reuse=tf.AUTO_REUSE):
                    input_i = self._gconv_pool(input_i, adj_i, fz_i, nb_filter, pool_size_i)

            input_i = tf.reshape(input_i, [self._batch_size, self._num_nodes, -1])
            gcnn_inputs.append(input_i)

        return gcnn_inputs


    def _gconv_pool(self, inputs, adj_mx1, feat_size, output_size, pool_size,
                    bias_start=1.0, is_training=False):
        """Graph convolution between input and the graph matrix.

        :param inputs: a Tensor: (batch_size, c_num_nodes, o_num_nodes, input_dim/state_dim)
        :param adj_mx1: adjacency matrix
        :param feat_size: the size of filters
        :param output_size: #filters
        :param pool_size: pooling size
        :param bias_start: bias
        :return:
        """

        supports1 = [calculate_scaled_laplacian(adj_mx1, lambda_max=None)]
        # if self._filter_type == "laplacian":
        #     supports1.append(calculate_scaled_laplacian(adj_mx1, lambda_max=None))
        # elif self._filter_type == "random_walk":
        #     supports1.append(calculate_random_walk_matrix(adj_mx1).T)
        # elif self._filter_type == "dual_random_walk":
        #     supports1.append(calculate_random_walk_matrix(adj_mx1).T)
        #     supports1.append(calculate_random_walk_matrix(adj_mx1.T).T)
        # else:
        #     supports1.append(calculate_scaled_laplacian(adj_mx1))

        supports = []
        for support in supports1:
            supports.append(self._build_sparse_matrix(support))

        # Reshape input and state to (batch_size, c_num_nodes, o_num_nodes, input_dim/state_dim)
        batch_size, c_nodes, o_nodes, feat_dim = inputs.get_shape()
        batch_size, c_nodes, o_nodes, feat_dim = int(batch_size), int(c_nodes), int(o_nodes), int(feat_dim)
        # input_channel = self._input_dim * 2
        # transform input into (batch_size, c_nodes, o_nodes * feature)
        inputs = tf.reshape(inputs, (batch_size, c_nodes, -1))
        input_size = inputs.get_shape()[2].value # o_nodes * feature
        input_channel = int(input_size / self._num_nodes)
        dtype = inputs.dtype
        # print("Input shape ", inputs.get_shape())
        # print("State shape ", state.get_shape())
        # print("Inputs and States Shape ", inputs_and_state.get_shape())
        x = inputs
        x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, total_arg_size, batch_size)
        x0 = tf.reshape(x0, shape=[c_nodes, input_size * batch_size])
        x = tf.expand_dims(x0, axis=0)

        if feat_size == 0:
            pass
        else:
            for support in supports:
                x1 = tf.sparse_tensor_dense_matmul(support, x0)
                x = self._concat(x, x1)

                for k in range(2, feat_size + 1):
                    x2 = 2 * tf.sparse_tensor_dense_matmul(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        # #filter size
        num_matrices = len(supports) * feat_size + 1  # Adds for x itself.
        # print("X shape is ", x.get_shape())
        x = tf.reshape(x, shape=[num_matrices, c_nodes, self._num_nodes, input_channel, batch_size])
        x = tf.transpose(x, perm=[4, 1, 2, 3, 0])  # (batch_size, num_nodes, o_node, in_chan, order)
        x = tf.reshape(x, shape=[batch_size * c_nodes * self._num_nodes, input_channel * num_matrices])

        weights = tf.get_variable(
            'weights', [input_channel * num_matrices, output_size], dtype=dtype)
        x = tf.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        biases = tf.get_variable("biases", [output_size], dtype=dtype,
            initializer=tf.constant_initializer(bias_start, dtype=dtype))
        x = tf.nn.bias_add(x, biases)

        x = tf.layers.dropout(x, rate=self._dropout, training=is_training)
        # reshape output back to 4D: (batch_size, num_node * o_node * state_dim) ->
        # (batch_size, num_node, o_node, state_dim)
        x = tf.reshape(x, shape=(batch_size, c_nodes, -1, output_size))
        x = self.pool(x, pool_size)
        x = self.act_fun(x)
        # x = tf.nn.sigmoid(x)
        # x = tf.nn.tanh(x)
        # (batch_size, num_node, o_node, state_dim)
        return x

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

        :param o_latent: 4D tensor: [batch_size, num_nodes, latent_dim, output_dim]
        :param d_latent: 4D tensor: [batch_size, num_nodes, latent_dim, output_dim]
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
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        return tf.sparse_reorder(L)

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

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
    def run_epoch(sess, model, inputs, labels, weights,
                  o_inputs, d_inputs, o_labels, d_labels, nums,
                  return_output=False, train_op=None, writer=None, shuffle=False):
        losses = []
        outputs = []
        trace_loss = []

        fetches = {
            'trace_loss': model.trace_loss,
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

        if shuffle:
            inputs, labels, weights, o_inputs, d_inputs, o_labels, d_labels, nums = \
                MGRNNModel.shuffle_data(inputs, labels, weights, o_inputs,
                    d_inputs, o_labels, d_labels, nums)

        for _, (x, y, w, o_i, d_i, o_l, d_l, num_i) in enumerate(zip(
                inputs, labels, weights, o_inputs,
                d_inputs, o_labels, d_labels, nums)):
            # train on one batch
            feed_dict = {
                model.inputs: x,
                model.labels: y,
                model.weight: w,
                model.o_inputs: o_i,
                model.d_inputs: d_i,
                model.o_labels: o_l,
                model.d_labels: d_l,
                model.num: num_i
            }
            vals = sess.run(fetches, feed_dict=feed_dict)

            losses.append(vals['loss'])
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
    def run_step(sess, model, inputs, labels, weights,
                 o_inputs, d_inputs, o_labels, d_labels, nums,
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
            model.o_inputs: o_inputs[rand_int],
            model.d_inputs: d_inputs[rand_int],
            model.o_labels: o_labels[rand_int],
            model.d_labels: d_labels[rand_int],
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
    def shuffle_data(inputs, labels, weights, o_inputs,
                     d_inputs, o_labels, d_labels, nums):
        num_elements = len(inputs)
        seq_ind = np.arange(num_elements, dtype=int)
        ran_ind = np.random.choice(seq_ind, num_elements, replace=False)
        inputs = inputs[ran_ind, ...]
        labels = labels[ran_ind, ...]
        weights = weights[ran_ind, ...]
        o_inputs = o_inputs[ran_ind, ...]
        d_inputs = d_inputs[ran_ind, ...]
        o_labels = o_labels[ran_ind, ...]
        d_labels = d_labels[ran_ind, ...]
        nums = nums[ran_ind, ...]

        return inputs, labels, weights, o_inputs, \
               d_inputs, o_labels, d_labels, nums

    @property
    def o_inputs(self):
        return self._o_inputs

    @property
    def d_inputs(self):
        return self._d_inputs

    @property
    def weight(self):
        return self._weight

    @property
    def o_labels(self):
        return self._o_labels

    @property
    def d_labels(self):
        return self._d_labels

    @property
    def num(self):
        return self._num