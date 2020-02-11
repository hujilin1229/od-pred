"""
Base class for tensorflow models for traffic forecasting.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from lib.utils import calculate_normalized_laplacian

class TFModel(object):
    def __init__(self, config, scaler=None, **kwargs):
        """
        Initialization including placeholders, learning rate,
        :param config:
        :param scaler: data z-norm normalizer
        :param kwargs:
        """
        self._config = dict(config)

        # Placeholders for input and output.
        self._inputs = None
        self._labels = None
        self._outputs = None

        # Scaler for data normalization.
        self._scaler = scaler

        # Train and loss
        self._loss = None
        self._mae = None
        self._train_op = None

        # Learning rate.
        learning_rate = config.get('learning_rate', 0.001)
        self._lr = tf.get_variable('learning_rate', shape=(), initializer=tf.constant_initializer(learning_rate),
                                   trainable=False)
        self._new_lr = tf.placeholder(tf.float32, shape=(), name='new_learning_rate')
        self._lr_update = tf.assign(self._lr, self._new_lr, name='lr_update')

        # Log merged summary
        self._merged = None

    def _get_norm_rank(self, o_output, d_output, adj_mat):
        """
        Calculate the rank loss for o_output and d_output

        :param o_output: latent tensor for O
        :param d_output: latent tensor for D
        :param adj_mat: adjacent matrix for OD
        :return: rank loss in sum
        """

        normalized_lap = calculate_normalized_laplacian(adj_mat)
        normalized_lap = normalized_lap.todense().astype(np.float32)
        o_output = tf.reshape(tf.transpose(o_output, perm=[1, 0, 2, 3, 4]), [self._num_nodes, -1])
        d_output = tf.reshape(tf.transpose(d_output, perm=[1, 0, 2, 3, 4]), [self._num_nodes, -1])
        o_norm = tf.matmul(tf.matmul(tf.transpose(o_output, perm=[1, 0]), normalized_lap), o_output)
        d_norm = tf.matmul(tf.matmul(tf.transpose(d_output, perm=[1, 0]), normalized_lap), d_output)

        return tf.trace(o_norm) + tf.trace(d_norm)

    @staticmethod
    def run_epoch(sess, model, inputs, labels, weights,
                  o_inputs, d_inputs, o_labels, d_labels, nums,
                  return_output=False, train_op=None, writer=None, shuffle=False):
        losses = []
        maes = []
        outputs = []

        fetches = {
            'mae': model.mae,
            'loss': model.loss,
            'global_step': tf.train.get_or_create_global_step()
        }
        if train_op:
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

        for _, (x, y) in enumerate(zip(inputs, labels)):
            feed_dict = {
                model.inputs: x,
                model.labels: y,
            }

            vals = sess.run(fetches, feed_dict=feed_dict)

            losses.append(vals['loss'])
            maes.append(vals['mae'])
            if writer is not None and 'merged' in vals:
                writer.add_summary(vals['merged'], global_step=vals['global_step'])
            if return_output:
                outputs.append(vals['outputs'])

        results = {
            'loss': np.mean(losses),
            'mae': np.mean(maes)
        }
        if return_output:
            results['outputs'] = outputs
        return results

    def get_lr(self, sess):
        return np.asscalar(sess.run(self._lr))

    def set_lr(self, sess, lr):
        sess.run(self._lr_update, feed_dict={
            self._new_lr: lr
        })

    @property
    def trace_loss(self):
        return self._trace_loss

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def seq_len(self):
        return self._seq_len

    @property
    def horizon(self):
        return self._horizon

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def num_coarsened_nodes(self):
        return self._num_coarsened_nodes

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def loss(self):
        return self._loss

    @property
    def lr(self):
        return self._lr

    @property
    def mae(self):
        return self._mae

    @property
    def merged(self):
        return self._merged

    @property
    def outputs(self):
        return self._outputs

    @property
    def train_op(self):
        return self._train_op
