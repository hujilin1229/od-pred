from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

from lib.utils import generate_graph_seq2seq_odsep_data_with_time
from lib import tf_utils, metrics, metrics_weight
from model.mgrnn_model import MGRNNModel
from model.tf_model_supervisor import TFModelSupervisor


class MGRNNSupervisor(TFModelSupervisor):
    """
    Do experiments using Graph Random Walk RNN model.
    """

    def __init__(self, traffic_reading_df, adj_mx, config,
                 origin_df_file, nodes, coarsed_dict=None):
        self._adj_mx = adj_mx
        self._nodes = nodes
        self._coarsed_dict = coarsed_dict
        self._origin_df_file = origin_df_file
        self._start_time = config['start_date']
        self._zone = config['zone']
        self._sample_rate = config['sample_rate']
        self._data_format = config['data_format']
        self._shuffle_training = config['shuffle_training']
        self._hopk = config['hopk']
        self._sigma = config['sigma']
        if not os.path.exists('./result/'):
            os.makedirs('./result/')
        self._result_dir = './result/'

        super(MGRNNSupervisor, self).__init__(config, df_data=traffic_reading_df)

    def _prepare_val_test_df(self, val_intervals, test_intervals):

        max_val = max(val_intervals)
        min_val = min(val_intervals)
        max_test = max(test_intervals)
        min_test = max(test_intervals)
        df_val_file = self._origin_df_file + 'df_val_{}_{}.csv'.format(min_val, max_val)
        df_test_file = self._origin_df_file + 'df_test_{}_{}.csv'.format(min_test, max_test)
        if os.path.exists(df_val_file) and os.path.exists(df_test_file):
            df_od_val = pd.read_csv(df_val_file)
            df_od_test = pd.read_csv(df_test_file)
        else:
            df_od = pd.read_csv(self._origin_df_file, parse_dates=['time'])
            start_time = datetime.strptime(self._start_time, '%Y-%m-%d')
            df_od = df_od.reset_index()
            df_od_timedelta = pd.to_timedelta(df_od['time'] - start_time)
            df_od['TI'] = (df_od_timedelta.dt.days * (60 * 24 / self._sample_rate) +
                           df_od_timedelta.dt.seconds / (60 * self._sample_rate)).astype(int)
            if self._zone == 'taxi_zone':
                pick_id = 'pickup_location_id'
                drop_id = 'dropoff_location_id'
            elif self._zone == 'lon10_lat9':
                pick_id = 'pickup_sr_id'
                drop_id = 'dropoff_sr_id'
            else:
                pick_id = 'pickup_nyct2010_gid'
                drop_id = 'dropoff_nyct2010_gid'

            df_od = df_od[['TI', pick_id, drop_id, 'pickup_longitude', 'pickup_latitude',
                           'dropoff_longitude', 'dropoff_latitude', 'manhat_dist', 'time_duration', 'speed']]
            df_od['o_id'] = df_od[pick_id].astype(int)
            df_od['d_id'] = df_od[drop_id].astype(int)
            del df_od[pick_id], df_od[drop_id]
            df_od_val = df_od[(df_od.TI <= max(val_intervals)) & (df_od.TI >= min(val_intervals))]
            df_od_test = df_od[(df_od.TI <= max(test_intervals)) & (df_od.TI >= min(test_intervals))]

            df_od_val.to_csv(df_val_file)
            df_od_test.to_csv(df_test_file)

        return df_od_val, df_od_test

    def _prepare_train_val_test_data(self):
        # Parsing model parameters.
        batch_size = self._get_config('batch_size')
        horizon = self._get_config('horizon')
        seq_len = self._get_config('seq_len')
        test_batch_size = self._get_config('test_batch_size')
        add_time_in_day = self._get_config('add_time_in_day')
        add_day_in_week = self._get_config('add_day_in_week')

        self._graphs = self._coarsed_dict['graphs']
        perm = self._coarsed_dict['perm']

        num_nodes = self._graphs[0].shape[0]
        x_train, y_train, weight_train, \
        self._ox_train, self._dx_train, \
        self._oy_train, self._dy_train, \
        self._inter_train_x, self._inter_train_y, self._nums_train = \
            generate_graph_seq2seq_odsep_data_with_time(self._train_dict,
                                                        batch_size=batch_size,
                                                        seq_len=seq_len,
                                                        horizon=horizon,
                                                        num_nodes=num_nodes,
                                                        scaler=self._scaler,
                                                        add_time_in_day=add_time_in_day,
                                                        add_day_in_week=add_day_in_week,
                                                        perm=perm)
        x_val, y_val, weight_val, \
        self._ox_val, self._dx_val, \
        self._oy_val, self._dy_val, \
        self._inter_val_x, self._inter_val_y, self._nums_val = \
            generate_graph_seq2seq_odsep_data_with_time(self._val_dict,
                                                        batch_size=test_batch_size,
                                                        seq_len=seq_len,
                                                        horizon=horizon,
                                                        num_nodes=num_nodes,
                                                        scaler=self._scaler,
                                                        add_time_in_day=add_time_in_day,
                                                        add_day_in_week=add_day_in_week,
                                                        perm=perm)
        x_test, y_test, weight_test, \
        self._ox_test, self._dx_test, \
        self._oy_test, self._dy_test, \
        self._inter_test_x, self._inter_test_y, self._nums_test = \
            generate_graph_seq2seq_odsep_data_with_time(self._test_dict,
                                                        batch_size=test_batch_size,
                                                        seq_len=seq_len,
                                                        horizon=horizon,
                                                        num_nodes=num_nodes,
                                                        scaler=self._scaler,
                                                        add_time_in_day=add_time_in_day,
                                                        add_day_in_week=add_day_in_week,
                                                        perm=perm)

        self._nb_batches = len(x_train)

        return x_train, y_train, weight_train, \
               x_val, y_val, weight_val, \
               x_test, y_test, weight_test

    def _build_train_val_test_models(self):
        # Builds the model.
        input_dim = self._x_train.shape[-1]
        num_nodes = self._adj_mx.shape[0]
        num_coarsened_nodes = self._graphs[0].shape[0]

        output_dim = self._get_config('output_dim')
        # input_dim = self._get_config('input_dim')
        train_batch_size = self._get_config('batch_size')
        test_batch_size = self._get_config('test_batch_size')
        train_config = dict(self._config)
        train_config.update({
            'batch_size': train_batch_size,
            'input_dim': input_dim,
            'num_nodes': num_nodes,
            'num_coarsened_nodes': num_coarsened_nodes,
            'output_dim': output_dim,
        })
        test_config = dict(self._config)
        test_config.update({
            'batch_size': test_batch_size,
            'input_dim': input_dim,
            'num_nodes': num_nodes,
            'num_coarsened_nodes': num_coarsened_nodes,
            'output_dim': output_dim,
        })
        with tf.name_scope('Train'):
            with tf.variable_scope('MGRNN', reuse=False):
                train_model = MGRNNModel(is_training=True, config=train_config, scaler=self._scaler,
                                         adj_mxs=self._graphs, adj_origin=self._adj_mx)
        with tf.name_scope('Val'):
            with tf.variable_scope('MGRNN', reuse=True):
                val_model = MGRNNModel(is_training=False, config=test_config, scaler=self._scaler,
                                       adj_mxs=self._graphs, adj_origin=self._adj_mx)
        with tf.name_scope('Test'):
            with tf.variable_scope('MGRNN', reuse=True):
                test_model = MGRNNModel(is_training=False, config=test_config, scaler=self._scaler,
                                        adj_mxs=self._graphs, adj_origin=self._adj_mx)

        return train_model, val_model, test_model

    def _convert_model_outputs_to_eval_df(self, y_preds, inter_y):
        # y_preds: list of (batch_size, horizon, num_nodes, num_nodes, output_dim)
        # inter_y: array: (epochs, batch_size, horizon)
        # horizon = y_preds.shape[2]

        # to make the order same with inter_y (epochs, batch_size, ...)
        y_preds = np.stack(y_preds, axis=0)
        nb_ypreds = y_preds.shape[0]
        bz_ypreds = y_preds.shape[1]
        h_ypreds = y_preds.shape[2]
        nb_epochs, batch_size, horizon = inter_y.shape
        assert nb_ypreds == nb_epochs
        assert bz_ypreds == batch_size
        assert h_ypreds == horizon

        num_nodes = y_preds.shape[3]
        output_dim = y_preds.shape[-1]
        if output_dim == 1:
            y_preds = np.reshape(y_preds, (nb_epochs * batch_size, horizon, num_nodes, num_nodes))
        else:
            y_preds = np.reshape(y_preds, (nb_epochs * batch_size, horizon, num_nodes, num_nodes, output_dim))
        inter_y = np.reshape(inter_y, (nb_epochs * batch_size, horizon))
        df_preds = {}
        for horizon_i in range(horizon):
            y_pred = np.reshape(y_preds[:, horizon_i, :], (nb_epochs * batch_size, num_nodes * num_nodes)).reshape(-1)
            o_id = np.tile(np.reshape(self._nodes, (num_nodes, 1)), (1, num_nodes)).reshape(-1)
            d_id = np.tile(np.reshape(self._nodes, (1, num_nodes)), (num_nodes, 1)).reshape(-1)
            o_ids = np.tile(np.reshape(o_id, (1, -1)), (nb_epochs * batch_size, 1)).reshape(-1)
            d_ids = np.tile(np.reshape(d_id, (1, -1)), (nb_epochs * batch_size, 1)).reshape(-1)
            inter_y_i = inter_y[:, horizon_i]
            inter_ys = np.tile(inter_y_i, (1, num_nodes * num_nodes)).reshape(-1)
            dict_df = {'pred': self._scaler.inverse_transform(y_pred),
                       'TI': inter_ys,
                       'o_id': o_ids,
                       'd_id': d_ids}
            df_pred = pd.DataFrame(dict_df)
            df_preds[horizon_i] = df_pred

        return df_preds

    def _train(self, sess, **kwargs):
        history = []
        min_val_loss = float('inf')
        wait = 0

        epochs = self._get_config('epochs')
        initial_lr = self._get_config('learning_rate')
        min_learning_rate = self._get_config('min_learning_rate')
        lr_decay_epoch = self._get_config('lr_decay_epoch')
        lr_decay = self._get_config('lr_decay')
        lr_decay_interval = self._get_config('lr_decay_interval')
        patience = self._get_config('patience')
        test_every_n_epochs = self._get_config('test_every_n_epochs')

        # configurations need to be defined in base_config.json
        save_model = self._get_config('save_model')
        max_to_keep = self._get_config('max_to_keep')
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)
        model_filename = self._get_config('model_filename')

        if model_filename is not None:
            saver.restore(sess, model_filename)
            self._train_model.set_lr(sess, self._get_config('learning_rate'))
            self._epoch = self._get_config('epoch') + 1
        else:
            sess.run(tf.global_variables_initializer())
            saver.save(sess, os.path.join(self._log_dir, 'model'))

        while self._epoch <= epochs:
            # Learning rate schedule.
            new_lr = self.calculate_scheduled_lr(initial_lr, epoch=self._epoch,
                                                 lr_decay=lr_decay, lr_decay_epoch=lr_decay_epoch,
                                                 lr_decay_interval=lr_decay_interval,
                                                 min_lr=min_learning_rate)
            if new_lr != initial_lr:
                self._logger.info('Updating learning rate to: %.6f' % new_lr)
                self._train_model.set_lr(sess=sess, lr=new_lr)
            sys.stdout.flush()

            start_time = time.time()
            train_results = MGRNNModel.run_epoch(sess, self._train_model,
                                                 inputs=self._x_train, labels=self._y_train, weights=self._wt_train,
                                                 o_inputs=self._ox_train, d_inputs=self._dx_train,
                                                 o_labels=self._oy_train, d_labels=self._dy_train,
                                                 nums=self._nums_train, train_op=self._train_model.train_op,
                                                 writer=self._writer, shuffle=self._shuffle_training)
            end_time = time.time()
            train_loss = train_results['loss']
            if train_loss > 1e5:
                self._logger.warn('Gradient explosion detected. Ending...')
                break

            global_step = sess.run(tf.train.get_or_create_global_step())
            # Compute validation error.
            val_results = MGRNNModel.run_epoch(sess, self._val_model, inputs=self._x_val,
                                               labels=self._y_val, weights=self._wt_val,
                                               o_inputs=self._ox_val, d_inputs=self._dx_val,
                                               o_labels=self._oy_val, d_labels=self._dy_val,
                                               nums=self._nums_val, return_output=True,
                                               train_op=None)
            val_loss = val_results['loss']
            val_trace_loss = val_results['trace_loss']
            tf_utils.add_simple_summary(self._writer,
                                        ['loss/train_loss', 'loss/val_loss'],
                                        [train_loss, val_loss], global_step=global_step)

            message = 'Epoch %d (%d) train_loss: %.4f, val_loss: %.4f, val_trace_loss: %.4f, %.6f s' % (
                self._epoch, global_step, train_loss, val_loss, val_trace_loss,
                (end_time - start_time)/self._x_train.shape[0])

            self._logger.info(message)
            if self._epoch % test_every_n_epochs == test_every_n_epochs - 1:
                self.test_and_write_result(sess=sess, global_step=global_step, epoch=self._epoch)

            if val_loss < min_val_loss:
                wait = 0
                if save_model > 0:
                    model_filename = self.save_model(sess, saver, val_loss)
                self._logger.info(
                    'Val loss decrease from %.4f to %.4f, saving to %s' % (min_val_loss, val_loss, model_filename))
                min_val_loss = val_loss
            # disable early stopping
            else:
                wait += 1
                if wait > patience:
                    model_filename = self.save_model(sess, saver, val_loss)
                    self._logger.warn('Early stopping at epoch: %d, saving to %s' % (self._epoch, model_filename))
                    break

            history.append(val_loss)
            # Increases epoch.
            self._epoch += 1

            sys.stdout.flush()
        return np.min(history)

    def _merge_pred_real(self, df_pred, df_real):
        """
        df_pred = {'pred': self._scaler.inverse_transform(y_pred),
                   'TI': inter_ys,
                   'o_id': o_ids,
                   'd_id': d_ids}
        df_real = df_od[['TI', 'o_id', 'd_id', 'pickup_longitude', 'pickup_latitude',
                           'dropoff_longitude', 'dropoff_latitude','manhat_dist', 'time_duration']]

        :param df_pred:
        :param df_real:
        :return:
        """

        df_merge = df_real.merge(df_pred, left_on=['TI', 'o_id', 'd_id'],
                                 right_on=['TI', 'o_id', 'd_id'], how='inner')
        if self._data_format == 'speed':
            df_merge['pred_duration'] = df_merge['manhat_dist'] / df_merge['pred']
        else:
            df_merge['pred_duration'] = df_merge['pred']

        return df_merge, df_merge['pred_duration'], df_merge['time_duration']

    def _test_and_write_result(self, sess, global_step, **kwargs):
        """
        Test and write the returned results

        :param sess:
        :param global_step:
        :param kwargs:
        :return:
        """
        null_val = self._get_config('null_val')
        start_time = time.time()
        test_results = MGRNNModel.run_epoch(sess, self._test_model, inputs=self._x_test,
                                            labels=self._y_test, weights=self._wt_test,
                                            o_inputs=self._ox_test, d_inputs=self._dx_test,
                                            o_labels=self._oy_test, d_labels=self._dy_test,
                                            nums=self._nums_test,
                                            return_output=True, train_op=None)
        end_time = time.time()
        inter_y = self._inter_test_y
        # y_preds:  a list of (batch_size, horizon, num_nodes, output_dim)
        test_loss, y_preds = test_results['loss'], test_results['outputs']
        tf_utils.add_simple_summary(self._writer, ['loss/test_loss'], [test_loss], global_step=global_step)

        mode = self._get_config('mode')
        if mode == 'avg':
            # Reshapes to (batch_size, epoch_size, horizon, num_node)
            df_preds = self._convert_model_outputs_to_eval_df(y_preds, inter_y)
            df_pred_duration = {}
            for i, df_pred in df_preds.items():
                df_test = self._test_origin
                self._logger.info("-----{}-------".format(i))
                df_pred_full, df_pred, df_test = self._merge_pred_real(df_pred, df_test)
                print(df_pred_full[['pred_duration', 'time_duration', 'pred', 'speed']])
                mae, mape, rmse = metrics.calculate_metrics(df_pred, df_test, null_val)

                tf_utils.add_simple_summary(self._writer,
                                            ['%s_%d' % (item, i + 1) for item in
                                             ['metric/rmse', 'metric/mape', 'metric/mae']],
                                            [rmse, mape, mae],
                                            global_step=global_step)
                end_time = time.time()
                message = '%s Horizon %d, mape:%.4f, rmse:%.4f, mae:%.4f, %.6f s' % (self._get_config.get('loss_func'),
                    i + 1, mape, rmse, mae, end_time - start_time)
                self._logger.info(message)
                # if self._data_format == 'speed':
                #     vmae, vmape, vrmse = metrics.calculate_metrics(
                #         df_pred_full['pred'], df_pred_full['speed'], null_val)
                #     message = 'Horizon %d, v_mape:%.4f, v_rmse:%.4f, v_mae:%.4f, %ds' % (
                #         i + 1, vmape, vrmse, vmae, end_time - start_time)
                #     self._logger.info(message)
                start_time = end_time
                df_pred_duration[i] = df_pred_full

            return df_pred_duration
        else:
            # get the shape of prediction: (batch_size, horizon, num_nodes,  num_nodes, input_dim)
            y_preds = np.concatenate(y_preds, axis=0)
            pred_shape = y_preds.shape

            for i in range(pred_shape[1]):
                label_i  = self._y_test[:, :, i, :, :, :pred_shape[-1]]
                pred_i = y_preds[:, i, :, :, :]
                label_i = np.reshape(label_i, pred_i.shape)
                wt_i = self._wt_test[:, :, i, :, :]
                wt_i = np.reshape(wt_i, pred_i.shape[:-1])
                kl, l2, emd, label_kl = metrics_weight.calculate_metrics_hist(pred_i, label_i, wt_i)
                self._logger.info("-----{}-------".format(i))
                tf_utils.add_simple_summary(self._writer,
                                            ['%s_%d' % (item, i + 1) for item in
                                             ['metric/kl', 'metric/l2', 'metric/emd']],
                                            [kl, l2, emd],
                                            global_step=global_step)

                message = 'Horizon %d, kl:%.4f, jsd:%.4f, emd:%.4f, l2:%.4f, %.6f s' % (
                    i + 1, kl, (label_kl + kl) / 2, emd, l2, (end_time - start_time)/pred_shape[0])
                self._logger.info(message)

                # # print the results
                # pred_i_j = pred_i[wt_i, :]
                # label_i_j = label_i[wt_i, :]
                # for j in range(10):
                #     print(j)
                #     print('label: ', label_i_j[j, :])
                #     print('pred: ', pred_i_j[j, :])

            return y_preds

    def test_and_write_results(self, sess, model_dir=None,
                               model_filename=None, dist_mx=None):

        # Newly added for restoring from the model
        if model_filename is not None and model_dir is not None:
            print("Now in testing and writing....")
            # new_saver = tf.train.import_meta_graph(os.path.join(model_dir, 'model.meta'))
            new_saver = tf.train.Saver()
            new_saver.restore(sess, os.path.join(model_dir, model_filename))

        null_val = self._get_config('null_val')
        start_time = time.time()
        test_results = MGRNNModel.run_epoch(sess, self._test_model, inputs=self._x_test,
                                            labels=self._y_test, weights=self._wt_test,
                                            o_inputs=self._ox_test, d_inputs=self._dx_test,
                                            o_labels=self._oy_test, d_labels=self._dy_test,
                                            nums=self._nums_test,
                                            return_output=True, train_op=None)
        end_time = time.time()
        inter_y = self._inter_test_y
        # y_preds:  a list of (batch_size, horizon, num_nodes, output_dim)
        test_loss, y_preds = test_results['loss'], test_results['outputs']

        mode = self._get_config('mode')
        if mode == 'avg':
            # Reshapes to (batch_size, epoch_size, horizon, num_node)
            df_preds = self._convert_model_outputs_to_eval_df(y_preds, inter_y)
            df_pred_duration = {}
            for i, df_pred in df_preds.items():
                df_test = self._test_origin
                self._logger.info("-----{}-------".format(i))
                df_pred_full, df_pred, df_test = self._merge_pred_real(df_pred, df_test)
                print(df_pred_full[['pred_duration', 'time_duration', 'pred', 'speed']])
                mae, mape, rmse = metrics.calculate_metrics(df_pred, df_test, null_val)
                end_time = time.time()
                message = '%s Horizon %d, mape:%.4f, rmse:%.4f, mae:%.4f, %ds' % (self._get_config.get('loss_func'),
                                                                                  i + 1, mape, rmse, mae,
                                                                                  end_time - start_time)
                self._logger.info(message)

                df_pred_duration[i] = df_pred_full

            return df_pred_duration
        else:
            # get the shape of prediction: (batch_size, horizon, num_nodes,  num_nodes, input_dim)
            y_preds = np.concatenate(y_preds, axis=0)
            pred_shape = y_preds.shape
            self._write_results_into_df(y_preds, dist_mx)

            for i in range(pred_shape[1]):
                label_i = self._y_test[:, :, i, :, :, :pred_shape[-1]]
                pred_i = y_preds[:, i, :, :, :]
                label_i = np.reshape(label_i, pred_i.shape)
                wt_i = self._wt_test[:, :, i, :, :]
                wt_i = np.reshape(wt_i, pred_i.shape[:-1])
                kl, l2, emd, jsd = metrics_weight.calculate_metrics_hist(pred_i, label_i, wt_i)
                self._logger.info("-----{}-------".format(i))

                message = 'Horizon %d, kl:%.4f, jsd:%.4f, emd:%.4f, l2:%.4f, %ds' % (
                    i + 1, kl, jsd, emd, l2, end_time - start_time)
                self._logger.info(message)

            return y_preds

    def _write_results_into_df(self, y_preds, dist_mx):

        head_column = ['KL', 'EMD', 'JS', 'TI', 'Horizon', 'O_id', 'D_id', 'Dist', 'Hopk', 'Sigma']
        pred_shape = y_preds.shape
        dest_df = os.path.join(
            self._result_dir, 'GODS_{0}_S{1}_s6h3_add_sigma_S3H3.csv'.format(self._zone, self._sample_rate))

        for i in range(pred_shape[1]):
            dict_horizon_i = {}
            for head_i in head_column:
                dict_horizon_i[head_i] = []
            label_i = self._y_test[:, :, i, :, :, :pred_shape[-1]]
            pred_i = y_preds[:, i, :, :, :]
            label_i = np.reshape(label_i, pred_i.shape)

            current_time_interval = self._inter_test_y[:, :, i]
            current_time_interval = np.reshape(current_time_interval, [pred_i.shape[0], 1, 1])

            kl, jsd, emd = metrics_weight.calculate_metrics_hist_matrix(
                pred_i, label_i)
            wt_i = self._wt_test[:, :, i, :, :]
            wt_i = np.reshape(wt_i, pred_i.shape[:-1])
            num_nodes = len(self._nodes)
            oids = np.tile(np.arange(num_nodes).reshape(num_nodes, 1), [1, num_nodes])
            dids = np.tile(np.arange(num_nodes).reshape(1, num_nodes), [num_nodes, 1])
            Oids = np.tile(np.reshape(oids, [1, num_nodes, num_nodes]), [pred_i.shape[0], 1, 1])
            Dids = np.tile(np.reshape(dids, [1, num_nodes, num_nodes]), [pred_i.shape[0], 1, 1])
            current_time_interval = np.tile(current_time_interval, [1, num_nodes, num_nodes])
            dist_mx_tile = np.expand_dims(dist_mx, axis=0)
            dist_mx_tile = np.tile(dist_mx_tile, [pred_i.shape[0], 1, 1])

            dict_horizon_i['KL'] = kl[wt_i].tolist()
            dict_horizon_i['EMD'] = emd[wt_i].tolist()
            dict_horizon_i['JS'] = jsd[wt_i].tolist()
            dict_horizon_i['O_id'] = Oids[wt_i].tolist()
            dict_horizon_i['D_id'] = Dids[wt_i].tolist()
            dict_horizon_i['Hopk'] = [self._hopk] * np.sum(wt_i)
            dict_horizon_i['Sigma'] = [self._sigma] * np.sum(wt_i)
            dict_horizon_i['TI'] = current_time_interval[wt_i].tolist()
            dict_horizon_i['Horizon'] = [i] * np.sum(wt_i)
            dict_horizon_i['Dist'] = dist_mx_tile[wt_i].tolist()
            df_result = pd.DataFrame(dict_horizon_i)

            if os.path.isfile(dest_df):
                with open(dest_df, 'a') as f:
                    df_result.to_csv(f, header=False)
            else:
                with open(dest_df, 'a') as f:
                    df_result.to_csv(f, header=True)

    @staticmethod
    def _generate_run_id(config):
        batch_size = config.get('batch_size')
        sample_rate = config.get('sample_rate')
        dropout = config.get('drop_out')
        learning_rate = config.get('learning_rate')
        zone = config.get('zone')
        data_format = config.get('data_format')
        loss_func = config.get('loss_func')
        c_filter_size = config.get('c_filter_size')
        nb_c_filter = config.get('nb_c_filter')
        seq_len = config.get('seq_len')
        fill_mean = config.get('fill_mean')
        lr_decay = config.get('lr_decay')
        lr_decay_epoch = config.get('lr_decay_epoch')
        sigma = config.get('sigma')
        hopk = config.get('hopk')
        activate_func = config.get('activate_func')
        pool_type = config.get('pool_type')
        optimizer = config.get('optimizer')
        structure = '-'.join(
            ['%d_%d' % (nb_c_filter[i], c_filter_size[i]) for i in range(len(c_filter_size))])
        time_id = time.strftime('%m%d%H%M%S')
        horizon = config.get('horizon')
        filter_type = config.get('filter_type')
        filter_type_abbr = 'L'
        if filter_type == 'random_walk':
            filter_type_abbr = 'R'
        elif filter_type == 'dual_random_walk':
            filter_type_abbr = 'DR'
        run_id = 'mgrnn_{}_h_{}_{}_lr_{}_LrDecay{}_LDE{}_bs_{}_' \
                 'd_{}_sl_{}_{}_{}_{}_{}_{}_{}_sigma{}_hopk{}_{}_{}_{}_{}/'.format(
            filter_type_abbr, horizon, structure, learning_rate, lr_decay,
            lr_decay_epoch, batch_size, dropout, seq_len, loss_func, zone,
            data_format, sample_rate, time_id, fill_mean, sigma, hopk, optimizer,
            activate_func, pool_type, config.get('trace_ratio'))

        return run_id
