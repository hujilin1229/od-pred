from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import pandas as pd
from lib import tf_utils, metrics, metrics_weight
from model.fcrnn_supervisor import FCRNNSupervisor
from statsmodels.tsa.api import ARIMA

class ARIMASupervisor(FCRNNSupervisor):
    """
    Do experiments using Graph Random Walk RNN model.
    """

    def __init__(self, traffic_reading_df, adj_mx, config,
                 origin_df_file, nodes):
        self._adj_mx = adj_mx
        self._nodes = nodes
        self._origin_df_file = origin_df_file
        self._start_time = config['start_date']
        self._zone = config['zone']
        self._sample_rate = config['sample_rate']
        self._data_format = config['data_format']
        self._horizon = config['horizon']
        self._output_dim = config['output_dim']
        self._num_nodes = len(nodes)
        self._mode = config['mode']
        self._seq_len = config['seq_len']
        # (p, d, q): order for ARIMA,
        # p: historical observation, d: difference, q: MA steps
        self._order = (self._seq_len, 1, 2)
        super(ARIMASupervisor, self).__init__(traffic_reading_df, adj_mx, config,
                 origin_df_file, nodes)

    def _build_train_val_test_models(self):
        # Get the training data and testing data
        train_data_array = self._train_dict['data']
        val_data_array = self._val_dict['data']
        test_seq_len_array = self._test_dict['data'][:self._seq_len, ...]
        self._histories = []
        self._models = []
        for node_i in range(self._num_nodes):
            history_row = []
            model_row = []
            for node_j in range(self._num_nodes):
                train_data = train_data_array[:, node_i, node_j, :self._output_dim]
                val_data = val_data_array[:, node_i, node_j, :self._output_dim]
                test_data = test_seq_len_array[:, node_i, node_j, :self._output_dim]
                history_i = [train_data[i, :] for i in range(train_data.shape[0])]
                history_i += [val_data[i, :] for i in range(val_data.shape[0])]
                history_i += [test_data[i, :] for i in range(test_data.shape[0])]
                history_row.append(history_i)
                input_seq = np.stack(history_i, axis=0)
                model_dim = []
                for dim_i  in range(self._output_dim):
                    try:
                        model_i = ARIMA(input_seq[:, dim_ind], order=self._order)
                        model_i_fit = model_i.fit()
                        model_dim.append(model_i_fit)
                    except:
                        model_dim.append(None)
                        continue
                model_row.append(model_dim)
            self._models.append(model_row)
            self._histories.append(history_row)

        return None, None, None

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

        test_data_array = self._test_dict['data'][self._seq_len:, ...]
        test_weight = self._test_dict['weight'][self._seq_len:, ...]
        dtype = test_data_array.dtype
        num_samples = test_data_array.shape[0]
        test_batch_size = self._get_config('test_batch_size')
        batch_len = num_samples // test_batch_size
        num_predictions = batch_len - self._seq_len - self._horizon
        test_predictions = np.zeros((
            num_predictions, self._horizon, self._num_nodes,
            self._num_nodes, self._output_dim), dtype=dtype)
        test_labels = np.zeros((
            num_predictions, self._horizon, self._num_nodes,
            self._num_nodes, self._output_dim), dtype=dtype)
        test_weights = np.zeros((
            num_predictions, self._horizon, self._num_nodes,
            self._num_nodes), dtype=np.bool)

        for test_i in range(num_predictions):
            for node_i in range(self._num_nodes):
                for node_j in range(self._num_nodes):
                    tmp_weight = test_weight[test_i: test_i+self._horizon, node_i, node_j]
                    if np.sum(tmp_weight) > 0:
                        test_weights[test_i, :, node_i, node_j] = \
                            test_weight[test_i: test_i + self._horizon, node_i, node_j]
                        test_labels[test_i, :, node_i, node_j, :] = \
                            test_data_array[test_i: test_i + self._horizon, node_i, node_j, :]
                        input_seq = np.stack(self._histories[node_i][node_j], axis=0)
                        for dim_ind in range(self._output_dim):
                            model_i_fit = self._models[node_i][node_j][dim_ind]
                            if model_i_fit is None:
                                continue
                            preds_i = model_i_fit.forecast(self._horizon, input_seq[-self._seq_len, dim_ind])
                            test_predictions[test_i, :, node_i, node_j, dim_ind] = preds_i

                    self._histories[node_i][node_j] += test_data_array[test_i, node_i, node_j, :]
        # make the prediction reasonable
        test_predictions[test_predictions < 0] = 0.
        test_predictions = test_predictions / np.sum(test_predictions, axis=-1, keepdims=True)
        test_predictions = np.where(np.isnan(test_predictions),
                                    np.zeros_like(test_predictions), test_predictions)
        self._analyze_result_hist(test_predictions, test_labels, test_weights)

        return None

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

    def _analyze_result_hist(self, y_preds, labels, weights):
        # get the shape of prediction: (num_preds, horizon, num_nodes,  num_nodes, input_dim)
        pred_shape = y_preds.shape
        for i in range(pred_shape[1]):
            label_i  = labels[:, i, :, :, :pred_shape[-1]]
            pred_i = y_preds[:, i, :, :, :]
            wt_i = weights[:, i, :, :]
            wt_i = np.reshape(wt_i, pred_i.shape[:-1])
            kl, l2, emd, jsd = metrics_weight.calculate_metrics_hist(pred_i, label_i, wt_i)
            self._logger.info("-----{}-------".format(i))
            message = 'Horizon %d, kl:%.4f, jsd:%.4f, emd:%.4f, l2:%.4f' % (
                i + 1, kl, jsd, emd, l2)
            self._logger.info(message)

            return y_preds

    def _test_and_write_result(self, y_preds):
        null_val = self._get_config('null_val')
        start_time = time.time()

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
                message = 'Horizon %d, mape:%.4f, rmse:%.4f, mae:%.4f, %ds' % (
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
                kl, l2, emd = metrics_weight.calculate_metrics_hist(pred_i, label_i, wt_i)
                self._logger.info("-----{}-------".format(i))
                tf_utils.add_simple_summary(self._writer,
                                            ['%s_%d' % (item, i + 1) for item in
                                             ['metric/kl', 'metric/l2', 'metric/emd']],
                                            [kl, l2, emd],
                                            global_step=global_step)
                end_time = time.time()
                message = 'Horizon %d, kl:%.4f, emd:%.4f, l2:%.4f, %ds' % (
                    i + 1, kl, l2, emd, end_time - start_time)
                self._logger.info(message)

            return y_preds

    @staticmethod
    def _generate_run_id(config):
        batch_size = config.get('batch_size')
        sample_rate = config.get('sample_rate')
        dropout = config.get('dropout')
        learning_rate = config.get('learning_rate')
        zone = config.get('zone')
        data_format = config.get('data_format')
        loss_func = config.get('loss_func')
        nb_c_filter = config.get('nb_c_filter')
        seq_len = config.get('seq_len')
        structure = '-'.join(
            ['FC%d' % (nb_c_filter[i]) for i in range(len(nb_c_filter))])
        time_id = time.strftime('%m%d%H%M%S')
        horizon = config.get('horizon')
        filter_type = config.get('filter_type')
        filter_type_abbr = 'L'
        if filter_type == 'random_walk':
            filter_type_abbr = 'R'
        elif filter_type == 'dual_random_walk':
            filter_type_abbr = 'DR'
        run_id = 'arima_{}_h_{}_{}_lr_{}_bs_{}_d_{}_sl_{}_{}_{}_{}_{}_{}/'.format(
            filter_type_abbr, horizon,
            structure, learning_rate, batch_size,
            dropout, seq_len, loss_func, zone,
            data_format, sample_rate, time_id)
        return run_id
