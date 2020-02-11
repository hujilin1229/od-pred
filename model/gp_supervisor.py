from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import pandas as pd
from lib import tf_utils, metrics, metrics_weight
from model.fcrnn_supervisor import FCRNNSupervisor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

class GPRSupervisor(FCRNNSupervisor):
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
        super(GPRSupervisor, self).__init__(traffic_reading_df, adj_mx, config,
                 origin_df_file, nodes)

    def _build_train_val_test_models(self):
        # get the shape info of the data
        epoch_size, batch_size, seq_len, nb_node, _, in_feat = self._x_train.shape
        train_data = np.transpose(self._x_train, [0, 1, 3, 4, 2, 5])
        train_label = np.transpose(self._y_train, [0, 1, 3, 4, 2, 5])

        train_weights = np.reshape(
            self._wt_train, (epoch_size * batch_size, self._horizon, nb_node, nb_node))

        train_data = np.reshape(
            train_data, (epoch_size * batch_size, nb_node, nb_node, seq_len * in_feat))
        train_label = np.reshape(
            train_label, (epoch_size * batch_size, nb_node, nb_node, self._horizon, self._output_dim))

        self._models = []
        for node_i in range(self._num_nodes):
            model_row = []
            for node_j in range(self._num_nodes):
                train_data_i_j = train_data[:, node_i, node_j, :]
                model_hi = []
                for h_i in range(self._horizon):
                    model_out = []
                    # get usable training data
                    train_y_all = train_label[:, node_i, node_j, h_i, :]
                    train_weight = train_weights[:, h_i, node_i, node_j]
                    if np.sum(train_weight) == 0:
                        model_hi.append(model_out)
                        continue
                    train_y_i_j_hi = train_y_all[train_weight, :]
                    train_data_i_j_hi = train_data_i_j[train_weight, :]
                    for out_i in range(self._output_dim):
                        # clf = SVR(C=1.0, epsilon=0.05)
                        kernel = DotProduct() + WhiteKernel()
                        gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
                        gpr.fit(train_data_i_j_hi, train_y_i_j_hi[:, out_i])
                        model_out.append(gpr)
                    model_hi.append(model_out)
                model_row.append(model_hi)
            self._models.append(model_row)

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
        # get the shape info of the data
        epoch_size, batch_size, seq_len, nb_node, _, in_feat = self._x_test.shape
        test_data = np.transpose(self._x_test, [0, 1, 3, 4, 2, 5])
        test_data = np.reshape(
            test_data, (epoch_size * batch_size, nb_node, nb_node, seq_len * in_feat))
        dtype = test_data.dtype
        test_labels = np.reshape(
            self._y_test, (epoch_size * batch_size, self._horizon, nb_node, nb_node, self._output_dim))
        test_weights = np.reshape(
            self._wt_test, (epoch_size * batch_size, self._horizon, nb_node, nb_node))
        test_predictions = np.zeros((
            epoch_size * batch_size, self._horizon, self._num_nodes,
            self._num_nodes, self._output_dim), dtype=dtype)

        for node_i in range(self._num_nodes):
            for node_j in range(self._num_nodes):
                test_data_i_j = test_data[:, node_i, node_j, :]
                for h_i in range(self._horizon):
                    h_i_models = self._models[node_i][node_j][h_i]
                    if len(h_i_models) == 0:
                        continue
                    for out_i in range(self._output_dim):
                        clf = h_i_models[out_i]
                        test_predictions[:, h_i, node_i, node_j, out_i] = clf.predict(test_data_i_j)

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
        self._logger.info('GPR Results...')
        pred_shape = y_preds.shape
        for i in range(pred_shape[1]):
            label_i  = labels[:, i, :, :, :self._output_dim]
            pred_i = y_preds[:, i, :, :, :]
            wt_i = weights[:, i, :, :]
            wt_i = np.reshape(wt_i, pred_i.shape[:-1])
            kl, l2, emd, jsd = metrics_weight.calculate_metrics_hist(pred_i, label_i, wt_i)
            self._logger.info("-----{}-------".format(i))
            message = 'Horizon %d, kl:%.4f, jsd:%.4f, emd:%.4f, l2:%.4f' % (i + 1, kl, jsd, emd, l2)
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
                kl, l2, emd, jsd = metrics_weight.calculate_metrics_hist(pred_i, label_i, wt_i)
                self._logger.info("-----{}-------".format(i))
                message = 'Horizon %d, kl:%.4f, jsd:%.4f, emd:%.4f, l2:%.4f' % (
                    i + 1, kl, jsd, emd, l2)
                self._logger.info(message)

            return y_preds

    def _write_results_into_df(self, y_preds, dist_mx):

        head_column = ['KL', 'EMD', 'JS', 'TI', 'Horizon', 'O_id', 'D_id', 'Dist', 'Hopk', 'Sigma']
        pred_shape = y_preds.shape
        dest_df = os.path.join(
            self._result_dir, 'GP_{0}_S{1}_s6h3_add_sigma_S3H3.csv'.format(self._zone, self._sample_rate))

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
        run_id = 'gpr_{}_h_{}_{}_lr_{}_bs_{}_d_{}_sl_{}_{}_{}_{}_{}_{}/'.format(
            filter_type_abbr, horizon,
            structure, learning_rate, batch_size,
            dropout, seq_len, loss_func, zone,
            data_format, sample_rate, time_id)
        return run_id
