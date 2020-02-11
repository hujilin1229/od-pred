from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sklearn
import sklearn.datasets
import sklearn.naive_bayes
import sklearn.linear_model
import sklearn.svm
import sklearn.neighbors
import sklearn.ensemble
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import pickle as pkl
from sqlalchemy import create_engine, TIMESTAMP
import configparser
import pandas as pd

from pathlib2 import Path
from sklearn.preprocessing import LabelEncoder
import sys
import logging
from lib import coarsening
import scipy.sparse as sp
from scipy.sparse import linalg

plt.switch_backend('agg')


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, raw_data, weight=None):
        if weight is not None:
            raw_data = raw_data[weight]
        self.mean = raw_data.mean()
        self.std = raw_data.std()

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class MaxMinScaler:
    """
    Standard the input
    """

    def __init__(self, raw_data, weight=None):
        if weight is not None:
            raw_data = raw_data[weight]
        self.max_value = max(raw_data)
        self.min_value = min(raw_data)
        assert self.max_value > self.min_value

    def transform(self, data):
        return (data - self.min_value) / (self.max_value - self.min_value)

    def inverse_transform(self, data):
        return data * (self.max_value - self.min_value) + self.min_value

def plot_cdf_pdf(series):

    fig, ax = plt.subplots()
    ax.set_xlim((ax.get_xlim()[0], series.max()))
    ax2 = ax.twinx()
    n, bins, patches = ax.hist(series, bins=20, density=True)
    ax.set_ylabel('PDF')
    n, bins, patches = ax2.hist(
        series, cumulative=1, density=True,
        histtype='step', bins=20, color='tab:orange')
    ax2.set_ylabel('CDF')

    plt.savefig('cdf_test.png')

def get_logger(log_dir, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, 'info.log'))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


def haversine_(lat1, lng1, lat2, lng2):
    """function to calculate haversine distance between two co-ordinates"""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return (h)


def manhattan_distance_pd(lat1, lng1, lat2, lng2):
    """function to calculate manhatten distance between pick_drop"""
    a = haversine_(lat1, lng1, lat1, lng2)
    b = haversine_(lat1, lng1, lat2, lng1)
    return a + b


def connect_sql_server(server, dir):
    """Connect to a sql server with configure file

    This function is used to connect to a sql server using sqlalchemy package by
    specifying a configure file and the a configure name in this file.

    :param server (string): the name of the sql server in configure file
    :param dir (string): the directory of configure file

    :return: the connected sql engine.
    """
    db_conf_file = os.path.join(dir, "dbconf.conf")
    db_conf = configparser.ConfigParser()
    db_conf.read(db_conf_file)
    connect_str = db_conf.get(server, "conn_str")
    engine = create_engine(connect_str)

    return engine


def apply_small_num(row, *args):
    """
    Apply Function: Replace small number in the cell of Dataframe with np.nan

    :param row: row in a pd.Dataframe
    :param args: Variable length argument list.
    :return: the modified row
    """
    for i, link_i in enumerate(args[0]):
        if row[link_i] < args[1]:
            row[link_i] = np.nan
    return row


def cvt_df_nan_2_list(df, cols, nb_bins):
    """
    Convert any nan cell in a pd.Dataframe into zero array with length of nb_bins

    :param df: pd.Dataframe, the target Dataframe
    :param cols: list, list of column names
    :param nb_bins: int, number of buckets for the histogram

    :return: pd.Dataframe, the converted Dataframe.
    """
    for col in cols:
        df.loc[df[col].isnull(), col] = \
            df.loc[df[col].isnull(), col].apply(
                lambda x: np.zeros(nb_bins - 1))
    return df


def vel_list(array_like):
    """
    Convert the array like content into list

    :param array_like (numpy.array or related):

    :return: list: List format of the argument.
    """
    return list(array_like.values)


def my_rolling_apply_list(frame, func, window, hist_range):
    """
    Construct histogram in all cells

    :param frame (pd.Dataframe): The source dataframe
    :param window (int): The size of the window
    :param func: The function need to be applied
    :param link_len (double): the length of the current link
    :param hist_range (np.array): the histogram buckets

    :return: pd.Series
    """
    index = frame.index[window - 1:]
    values = [func(frame.iloc[i:i + window], hist_range)
              for i in range(len(frame) - window + 1)]
    return pd.Series(data=values, index=index).reindex(frame.index)


def my_rolling_apply_avg(frame, func, window):
    """
    Construct average value in all cells

    :param frame (pd.Dataframe): The source dataframe
    :param func: The function need to be applied
    :param window (int): The size of the window

    :return: pd.Series
    """
    index = frame.index[window - 1:]
    values = [func(frame.iloc[i:i + window])
              for i in range(len(frame) - window + 1)]

    return pd.Series(data=values, index=index).reindex(frame.index)


def convert_multi_channel_array(df_all_array, nb_bins):
    multi_channel_array = []
    all_shape = df_all_array.shape
    if nb_bins == 1:
        df_all_array = np.expand_dims(df_all_array, axis=-1)

    for i in range(all_shape[0]):
        channel_i = np.zeros((all_shape[1], nb_bins))
        for j in range(all_shape[1]):
            if pd.isnull([df_all_array[i, j]]).any():
                continue
            else:
                for k in range(nb_bins):
                    channel_i[j, k] = df_all_array[i, j][k]
        multi_channel_array.append(channel_i)
    multi_channel_array = np.array(multi_channel_array)

    return multi_channel_array


def train_val_test_split_df(df, val_ratio=0.1, test_ratio=0.2):
    n_sample = df.shape[0]
    n_val = int(round(n_sample * val_ratio))
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_val - n_test
    train_data, val_data, test_data = df[:n_train, ...], df[n_train: n_train + n_val, ...], df[-n_test:, ...]
    return train_data, val_data, test_data


def generate_graph_seq2seq_odsep_data_with_time(dict_data, batch_size, seq_len, horizon, num_nodes, scaler=None,
                                                add_time_in_day=True, add_day_in_week=False, perm=None):
    """

    :param df: (2197, 67, 67, 1)
    :param batch_size:
    :param seq_len:
    :param horizon:
    :param scaler:
    :param add_day_in_week:
    :return:
    x, y, both are 5-D tensors with size (epoch_size, batch_size, seq_len, num_sensors, input_dim).
    Adjacent batches are continuous sequence, i.e., x[i, j, :, :] is before x[i+1, j, :, :]
    """

    df = dict_data['data']
    if scaler:
        df = scaler.transform(df)
    num_samples = df.shape[0]
    data = df
    batch_len = num_samples // batch_size
    # data = np.expand_dims(data, axis=-1)
    # coarsen data

    weight = dict_data['weight']
    num = dict_data['num']

    interval = dict_data['TI']
    origin_nodes = data.shape[1]
    feature_size = data.shape[-1]
    if perm is not None:
        # input data shape is (Samples, nb_nodes, nb_nodes, nb_intervals)
        # with perm operation, the output will be (Samples, nb_new_nodes, nb_new_nodes, nb_intervals)
        o_data = np.transpose(data, [0, 1, 3, 2])
        o_data = np.reshape(o_data, (-1, origin_nodes))
        o_data = coarsening.perm_data(o_data, perm)
        o_data = np.reshape(o_data, (num_samples, origin_nodes, feature_size, num_nodes))
        o_data = np.transpose(o_data, [0, 1, 3, 2])

        d_data = np.transpose(data, [0, 3, 2, 1])
        d_data = np.reshape(d_data, (-1, origin_nodes))
        d_data = coarsening.perm_data(d_data, perm)
        d_data = np.reshape(d_data, (num_samples, feature_size, origin_nodes, num_nodes))
        d_data = np.transpose(d_data, [0, 3, 2, 1])
    else:
        o_data = data
        d_data = data

    data_list = [data]
    o_data_list = [o_data]
    d_data_list = [d_data]

    if add_time_in_day:
        time_ind = interval % int(24 * 60 / 40)
        time_in_day = np.tile(time_ind, [1, origin_nodes, origin_nodes, 1]).transpose((3, 2, 1, 0))
        data_list.append(time_in_day)
        time_in_day = np.tile(time_ind, [1, origin_nodes, num_nodes, 1]).transpose((3, 2, 1, 0))
        o_data_list.append(time_in_day)
        time_in_day = np.tile(time_ind, [1, num_nodes, origin_nodes, 1]).transpose((3, 2, 1, 0))
        d_data_list.append(time_in_day)

    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, origin_nodes, origin_nodes, 7))
        day_in_week[np.arange(num_samples), :, :, dict_data['DoW']] = 1
        data_list.append(day_in_week)
        day_in_week = np.zeros(shape=(num_samples, origin_nodes, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, :, dict_data['DoW']] = 1
        o_data_list.append(day_in_week)
        day_in_week = np.zeros(shape=(num_samples, num_nodes, origin_nodes, 7))
        day_in_week[np.arange(num_samples), :, :, dict_data['DoW']] = 1
        d_data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    data = data[:batch_size * batch_len, ...].reshape((batch_size, batch_len, origin_nodes, origin_nodes, -1))
    o_data = np.concatenate(o_data_list, axis=-1)
    o_data = o_data[:batch_size * batch_len, ...].reshape((batch_size, batch_len, origin_nodes, num_nodes, -1))
    d_data = np.concatenate(d_data_list, axis=-1)
    d_data = d_data[:batch_size * batch_len, ...].reshape((batch_size, batch_len, num_nodes, origin_nodes, -1))
    interval = interval[:batch_size * batch_len].reshape((batch_size, batch_len))

    weight = weight[:batch_size * batch_len, ...].reshape((batch_size, batch_len, origin_nodes, origin_nodes))
    num = num[:batch_size * batch_len, ...].reshape((batch_size, batch_len, origin_nodes, origin_nodes))
    epoch_size = batch_len - seq_len - horizon
    x, y, w, nums = [], [], [], []
    o_x, d_x, o_y, d_y = [], [], [], []
    inter_x, inter_y = [], []
    for i in range(epoch_size):
        w_i = weight[:, i + seq_len: i + seq_len + horizon, ...]
        x_i = data[:, i: i + seq_len, ...]
        # check if has data or not, if not, no need to train
        if np.sum(w_i) == 0 or np.sum(x_i) == 0:
            continue
        x_i = data[:, i: i + seq_len, ...]
        ox_i = o_data[:, i: i + seq_len, ...]
        dx_i = d_data[:, i: i + seq_len, ...]
        y_i = data[:, i + seq_len: i + seq_len + horizon, ...]
        oy_i = o_data[:, i + seq_len: i + seq_len + horizon, ...]
        dy_i = d_data[:, i + seq_len: i + seq_len + horizon, ...]
        num_i = num[:, i + seq_len: i + seq_len + horizon, ...]

        inter_x_i = interval[:, i: i + seq_len]
        inter_y_i = interval[:, i + seq_len: i + seq_len + horizon]
        x.append(x_i)
        y.append(y_i)
        w.append(w_i)
        nums.append(num_i)

        o_x.append(ox_i)
        d_x.append(dx_i)
        o_y.append(oy_i)
        d_y.append(dy_i)

        inter_x.append(inter_x_i)
        inter_y.append(inter_y_i)

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    w = np.stack(w, axis=0)
    nums = np.stack(nums, axis=0)
    o_x = np.stack(o_x, axis=0)
    d_x = np.stack(d_x, axis=0)
    o_y = np.stack(o_y, axis=0)
    d_y = np.stack(d_y, axis=0)

    inter_y = np.stack(inter_y, axis=0)
    inter_x = np.stack(inter_x, axis=0)

    return x, y, w, o_x, d_x, o_y, d_y, inter_x, inter_y, nums


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d += np.spacing(np.array(0, adj.dtype))  # get a small number in case to be divided by zero
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum(adj_mx, adj_mx.T)
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


# Helpers to quantify classifier's quality.
def baseline(train_data, train_labels, test_data, test_labels, omit=[]):
    """Train various classifiers to get a baseline."""
    clf, train_accuracy, test_accuracy, train_f1, test_f1, exec_time = [], [], [], [], [], []
    clf.append(sklearn.neighbors.KNeighborsClassifier(n_neighbors=10))
    clf.append(sklearn.linear_model.LogisticRegression())
    clf.append(sklearn.naive_bayes.BernoulliNB(alpha=.01))
    clf.append(sklearn.ensemble.RandomForestClassifier())
    clf.append(sklearn.naive_bayes.MultinomialNB(alpha=.01))
    clf.append(sklearn.linear_model.RidgeClassifier())
    clf.append(sklearn.svm.LinearSVC())
    for i, c in enumerate(clf):
        if i not in omit:
            t_start = time.process_time()
            c.fit(train_data, train_labels)
            train_pred = c.predict(train_data)
            test_pred = c.predict(test_data)
            train_accuracy.append('{:5.2f}'.format(
                100 * sklearn.metrics.accuracy_score(train_labels, train_pred)))
            test_accuracy.append('{:5.2f}'.format(
                100 * sklearn.metrics.accuracy_score(test_labels, test_pred)))
            train_f1.append('{:5.2f}'.format(
                100 * sklearn.metrics.f1_score(train_labels, train_pred, average='weighted')))
            test_f1.append('{:5.2f}'.format(
                100 * sklearn.metrics.f1_score(test_labels, test_pred, average='weighted')))
            exec_time.append('{:5.2f}'.format(time.process_time() - t_start))
    print('Train accuracy:      {}'.format(' '.join(train_accuracy)))
    print('Test accuracy:       {}'.format(' '.join(test_accuracy)))
    print('Train F1 (weighted): {}'.format(' '.join(train_f1)))
    print('Test F1 (weighted):  {}'.format(' '.join(test_f1)))
    print('Execution time:      {}'.format(' '.join(exec_time)))


def grid_search(params, grid_params, train_data, train_labels, train_label_weight,
                val_data, val_labels, val_label_weight,
                test_data, test_labels, test_label_weight, model, result_file,
                train_embed_data=None, test_embed=None):
    """Explore the hyper-parameter space with an exhaustive grid search."""
    param = params.copy()
    train_accuracys, test_accuracys, train_f1, test_f1 = [], [], [], []
    grid = sklearn.model_selection.ParameterGrid(grid_params)
    print('grid search: {} combinations to evaluate'.format(len(grid)))
    for grid_param in grid:
        result_dict = {}
        for param_key in grid[0].keys():
            result_dict[param_key] = []
        train_accuracy, test_accuracy, train_f1, test_f1 = [], [], [], []

        param.update(grid_param)
        name = '{}'.format(grid)
        print('\n\n  {}  \n\n'.format(grid_param))
        for grid_key, grid_item in grid_param.items():
            result_dict[grid_key].append(grid_item)

        m = model(**param)
        m.fit(train_data, train_labels, val_data, val_labels,
              train_label_weight, val_label_weight,
              train_embed_data, test_embed)

        string, accuracy, loss = m.evaluate(train_data, train_labels,
                                            embed_data=train_embed_data,
                                            y_weight=train_label_weight)
        train_accuracy.append('{:5.2f}'.format(accuracy))
        train_accuracys.append('{:5.2f}'.format(accuracy))
        print('train {}'.format(string))

        string, accuracy, loss = m.evaluate(test_data, test_labels,
                                            embed_data=test_embed,
                                            y_weight=test_label_weight)
        test_accuracy.append('{:5.2f}'.format(accuracy))
        test_accuracys.append('{:5.2f}'.format(accuracy))
        print('test  {}'.format(string))

        result_dict['training_kl'] = train_accuracy
        result_dict['test_kl'] = test_accuracy
        result_df = pd.DataFrame.from_dict(result_dict)

        if Path(result_file).is_file():
            result_df.to_csv(result_file, mode='a', sep='\t', header=False)
        else:
            result_df.to_csv(result_file, sep='\t')

    print('\n\n')
    print('Train KL div:      {}'.format(' '.join(train_accuracys)))
    print('Test KL div:       {}'.format(' '.join(test_accuracys)))
    for i, grid_params in enumerate(grid):
        print("KL Divergence...")
        print('{} --> {} {} '.format(grid_params,
                                     train_accuracys[i],
                                     test_accuracys[i]))


def grid_search_avg(params, grid_params, train_data, train_labels, train_label_weight,
                    val_data, val_labels, val_label_weight,
                    test_data, test_labels, test_label_weight, model, result_file, yscaler,
                    train_embed_data=None, test_embed=None):
    """Explore the hyper-parameter space with an exhaustive grid search."""
    param = params.copy()
    train_accuracy, test_accuracy, train_f1, test_f1 = [], [], [], []
    grid = sklearn.model_selection.ParameterGrid(grid_params)
    print('grid search: {} combinations to evaluate'.format(len(grid)))
    result_dict = {}
    for param_key in grid[0].keys():
        result_dict[param_key] = []
    for grid_param in grid:
        param.update(grid_param)
        print('\n\n  {}  \n\n'.format(grid_param))
        for grid_key, grid_item in grid_param.items():
            result_dict[grid_key].append(grid_item)

        m = model(**param)
        m.fit(train_data, train_labels, val_data, val_labels,
              train_label_weight, val_label_weight,
              train_embed_data, test_embed)

        string, _, _, _, train_pred = m.evaluate(train_data, train_labels,
                                                 embed_data=train_embed_data,
                                                 y_weight=train_label_weight)

        accuracy = weighted_mape(train_labels, train_pred,
                                 train_label_weight, yscaler)
        train_accuracy.append('{:5.2f}'.format(accuracy))
        print('train {}'.format(accuracy))

        string, _, _, _, test_pred = m.evaluate(test_data, test_labels,
                                                embed_data=test_embed,
                                                y_weight=test_label_weight)
        accuracy = weighted_mape(test_labels, test_pred,
                                 test_label_weight, yscaler)
        test_accuracy.append('{:5.2f}'.format(accuracy))
        print('test  {}'.format(accuracy))

    print('\n\n')
    print('Train MAPE:      {}'.format(' '.join(train_accuracy)))
    print('Test MAPE:       {}'.format(' '.join(test_accuracy)))
    for i, grid_params in enumerate(grid):
        print("MAPE...")
        print('{} --> {} {} '.format(grid_params,
                                     train_accuracy[i],
                                     test_accuracy[i]))

    result_dict['training_mape'] = train_accuracy
    result_dict['test_mape'] = test_accuracy
    result_df = pd.DataFrame.from_dict(result_dict)

    if Path(result_file).is_file():
        result_df.to_csv(result_file, mode='a', sep='\t')
    else:
        result_df.to_csv(result_file, sep='\t')


class model_perf(object):
    def __init__(s):
        s.names, s.params = set(), {}
        s.fit_accuracies, s.fit_losses, s.fit_time = {}, {}, {}
        s.train_accuracy, s.train_f1, s.train_loss = {}, {}, {}
        s.test_accuracy, s.test_f1, s.test_loss = {}, {}, {}
        # store the prediction
        s.train_mape, s.test_mape, s.pred_val = {}, {}, {}

    def test(s, model, name, params, train_data, train_labels,
             val_data, val_labels, test_data, test_labels, train_label_weight,
             val_label_weight=None, test_label_weight=None,
             train_embed_data=None, validate_embed_data=None, test_embed_data=None,
             classify=False, y_scaler=None, write=True, test_all=False):
        s.params[name] = params
        s.fit_accuracies[name], s.fit_losses[name], s.fit_time[name] = \
            model.fit(train_data, train_labels, val_data, val_labels,
                      train_label_weight, val_label_weight,
                      train_embed_data, validate_embed_data)
        string, s.train_accuracy[name], s.train_f1[name], s.train_loss[name], train_pred = \
            model.evaluate(train_data, train_labels, None, train_embed_data, train_label_weight)
        print('train {}'.format(string))

        string, s.test_accuracy[name], s.test_f1[name], s.test_loss[name], test_pred = \
            model.evaluate(test_data, test_labels, None, test_embed_data, test_label_weight)
        print('test  {}'.format(string))

        if not classify:
            s.train_mape[name] = weighted_mape(train_labels, train_pred, train_label_weight, y_scaler)
            s.test_mape[name] = weighted_mape(test_labels, test_pred, test_label_weight, y_scaler)
            s.pred_val[name] = y_scaler.inverse_transform(test_pred)
            print("Training MAPE: ", s.train_mape[name])
            print("Testing MAPE: ", s.test_mape[name])
        s.names.add(name)

    def od_test(s, model, name, params, train_data, train_labels,
                val_data, val_labels, test_data, test_labels, train_label_weight,
                val_label_weight=None, test_label_weight=None,
                train_embed_data=None, validate_embed_data=None, test_embed_data=None,
                classify=False, y_scaler=None, write=True, test_all=False):
        s.params[name] = params
        s.fit_accuracies[name], s.fit_losses[name], s.fit_time[name] = \
            model.fit(train_data, train_labels, val_data, val_labels,
                      train_label_weight, val_label_weight,
                      train_embed_data, validate_embed_data)
        string, s.train_accuracy[name], s.train_f1[name], s.train_loss[name], train_pred = \
            model.evaluate(train_data, train_labels, None, train_embed_data, train_label_weight)
        print('train {}'.format(string))

        string, s.test_accuracy[name], s.test_f1[name], s.test_loss[name], test_pred = \
            model.evaluate(test_data, test_labels, None, test_embed_data, test_label_weight)
        print('test  {}'.format(string))

        if not classify:
            s.train_mape[name] = weighted_mape(train_labels, train_pred, train_label_weight, y_scaler)
            s.test_mape[name] = weighted_mape(test_labels, test_pred, test_label_weight, y_scaler)
            s.pred_val[name] = y_scaler.inverse_transform(test_pred)
            print("Training MAPE: ", s.train_mape[name])
            print("Testing MAPE: ", s.test_mape[name])
        s.names.add(name)

    def hist_test(s, model, name, params, train_data, train_labels,
                  val_data, val_labels, test_data, test_labels, train_label_weight,
                  test_label_weight=None, val_label_weight=None,
                  train_embed_data=None, val_embed=None, test_embed=None):
        s.params[name] = params
        print("num of nan values in train_data", np.sum(np.isnan(train_data)))
        print("num of nan values in val_data", np.sum(np.isnan(val_data)))
        s.fit_accuracies[name], s.fit_losses[name], s.fit_time[name] = \
            model.fit(train_data, train_labels, val_data, val_labels,
                      train_label_weight, val_label_weight,
                      train_embed_data, val_embed)
        string, s.train_accuracy[name], s.train_loss[name], _ = \
            model.evaluate(train_data, train_labels,
                           embed_data=train_embed_data, y_weight=train_label_weight)

        print('train {}'.format(string))
        string, s.test_accuracy[name], s.test_loss[name], test_pred = \
            model.evaluate(test_data, test_labels,
                           embed_data=test_embed,
                           y_weight=test_label_weight)
        s.pred_val[name] = test_pred
        print('test  {}'.format(string))
        s.names.add(name)

    def lsm_test(s, model, name, params, train_data, train_labels, val_labels,
                 train_mean=None, small_threshold=3.0, large_threshold=40.0):

        s.params[name] = params
        s.fit_losses[name] = \
            model.fit(train_data, train_labels, val_labels)

        train_data = [train_data_i.todense() for train_data_i in train_data]
        val_labels = [val_label.todense() for val_label in val_labels]

        string, s.train_loss[name] = \
            model.evaluate(train_data, val_labels,
                           train_mean, small_threshold,
                           large_threshold)
        print('training loss: ', s.fit_losses[name])
        print('final loss: {}'.format(string))
        s.names.add(name)

    def lsm_list_test(s, model, name, params, list_G_ts, list_trainY_ts,
                      list_valY_ts, train_mean=None,
                      small_threshold=3.0, large_threshold=40.0):
        s.params[name] = params
        fit_losses = []
        mapes = []
        complete_list = []
        for G_ind, G_ts in enumerate(list_G_ts):
            train_data = G_ts
            train_labels = list_trainY_ts[G_ind]
            val_labels = list_valY_ts[G_ind]
            tmp_fit_loss = model.fit(train_data, train_labels, val_labels)

            train_data = [train_data_i.todense() for train_data_i in train_data]
            val_labels = [val_label.todense() for val_label in val_labels]

            fit_losses.append(tmp_fit_loss)
            _, mape, complete = model.evaluate(train_data, val_labels,
                                               train_mean, small_threshold,
                                               large_threshold)
            mapes.append(mape)
            complete_list.append(complete)
            # print("{}th mape".format(G_ind), mape)

        s.fit_losses[name] = np.nanmean(fit_losses)
        s.test_mape[name] = np.nanmean(mapes)
        s.pred_val[name] = complete_list
        print("The average mape for LSM + HA is ", s.test_mape[name])

        s.names.add(name)

    def show(s, fontsize=None):
        if fontsize:
            plt.rc('pdf', fonttype=42)
            plt.rc('ps', fonttype=42)
            plt.rc('font', size=fontsize)  # controls default text sizes
            plt.rc('axes', titlesize=fontsize)  # fontsize of the axes title
            # fontsize of the x any y labels
            plt.rc('axes', labelsize=fontsize)
            plt.rc('xtick', labelsize=fontsize)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=fontsize)  # fontsize of the tick labels
            plt.rc('legend', fontsize=fontsize)  # legend fontsize
            plt.rc('figure', titlesize=fontsize)  # size of the figure title
        print('  accuracy        F1             loss        time [ms]  name')
        print('test  train   test  train   test     train')
        for name in sorted(s.names):
            print('{:5.2f} {:5.2f}   {:5.2f} {:5.2f}   {:.2e} {:.2e}   {:3.0f}   {}'.format(
                s.test_accuracy[name], s.train_accuracy[name],
                s.test_f1[name], s.train_f1[name],
                s.test_loss[name], s.train_loss[name], s.fit_time[name] * 1000, name))

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        for name in sorted(s.names):
            steps = np.arange(len(s.fit_accuracies[name])) + 1
            steps *= s.params[name]['eval_frequency']
            ax[0].plot(steps, s.fit_accuracies[name], '.-', label=name)
            ax[1].plot(steps, s.fit_losses[name], '.-', label=name)
        ax[0].set_xlim(min(steps), max(steps))
        ax[1].set_xlim(min(steps), max(steps))
        ax[0].set_xlabel('step')
        ax[1].set_xlabel('step')
        ax[0].set_ylabel('validation accuracy')
        ax[1].set_ylabel('training loss')
        ax[0].legend(loc='lower right')
        ax[1].legend(loc='upper right')
        # fig.savefig('training.pdf')

    def predict_show(s, fontsize=None):
        if fontsize:
            plt.rc('pdf', fonttype=42)
            plt.rc('ps', fonttype=42)
            plt.rc('font', size=fontsize)  # controls default text sizes
            plt.rc('axes', titlesize=fontsize)  # fontsize of the axes title
            # fontsize of the x any y labels
            plt.rc('axes', labelsize=fontsize)
            plt.rc('xtick', labelsize=fontsize)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=fontsize)  # fontsize of the tick labels
            plt.rc('legend', fontsize=fontsize)  # legend fontsize
            plt.rc('figure', titlesize=fontsize)  # size of the figure title
        print('        MAPE            loss        time [ms]  name')
        print('     test  train     test  train')
        for name in sorted(s.names):
            print('{:.2e} {:.2e}   {:.2e} {:.2e}   {:3.0f}     {}'.format(
                s.test_mape[name], s.train_mape[name],
                s.test_loss[name], s.train_loss[name], s.fit_time[name] * 1000, name))

        fig, ax = plt.subplots(figsize=(15, 5))
        for name in sorted(s.names):
            steps = np.arange(len(s.fit_losses[name])) + 1
            steps *= s.params[name]['eval_frequency']
            ax.plot(steps, s.fit_losses[name], '.-', label=name)
        ax.set_xlim(min(steps), max(steps))
        ax.set_xlabel('step')
        ax.set_ylabel('training loss')
        ax.legend(loc='upper right')
        fig.savefig('training.pdf')

    def hist_show(s, fontsize=None):
        if fontsize:
            plt.rc('pdf', fonttype=42)
            plt.rc('ps', fonttype=42)
            plt.rc('font', size=fontsize)  # controls default text sizes
            plt.rc('axes', titlesize=fontsize)  # fontsize of the axes title
            # fontsize of the x any y labels
            plt.rc('axes', labelsize=fontsize)
            plt.rc('xtick', labelsize=fontsize)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=fontsize)  # fontsize of the tick labels
            plt.rc('legend', fontsize=fontsize)  # legend fontsize
            plt.rc('figure', titlesize=fontsize)  # size of the figure title
        print('        KL_DIV            loss        time [ms]  name')
        print('     test  train     test  train')
        for name in sorted(s.names):
            print('{:.2e} {:.2e}   {:.2e} {:.2e}   {:3.0f}     {}'.format(
                s.test_accuracy[name], s.train_accuracy[name],
                s.test_loss[name], s.train_loss[name], s.fit_time[name] * 1000, name))

        fig, ax = plt.subplots(figsize=(15, 5))
        for name in sorted(s.names):
            steps = np.arange(len(s.fit_losses[name])) + 1
            steps *= s.params[name]['eval_frequency']
            ax.plot(steps, s.fit_losses[name], '.-', label=name)
        ax.set_xlim(min(steps), max(steps))
        ax.set_xlabel('step')
        ax.set_ylabel('training loss')
        ax.legend(loc='upper right')
        fig.savefig('training.pdf')


# Helpers to fetch the data.
def construct_today(df_link_tt, nodes):
    re_order_nodes = nodes.copy()
    re_order_nodes = [str(i) for i in re_order_nodes]
    df_link_tt = df_link_tt[re_order_nodes]
    array_link_tt = df_link_tt.values

    array_x_all = array_link_tt
    array_y_all = array_link_tt

    return array_x_all, array_y_all


def construct_tomorrow(df_link_tt, nodes):
    # reorder this dataframe with the corresponding node order
    re_order_nodes = nodes.copy()
    re_order_nodes.append('inter_index')
    re_order_nodes = [str(i) for i in re_order_nodes]
    df_link_tt = df_link_tt[re_order_nodes]
    array_link_tt = df_link_tt.values
    array_x_all = []
    array_y_all = []
    for i in range(array_link_tt.shape[0] - 1):
        if array_link_tt[i, -1] + 1 == array_link_tt[i + 1, -1]:
            array_x_all.append(array_link_tt[i, :-1])
            array_y_all.append(array_link_tt[i + 1, :-1])
    array_x_all = np.array(array_x_all)
    array_y_all = np.array(array_y_all)

    return array_x_all, array_y_all


# clean data
def clean_data(x_all, scale=2, rm_big=True):
    x_var = np.var(x_all, 0)
    x_median = np.median(x_all, 0)
    for i in range(x_all.shape[1]):
        if rm_big:
            b_remove = x_all[:, i] > x_median[i] * scale
            x_all[b_remove, i] = x_median[i] * scale
        else:
            b_remove = x_all[:, i] < x_median[i] * scale
            x_all[b_remove, i] = x_median[i] * scale
    return x_all


# expand one row into multiple rows with dataframe apply


# def f(group, *args):
#     row = group.iloc[0]
#     Dict = {}
#     row_dict = row.to_dict()
#     for item in row_dict:
#         Dict[item] = [row[item]] * int(args[1])
#     time_base = datetime.strptime(row['date'], '%Y-%m-%d') + \
#         datetime.timedelta(hours=int(row[args[0]]))
#     Dict['time'] = [time_base +
#                     datetime.timedelta(minutes=i * 20) for i in range(args[1])]
#     return pd.DataFrame(Dict)

def categorical_trans_file(data, cat_head):
    # make the training data categorical
    les = {}
    for i in range(data.shape[1]):
        le = LabelEncoder()
        data[:, i] = le.fit_transform(data[:, i])
        les[cat_head[i]] = le

    with open('les.pickle', 'wb') as f:
        pkl.dump(les, f, -1)

    return les, data


def fetch_needed_data(dir):
    """Get classes, training and validating data from file"""

    dict_list = ['dict_normal', 'train_data_dict', 'validate_data_dict']
    dicts = {}
    for i, dict_i in enumerate(dict_list):
        path_dict_i = os.path.join(dir, dict_i + '.pickle')
        if os.path.exists(path_dict_i):
            try:
                with open(path_dict_i, 'rb') as f:
                    dicts[dict_i] = pkl.load(f)
            except EOFError:
                return None
        else:
            return None

    return dicts


def weighted_kl_div(y_true, y_pred, weight, epsilon=1e-8):
    N, M, B = y_pred.shape
    N, M, B = int(N), int(M), int(B)

    w_N, w_M = weight.shape
    w_N, w_M = int(w_N), int(w_M)

    assert w_N == N, w_M == M

    log_op = np.log(y_pred + epsilon) - np.log(y_true + epsilon)
    mul_op = np.multiply(y_pred, log_op)
    sum_hist = np.sum(mul_op, 2)
    weight_sum = np.multiply(weight, sum_hist)

    # sum_kl_row = np.sum(weight_sum, 1)
    # sum_weight_row = np.sum(weight, 1)
    # kl_weight_sum = np.multiply(sum_kl_row, sum_weight_row)
    avg_kl_div = np.sum(weight_sum) / np.sum(weight)

    # avg_kl_div = np.mean(weight_sum)
    # num_weighted = np.sum(weight[0, :])
    # num_total = weight.shape[1]
    # avg_kl_div = kl_weight_sum * num_total / num_weighted

    return avg_kl_div


def weighted_mape(y_true, y_pred, weight, y_scaler):
    predict = y_scaler.inverse_transform(y_pred)
    real = y_scaler.inverse_transform(y_true)

    avg_mape = np.sum(np.abs((real - predict) / real)
                      * weight) / np.sum(weight) * 100

    # print out the available mape
    predict_weight = predict[weight > 0]
    real_weight = real[weight > 0]
    weight_weight = weight[weight > 0]
    # unreasonable value in testing data

    mape = np.abs(predict_weight - real_weight) / real_weight
    reasonable = real_weight >= y_scaler.data_min_
    predict_weight = predict_weight[reasonable]
    real_weight = real_weight[reasonable]
    weight_weight = weight_weight[reasonable]

    # print("predicted: ", predict_weight)
    # print("real: ", real_weight)
    # print("weight: ", weight_weight)
    # print("sum mape: ", np.abs(predict_weight - real_weight) / real_weight)
    # print("total number: ", np.sum(weight_weight))
    avg_mape = np.average(np.abs(predict_weight - real_weight) / real_weight)

    return avg_mape


def weighted_rmsle(y_true, y_pred, weight, y_scaler):
    predict = y_scaler.inverse_transform(y_pred)
    real = y_scaler.inverse_transform(y_true)

    avg_mape = np.sum(np.abs((real - predict) / real)
                      * weight) / np.sum(weight) * 100

    # print out the available mape
    predict_weight = predict[weight > 0]
    real_weight = real[weight > 0]
    weight_weight = weight[weight > 0]
    # unreasonable value in testing data
    reasonable = real_weight >= y_scaler.data_min_
    predict_weight = predict_weight[reasonable]
    real_weight = real_weight[reasonable]

    rmsle = np.average(np.square(np.log(predict_weight + 1) - np.log(real_weight + 1)))

    return rmsle


def get_avg_predictions(y_pred, y_scaler):
    predict = y_scaler.inverse_transform(y_pred)

    return predict


def weighted_emd(y_true, y_pred, weight, epsilon=1e-8):
    N, M, B = y_pred.shape
    N, M, B = int(N), int(M), int(B)

    w_N, w_M = weight.shape
    w_N, w_M = int(w_N), int(w_M)

    assert w_N == N, w_M == M

    log_op = np.log(y_pred + epsilon) - np.log(y_true + epsilon)
    mul_op = np.multiply(y_pred, log_op)
    sum_hist = np.sum(mul_op, 2)
    weight_sum = np.multiply(weight, sum_hist)

    # sum_kl_row = np.sum(weight_sum, 1)
    # sum_weight_row = np.sum(weight, 1)
    # kl_weight_sum = np.multiply(sum_kl_row, sum_weight_row)
    avg_kl_div = np.sum(weight_sum) / np.sum(weight)

    # avg_kl_div = np.mean(weight_sum)
    # num_weighted = np.sum(weight[0, :])
    # num_total = weight.shape[1]
    # avg_kl_div = kl_weight_sum * num_total / num_weighted

    return avg_kl_div


def get_vel_avg_rolling(pdSeries_like, min_nb=5, unit=1.0):
    data_lists = pdSeries_like.values.flatten()
    data_list = []
    for i, item in enumerate(data_lists):
        if type(item) == list or type(item) == np.ndarray:
            for j, item_j in enumerate(item):
                if not np.isnan(item_j):
                    data_list.append(item_j)
        else:
            if not np.isnan(item):
                data_list.append(item)

    # To align the histogram data construction,
    # and make sure the average value is meaningful
    if len(data_list) < min_nb:
        return np.nan

    data_array = np.array(data_list)
    data_array = (data_array) * unit
    # data_keep = (data_array < 40) & (
    #     data_array >= self.small_threshold)
    # data_array = data_array[data_keep]

    return np.nanmean(data_array)

def get_vel_num_rolling(pdSeries_like):
    data_lists = pdSeries_like.values.flatten()
    data_list = []
    for i, item in enumerate(data_lists):
        if type(item) == list or type(item) == np.ndarray:
            for j, item_j in enumerate(item):
                if not np.isnan(item_j):
                    data_list.append(item_j)
        else:
            if not np.isnan(item):
                data_list.append(item)

    return len(data_list)

def get_vel_mape_rolling(pdSeries_like, min_nb=5):
    data_lists = pdSeries_like.values.flatten()
    data_list = []
    for i, item in enumerate(data_lists):
        if type(item) == list or type(item) == np.ndarray:
            for j, item_j in enumerate(item):
                data_list.append(item_j)
        else:
            data_list.append(item)

    # To align the histogram data construction,
    # and make sure the average value is meaningful
    if len(data_list) < min_nb:
        return np.nan

    data_array = np.array(data_list)
    mape = np.abs(data_array - data_array.mean()) / data_array
    # data_keep = (data_array < 40) & (
    #     data_array >= self.small_threshold)
    # data_array = data_array[data_keep]

    return np.nanmean(mape)


def get_vel_hist_rolling(pdSeries_like, hist_bin,
                         min_nb=5, big_threshold=21.0,
                         small_threshold=0.0):
    data_lists = pdSeries_like.values.flatten()
    data_list = []
    for i, item in enumerate(data_lists):
        if type(item) == list or type(item) == np.ndarray:
            for j, item_j in enumerate(item):
                data_list.append(item_j)
        else:
            data_list.append(item)

    tt_array = np.array(data_list)
    # print('tt_array: ', tt_array)
    data_keep = (tt_array < big_threshold) & (
        tt_array >= small_threshold)
    tt_array = tt_array[data_keep]

    if len(tt_array) < min_nb:
        # print("length smaller than {}...".format(self.min_nb))
        return np.zeros(len(hist_bin)-1)

    hist, bin_edges = np.histogram(tt_array, hist_bin, density=True)

    if np.isnan(hist).any():
        return np.zeros(len(hist_bin)-1)

    hist *= hist_bin[1] - hist_bin[0]

    return hist


def construct_o_matrix_list(df_od_gb, oid, sample_rate, graph_edges):
    """
    Construct the Origin-Time matrix with list of elements

    :param df_od_gb: group-by dataframe, with key (O_id, D_id)
    :param oid: id, Origin id
    :return: Origin-Time dataframe
    """

    sample_str = '{}T'.format(sample_rate)
    list_dfs = []
    for did in graph_edges:
        od_key = (oid, did)
        od_pair = 'O{0}_D{1}'.format(oid, did)
        od_vel = pd.DataFrame()
        if od_key in df_od_gb.groups:
            df_od_group = df_od_gb.get_group(od_key)
            od_vel[od_pair] = df_od_group.speed.resample(
                sample_str).apply(vel_list)
        else:
            od_vel[od_pair] = None
        list_dfs.append(od_vel)
    df_od_tb = pd.concat(list_dfs, axis=1, join='outer')

    return df_od_tb


def construct_OD_time_dataset(
        df_of, server_name, num_bins, sample_rate, graph_edges):
    """
    Construct the 4D-tensor for NN

    :param df_of: pandas dataframe
    :param dataset_file: hdf5 file name
    :return: hdf5 format data set
    """

    hist_range = np.linspace(0, 21, num_bins+1)
    dist_unit = 1.0
    if server_name == 'nyc':
        dist_unit = 1609.34

    df_of['speed'] = df_of['dist'] * dist_unit / df_of['tt']
    print("ODT Tensor Constructing...")
    print(df_of.head())
    st_time_stamp_str = pd.to_datetime(df_of.time).min().strftime("%Y-%m-%d")
    end_time_stamp_str = pd.to_datetime(df_of.time).max().strftime("%Y-%m-%d")
    datetime_index = pd.date_range(
        st_time_stamp_str, end_time_stamp_str, freq='{}T'.format(sample_rate))
    # graph_edges = sorted(list(set(df_of.pick_id.tolist()).union(set(df_of.drop_id.tolist()))))
    # print(graph_edges)
    df_of = df_of.set_index('time')
    df_od_gb = df_of.groupby(['pick_id', 'drop_id'])
    print("Number of groups actually: ", len(df_od_gb.groups.keys()))
    odt_tensor = []
    odt_tensor_mape = []
    odt_tensor_num = []

    for oid in graph_edges:
        print("O: ", oid)
        df_od_tb = construct_o_matrix_list(
            df_od_gb, oid, sample_rate, graph_edges)

        df_od_tb = df_od_tb.reindex(datetime_index)
        # convert the dataframe with list into array
        df_od_tb_mape = df_od_tb.apply(
            my_rolling_apply_avg, axis=0,
            args=(get_vel_mape_rolling, 1))
        df_od_tb_num = df_od_tb.apply(
            my_rolling_apply_avg, axis=0,
            args=(get_vel_num_rolling, 1))

        df_od_tb = df_od_tb.apply(
            my_rolling_apply_list, axis=0,
            args=(get_vel_hist_rolling, 1,
                  hist_range))

        o_time_matrix = df_od_tb.values
        # print("O_time matrixx shape is ", o_time_matrix.shape)
        # Check here and notice the order of O and D
        # the axis for expand_dims should be 1 such that OD matrix is (O, D) !!!
        o_time_matrix = convert_multi_channel_array(o_time_matrix, num_bins)
        o_time_matrix = np.expand_dims(o_time_matrix, axis=1)
        odt_tensor.append(o_time_matrix)

        o_mape_matrix = df_od_tb_mape.values
        o_num_matrix = df_od_tb_num.values
        o_mape_matrix = np.expand_dims(o_mape_matrix, axis=1)
        o_num_matrix = np.expand_dims(o_num_matrix, axis=1)
        odt_tensor_mape.append(o_mape_matrix)
        odt_tensor_num.append(o_num_matrix)

    odt_tensor = np.concatenate(odt_tensor, axis=1)

    odt_tensor_mape = np.concatenate(odt_tensor_mape, axis=1)
    odt_tensor_num = np.concatenate(odt_tensor_num, axis=1)

    sum_last_axis = np.sum(odt_tensor, axis=-1)
    non_zeros_pos = sum_last_axis > 0
    delta_days = [(date_i - datetime_index.date[0]).days for date_i in datetime_index.date]
    delta_days = np.array(delta_days)
    time_inter = (delta_days * (60*24 / sample_rate) +
                 datetime_index.hour * (60 / sample_rate) +
                 datetime_index.minute / sample_rate).astype(int)
    dayofweek = datetime_index.dayofweek.values

    return odt_tensor, non_zeros_pos, time_inter

    # dataset_f = h5py.File(dataset_file, 'w')
    # dataset_f.create_dataset('data', data=odt_tensor)
    # dataset_f.create_dataset('weight', data=non_zeros_pos)
    # dataset_f.create_dataset('TI', data=time_inter)
    # dataset_f.create_dataset('DoW', data=dayofweek)
    # dataset_f.create_dataset('mape', data=odt_tensor_mape)
    # dataset_f.create_dataset('num', data=odt_tensor_num)
    # dataset_f.close()
    #
    # return dataset_f