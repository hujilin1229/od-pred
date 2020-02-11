from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim
import argparse
import itertools
import os
import sys
import numpy as np
import h5py
import pickle
from sklearn.preprocessing import MinMaxScaler
from lib.utils import construct_OD_time_dataset
from lib import metrics_weight


L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
# Data loaded
server_name = 'chengdu'
borough = 'sr'
start_date = '2014-08-03'
end_date = '2014-08-31'

sample_rate = 15
batch_size = 1024
epochs = 1000
num_lats = num_lons = 20
NUM_BINS = 7

class batch_data(object):
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:
    - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])
    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    def __init__(self, input_dict, output_dict):

        self.input_dict = input_dict
        self.output_dict = output_dict
        self._input_keys = input_dict.keys()
        self._output_keys = output_dict.keys()
        self._random_key = list(input_dict.keys())[0]
        self._data_len = self.input_dict[self._random_key].shape[0]
        self._batch_id = 0

    @property
    def batch_id(self):
        return self._batch_id

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self._batch_id == self._data_len:
            self._batch_id = 0
        batch_input = {}
        batch_output = {}
        for key_i in self._input_keys:
            batch_input[key_i] = self.input_dict[key_i][
                                  self._batch_id:min(self._batch_id + batch_size, self._data_len)]
        for key_i in self._output_keys:
            batch_output[key_i] = self.output_dict[key_i][
                                  self._batch_id:min(self._batch_id + batch_size, self._data_len)]
        self._batch_id = min(self._batch_id + batch_size, self._data_len)
        return batch_input, batch_output


def train_val_test_split_df(df, val_ratio=0.1, test_ratio=0.2):
    n_sample = df.shape[0]
    n_val = int(round(n_sample * val_ratio))
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_val - n_test
    train_data, val_data, test_data = df[:n_train, ...], df[n_train: n_train + n_val, ...], df[-n_test:, ...]
    return train_data, val_data, test_data


def prepare_input_output_data(df, deepwalk_embed_mat,
                              dist_scaler, time_scaler):
    input_keys = ['o_edge', 'd_edge', 'pick_lat_ind', 'pick_lon_ind',
                  'drop_lat_ind', 'drop_lon_ind', 'time_interval', 'dow']
    input_dict = {}
    output_dict = {}

    for key_i in input_keys[:2]:
        input_dict[key_i] = deepwalk_embed_mat[df[key_i].astype(int).values]
    for key_i in input_keys[2:]:
        input_dict[key_i] = np.expand_dims(df[key_i].astype(int).values, -1)

    label_dist = dist_scaler.transform(np.reshape(df['trip_distance'].values, (-1, 1)))
    label_tt = time_scaler.transform(np.reshape(df['time_duration'].values, (-1, 1)))
    output_dict['trip_distance'] = label_dist
    output_dict['time_duration'] = label_tt

    return input_dict, output_dict


def construct_laplacian_mat(num_rows, num_cols):
    """
    Construct Laplacian matrix for grid with adjacency of 1

    :param num_rows:
    :param num_cols:
    :return:
    """
    dim = int(num_rows * num_cols)
    adj_mat = np.zeros((dim, dim))
    for i in range(num_rows):
        for j in range(num_cols):
            adj_mat[i][j] = 1
            adj_mat[max(0, i - 1)][j] = 1
            adj_mat[min(num_rows - 1, i + 1)][j] = 1
            adj_mat[i][max(0, j - 1)] = 1
            adj_mat[i][min(num_cols - 1, j + 1)] = 1
    degree_mat = np.diag(np.sum(adj_mat, axis=1))
    lap_mat = degree_mat - adj_mat

    return lap_mat


def load_data(server_name, borough, start_date, end_date,
              sample_rate, batch_size, num_lats, num_lons):

    base_dir = './data/{0}/MURA/'.format(server_name)
    train_file = os.path.join(base_dir, 'train_dataset.mat')
    val_file = os.path.join(base_dir, 'val_dataset.mat')
    test_file = os.path.join(base_dir, 'test_dataset.mat')
    tt_scaler = os.path.join(base_dir, 'tt_scaler')
    dist_scaler = os.path.join(base_dir, 'dist_scaler')

    if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file) and os.path.exists(
            tt_scaler) and os.path.exists(dist_scaler):
        with open(train_file, 'rb') as f:
            train_mat = pickle.load(f)
        train_dataset = (train_mat['inputs'], train_mat['outputs'])
        with open(val_file, 'rb') as f:
            val_mat = pickle.load(f)
        val_dataset = (val_mat['inputs'], val_mat['outputs'])
        with open(test_file, 'rb') as f:
            test_mat = pickle.load(f)
        test_dataset = (test_mat['inputs'], test_mat['outputs'])
        with open(tt_scaler, 'rb') as f:
            time_duration_scaler = pickle.load(f)
        with open(dist_scaler, 'rb') as f:
            trip_dist_scaler = pickle.load(f)

    else:
        # split the data evenly according to the time index
        df_data = pd.read_csv('./data/{0}/MURA/{0}_{1}_mura.csv'.format(server_name, borough))

        # same data preprocessing
        df_data.time_duration[df_data.time_duration < 0] += 3600
        del df_data['c_path']
        print("Constructed DF shape: ", df_data.shape)
        df_data = df_data.dropna()
        df_data = df_data[(df_data.time_duration >= 60) &
                      (df_data.time_duration <= 10000)]
        # the trip_distance should be larger than 0.0
        df_data = df_data[df_data.trip_distance > 0.0]
        print("Processed DF shape: ", df_data.shape) # nyc 14,165,446

        df_data.time = pd.to_datetime(df_data.time)
        # construct latitude and longitude index, respectively
        lat_max = max(df_data.pickup_latitude.max(), df_data.dropoff_latitude.max())
        lat_min = max(df_data.pickup_latitude.min(), df_data.dropoff_latitude.min())
        lon_max = max(df_data.pickup_longitude.max(), df_data.dropoff_longitude.max())
        lon_min = max(df_data.pickup_longitude.min(), df_data.dropoff_longitude.min())
        lat_sep = (lat_max - lat_min) / num_lats
        lon_sep = (lon_max - lon_min) / num_lons

        df_data['pick_lat_ind'] = ((df_data.pickup_latitude - lat_min) / lat_sep).apply(np.floor)
        df_data['pick_lon_ind'] = ((df_data.pickup_longitude - lon_min) / lon_sep).apply(np.floor)
        df_data['drop_lat_ind'] = ((df_data.dropoff_latitude - lat_min) / lat_sep).apply(np.floor)
        df_data['drop_lon_ind'] = ((df_data.dropoff_longitude - lon_min) / lon_sep).apply(np.floor)
        df_data['pick_lat_ind'][df_data['pick_lat_ind'] == num_lats] = num_lats - 1
        df_data['pick_lon_ind'][df_data['pick_lon_ind'] == num_lons] = num_lons - 1
        df_data['drop_lat_ind'][df_data['drop_lat_ind'] == num_lats] = num_lats - 1
        df_data['drop_lon_ind'][df_data['drop_lon_ind'] == num_lons] = num_lons - 1
        df_data['pick_lat_ind'][df_data['pick_lat_ind'] < 0] = 0
        df_data['pick_lon_ind'][df_data['pick_lon_ind'] < 0] = 0
        df_data['drop_lat_ind'][df_data['drop_lat_ind'] < 0] = 0
        df_data['drop_lon_ind'][df_data['drop_lon_ind'] < 0] = 0

        st_time_stamp_str = start_date
        end_time_stamp_str = end_date
        datetime_index = pd.date_range(
            st_time_stamp_str, end_time_stamp_str,
            freq='{}T'.format(sample_rate))
        train_date, val_date, test_date = train_val_test_split_df(datetime_index)
        df_train = df_data[df_data.time < val_date.min()]
        df_val = df_data[(df_data.time >= val_date.min()) & (df_data.time < test_date.min())]
        df_test = df_data[df_data.time >= test_date.min()]

        # load the deep walk embedding results
        # 1. use deepwalk to running the link graph embedding
        # 2. read the resulted file with pandas read_csv
        # 3. get the graph embedding results
        deepwalk_embed_mat = pd.read_csv(
            './data/{0}/MURA/{1}.embedding'.format(server_name, borough),
            skiprows=1, sep=' ', header=None, index_col=0).sort_index(axis=0).values

        # MinMaxNormalizer
        whole_trip_dist = df_data['trip_distance'].values
        whole_time_duration = df_data['time_duration'].values
        trip_dist_scaler = MinMaxScaler().fit(np.reshape(whole_trip_dist, (-1, 1)))
        time_duration_scaler = MinMaxScaler().fit(np.reshape(whole_time_duration, (-1, 1)))
        # construct the training data and validating data
        # inputs column names: ['o_edge', 'd_edge', 'lat_ind', 'lon_ind', 'time_interval', 'dow']
        train_dataset = prepare_input_output_data(df_train, deepwalk_embed_mat,
                                                  trip_dist_scaler, time_duration_scaler)
        val_dataset = prepare_input_output_data(df_val, deepwalk_embed_mat,
                                                trip_dist_scaler, time_duration_scaler)
        test_dataset = prepare_input_output_data(df_test, deepwalk_embed_mat,
                                                 trip_dist_scaler, time_duration_scaler)
        test_dataset[0]['pick_id'] = df_test['pickup_location_id'].astype(int).values
        test_dataset[0]['drop_id'] = df_test['dropoff_location_id'].astype(int).values
        test_dataset[0]['time'] = df_test['time'].values

        with open(train_file, 'wb') as handle:
            pickle.dump({'inputs': train_dataset[0],
                         'outputs': train_dataset[1]}, handle)

        with open(val_file, 'wb') as handle:
            pickle.dump({'inputs': val_dataset[0],
                         'outputs': val_dataset[1]}, handle)

        with open(test_file, 'wb') as handle:
            pickle.dump({'inputs': test_dataset[0],
                         'outputs': test_dataset[1]}, handle)
        with open(tt_scaler, 'wb') as f:
            pickle.dump(time_duration_scaler, f)
        with open(dist_scaler, 'wb') as f:
            pickle.dump(trip_dist_scaler, f)


    # load/construct the graph laplacian matrix for spatial and temporal graph
    laplacian_spatial = construct_laplacian_mat(num_lats, num_lons)
    laplacian_temporal = construct_laplacian_mat(7, int(60 / sample_rate * 24))

    return train_dataset, val_dataset, test_dataset, \
           laplacian_spatial, laplacian_temporal, \
           time_duration_scaler, trip_dist_scaler


def load_data_cd(server_name, borough, start_date, end_date,
              sample_rate, batch_size, num_lats, num_lons):

    base_dir = './data/{0}/MURA/'.format(server_name)
    train_file = os.path.join(base_dir, 'train_dataset.mat')
    val_file = os.path.join(base_dir, 'val_dataset.mat')
    test_file = os.path.join(base_dir, 'test_dataset.mat')
    tt_scaler = os.path.join(base_dir, 'tt_scaler')
    dist_scaler = os.path.join(base_dir, 'dist_scaler')

    if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file) and os.path.exists(
            tt_scaler) and os.path.exists(dist_scaler):
        with open(train_file, 'rb') as f:
            train_mat = pickle.load(f)
        train_dataset = (train_mat['inputs'], train_mat['outputs'])
        with open(val_file, 'rb') as f:
            val_mat = pickle.load(f)
        val_dataset = (val_mat['inputs'], val_mat['outputs'])
        with open(test_file, 'rb') as f:
            test_mat = pickle.load(f)
        test_dataset = (test_mat['inputs'], test_mat['outputs'])
        with open(tt_scaler, 'rb') as f:
            time_duration_scaler = pickle.load(f)
        with open(dist_scaler, 'rb') as f:
            trip_dist_scaler = pickle.load(f)

    else:
        # Read map matching results
        # df_mr_cd = pd.read_csv('./data/{0}/MURA/mr_{0}.txt'.format(server_name), sep=';')
        # engine = create_engine('postgresql://jilin:jilin@172.19.18.4:5432/chengdu')
        # sql_trips = 'select * from od_trips_srid_poly'
        # df_trips = pd.read_sql(sql_trips, engine)
        # df_combine = df_trips.join(df_mr_cd.set_index('id'), on='id')
        # df_o_path = df_combine['o_path'].str.split(",", n=1, expand=True)
        # df_combine['o_edge'] = df_o_path[0]
        # df_combine['d_edge'] = df_o_path[1]
        # df_combine['pickup_location_id'] = df_combine['pickup_sr_id'].astype(int)
        # df_combine['dropoff_location_id'] = df_combine['dropoff_sr_id'].astype(int)
        # df_combine['time'] = pd.to_datetime(df_combine.pickup_time)
        # df_combine['dow'] = df_combine.time.dt.dayofweek
        # df_combine['time_interval'] = df_combine.time.dt.hour * 4 + (df_combine.time.dt.minute / 15).apply(np.floor)
        # df_combine['time_duration'] = df_combine.total_time
        # del df_combine['total_time']
        # df_combine.to_csv('./data/{0}/MURA/{0}_{1}_mura.csv'.format(server_name, borough))

        # split the data evenly according to the time index
        df_data = pd.read_csv('./data/{0}/MURA/{0}_{1}_mura.csv'.format(server_name, borough))

        # same data preprocessing
        df_data.time_duration[df_data.time_duration < 0] += 3600
        del df_data['c_path']
        print("Constructed DF shape: ", df_data.shape)
        df_data = df_data.dropna()
        df_data = df_data[(df_data.time_duration >= 60) &
                      (df_data.time_duration <= 10000)]
        # the trip_distance should be larger than 0.0
        if server_name == 'nyc':
            df_data = df_data[df_data.trip_distance > 0.0]
        else:
            df_data = df_data[(df_data.trip_distance > 1000) &
                              (df_data.trip_distance < 15000)]
            df_data['speed'] = df_data['trip_distance'] / df_data['time_duration']
            df_data = df_data[(df_data.speed >= 1.0) & (df_data.speed <= 40.)]

        print("Processed DF shape: ", df_data.shape) # nyc 14,165,446

        df_data.time = pd.to_datetime(df_data.time)
        # construct latitude and longitude index, respectively
        lat_max = max(df_data.pickup_latitude.max(), df_data.dropoff_latitude.max())
        lat_min = max(df_data.pickup_latitude.min(), df_data.dropoff_latitude.min())
        lon_max = max(df_data.pickup_longitude.max(), df_data.dropoff_longitude.max())
        lon_min = max(df_data.pickup_longitude.min(), df_data.dropoff_longitude.min())
        lat_sep = (lat_max - lat_min) / num_lats
        lon_sep = (lon_max - lon_min) / num_lons

        df_data['pick_lat_ind'] = ((df_data.pickup_latitude - lat_min) / lat_sep).apply(np.floor)
        df_data['pick_lon_ind'] = ((df_data.pickup_longitude - lon_min) / lon_sep).apply(np.floor)
        df_data['drop_lat_ind'] = ((df_data.dropoff_latitude - lat_min) / lat_sep).apply(np.floor)
        df_data['drop_lon_ind'] = ((df_data.dropoff_longitude - lon_min) / lon_sep).apply(np.floor)
        df_data['pick_lat_ind'][df_data['pick_lat_ind'] == num_lats] = num_lats - 1
        df_data['pick_lon_ind'][df_data['pick_lon_ind'] == num_lons] = num_lons - 1
        df_data['drop_lat_ind'][df_data['drop_lat_ind'] == num_lats] = num_lats - 1
        df_data['drop_lon_ind'][df_data['drop_lon_ind'] == num_lons] = num_lons - 1
        df_data['pick_lat_ind'][df_data['pick_lat_ind'] < 0] = 0
        df_data['pick_lon_ind'][df_data['pick_lon_ind'] < 0] = 0
        df_data['drop_lat_ind'][df_data['drop_lat_ind'] < 0] = 0
        df_data['drop_lon_ind'][df_data['drop_lon_ind'] < 0] = 0

        st_time_stamp_str = start_date
        end_time_stamp_str = end_date
        datetime_index = pd.date_range(
            st_time_stamp_str, end_time_stamp_str,
            freq='{}T'.format(sample_rate))
        train_date, val_date, test_date = train_val_test_split_df(datetime_index)
        df_train = df_data[df_data.time < val_date.min()]
        df_val = df_data[(df_data.time >= val_date.min()) & (df_data.time < test_date.min())]
        df_test = df_data[df_data.time >= test_date.min()]

        # load the deep walk embedding results
        # 1. use deepwalk to running the link graph embedding
        # 2. read the resulted file with pandas read_csv
        # 3. get the graph embedding results
        deepwalk_embed_mat = pd.read_csv(
            './data/{0}/MURA/{1}.embedding'.format(server_name, borough),
            skiprows=1, sep=' ', header=None, index_col=0).sort_index(axis=0).values

        # MinMaxNormalizer
        whole_trip_dist = df_data['trip_distance'].values
        whole_time_duration = df_data['time_duration'].values
        trip_dist_scaler = MinMaxScaler().fit(np.reshape(whole_trip_dist, (-1, 1)))
        time_duration_scaler = MinMaxScaler().fit(np.reshape(whole_time_duration, (-1, 1)))
        # construct the training data and validating data
        # inputs column names: ['o_edge', 'd_edge', 'lat_ind', 'lon_ind', 'time_interval', 'dow']
        train_dataset = prepare_input_output_data(df_train, deepwalk_embed_mat,
                                                  trip_dist_scaler, time_duration_scaler)
        val_dataset = prepare_input_output_data(df_val, deepwalk_embed_mat,
                                                trip_dist_scaler, time_duration_scaler)
        test_dataset = prepare_input_output_data(df_test, deepwalk_embed_mat,
                                                 trip_dist_scaler, time_duration_scaler)
        test_dataset[0]['pick_id'] = df_test['pickup_location_id'].astype(int).values
        test_dataset[0]['drop_id'] = df_test['dropoff_location_id'].astype(int).values
        test_dataset[0]['time'] = df_test['time'].values

        with open(train_file, 'wb') as handle:
            pickle.dump({'inputs': train_dataset[0],
                         'outputs': train_dataset[1]}, handle)

        with open(val_file, 'wb') as handle:
            pickle.dump({'inputs': val_dataset[0],
                         'outputs': val_dataset[1]}, handle)

        with open(test_file, 'wb') as handle:
            pickle.dump({'inputs': test_dataset[0],
                         'outputs': test_dataset[1]}, handle)
        with open(tt_scaler, 'wb') as f:
            pickle.dump(time_duration_scaler, f)
        with open(dist_scaler, 'wb') as f:
            pickle.dump(trip_dist_scaler, f)


    # load/construct the graph laplacian matrix for spatial and temporal graph
    laplacian_spatial = construct_laplacian_mat(num_lats, num_lons)
    laplacian_temporal = construct_laplacian_mat(7, int(60 / sample_rate * 24))

    return train_dataset, val_dataset, test_dataset, \
           laplacian_spatial, laplacian_temporal, \
           time_duration_scaler, trip_dist_scaler


def fc_block(input_tensor, units, stage):
    """A block that has two fully-connected layer.
    # Arguments
        input_tensor: input tensor
        units: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the second conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the second conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """

    bn_axis = 1
    fc_name_base = 'res' + str(stage) + '_branch'
    bn_name_base = 'bn' + str(stage) + '_branch'

    x = tf.layers.batch_normalization(input_tensor, axis=bn_axis,
                                      momentum=BATCH_NORM_DECAY,
                                      epsilon=BATCH_NORM_EPSILON,
                                      name=bn_name_base + '1')
    x = tf.nn.relu(x)
    x = tf.layers.dense(x, units, name=fc_name_base + '1')

    x = tf.layers.batch_normalization(x, axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2')
    x = tf.nn.relu(x)
    x = tf.layers.dense(x, units, name=fc_name_base + '2')

    x = tf.concat([x, input_tensor], axis=-1)

    return x


def resnet11(o_edge_embedding, d_edge_embedding,
             pick_lon_input, pick_lat_input,
             drop_lon_input, drop_lat_input,
             ti_input, dow_input, lap_spatial,
             lap_temporal):
    """Instantiates the ResNet11 architecture.
    Args:
      num_classes: `int` number of classes for image classification.
    Returns:
        A Keras model instance.
    """

    lon_embedding = tf.Variable(tf.random_normal([num_lons, 20]), name='lon_embed')
    lat_embedding = tf.Variable(tf.random_normal([num_lats, 20]), name='lat_embed')
    ti_embedding = tf.Variable(tf.random_normal([int(60 / sample_rate) * 24, 20]), name='ti_embed')
    dow_embedding = tf.Variable(tf.random_normal([7, 20]), name='dow_embed')

    pick_lon_input = tf.layers.flatten(tf.nn.embedding_lookup(lon_embedding, pick_lon_input))
    pick_lat_input = tf.layers.flatten(tf.nn.embedding_lookup(lat_embedding, pick_lat_input))
    drop_lon_input = tf.layers.flatten(tf.nn.embedding_lookup(lon_embedding, drop_lon_input))
    drop_lat_input = tf.layers.flatten(tf.nn.embedding_lookup(lat_embedding, drop_lat_input))
    ti_input = tf.layers.flatten(tf.nn.embedding_lookup(ti_embedding, ti_input))
    dow_input = tf.layers.flatten(tf.nn.embedding_lookup(dow_embedding, dow_input))

    inputs = tf.concat([o_edge_embedding, d_edge_embedding,
                        pick_lon_input, pick_lat_input,
                        drop_lon_input, drop_lat_input,
                        ti_input, dow_input], axis=-1)

    x = fc_block(inputs, 1024, stage=1)
    x = fc_block(x, 1024, stage=2)
    x = fc_block(x, 1024, stage=3)
    x = fc_block(x, 1024, stage=4)
    x = fc_block(x, 1024, stage=5)
    tt = tf.layers.dense(x, 1, activation=tf.nn.sigmoid, name='travel_time')
    dist = tf.layers.dense(x, 1, activation=tf.nn.sigmoid, name='travel_distance')

    # Construct the Graph Laplacian regularizers loss
    with tf.name_scope('spatial'):
        spatial_error = construct_graph_lap_regularizer(tf.convert_to_tensor(lat_embedding),
                                                        tf.convert_to_tensor(lon_embedding), lap_spatial)

    with tf.name_scope('temporal'):
        temporal_error = construct_graph_lap_regularizer(tf.convert_to_tensor(dow_embedding),
                                                         tf.convert_to_tensor(ti_embedding), lap_temporal)

    # Create model.
    return tt, dist, spatial_error, temporal_error


def construct_graph_lap_regularizer(row_embed, col_embed, lap):
    """
    Compute the graph laplacian regularizer value

    :param row_embed: embedding for row
    :param col_embed: embedding for col
    :param lap: laplacian matrix for graph
    :return: scalar value
    """
    num_rows = int(row_embed.get_shape()[0])
    num_cols = int(col_embed.get_shape()[0])
    assert (num_rows * num_cols) == lap.get_shape()[0]

    # the following operation is implement np.repeat(row_embed, (num_cols, 1))
    order = list(range(0, num_cols * num_rows, num_rows))
    order = [[x + i for x in order] for i in range(num_rows)]
    order = list(itertools.chain.from_iterable(order))
    row_embed_expand = tf.gather(tf.tile(row_embed, [num_cols, 1]), order)
    col_embed_expand = tf.tile(col_embed, [num_rows, 1])
    row_col_combine = tf.concat([row_embed_expand, col_embed_expand], axis=1)

    trace_error = tf.linalg.trace(tf.matmul(tf.matmul(tf.transpose(row_col_combine), lap), row_col_combine))

    return trace_error

def loss_func(tt, dist, spatial_error, temporal_error,
              tt_label, dist_label, spatial_lambda,
              temporal_lambda):

    normal_loss = tf.reduce_mean(tf.abs(tt - tt_label)) + \
                      tf.reduce_mean(tf.abs(dist - dist_label))

    multi_task_loss = normal_loss + spatial_lambda * tf.reduce_sum(spatial_error) \
                     + temporal_lambda * tf.reduce_sum(temporal_error)

    return multi_task_loss

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def write_results_into_df(
        y_preds, y_gt, current_time_interval, wt_i, nodes, dist_mx):
    head_column = ['KL', 'EMD', 'JS', 'TI', 'Horizon', 'O_id', 'D_id', 'Dist']
    dest_df = os.path.join(
        './result', 'MURA_OD_{0}_{1}.csv'.format(server_name, sample_rate))

    dict_horizon_i = {}
    for head_i in head_column:
        dict_horizon_i[head_i] = []

    current_time_interval = np.reshape(current_time_interval, [y_preds.shape[0], 1, 1])

    kl, jsd, emd = metrics_weight.calculate_metrics_hist_matrix(y_preds, y_gt)

    num_nodes = len(nodes)
    oids = np.tile(np.arange(num_nodes).reshape(num_nodes, 1), [1, num_nodes])
    dids = np.tile(np.arange(num_nodes).reshape(1, num_nodes), [num_nodes, 1])
    Oids = np.tile(np.reshape(oids, [1, num_nodes, num_nodes]), [y_preds.shape[0], 1, 1])
    Dids = np.tile(np.reshape(dids, [1, num_nodes, num_nodes]), [y_preds.shape[0], 1, 1])
    current_time_interval = np.tile(current_time_interval, [1, num_nodes, num_nodes])
    dist_mx_tile = np.expand_dims(dist_mx, axis=0)
    dist_mx_tile = np.tile(dist_mx_tile, [y_preds.shape[0], 1, 1])

    dict_horizon_i['KL'] = kl[wt_i].tolist()
    dict_horizon_i['EMD'] = emd[wt_i].tolist()
    dict_horizon_i['JS'] = jsd[wt_i].tolist()
    dict_horizon_i['O_id'] = Oids[wt_i].tolist()
    dict_horizon_i['D_id'] = Dids[wt_i].tolist()
    dict_horizon_i['TI'] = current_time_interval[wt_i].tolist()
    dict_horizon_i['Horizon'] = [0] * np.sum(wt_i)
    dict_horizon_i['Dist'] = dist_mx_tile[wt_i].tolist()
    df_result = pd.DataFrame(dict_horizon_i)

    with open(dest_df, 'w') as f:
        df_result.to_csv(f, header=True)


def construct_compare_od_hist():
    train_dataset, val_dataset, test_dataset, \
    laplacian_spatial, laplacian_temporal, \
    tt_scaler, dist_scaler = load_data(server_name, borough, start_date,
                                       end_date, sample_rate, batch_size,
                                       num_lats, num_lons)

    with open('./exp/{}/mura_result.pickle'.format(server_name), 'rb') as f:
        results = pickle.load(f)
        # results['tt'] = np.reshape(tt_scaler.inverse_transform(results['tt']), -1)
        # TODO: In current case, no inverse_transform here
        results['tt'] = np.reshape(results['tt'], -1)
        results['dist'] = np.reshape(dist_scaler.inverse_transform(results['dist']), -1)
        print(results.keys())
    # Read node and dist_adj from file
    with open('./data/chengdu/MURA/edges_hop10_sigma_9.pickle', 'rb') as f:
        edges = pickle.load(f)
    with open('./data/chengdu/MURA/edge_dist.pickle', 'rb') as f:
        dist_mx = pickle.load(f)

    y_preds, w_preds, ti = construct_OD_time_dataset(
        pd.DataFrame(results), server_name, NUM_BINS, sample_rate, edges)
    results['tt'] = tt_scaler.inverse_transform(test_dataset[1]['time_duration'])
    results['dist'] = dist_scaler.inverse_transform(test_dataset[1]['trip_distance'])
    results['tt'] = np.reshape(results['tt'], -1)
    results['dist'] = np.reshape(results['dist'], -1)
    y_gt, _, _ = construct_OD_time_dataset(
        pd.DataFrame(results), server_name, NUM_BINS, sample_rate, edges)

    kl, l2, emd, jsd = metrics_weight.calculate_metrics_hist(y_preds, y_gt, w_preds)
    message = 'kl:%.4f, jsd:%.4f, emd:%.4f, l2:%.4f' % (kl, jsd, emd, l2)
    print(message)

    write_results_into_df(y_preds, y_gt, ti, w_preds, edges, dist_mx)

    dataset_file = './exp/{}/predicted_results.mat'.format(server_name)
    dataset_f = h5py.File(dataset_file, 'w')
    dataset_f.create_dataset('pred_od', data=y_preds)
    dataset_f.create_dataset('gt_od', data=y_gt)
    dataset_f.create_dataset('w_od', data=w_preds)
    dataset_f.close()


def mura_basic(log_dir):
    """Example of building, training and visualizing a word2vec model."""
    # Create the directory for TensorBoard variables if there is not.
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    train_dataset, val_dataset, test_dataset, \
    laplacian_spatial, laplacian_temporal, \
    tt_scaler, dist_scaler = load_data_cd(server_name, borough, start_date,
                                          end_date, sample_rate, batch_size,
                                          num_lats, num_lons)

    laplacian_spatial = construct_laplacian_mat(num_lats, num_lons)
    laplacian_temporal = construct_laplacian_mat(7, int(60 / sample_rate * 24))

    # tf Graph Input
    input_keys = ['o_edge', 'd_edge', 'pick_lat_ind', 'pick_lon_ind',
                  'drop_lat_ind', 'drop_lon_ind', 'time_interval', 'dow']
    output_keys = ['time_duration', 'trip_distance']

    o_edge_embedding = tf.placeholder(tf.float32, [None, 40], name='o_edge')
    d_edge_embedding = tf.placeholder(tf.float32, [None, 40], name='d_edge')
    pick_lon_input = tf.placeholder(tf.int32, [None, 1], name='pick_lon_ind')
    pick_lat_input = tf.placeholder(tf.int32, [None, 1], name='pick_lat_ind')
    drop_lon_input = tf.placeholder(tf.int32, [None, 1], name='drop_lon_ind')
    drop_lat_input = tf.placeholder(tf.int32, [None, 1], name='drop_lat_ind')
    ti_input = tf.placeholder(tf.int32, [None, 1], name='ti')
    dow_input = tf.placeholder(tf.int32, [None, 1], name='dow')
    # output placeholder
    tt_output = tf.placeholder(tf.float32, [None, 1], name='time_duration')
    dist_output = tf.placeholder(tf.float32, [None, 1], name='trip_distance')
    learning_rate = tf.placeholder(tf.float32, shape=[])

    input_ph_dict = {'o_edge': o_edge_embedding,
                     'd_edge': d_edge_embedding,
                     'pick_lat_ind': pick_lat_input,
                     'pick_lon_ind': pick_lon_input,
                     'drop_lon_ind': drop_lon_input,
                     'drop_lat_ind': drop_lat_input,
                     'time_interval': ti_input,
                     'dow': dow_input}

    output_ph_dict = {'time_duration': tt_output,
                      'trip_distance': dist_output}
    # constant placeholder for two graph laplacian
    laplacian_spatial_ph = tf.constant(laplacian_spatial, tf.float32, name='spatial_lap')
    laplacian_temporal_ph = tf.constant(laplacian_temporal, tf.float32, name='temp_lap')

    tt, dist, spatial_error, \
    temporal_error = resnet11(o_edge_embedding, d_edge_embedding,
                              pick_lon_input, pick_lat_input,
                              drop_lon_input, drop_lat_input,
                              ti_input, dow_input, laplacian_spatial_ph,
                              laplacian_temporal_ph)
    # Define loss and optimizer
    training_steps = 5000
    display_step = 100
    cost = loss_func(tt, dist, spatial_error,
                     temporal_error, tt_output, dist_output, 1e-7, 1e-7)

    print("Initialized...")
    lr_decay = 0.2
    cur_lr = 1e-5
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    model_summary()
    print("optimizing....")
    tt_loss = tf.reduce_mean(tf.abs(tt_output - tt))
    dist_loss = tf.reduce_mean(tf.abs(dist_output - dist))
    train_batch = batch_data(train_dataset[0], train_dataset[1])
    # Initialize the variables (i.e. assign their default value)
    init_op = tf.global_variables_initializer()

    num_epochs = -1
    step = 0
    # Launch the graph
    sess = tf.Session()
    # Run the initializer
    sess.run(init_op)
    while num_epochs <= 100:
        if train_batch.batch_id == 0:
            num_epochs += 1
        if num_epochs % 2 == 0:
            cur_lr = cur_lr*lr_decay
        step += 1
        batch_input_dict, batch_output_dict = train_batch.next(batch_size)
        # Run optimization op (backprop)
        feed_dict = {}
        for key_i in input_keys:
            feed_dict[input_ph_dict[key_i]] = batch_input_dict[key_i]
        for key_i in output_keys:
            feed_dict[output_ph_dict[key_i]] = batch_output_dict[key_i]
        feed_dict[learning_rate] = cur_lr
        sess.run(optimizer, feed_dict=feed_dict)
        if step % display_step == 0 or step == 1:
            # Calculate batch accuracy & loss
            # Calculate accuracy
            val_feed_dict = {}
            for key_i in input_keys:
                val_feed_dict[input_ph_dict[key_i]] = val_dataset[0][key_i]

            val_feed_dict[output_ph_dict['time_duration']] = val_dataset[1]['time_duration']
            val_feed_dict[output_ph_dict['trip_distance']] = val_dataset[1]['trip_distance']

            pred_tt, val_loss = sess.run([tt, cost], feed_dict=val_feed_dict)
            gt_tt = tt_scaler.inverse_transform(val_dataset[1]['time_duration'])
            pred_tt = tt_scaler.inverse_transform(pred_tt)
            pred_mae = np.mean(np.abs(gt_tt - pred_tt) / gt_tt)

            feed_dict[tt_output] = batch_output_dict['time_duration']
            acc, d_loss, loss = sess.run([tt_loss, dist_loss, cost], feed_dict=feed_dict)
            print("Step " + str(step * batch_size) + ", Minibatch Total Loss= " + \
                  "{:.6f}".format(loss) + ", Training TT Loss = " + \
                  "{:.5f}".format(acc) + ", Val Toal Loss = {:.6f}".format(val_loss) + \
                  "dist loss = {:.5f}".format(d_loss) + ", Val Toal Loss = {:.6f}".format(val_loss) + \
                  "Validating MAPE {:.5f} ".format(pred_mae))

    print("Optimization Finished!")

    # Calculate accuracy
    feed_dict = {}
    for key_i in input_keys:
        feed_dict[input_ph_dict[key_i]] = test_dataset[0][key_i]
    pred_tt, pred_dist = sess.run([tt, dist], feed_dict=feed_dict)
    results = {'tt': pred_tt,
               'dist': pred_dist,
               'pick_id': test_dataset[0]['pick_id'],
               'drop_id': test_dataset[0]['drop_id'],
               'time': test_dataset[0]['time']
               }
    with open('./exp/{}/mura_result2.pickle'.format(server_name), 'wb') as f:
        pickle.dump(results, f)


    gt_tt = tt_scaler.inverse_transform(test_dataset[1]['time_duration'])
    pred_tt = tt_scaler.inverse_transform(pred_tt)
    pred_mae = np.mean(np.abs(gt_tt - pred_tt) / gt_tt)
    print("Testing Accuracy:", pred_mae)


# All functionality is run after tf.app.run() (b/122547914). This could be split
# up but the methods are laid sequentially with their usage for clarity.
def main(unused_argv):
    # Give a folder path as an argument with '--log_dir' to save
    # TensorBoard summaries. Default is a log folder in current directory.
    current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(current_path, 'log'),
        help='The log directory for TensorBoard summaries.')
    flags, unused_flags = parser.parse_known_args()
    mura_basic(flags.log_dir)
    construct_compare_od_hist()


if __name__ == '__main__':
    tf.app.run()
