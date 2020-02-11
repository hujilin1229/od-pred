from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import json
import os
from lib.utils import get_logger
from lib import preprocessing as prep
from model.mgrnn_supervisor_all import MGRNNSupervisor

# flags
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 20, 'Batch size')
flags.DEFINE_integer('cl_decay_steps', 100,
                     'Parameter to control the decay speed of probability of feeding groundth instead of model output.')
flags.DEFINE_integer('epochs', 400, 'Maximum number of epochs to train.')
flags.DEFINE_string('filter_type', 'laplacian', 'laplacian/random_walk/dual_random_walk.')
flags.DEFINE_string('activate_func', 'tanh', 'tanh/sigmoid/relu.')
flags.DEFINE_string('pool_type', '_mpool', '_mpool/_apool.')
flags.DEFINE_integer('horizon', 1, 'Maximum number of timestamps to prediction.')
flags.DEFINE_float('l1_decay', -1.0, 'L1 Regularization')
flags.DEFINE_float('lr_decay', -1.0, 'Learning rate decay.')
flags.DEFINE_integer('lr_decay_epoch', -1, 'The epoch that starting decaying the parameter.')
flags.DEFINE_integer('lr_decay_interval', -1, 'Interval beteween each deacy.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate. -1: select by hyperopt tuning.')
flags.DEFINE_float('drop_out', 0.0, 'drop_out 0.0: select by hyperopt tuning.')
flags.DEFINE_string('log_dir', None, 'Log directory for restoring the model from a checkpoint.')
flags.DEFINE_string('loss_func', 'L2', 'KL/L2/EMD: loss function.')
flags.DEFINE_string('optimizer', 'adam', 'adam/sgd/ada.')
flags.DEFINE_float('min_learning_rate', -1, 'Minimum learning rate')
flags.DEFINE_integer('nb_weeks', 17, 'How many week\'s data should be used for train/test.')
flags.DEFINE_integer('patience', -1,
                     'Maximum number of epochs allowed for non-improving validation error before early stopping.')
flags.DEFINE_integer('seq_len', 3, 'Sequence length.')
flags.DEFINE_integer('test_every_n_epochs', 10, 'Run model on the testing dataset every n epochs.')
flags.DEFINE_bool('coarsen', True, 'Apply coarsen on input data.')
flags.DEFINE_integer('coarsening_levels', 4, 'Number of coarsened graph.')
flags.DEFINE_integer('num_gpus', 2, 'How many GPUs to use.')
flags.DEFINE_bool('use_cpu_only', False, 'Set to true to only use cpu.')
flags.DEFINE_bool('shuffle_training', False, 'shuffle_training False: select by hyperopt tuning.')
flags.DEFINE_bool('use_curriculum_learning', None, 'Set to true to use Curriculum learning in decoding stage.')
flags.DEFINE_integer('verbose', -1, '1: to log individual sensor information.')

# flags for data related
flags.DEFINE_string('server_name', 'chengdu', 'The name of dataset to be processed')
flags.DEFINE_string('borough', 'SecRing', 'Selected area')
flags.DEFINE_string('zone', 'polygon', 'lon10_lat9, lon20_lat18, polygons, map partition method')
flags.DEFINE_integer('sample_rate', 20, 'Sample rate to condense the data')
flags.DEFINE_string('mode', 'hist', 'avg: for single value, hist: for multi values')
flags.DEFINE_string('data_format', 'speed', 'speed or duration')
flags.DEFINE_bool('duration_log', False, 'Apply log10 to the data, True when data_form is duration')
flags.DEFINE_bool('fill_mean', False, 'Fill HA to the data, True when need to fill in mean values')
flags.DEFINE_bool('sparse_removal', False, 'Apply sparse removal to the data, True when need to remove sparse regions')
flags.DEFINE_string('scaler', 'maxmin', 'maxmin: MaxMinScaler, std: Standard scaler')

flags.DEFINE_string('config_filename', './conf/base_config_chengdu.json',
                    'Configuration filename for restoring the model.')
flags.DEFINE_bool('add_time_in_day', True, 'add time in day as additional input feature')
flags.DEFINE_bool('add_day_in_week', True, 'add day in week as additional input feature')

# flags for the graph construction
flags.DEFINE_string('model_filename', None, 'model file name to restore.')
flags.DEFINE_string('model_dir', None, 'model dir to restore.')
flags.DEFINE_integer('hopk', 4, 'Hopk to construct the adjacent matrix')
flags.DEFINE_integer('sigma', 9, 'sigma used to construct the adj matrix')

flags.DEFINE_float('trace_ratio', 0.0, 'Trace ratio in loss')
flags.DEFINE_bool('is_restore', False, 'Whether in training or test model directly')

def main(_):
    # Reads graph data.
    with open(FLAGS.config_filename) as f:
        # load configuration
        data_model_config = json.load(f)
        data_config = data_model_config['data']
        # load data: include graph and data array
        for name in ['server_name', 'hopk', 'sigma', 'mode', 'zone', 'coarsen',
                     'coarsening_levels', 'borough', 'data_format', 'duration_log',
                     'sample_rate']:
            data_config[name] = getattr(FLAGS, name)
        data_config['window_size'] = getattr(FLAGS, 'seq_len')
        data_config['predict_size'] = getattr(FLAGS, 'horizon')
        data_config['base_dir'] = os.path.join(data_config['base_dir'],
                                               data_config['server_name'],
                                               data_config['borough'],
                                               data_config['zone'])
        # data_config['data_dir'] = os.path.join(data_config['base_dir'],
        #                                        'S{}'.format(data_config['sample_rate']),
        #                                        data_config['data_format'],
        #                                        data_config['mode'],
        #                                        'W{}_P{}'.format(data_config['window_size'],
        #                                                         data_config['predict_size']),
        #                                        'rm{}'.format(data_config['data_rm_ratio'])
        #                                        )
        logger = get_logger('./logs/', 'info.log')

        logger.info('Loading graph...')
        dataset = prep.CDData(**data_config)
        logger.info('Loading graph tensor data with {} Mean-Fill...'.format(FLAGS.fill_mean))
        dataset_f = dataset.gcnn_lstm_data_construction(sparse_removal=FLAGS.sparse_removal)
        # make the output dimension to be 6
        # output_dim = 5
        # dataset_f['data'] = dataset_f['data'][..., :output_dim]
        # dataset_f['data'] = dataset_f['data'] / np.sum(dataset_f['data'], axis=-1, keepdims=True)
        # dataset_f['data'] = np.where(np.isnan(dataset_f['data']),
        #                              np.zeros_like(dataset_f['data']), dataset_f['data'])
        # if 'HA' in dataset_f.keys():
        #     dataset_f['HA'] = dataset_f['HA'][..., :output_dim]

        adj_mx = dataset.adj_matrix
        dist_mx = dataset.dist_matrix
        logger.info('Construct model and train...')
        nodes = dataset.nodes
        print("Number of edges is ", len(nodes))
        print("Shape of adjacency matrix is ", adj_mx.shape)
        print("The Adjacent Matrix is ", adj_mx)
        supervisor_config = data_model_config['model']
        # Manually set the output_dim to be 6
        supervisor_config['output_dim'] = dataset.output_dim

        supervisor_config['start_date'] = data_config['start_date']
        # setting for training
        supervisor_config['use_cpu_only'] = FLAGS.use_cpu_only
        if FLAGS.log_dir:
            supervisor_config['log_dir'] = FLAGS.log_dir
        if FLAGS.use_curriculum_learning is not None:
            supervisor_config['use_curriculum_learning'] = FLAGS.use_curriculum_learning
        if FLAGS.loss_func:
            supervisor_config['loss_func'] = FLAGS.loss_func
        if FLAGS.filter_type:
            supervisor_config['filter_type'] = FLAGS.filter_type
        # Overwrites space with specified parameters.
        for name in ['batch_size', 'cl_decay_steps', 'epochs', 'horizon', 'learning_rate', 'l1_decay',
                     'lr_decay', 'lr_decay_epoch', 'lr_decay_interval', 'sample_rate', 'min_learning_rate',
                     'patience', 'seq_len', 'test_every_n_epochs', 'verbose', 'coarsen', 'coarsening_levels',
                     'zone', 'scaler', 'data_format', 'num_gpus', 'mode', 'fill_mean', 'activate_func',
                     'hopk', 'sigma', 'drop_out', 'cl_decay_steps', 'shuffle_training', 'optimizer',
                     'pool_type', 'trace_ratio']:
            if type(getattr(FLAGS, name)) == str or getattr(FLAGS, name) >= 0:
                supervisor_config[name] = getattr(FLAGS, name)

        tf_config = tf.ConfigProto(allow_soft_placement=True)
        if FLAGS.use_cpu_only:
            tf_config = tf.ConfigProto(device_count={'GPU': 0})
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            supervisor = MGRNNSupervisor(traffic_reading_df=dataset_f, adj_mx=adj_mx,
                                         config=supervisor_config,
                                         origin_df_file=dataset.origin_df_file,
                                         nodes=nodes,
                                         coarsed_dict=dataset.coarsed_dict)
            if not FLAGS.is_restore:
                supervisor.train(sess=sess)
            else:
                supervisor.test_and_write_results(
                    sess=sess, model_filename=FLAGS.model_filename,
                    model_dir=FLAGS.model_dir, dist_mx=dist_mx)


if __name__ == '__main__':
    tf.app.run()
