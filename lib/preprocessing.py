import h5py
import networkx as nx
from .utils import *
from lib import coarsening
from .graph import get_proximity_matrix
import distutils.dir_util
import pickle
from datetime import datetime

class DataSet(object):

    def __init__(self, base_dir, server_name, conf_dir, random_node,
                 cat_head, con_head, start_date, end_date, data_format='duration',
                 duration_log=True,
                 hopk=500, sigma=25, sample_rate=5, windos_size=4, predict_size=4,
                 small_threshold=0.0, big_threshold=40.0, mode='avg',
                 directed=True, min_nb=5, unit=1.0, coarsen=True, coarsening_levels=4):
        self.base_dir = base_dir
        self.random_node = random_node
        self.hopk = hopk
        self.sigma = sigma
        self.mode = mode
        self.data_format = data_format
        self.duration_log = duration_log

        self.directed = directed
        self.min_nb = min_nb
        self._min_duration = 60
        self._max_duration = 10000
        self.unit = unit
        self.engine = connect_sql_server(server_name, conf_dir)
        # Get a road graph with the max number of nodes
        self._edges_adj, self._graph_edges, self._dist_matrix = self.construct_region_graph()
        self._coarsed_dict = self.coarse_graph(coarsen, coarsening_levels)
        self.cat_head = cat_head
        self.con_head = con_head
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.sample_rate = sample_rate
        self.window_size = windos_size
        self.predict_size = predict_size
        self.small_threshold = small_threshold
        self.big_threshold = big_threshold
        # self.scaler = MinMaxScaler(self.small_threshold, self.big_threshold)

    @property
    def coarsed_dict(self):
        return self._coarsed_dict

    @property
    def adj_matrix(self):
        return self._edges_adj

    @property
    def dist_matrix(self):
        return self._dist_matrix

    @property
    def output_dim(self):
        if self.mode == 'hist':
            return len(self._hist_range) - 1
        else:
            return 1

    @property
    def nodes(self):
        return self._graph_edges

    def coarse_graph(self, coarsened=False, coarsening_levels=0):
        if coarsened:
            coarsed_dict_file = os.path.join(
                self.base_dir, 'coarsed_sigma{}_hopk{}_{}.pickle'.format(
                    self.sigma, self.hopk, coarsening_levels))
            if os.path.isfile(coarsed_dict_file):
                with open(coarsed_dict_file, 'rb') as f:
                    coarsed_dict = pickle.load(f)
            else:
                graphs, perm = coarsening.coarsen(
                    self._edges_adj, levels=coarsening_levels,
                    self_connections=True)
                coarsed_dict = {'graphs': graphs,
                                'perm': perm}
                with open(coarsed_dict_file, 'wb') as f:
                    pickle.dump(coarsed_dict, f)
            for i, A in enumerate(coarsed_dict['graphs']):
                Mnew, Mnew = A.shape
                print('Layer {0}: M_{0} = |V| = {1} nodes'.format(i, Mnew))
        else:
            perm = None
            graphs = [self._edges_adj] * coarsening_levels
            coarsed_dict = {'graphs': graphs,
                            'perm': perm}

        return coarsed_dict


    def _qurey_necessary_rows(self, st_time_stamp, end_time_stamp):
        raise NotImplementedError

    def _construct_od_matrix_list(self, path, df_od):
        raise NotImplementedError

    def _get_groupby_df(self, df_od):
        raise NotImplementedError

    def _construct_o_matrix_list(self, df_od_gb, oid):
        """
        Construct the Origin-Time matrix with list of elements

        :param df_od_gb: group-by dataframe, with key (O_id, D_id)
        :param oid: id, Origin id
        :return: Origin-Time dataframe
        """

        sample_str = '{}T'.format(self.sample_rate)
        list_dfs = []
        for did in self._graph_edges:
            od_key = (oid, did)
            od_pair = 'O{0}_D{1}'.format(oid, did)
            od_vel = pd.DataFrame()
            if od_key in df_od_gb.groups:
                df_od_group = df_od_gb.get_group(od_key)
                if self.data_format == 'speed':
                    od_vel[od_pair] = df_od_group.speed.resample(
                        sample_str).apply(vel_list)
                else:
                    od_vel[od_pair] = df_od_group.time_duration.resample(
                        sample_str).apply(vel_list)
            else:
                od_vel[od_pair] = None
            list_dfs.append(od_vel)
        df_od_tb = pd.concat(list_dfs, axis=1, join='outer')

        return df_od_tb

    def gcnn_lstm_data_construction(self, fill_mean=False, sparse_removal=False):
        """
        Construct the 4D-tensor for NN

        :return: hdf5 format data set
        """

        df_vel_path = os.path.join(self.base_dir, 'S{}'.format(self.sample_rate),
                                   self.data_format, self.mode)
        if self.mode == 'hist':
            dataset_file = os.path.join(df_vel_path,
                                        'tensor_dataset_{}_{}_{}.mat'.format(
                                            self._hist_range[0],
                                            self._hist_range[-1],
                                            self._hist_range[-1] - self._hist_range[-2]))
        else:
            dataset_file = os.path.join(df_vel_path, 'tensor_dataset_{}.mat'.format(self.unit))

        directory = os.path.dirname(dataset_file)
        distutils.dir_util.mkpath(directory)
        st_time_stamp_str = self.start_date.strftime("%Y-%m-%d")
        end_time_stamp_str = self.end_date.strftime("%Y-%m-%d")
        df_vel_path = os.path.join(self.base_dir, '..', '{0}_{1}.csv'.format(
            st_time_stamp_str, end_time_stamp_str))
        self._origin_df_file = df_vel_path
        try:
            dataset_f = h5py.File(dataset_file, 'r')
        except OSError:
            df_od = self._qurey_necessary_rows(self.start_date, self.end_date)
            self._construct_OD_time_dataset(df_od, dataset_file)
            dataset_f = h5py.File(dataset_file, 'r')
        data_dict = {}
        for key in list(dataset_f.keys()):
            data_dict[key] = dataset_f[key].value
        dataset_f.close()

        if sparse_removal:
            print("In sparse removal...")
            data_dict = self.sparse_remove(data_dict)

        if fill_mean:
            data_dict = self.fill_mean_avg(data_dict)

        return data_dict

    def sparse_remove(self, data_dict):
        return self._sparse_remove(data_dict)

    def _sparse_remove(self, data_dict):
        raise NotImplementedError

    def fill_mean_avg(self, data_dict):
        data = data_dict['data']
        num = data_dict['num']
        weight = data_dict['weight']
        num[weight == False] = 0
        data_dict['num'] = num
        shape_data = data.shape

        num = np.expand_dims(num, axis=-1)
        num = np.tile(num, [1]*(len(shape_data) - 1) + [shape_data[-1]])
        weighted_data = data * num
        sum_data = np.sum(weighted_data, axis=0)
        sum_num = np.sum(num, axis=0)
        if self.mode == 'avg':
            avg_data = sum_data / sum_num
        else:
            avg_data = sum_data / np.sum(sum_data, axis=-1, keepdims=True)
        # make sure there's no nan in the data
        avg_data = np.where(np.isnan(avg_data), np.zeros_like(avg_data), avg_data)
        avg_data = np.expand_dims(avg_data, 0)
        avg_data = np.tile(avg_data, [shape_data[0]] + [1]*(len(shape_data) - 1))
        data[weight == False, :] = avg_data[weight == False, :]
        data_dict['data'] = data
        data_dict['HA'] = avg_data

        # assert np.sum(num > 0) == np.sum(weight)

        return data_dict.copy()

    def _remove_unreasonable_rows(self, df_od):
        """
        Clean the data further by removing some unreasonable rows

        :param df_od:
        :return:
        """
        # the trip should be larger than 60 seconds and less than max_duration(10000), otherwise we regard it as a outlier
        df_od = df_od[(df_od.time_duration >= self._min_duration) & (df_od.time_duration <= self._max_duration)]
        # the trip_distance should be larger than 0.0
        df_od = df_od[df_od.trip_distance > 0.0]
        # the speed should be larger than 0.5m/s
        df_od = df_od[df_od.speed > 0.0005]

        return df_od

    def _construct_OD_time_dataset(self, df_of, dataset_file):
        """
        Construct the 4D-tensor for NN

        :param df_of: pandas dataframe
        :param dataset_file: hdf5 file name
        :return: hdf5 format data set
        """
        if self.mode == 'hist':
            assert self._hist_range is not None
            nb_bins = len(self._hist_range) - 1
            # convert the speed to m/s
            df_of['speed'] = df_of['trip_distance'] * self._dist_unit / df_of['time_duration']
        elif self.mode == 'avg':
            nb_bins = 1
            df_of = self._remove_unreasonable_rows(df_of)
        else:
            raise Exception("[!] Unkown method: {}".format(self.mode))

        print("ODT Tensor Constructing...")
        df_od_gb = self._get_groupby_df(df_of)
        print("Number of groups theoretically: ", len(self.nodes)**2)
        print("Number of groups actually: ", len(df_od_gb.groups.keys()))

        st_time_stamp_str = self.start_date.strftime("%Y-%m-%d")
        end_time_stamp_str = self.end_date.strftime("%Y-%m-%d")

        datetime_index = pd.date_range(
            st_time_stamp_str, end_time_stamp_str,
            freq='{}T'.format(self.sample_rate))
        odt_tensor = []
        odt_tensor_mape = []
        odt_tensor_num = []
        for oid in self._graph_edges:
            print("O: ", oid)
            df_od_tb = self._construct_o_matrix_list(df_od_gb, oid)
            df_od_tb = df_od_tb.reindex(datetime_index)
            # convert the dataframe with list into array
            df_od_tb_mape = df_od_tb.apply(
                my_rolling_apply_avg, axis=0,
                args=(self.get_vel_mape_rolling, 1))
            df_od_tb_num = df_od_tb.apply(
                my_rolling_apply_avg, axis=0,
                args=(self.get_vel_num_rolling, 1))

            if self.mode == 'avg':
                df_od_tb = df_od_tb.apply(
                    my_rolling_apply_avg, axis=0,
                    args=(self.get_vel_avg_rolling, 1))
            elif self.mode == 'hist':
                df_od_tb = df_od_tb.apply(
                    my_rolling_apply_list, axis=0,
                    args=(self.get_vel_hist_rolling, 1,
                          self._hist_range))
            else:
                print("Please specify a mode: avg or hist")
                raise NotImplementedError

            o_time_matrix = df_od_tb.values
            # print("O_time matrixx shape is ", o_time_matrix.shape)
            # Check here and notice the order of O and D
            # the axis for expand_dims should be 1 such that OD matrix is (O, D) !!!
            o_time_matrix = convert_multi_channel_array(o_time_matrix, nb_bins)
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
        time_inter = (delta_days * (60*24 / self.sample_rate) +
                     datetime_index.hour * (60 / self.sample_rate) +
                     datetime_index.minute / self.sample_rate).astype(int)
        dayofweek = datetime_index.dayofweek.values

        dataset_f = h5py.File(dataset_file, 'w')
        dataset_f.create_dataset('data', data=odt_tensor)
        dataset_f.create_dataset('weight', data=non_zeros_pos)
        dataset_f.create_dataset('TI', data=time_inter)
        dataset_f.create_dataset('DoW', data=dayofweek)
        dataset_f.create_dataset('mape', data=odt_tensor_mape)
        dataset_f.create_dataset('num', data=odt_tensor_num)
        dataset_f.close()

        return dataset_f

    def _construct_overlap_avg_link_tt(self):
        return NotImplementedError

    def _get_edge_connection(self):
        raise NotImplementedError

    def _get_graph_edges(self):
        return NotImplementedError

    def _get_adj_matrix(self, zone_ids):
        return NotImplementedError

    def get_effective_edges(self, df_vel_path, least_threshold=0.5):
        """
        Construct sequence data from file

        :param df_vel_path: directory store the dataframe data
        :param val_start_date: validate date start date
        :param val_end_date: validate data end date
        :param least: bool, whether to keep all the training data the same with different data_rms
        :param least_threshold: float, if least is True, what's the least percentage
        :return: training data and validate data.
        """

        print("Get effective edges...")
        print("The number of graph edges previously is ", len(self._graph_edges))
        useful_edges = []
        min_max_dests = int(len(self._graph_edges) * least_threshold)
        for i, oid in enumerate(self._graph_edges):
            print("O: ", oid)
            df_vel_file = os.path.join(
                df_vel_path, 'df_vel_O{}.pickle'.format(oid))
            with open(df_vel_file, 'rb') as f:
                df_vel = pkl.load(f)
            included = self._get_effective_dests(
                df_vel, min_max_dests)
            if included:
                useful_edges.append(oid)

        assert len(useful_edges) > 0
        self._graph_edges = np.array(useful_edges)
        print("The number of satisfied edges is ", len(self._graph_edges))

        return useful_edges

    def _get_effective_dests(self, df_all, least_threshold=80):
        """
        Convert the Dataframe of single origin to sequence data format.

        :param df_all: pd.Dataframe, average speed from single origin to all destinations.
        :param least_threshold: int, the max number of destinations that can meet currently.
        :return: bool, whether include this origin or not.
        """

        df_all.loc[:, df_all.columns != 'time'] = \
            df_all.loc[:, df_all.columns != 'time'].replace(
                [np.inf, -np.inf], np.nan)

        data_array = df_all.loc[:, df_all.columns != 'time'].values
        print("The shape of data array is ", data_array.shape)
        row_notnull = pd.notnull(data_array)

        max_dests = np.max(row_notnull.sum(axis=1))

        if max_dests >= least_threshold:
            return True
        else:
            return False

    def construct_region_graph(self):
        """
        Construct a edge-noded graph

        :return: csr matrix, list: Adjacency Matrix, list of nodes
        """

        edge_adj_file = os.path.join(self.base_dir,
                                     'edge_adj_hop{}_sigma_{}.pickle'.format(
                                         self.hopk, self.sigma))
        edges_file = os.path.join(self.base_dir, 'edges_hop{}_sigma_{}.pickle'.format(
            self.hopk, self.sigma))

        edge_dist_file = os.path.join(self.base_dir, 'edge_dist.pickle')

        if os.path.exists(edge_adj_file) and os.path.exists(edges_file) \
                and os.path.exists(edge_dist_file):
            print("Reading road graph from existing file...")
            print('{}\n{}'.format(edge_adj_file, edges_file))
            with open(edge_adj_file, 'rb') as f:
                edges_adj = pkl.load(f)
            with open(edges_file, 'rb') as f:
                edges = pkl.load(f)
            with open(edge_dist_file, 'rb') as f:
                dist_matrix = pkl.load(f)
        else:
            print("Querying the DB...")
            edges = self._get_graph_edges()
            adj_matrix, dist_matrix = self._get_adj_matrix(edges)
            edges_adj = get_proximity_matrix(
                adj_matrix, dist_matrix, self.hopk, self.sigma)
            print("adj matrix is ", edges_adj)
            directory = os.path.dirname(edge_adj_file)
            if not os.path.exists(directory):
                os.makedirs(directory)

            with open(edge_adj_file, 'wb') as f:
                pkl.dump(edges_adj, f)
            with open(edges_file, 'wb') as f:
                pkl.dump(edges, f)
            with open(edge_dist_file, 'wb') as f:
                pkl.dump(dist_matrix, f)

        return edges_adj, edges, dist_matrix

    def _construct_od_log_duration_list(self, path):
        return NotImplementedError

    def _construct_od_speed_list(self, path):
        return NotImplementedError

    def construct_max_min_scaler(self, path, df_vel):
        """
        Construct the MaxMinScaler

        :param path: the folder directory that needs to store the re-sampled velocity
        :return:
        """

        max_min_threshold_file = os.path.join(path, 'max_min_threshold.pickle')

        if os.path.exists(max_min_threshold_file):
            with open(max_min_threshold_file, 'rb') as f:
                max_min_threshold = pkl.load(f)
            self.big_threshold = max_min_threshold[0]
            self.small_threshold = max_min_threshold[-1]

            return True

        biggest_speed = self.big_threshold
        smallest_speed = self.small_threshold

        df_all_array = df_vel.loc[:, self._graph_edges].values
        tmp_big = np.nanmax(df_all_array)
        tmp_small = np.nanmin(df_all_array)
        if biggest_speed < tmp_big:
            biggest_speed = tmp_big
        if smallest_speed > tmp_small:
            smallest_speed = tmp_small

        # To make sure there's no too big values this is unrealistic
        if self.big_threshold > biggest_speed:
            self.big_threshold = biggest_speed
        if self.small_threshold < smallest_speed:
            self.small_threshold = smallest_speed

        max_min_threshold = [self.big_threshold, self.small_threshold]

        with open(max_min_threshold_file, 'wb') as f:
            pkl.dump(max_min_threshold, f)

        return True

    def get_max_min_scaler(self, path):

        max_min_threshold_file = os.path.join(path, 'max_min_threshold.pickle')

        with open(max_min_threshold_file, 'rb') as f:
            max_min_threshold = pkl.load(f)

        self.scaler.fit(np.array(max_min_threshold).reshape(-1, 1))

        return self.scaler

    def get_effective_inters(self, df_all, least=True, least_threshold=0.5):
        """
        Convert the dataframe of single origin to sequence data format.

        :param df_all: pd.dataframe, average speed from single origin to all destinations.
        :param least: bool, whether to keep all the data or not.
        :param least_threshold: float, the threshold of the minimum percentage needed.
        :return: list, "inter"s that meet the requirement of least threshold
        """

        nb_rows = df_all.shape[0]
        # select the destinations that meet the requirement
        columns_selected = self._graph_edges
        columns_selected.append('time')
        df_all = df_all.loc[:, columns_selected]
        df_all.loc[:, df_all.columns != 'time'] = \
            df_all.loc[:, df_all.columns != 'time'].replace(
                [np.inf, -np.inf], np.nan)

        data_array = df_all.loc[:, df_all.columns != 'time'].values
        print("The shape of data array is ", data_array.shape)
        row_notnull = pd.notnull(data_array)
        row_keep = np.array([True] * nb_rows)
        tmp_row_notnull = row_notnull.copy()
        df_all['inter'] = ((df_all['time'].dt.date -
                            self.start_date.date()).dt.days * (60 / self.sample_rate) * 24 +
                           df_all['time'].dt.hour * (60 / self.sample_rate) +
                           df_all['time'].dt.minute / self.sample_rate).astype(int)

        # if least if true, a maximum least threshold should be met.
        if least:
            num_needed = int(len(self._graph_edges) * (1 - least_threshold))
        else:
            num_needed = len(self._graph_edges) - self.data_rm[0]
        # Since df_all contains one column of 'time', we need num_needed "+ 1"
        # here.
        curr_row_keep = tmp_row_notnull.sum(axis=1) >= num_needed
        print("The maximum number of records is ",
              np.max(tmp_row_notnull.sum(axis=1)))
        row_keep = np.logical_and(row_keep, curr_row_keep)
        df_all = df_all[row_keep]
        df_all = df_all.sort_values('inter', ascending=True)
        start_inters = []
        for i in range(df_all.shape[0] - self.window_size - self.predict_size):
            # TODO: the preprocessing size should be modofied accordingly~
            start_ind = int(i)
            end_ind = int(i + self.window_size + self.predict_size)
            inters = df_all.iloc[start_ind: end_ind]['inter'].tolist()
            # print(inters)
            start_inter = inters[0]
            end_inter = inters[-1]
            if (end_inter - start_inter + 1) == (self.window_size + self.predict_size):
                start_inters.append(start_inter)

        return start_inters

    def tailor_data_needed(self, row_array):

        row_notnull = pd.notnull(row_array)
        for row_i in range(row_notnull.shape[0]):
            not_selected_bool = np.array([True] * row_array.shape[1])
            notnull_idx_i = np.where(row_notnull[row_i, :])[0]
            rand_choice_idx = np.random.choice(
                notnull_idx_i, self.data_rm[0], replace=False)
            not_selected_bool[rand_choice_idx] = False
            row_array[row_i, not_selected_bool] = 0

        return row_array

    def tailor_predicted_val_weights(self, row_array):

        row_notnull = pd.notnull(row_array)
        y_weight = np.zeros(row_notnull.shape, dtype=np.int)
        y_weight[row_notnull] = 1
        row_array[row_notnull == False] = 0.0

        return y_weight, row_array

    def scaler_transform(self, data):
        # flatten the numpy array when do scaler fitting
        data_shape = data.shape
        reshape_row_selected = data.flatten()
        # make the data fall into range(0, 1)
        data = self.scaler.transform(reshape_row_selected.reshape(-1, 1))
        data = data.reshape(data_shape)

        return data

    def construct_predict_data_structure(self, df_vel, inter_i):

        df_vel = df_vel.set_index('time')
        df_vel = df_vel.replace([np.inf, -np.inf], np.nan)

        dt_series = pd.Series([pd.Timedelta(minutes=i * self.sample_rate)
                               for i in range(
            inter_i, inter_i + self.window_size + self.predict_size)])
        dt_series += self.start_date
        row_selected = df_vel.loc[dt_series].values
        x_data = self.tailor_data_needed(row_selected[:self.window_size, :])
        y_weight, y_data = self.tailor_predicted_val_weights(
            row_selected[self.window_size:, :])

        x_data = self.scaler_transform(x_data)
        y_data = self.scaler_transform(y_data)

        return x_data, y_data, y_weight

    def get_vel_hist_rolling(self, pdSeries_like, hist_bin):
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
        data_keep = (tt_array < self.big_threshold) & (
            tt_array >= self.small_threshold)
        tt_array = tt_array[data_keep]

        if len(tt_array) < self.min_nb:
            # print("length smaller than {}...".format(self.min_nb))
            return np.zeros(len(hist_bin)-1)

        hist, bin_edges = np.histogram(tt_array, hist_bin, density=True)

        if np.isnan(hist).any():
            return np.zeros(len(hist_bin)-1)

        hist *= hist_bin[1] - hist_bin[0]

        return hist

    def get_vel_avg_rolling(self, pdSeries_like):
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
        if len(data_list) < self.min_nb:
            return np.nan

        data_array = np.array(data_list)
        data_array = (data_array) * self.unit
        # data_keep = (data_array < 40) & (
        #     data_array >= self.small_threshold)
        # data_array = data_array[data_keep]

        return np.nanmean(data_array)

    def get_vel_num_rolling(self, pdSeries_like):
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

    def get_vel_mape_rolling(self, pdSeries_like):
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
        if len(data_list) < self.min_nb:
            return np.nan

        data_array = np.array(data_list)
        mape = np.abs(data_array - data_array.mean()) / data_array
        # data_keep = (data_array < 40) & (
        #     data_array >= self.small_threshold)
        # data_array = data_array[data_keep]

        return np.nanmean(mape)

    def get_proximity_matrix(self, A):
        """
        Get a proximity matrix from adjacency matrix

        :param A: 2d dimension of array
        :return: 2d numpy array
        """
        assert A.ndim == 2
        W = np.zeros(A.shape, dtype=np.float32)
        m = np.sum(A)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                W[i, j] = np.sum(A[i, :]) * np.sum(A[:, j]) / (2 * m)
        return W

    def get_hop_proximity_matrix(self, G, hop=2):
        """
        Get a proximity matrix from adjacency matrix

        :param A: 2d dimension of array
        :return: 2d numpy array
        """
        nodes = G.nodes()
        W = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
        for node_i in nodes:
            length = nx.single_source_shortest_path_length(G, node_i, hop)
            for node_j, length_j in length.items():
                if length_j > 0:
                    W[node_i, node_j] = 1
        W = np.maximum(W, W.transpose())
        return W


class NYCData(DataSet):

    def __init__(self, base_dir, server_name, conf_dir, random_node,
                 cat_head, con_head, start_date, end_date, data_format='duration',
                 duration_log=True,
                 hopk=2, sigma=25, sample_rate=15, window_size=20, predict_size=4,
                 small_threshold=0.0, big_threshold=50.0, mode='avg', borough='Manhattan',
                 directed=True, min_nb=5, unit=1609.34, zone='taxi_zone', coarsen=True,
                 coarsening_levels=4):
        self._name = 'nyc'
        self._zone = zone
        self._borough = borough
        self._hist_range = np.arange(0, 22, 3)
        self._dist_unit = 1609.34
        super().__init__(base_dir, server_name, conf_dir, random_node,
                         cat_head, con_head, start_date, end_date, data_format,
                         duration_log, hopk, sigma,
                         sample_rate, window_size, predict_size, small_threshold,
                         big_threshold, mode, directed, min_nb, unit, coarsen,
                         coarsening_levels)

    def _sparse_remove(self, data_dict):
        data = data_dict['data']
        data_sum_row = np.sum(data, axis=-1)
        data_sum_all = np.sum(data_sum_row, axis=0)
        # make sure that each row and col have a minimum of
        # 20 records within whole dataset
        data_sum_threshold = data_sum_all > 100

        row_ind = np.sum(data_sum_threshold, axis=0) > 20
        col_ind = np.sum(data_sum_threshold, axis=1) > 20
        keep_ind = row_ind & col_ind
        data_dict['data'] = data_dict['data'][:, keep_ind, :, :]
        data_dict['data'] = data_dict['data'][:, :, keep_ind, :]
        data_dict['mape'] = data_dict['mape'][:, keep_ind, :]
        data_dict['num'] = data_dict['num'][:, keep_ind, :]
        data_dict['weight'] = data_dict['weight'][:, keep_ind, :]
        data_dict['mape'] = data_dict['mape'][:, :, keep_ind]
        data_dict['num'] = data_dict['num'][:, :, keep_ind]
        data_dict['weight'] = data_dict['weight'][:, :, keep_ind]
        self._edges_adj = self._edges_adj[keep_ind, :]
        self._edges_adj = self._edges_adj[:, keep_ind]
        self._graph_edges = self._graph_edges[keep_ind]

        return data_dict

    def _get_adj_matrix(self, zone_ids):
        """
        Construct the adjacency matrix with the given edge id

        :param zone_ids: list, edge ids
        :return: adjacency matrix and distance matrix
        """

        zone_ids = np.unique(zone_ids)
        zone_inds = np.array(zone_ids) - 1

        if self._zone == 'taxi_zone':
            conn = self.engine.connect()
            sql_adj = "select * from {}_adj".format(self._zone)
            adj_matrix_df = pd.read_sql(sql_adj, conn)
            print("The shape of adj_matrix is ", adj_matrix_df.shape)
            sql_dist = "select * from {}_dist".format(self._zone)
            dist_matrix_df = pd.read_sql(sql_dist, conn)
            print("The shape of dist_matrix_df is ", dist_matrix_df.shape)
            conn.close()
        else:
            adj_matrix_df = pd.read_csv(os.path.join(
                self.base_dir, 'nyct2010_adj.csv'), index_col=0)
            print("The shape of adj_matrix is ", adj_matrix_df.shape)
            dist_matrix_df = pd.read_csv(os.path.join(
                self.base_dir, 'nyct2010_dist.csv'), index_col=0)
            print("The shape of dist_matrix_df is ", dist_matrix_df.shape)


        # make the dataframe into numpy.array
        adj_matrix = adj_matrix_df.values
        dist_matrix = dist_matrix_df.values
        # select the corresponding rows and cols
        adj_matrix = adj_matrix[zone_inds, :]
        adj_matrix = adj_matrix[:, zone_inds]
        dist_matrix = dist_matrix[zone_inds, :]
        dist_matrix = dist_matrix[:, zone_inds]
        adj_shape = adj_matrix.shape
        dist_shape = dist_matrix.shape
        assert adj_shape[0] == adj_shape[1]
        assert adj_shape[0] == len(zone_ids)
        assert dist_shape[0] == dist_shape[1]
        assert dist_shape[0] == len(zone_ids)

        print("The shape of adj_matrix is ", adj_matrix.shape)

        return adj_matrix, dist_matrix

    def _get_graph_edges(self):
        conn = self.engine.connect()
        if self._zone == 'taxi_zone':
            sql_get_locationid = "select locationid " \
                                 "from {}s " \
                                 "where borough = '{}' " \
                                 "order by locationid".format(
                self._zone, self._borough)
            locations = pd.read_sql(sql_get_locationid, conn)
            zone_ids = locations['locationid'].tolist()
        else:
            sql_get_locationid = "select gid " \
                                 "from {} " \
                                 "where boroname = '{}' " \
                                 "order by gid".format(
                self._zone, self._borough)
            locations = pd.read_sql(sql_get_locationid, conn)
            zone_ids = locations['gid'].tolist()

        conn.close()
        zone_ids = np.unique(zone_ids)

        return zone_ids

    def _get_groupby_df(self, df_od):
        if self._zone == 'taxi_zone':
            pick_id = 'pickup_location_id'
            drop_id = 'dropoff_location_id'
        else:
            pick_id = 'pickup_nyct2010_gid'
            drop_id = 'dropoff_nyct2010_gid'

        df_od_gb = df_od.groupby([pick_id, drop_id])

        return df_od_gb

    @property
    def origin_df_file(self):
        # get the origin record file
        # df_od = pd.read_csv(df_vel_path, parse_dates=['time'])
        return self._origin_df_file

    def _qurey_necessary_rows(self, st_time_stamp, end_time_stamp):
        """
        Query data from database within a time range

        :param st_time_stamp:
        :param end_time_stamp:
        :return:
        """

        if self._zone == 'taxi_zone':
            pick_id = 'pickup_location_id'
            drop_id = 'dropoff_location_id'
        else:
            pick_id = 'pickup_nyct2010_gid'
            drop_id = 'dropoff_nyct2010_gid'

        st_time_stamp_str = st_time_stamp.strftime("%Y-%m-%d")
        end_time_stamp_str = end_time_stamp.strftime("%Y-%m-%d")
        df_vel_path = os.path.join(self.base_dir, '..', '{0}_{1}.csv'.format(
            st_time_stamp_str, end_time_stamp_str))
        if os.path.exists(df_vel_path):
            print("Read Datafrom exising file...")
            df_od = pd.read_csv(df_vel_path, parse_dates=['time'])
        else:
            print("Query Data from Database...")
            # filter data with start and end date
            od_sql = "select pickup_datetime, dropoff_datetime, " \
                     "pickup_longitude, pickup_latitude, " \
                     "dropoff_longitude, dropoff_latitude," \
                     "pickup_location_id, dropoff_location_id, " \
                     "pickup_nyct2010_gid, dropoff_nyct2010_gid, " \
                     "trip_distance, fare_amount from trips where " \
                     "pickup_datetime > '{0}' and dropoff_datetime < '{1}' and " \
                     "dropoff_datetime > '{0}' and pickup_datetime < '{1}' and trip_distance > 0" \
                .format(st_time_stamp_str, end_time_stamp_str)
            conn = self.engine.connect()
            df_od = pd.read_sql(od_sql, conn, parse_dates=[
                'pickup_datetime', 'dropoff_datetime'])
            conn.close()

            df_od = df_od[df_od[pick_id].isin(self._graph_edges)]
            df_od = df_od[df_od[drop_id].isin(self._graph_edges)]
            df_od['time_duration'] = (df_od['dropoff_datetime'] -
                                      df_od['pickup_datetime']).astype('timedelta64[s]')
            # calculate the manhattan distance by giving O_ll and D_ll
            df_od['manhat_dist'] = manhattan_distance_pd(df_od['pickup_latitude'].values,
                                                         df_od['pickup_longitude'].values,
                                                         df_od['dropoff_latitude'].values,
                                                         df_od['dropoff_longitude'].values)

            df_od['time'] = df_od['pickup_datetime']
            # time duration <0 due to the time saving mechanism add hour or minus one hour.

            df_od.time_duration[df_od.time_duration<0] += 3600
            df_od_wrong = df_od[df_od.time_duration < 0]
            print("Constructed DF shape: ", df_od.shape)
            print("Some wrong records...")
            print(df_od_wrong.head())
            df_od = df_od.dropna()
            df_od = df_od[(df_od.manhat_dist>0)]
            df_od = df_od[(df_od.time_duration >= self._min_duration) &
                          (df_od.time_duration <= self._max_duration)]
            # the trip_distance should be larger than 0.0
            df_od = df_od[df_od.trip_distance > 0.0]
            df_od['manhat_trip_ratio'] = df_od['manhat_dist'] / df_od['trip_distance']
            r_std = df_od['manhat_trip_ratio'].std()
            r_mean = df_od['manhat_trip_ratio'].mean()
            df_od['speed'] = df_od['trip_distance'] / df_od['time_duration']
            df_od = df_od[(df_od.manhat_trip_ratio > (r_mean - r_std)) &
                          (df_od.manhat_trip_ratio < (r_mean + r_std)) &
                          (df_od.speed < 0.03) & (df_od.speed > 0.0052-0.00253)]
            print("Processed DF shape: ", df_od.shape)

            df_od.to_csv(df_vel_path)

        df_od = df_od.set_index('time')
        print("Dataframe load successfully, with shape ", df_od.shape)

        return df_od

class CDData(DataSet):

    def __init__(self, base_dir, server_name, conf_dir, random_node,
                 cat_head, con_head, start_date, end_date, data_format='duration',
                 duration_log=True,
                 hopk=2, sigma=25, sample_rate=15, window_size=20, predict_size=4,
                 small_threshold=0.0, big_threshold=18.0, mode='avg', borough='second_ring',
                 directed=True, min_nb=5, unit=1.00, zone='lon10_lat9', coarsen=True,
                 coarsening_levels=4):
        self._name = 'chengdu'
        self._zone = zone
        self._borough = borough
        self._dist_unit = 1.
        self._hist_range = np.linspace(small_threshold, big_threshold, 8)
        print("The hist range is ", self._hist_range)
        super().__init__(base_dir, server_name, conf_dir, random_node,
                         cat_head, con_head, start_date, end_date, data_format,
                         duration_log, hopk, sigma,
                         sample_rate, window_size, predict_size, small_threshold,
                         big_threshold, mode, directed, min_nb, unit, coarsen,
                         coarsening_levels)

    def _sparse_remove(self, data_dict):
        data = data_dict['data']
        data_sum_row = np.sum(data, axis=-1)
        data_sum_all = np.sum(data_sum_row, axis=0)
        # make sure that each row and col have a minimum of
        # 20 records within whole dataset
        data_sum_threshold = data_sum_all > 40

        row_ind = np.sum(data_sum_threshold, axis=0) > 5
        col_ind = np.sum(data_sum_threshold, axis=1) > 5
        keep_ind = row_ind & col_ind
        data_dict['data'] = data_dict['data'][:, keep_ind, :, :]
        data_dict['data'] = data_dict['data'][:, :, keep_ind, :]
        data_dict['mape'] = data_dict['mape'][:, keep_ind, :]
        data_dict['num'] = data_dict['num'][:, keep_ind, :]
        data_dict['weight'] = data_dict['weight'][:, keep_ind, :]
        data_dict['mape'] = data_dict['mape'][:, :, keep_ind]
        data_dict['num'] = data_dict['num'][:, :, keep_ind]
        data_dict['weight'] = data_dict['weight'][:, :, keep_ind]
        self._edges_adj = self._edges_adj[keep_ind, :]
        self._edges_adj = self._edges_adj[:, keep_ind]
        self._graph_edges = self._graph_edges[keep_ind]

        return data_dict

    def _get_adj_matrix(self, zone_ids):
        """
        Construct the adjacency matrix with the given edge id

        :param zone_ids: list, edge ids
        :return: adjacency matrix and distance matrix
        """

        conn = self.engine.connect()
        sql_adj = "select * from second_ring_{}_adj".format(self._zone)
        adj_matrix_df = pd.read_sql(sql_adj, conn)
        print("The shape of adj_matrix is ", adj_matrix_df.shape)
        sql_dist = "select * from second_ring_{}_dist".format(self._zone)
        dist_matrix_df = pd.read_sql(sql_dist, conn)
        print("The shape of dist_matrix_df is ", dist_matrix_df.shape)
        conn.close()

        # make the dataframe into numpy.array
        adj_matrix = adj_matrix_df.values
        dist_matrix = dist_matrix_df.values
        adj_shape = adj_matrix.shape
        dist_shape = dist_matrix.shape
        assert adj_shape[0] == adj_shape[1]
        assert adj_shape[0] == len(zone_ids)
        assert dist_shape[0] == dist_shape[1]
        assert dist_shape[0] == len(zone_ids)

        return adj_matrix, dist_matrix

    def _get_graph_edges(self):
        conn = self.engine.connect()

        if self._zone == 'polygon':
            sql_get_locationid = 'select id ' \
                                 'from second_ring_polygons_geom ' \
                                 'order by id'
            locations = pd.read_sql(sql_get_locationid, conn)
            zone_ids = locations['id'].tolist()
        else:
            sql_get_locationid = 'select region_id, polygons, geom ' \
                                 'from second_ring_{}_geom ' \
                                 'order by region_id'.format(self._zone)
            locations = pd.read_sql(sql_get_locationid, conn)
            zone_ids = locations['region_id'].tolist()

        conn.close()
        zone_ids = np.unique(zone_ids)

        # # Manullay set the first ring with 500 meters accuracy
        # first_row = np.arange(85, 94, 1, dtype=np.int)
        # zone_ids = []
        # for i in range(10):
        #     zone_ids += list(first_row)
        #     first_row += 20
        # zone_ids = np.array(zone_ids)

        return zone_ids

    def _get_groupby_df(self, df_od):
        pick_id = 'pickup_sr_id'
        drop_id = 'dropoff_sr_id'

        df_od_gb = df_od.groupby([pick_id, drop_id])

        return df_od_gb

    @property
    def origin_df_file(self):
        # get the origin record file
        # df_od = pd.read_csv(df_vel_path, parse_dates=['time'])
        return self._origin_df_file

    def _qurey_necessary_rows(self, st_time_stamp, end_time_stamp):
        """
        Query data from database within a time range

        :param st_time_stamp:
        :param end_time_stamp:
        :return:
        """

        st_time_stamp_str = st_time_stamp.strftime("%Y-%m-%d")
        end_time_stamp_str = end_time_stamp.strftime("%Y-%m-%d")
        df_vel_path = os.path.join(self.base_dir, '..', '{2}_{0}_{1}.csv'.format(
            st_time_stamp_str, end_time_stamp_str, self._zone))
        if os.path.exists(df_vel_path):
            print("Read Datafrom exising file...")
            df_od = pd.read_csv(df_vel_path, parse_dates=['time'])
        else:
            print("Query Data from Database...")
            # filter data with start and end date
            # second ring coarse is from table od_trips_dist_sr_region,
            # od_trips_dist_sr_region_500 otherwise
            od_sql = "select pickup_time, dropoff_time, " \
                     "pickup_longitude, pickup_latitude, " \
                     "dropoff_longitude, dropoff_latitude," \
                     "pickup_sr_id, dropoff_sr_id, " \
                     "trip_distance, total_time from od_trips_dist_sr_region_500 " \
                     "where total_time > 0"
            if self._zone == 'polygon':
                od_sql = "select pickup_time, dropoff_time, " \
                         "pickup_longitude, pickup_latitude, " \
                         "dropoff_longitude, dropoff_latitude," \
                         "pickup_sr_id, dropoff_sr_id, " \
                         "trip_distance, total_time from od_trips_dist_sr_polygon_region " \
                         "where total_time > 0"
            conn = self.engine.connect()
            df_od = pd.read_sql(od_sql, conn, parse_dates=[
                'pickup_time', 'dropoff_time'])
            conn.close()
            df_od['manhat_dist'] = manhattan_distance_pd(df_od['pickup_latitude'].values,
                                                         df_od['pickup_longitude'].values,
                                                         df_od['dropoff_latitude'].values,
                                                         df_od['dropoff_longitude'].values)
            df_od = df_od[df_od.pickup_sr_id.isin(self._graph_edges)]
            df_od = df_od[df_od.dropoff_sr_id.isin(self._graph_edges)]
            # df_od = df_od[df_od[pick_id].isin(self._graph_edges)]
            # df_od = df_od[df_od[drop_id].isin(self._graph_edges)]
            df_od['time_duration'] = (df_od['dropoff_time'] -
                                      df_od['pickup_time']).astype('timedelta64[s]')
            # calculate the manhattan distance by giving O_ll and D_ll
            df_od = df_od[df_od.time_duration <= 3600]
            df_od = df_od[(df_od.trip_distance > 1000) & (df_od.trip_distance < 15000) ]
            df_od['time'] = df_od['pickup_time']
            # the trip_distance should be larger than 0.0
            df_od['speed'] = df_od['trip_distance'] / df_od['time_duration']
            df_od = df_od[df_od.speed >= 1.0]
            # time duration <0 due to the time saving mechanism add hour or minus one hour.
            df_od = df_od[df_od.speed <= self.big_threshold]

            # the trip_distance should be larger than 0.0
            # Follow the filtering rule as in NYC
            df_od['manhat_trip_ratio'] = df_od['manhat_dist'] / df_od['trip_distance']
            df_od['speed'] = df_od['trip_distance'] / df_od['time_duration']
            df_od = df_od[(df_od.manhat_trip_ratio < 2.5) & (df_od.speed > 2.5)]
            print("Processed DF shape: ", df_od.shape)

            df_od.to_csv(df_vel_path)
        df_od = df_od[df_od.speed >= 2.5]
        df_od['manhat_trip_ratio'] = df_od['manhat_dist'] / df_od['trip_distance']
        df_od['speed'] = df_od['trip_distance'] / df_od['time_duration']
        df_od = df_od[(df_od.manhat_trip_ratio < 2.5) & (df_od.speed > 2.5)]

        df_od = df_od.set_index('time')
        print("Dataframe load successfully, with shape ", df_od.shape)

        return df_od