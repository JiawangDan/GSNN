
import torch
import torch.utils.data
import os
import tarfile
import numpy as np
import pickle
from common_structures_seq import PointData, PointDataTest
import label_utils_seq
import feat_utils_seq
import constants
from tqdm import tqdm
import multiprocessing as mp
import point_data_utils_seq
import common_utils_seq
import random
import pandas as pd
from dyg_utils import get_neighbor_finder
import importlib

def read_tar_names(param):
    filename = param
    with tarfile.open(filename, 'r') as fin:
        return fin.getnames()
    pass


class DygDataset(torch.utils.data.Dataset):

    def get_point_data_neg(self, eid) -> PointData:
        ts = self.timestamps[eid]
        current_sample = [
            self.src_nodes[eid], self.dst_nodes[eid], self.edge_types[eid], ts]

        while True:
            choose_change_idx = np.random.choice(3)
            choices = [self.uniq_src_nodes, self.uniq_dst_nodes,
                       self.num_edge_types]
            current_sample[choose_change_idx] = np.random.choice(
                choices[choose_change_idx])

            point_data = point_data_utils_seq.get_point_data(
                current_sample, self.src_nodes, self.dst_nodes, self.edge_types,
                self.timestamps, self.config, self.triplet_index,
                self.triplet_index_bilateral, self.pair_index, self.node_index,
                self.trip_search_index, self.trip_search_index_map,
                self.pair_search_index, self.pair_search_index_map,
                self.src_node_search_index, self.src_node_search_index_map,
                self.dst_node_search_index, self.dst_node_search_index_map)

            if len(point_data.history_edges_triplet) > 0 or \
               len(point_data.history_edges_pair) > 0 or \
               len(point_data.history_edges_src) > 0 or \
               len(point_data.history_edges_dst) > 0:
                break
            pass
        return point_data

    def get_point_data_pos(self, eid) -> PointData:
        ts = self.timestamps[eid]
        current_sample = [
            self.src_nodes[eid], self.dst_nodes[eid], self.edge_types[eid], ts]

        point_data = point_data_utils_seq.get_point_data(
            current_sample, self.src_nodes, self.dst_nodes, self.edge_types,
            self.timestamps, self.config, self.triplet_index,
            self.triplet_index_bilateral, self.pair_index, self.node_index,
            self.trip_search_index, self.trip_search_index_map,
            self.pair_search_index, self.pair_search_index_map,
            self.src_node_search_index, self.src_node_search_index_map,
            self.dst_node_search_index, self.dst_node_search_index_map)
        return point_data

    def get_point_graph_data(self, idx) :
        filename_user = os.path.join(self.folder, 'graphfeature_user_{}.npy'.format(idx // 100*100+1))
        filename_oppo = os.path.join(self.folder, 'graphfeature_oppo_{}.npy'.format(idx // 100 * 100 + 1))

        user_graph = np.load(filename_user)
        oppo_graph = np.load(filename_oppo)

        return user_graph[idx%100-1, :], oppo_graph[idx%100-1, :]


    def get_start_end(self, data_folder):
        filenames = sorted(os.listdir(data_folder))

        with tarfile.open(os.path.join(data_folder, filenames[0]), 'r') as tar:
            first = int(tar.getnames()[0].split('.')[0])
        with tarfile.open(os.path.join(data_folder, filenames[-1]), 'r') as tar:
            last = int(tar.getnames()[-1].split('.')[0])
            pass

        return [first, last]

    def get_all_positive_eids(self,):

        # should not be new node
        past_nodes = set()
        is_new_node = np.zeros(len(self.src_nodes), dtype='bool')
        for eid, (snode, dnode) in enumerate(
                zip(self.src_nodes, tqdm(self.dst_nodes))):
            if snode not in past_nodes or dnode not in past_nodes:
                is_new_node[eid] = True
                pass

            past_nodes.add(snode)
            past_nodes.add(dnode)
            pass

        is_before_train_start = self.timestamps < \
                                common_utils_seq.dt2ts(self.config['train_start'])

        eids = np.arange(len(self.src_nodes), dtype='int32')
        eids = eids[np.logical_not(is_new_node | is_before_train_start)]
        return eids

    def __init__(self, config, split, valid_percent=0.1, num=1000):
        folder = config['dataset_path']
        self.folder = folder
        self.data_folder = os.path.join(folder, 'data')
        self.config = config


        self.timestamps = np.load(os.path.join(folder, 'timestamps.npy'))
        self.src_nodes = np.load(os.path.join(folder, 'src_nodes.npy'))
        self.dst_nodes = np.load(os.path.join(folder, 'dst_nodes.npy'))
        self.edge_types = np.load(os.path.join(folder, 'edge_types.npy'))
        self.triplet_index = np.load(os.path.join(folder, 'triplet_index.npy'))
        self.triplet_index_bilateral = np.load(
            os.path.join(folder, 'triplet_index_bilateral.npy'))
        self.pair_index = np.load(os.path.join(folder, 'pair_index.npy'))
        self.node_index = np.load(os.path.join(folder, 'node_index.npy'))
        self.trip_search_index = np.load(os.path.join(
            folder, 'trip_search_index.npy'))
        self.trip_search_index_map = np.load(os.path.join(
            folder, 'trip_search_index_map.npy'))
        self.pair_search_index = np.load(os.path.join(
            folder, 'pair_search_index.npy'))
        self.pair_search_index_map = np.load(os.path.join(
            folder, 'pair_search_index_map.npy'))
        self.src_node_search_index = np.load(os.path.join(
            folder, 'src_node_search_index.npy'))
        self.src_node_search_index_map = np.load(os.path.join(
            folder, 'src_node_search_index_map.npy'))
        self.dst_node_search_index = np.load(os.path.join(
            folder, 'dst_node_search_index.npy'))
        self.dst_node_search_index_map = np.load(os.path.join(
            folder, 'dst_node_search_index_map.npy'))

        with open(os.path.join(folder, 'edge_type_map.pk'), 'rb') as fin:
            self.edge_type_map = pickle.load(fin)
            self.num_edge_types = len(self.edge_type_map)
            pass

        with open(os.path.join(folder, 'node_map.pk'), 'rb') as fin:
            self.node_map = pickle.load(fin)
            self.num_nodes = len(self.node_map)
            pass

        self.node_map_reverse =  dict(zip(self.node_map.values(), self.node_map.keys()))
        self.edgetype_map_reverse = dict(zip(self.edge_type_map.values(), self.edge_type_map.keys()))

        if 'node_feat_file' in config:
            # dataset a
            self.edge_type_feat = np.load(os.path.join(
                folder, 'edge_type_feat.npy'))
            self.node_feat = np.load(os.path.join(folder, 'node_feat.npy'))
            pass
        else:
            # dataset b
            self.edge_feat = np.load(os.path.join(folder, 'edge_feat.npy'))
            edge_feat_ids = np.load(os.path.join(folder, 'edge_feat_ids.npy'))
            self.edge_feat_idmap = {eid: i for i, eid in enumerate(edge_feat_ids)}
            pass

        self.all_positive_eids = self.get_all_positive_eids()

        num_train = len(self.all_positive_eids)
        num_valid_test = int(num_train * valid_percent)
        num_valid_train = num_train - num_valid_test

        if split == 'train':
            self.positive_eids = self.all_positive_eids
            pass
        elif split == 'valid_train':
            self.positive_eids = self.all_positive_eids[:num_valid_train]
            pass
        elif split == 'valid_test':
            self.positive_eids = self.all_positive_eids[num_valid_train:]
            pass
        else:
            raise RuntimeError(f'no recognize split: {split}')

        print(len(self.positive_eids))
        self.split = split

        self.num = num

        self.uniq_src_nodes = np.unique(self.src_nodes)
        self.uniq_dst_nodes = np.unique(self.dst_nodes)


        #############graph#############
        self.folder_graph = config['graph_dataset_path']

        self.full_data_graph = pd.read_csv('{}/full_data_edge.csv'.format(self.folder_graph))
        self.edgeidx_2_feat_dict = np.load('{}/edgeidx_2_feat.npy'.format(config['graph_dataset_path'])).astype(np.float16)
        self.ngh_finder = get_neighbor_finder(self.full_data_graph, uniform=False)
        self.neighbors_num = config['neighbors']

        self.index_start_graph = np.min(np.where(self.full_data_graph.train_flag == 1))
        self.index_end_graph = np.max(np.where(self.full_data_graph.train_flag == 1))

        with open(os.path.join(self.folder_graph, 'graph_node_map.pk'), 'rb') as fin:
            self.graph_node_map = pickle.load(fin)

        pass

    def encode_history_edges(self, edges, current_ts):
        edges_ts = self.timestamps[edges[:, 0]]

        edges_feat = feat_utils_seq.time_encoding(current_ts, edges_ts)
        edges_direction = np.zeros(edges_feat.shape[0], dtype='int64')
        edges_direction = edges[:, 1]

        edges_feat = np.concatenate(
            (edges_feat, edges_direction.reshape(-1, 1)), axis=-1)
        
        return edges_feat

    def get_history_edge_feat_b(self, edges):
        feat = np.zeros(
            (len(edges), self.config['extra_feat_dim']),
            dtype='float32')

        if len(edges) > 0:
            edges = edges[:, 0]

            fids = []
            findex = []
            for i, e in enumerate(edges):
                fid = self.edge_feat_idmap.get(e)
                if fid is not None:
                    fids.append(int(fid))
                    findex.append(i)
                    pass
                pass

            fids = np.array(fids, dtype='int64')
            findex = np.array(findex, dtype='int64')

            feat[findex] = self.edge_feat[fids]
            pass
        
        return feat

    def get_edge_feat(self, src_node, dst_node, edge_type):
        if 'node_feat_file' in self.config:
            src_feat = self.node_feat[src_node] + 1 # 1 for -1
            dst_feat = self.node_feat[dst_node] + 1
            edge_type_feat = self.edge_type_feat[edge_type] + 1

            edge_feat = np.hstack((src_feat, dst_feat, edge_type_feat))
            edge_feat = feat_utils_seq.merge_category(
                edge_feat, self.config['edge_feat_dim'])
            pass
        else:
            edge_feat = np.zeros((1, 1), dtype='int64')
            pass
        
        return edge_feat

    def get_pair_feat_extra(self, current_edge_type, pair_edges):
        
        if len(pair_edges) > 0:
            pair_edge_types = self.edge_types[pair_edges[:, 0]]
            is_predict_edge_type = np.array(pair_edge_types == current_edge_type, dtype='int32')
            pair_edge_types_feat = pair_edge_types[:, None]

            pair_feat = np.concatenate(
                (pair_edge_types_feat, is_predict_edge_type[:, None]), axis=-1)
        else:
            pair_feat = np.zeros((0, 2), dtype='int64')
            pass
        
        return pair_feat
    
    def remove_edges_before(self, edges, timestamps, ts):
        if len(edges) == 0 :
            return edges
        else:
            return edges[timestamps[edges[:, 0]] < ts]
        pass

    def get_max_history_ts(self, trip_edges, pair_edges, src_edges, dst_edges, min_label_ts):
        max_history_ts = None
        if len(trip_edges) > 0:
            max_history_ts = np.max(self.timestamps[trip_edges[:, 0]])
            pass

        for edges in (pair_edges, src_edges, dst_edges):
            if len(pair_edges) > 0:# and max_history_ts is None:
                max_history_ts_edges = np.max(self.timestamps[edges[:, 0]])
                if max_history_ts is not None:
                    max_history_ts = max(
                        max_history_ts,
                        max_history_ts_edges)
                else:
                    max_history_ts = max_history_ts_edges
                    pass
                pass
        
        if max_history_ts is None:
            return min_label_ts
        else:
            return max_history_ts
        pass

    def get_node_history_feat_extra(self, history_edges, current_edge_type, other_node):
        feat = np.zeros((len(history_edges), 4), dtype='int64')

        for i, (eid, direction) in enumerate(history_edges):
            feat[i, 1] = self.edge_types[eid]
            if direction == 0:
                feat[i, 0] = self.dst_nodes[eid]
                feat[i, 3] = int(self.dst_nodes[eid] == other_node)
            else:
                feat[i, 0] = self.src_nodes[eid]
                feat[i, 3] = int(self.src_nodes[eid] == other_node)

                pass
            feat[i, 2] = int(self.edge_types[eid] == current_edge_type)
            pass

        return feat
        pass
    
    def encode_node_history_edges(
            self, history_edges, max_history_ts, current_edge_type,
            other_node):
        feat = self.encode_history_edges(
            history_edges,
            max_history_ts)
        feat_extra = self.get_node_history_feat_extra(
            history_edges, current_edge_type, other_node)
        feat = np.concatenate(
            (feat, feat_extra),
            axis=-1)
        feat = feat_utils_seq.merge_category(feat, self.config['node_history_feat_dim'])
        return feat
    
    def __getitem__(self, idx):
        eid = np.random.choice(self.positive_eids)
        flag_neg = True
        if random.random() < self.config['neg_sample_proba']:
            point_data = self.get_point_data_neg(eid)
            pass
        else:
            point_data = self.get_point_data_pos(eid)
            flag_neg = False
            pass
        
        config = self.config

        label_bins, label = label_utils_seq.get_label(
            point_data.target_ts, point_data.max_history_ts,
            point_data.max_future_ts, self.config['neg_sample_num'],
            self.config['label_bin_size'])

        edge_feat = self.get_edge_feat(
            point_data.src_node, point_data.dst_node,
            point_data.edge_type)

        history_edges_trip = point_data.history_edges_triplet
        history_edges_pair = point_data.history_edges_pair
        history_edges_src = point_data.history_edges_src
        history_edges_dst = point_data.history_edges_dst

        max_history_ts = point_data.max_history_ts
        
        trip_feat = self.encode_history_edges(
            history_edges_trip,
            max_history_ts)
        
        trip_feat = feat_utils_seq.merge_category(
            trip_feat, config['trip_feat_dim'])

        if 'node_feat_file' not in self.config:
            trip_feat_extra_b = self.get_history_edge_feat_b(
                history_edges_trip)
            pair_feat_extra_b = self.get_history_edge_feat_b(
                history_edges_pair)
            src_feat_extra_b = self.get_history_edge_feat_b(
                history_edges_src)
            dst_feat_extra_b = self.get_history_edge_feat_b(
                history_edges_dst)
            pass
        else:
            trip_feat_extra_b = np.zeros((len(history_edges_trip), 1), dtype='float32')
            pair_feat_extra_b = np.zeros((len(history_edges_pair), 1), dtype='float32')
            src_feat_extra_b = np.zeros((len(history_edges_src), 1), dtype='float32')
            dst_feat_extra_b = np.zeros((len(history_edges_dst), 1), dtype='float32')
            pass
                    
        pair_feat = self.encode_history_edges(
            history_edges_pair,
            max_history_ts)

        pair_feat_extra = self.get_pair_feat_extra(
            point_data.edge_type, history_edges_pair)

        pair_feat = np.concatenate(
            (pair_feat, pair_feat_extra),
            axis=-1)

        pair_feat = feat_utils_seq.merge_category(
            pair_feat, config['pair_feat_dim'])

        src_feat = self.encode_node_history_edges(
            history_edges_src, max_history_ts, point_data.edge_type,
            point_data.dst_node)
        dst_feat = self.encode_node_history_edges(
            history_edges_dst, max_history_ts, point_data.edge_type,
            point_data.src_node)
        
        label_feat = feat_utils_seq.get_label_feat(
            label_bins, config['label_bin_size'],
            max_history_ts,
            config['max_label_class']
        )

        labels_time = label_bins * config['label_bin_size']
        labels_time = np.reshape(labels_time, [-1, 1])

        label_feat = feat_utils_seq.merge_category(
            label_feat, config['label_feat_dim'])

        #########graph item#############

        graph_src_id = self.graph_node_map[self.node_map_reverse[point_data.src_node]]
        graph_dst_id =self.graph_node_map[self.node_map_reverse[point_data.dst_node]]
        graph_timestamp = self.full_data_graph.st_timestamps[eid]
        graph_edge_type = self.edgetype_map_reverse[point_data.edge_type] + 1

        graph_src_id = np.array([graph_src_id])
        graph_dst_id = np.array([graph_dst_id])
        graph_edge_type = np.array([graph_edge_type])
        graph_timestamp = np.array([graph_timestamp])

        src_neighbor_one_hop = self.ngh_finder.get_temporal_neighbor(graph_src_id, graph_timestamp,
                                                            n_neighbors=self.neighbors_num)
        src_one_hop_source = src_neighbor_one_hop[0].reshape([-1])
        src_one_hop_ts = src_neighbor_one_hop[2].reshape([-1])
        src_neighbor_two_hop = self.ngh_finder.get_temporal_neighbor(src_one_hop_source, src_one_hop_ts, n_neighbors=self.neighbors_num)
        src_two_hop_source = src_neighbor_two_hop[0].reshape([-1, self.neighbors_num * self.neighbors_num])
        src_two_hop_idx = src_neighbor_two_hop[1].reshape([-1, self.neighbors_num * self.neighbors_num])
        src_two_hop_ts = src_neighbor_two_hop[2].reshape([-1, self.neighbors_num * self.neighbors_num])
        src_full_hop_info = np.concatenate([eid.reshape([-1, 1]), src_neighbor_one_hop[0], src_two_hop_source, src_neighbor_one_hop[1], src_two_hop_idx
                                           , src_neighbor_one_hop[2], src_two_hop_ts], axis=1)

        dst_neighbor_one_hop = self.ngh_finder.get_temporal_neighbor(graph_dst_id, graph_timestamp,
                                                                     n_neighbors=self.neighbors_num)
        dst_one_hop_source = dst_neighbor_one_hop[0].reshape([-1])
        dst_one_hop_ts = dst_neighbor_one_hop[2].reshape([-1])
        dst_neighbor_two_hop = self.ngh_finder.get_temporal_neighbor(dst_one_hop_source, dst_one_hop_ts,
                                                                     n_neighbors=self.neighbors_num)
        dst_two_hop_source = dst_neighbor_two_hop[0].reshape([-1, self.neighbors_num * self.neighbors_num])
        dst_two_hop_idx = dst_neighbor_two_hop[1].reshape([-1, self.neighbors_num * self.neighbors_num])
        dst_two_hop_ts = dst_neighbor_two_hop[2].reshape([-1, self.neighbors_num * self.neighbors_num])
        dst_full_hop_info = np.concatenate(
            [eid.reshape([-1, 1]), dst_neighbor_one_hop[0], dst_two_hop_source, dst_neighbor_one_hop[1], dst_two_hop_idx
                , dst_neighbor_one_hop[2], dst_two_hop_ts], axis=1)

        src_full_hop_info = src_full_hop_info.astype(int)
        dst_full_hop_info = dst_full_hop_info.astype(int)
        # proprocess edgeidx_2_feat
        idx_hop1_start = 1 + self.neighbors_num + self.neighbors_num * self.neighbors_num
        idx_hop1_end = 1 + 2 * self.neighbors_num + self.neighbors_num * self.neighbors_num
        idx_hop2_end = 1 + 2 * self.neighbors_num + 2 * self.neighbors_num * self.neighbors_num

        user_neighbor_1hop_edgeidx = src_full_hop_info[:, idx_hop1_start:idx_hop1_end]
        user_neighbor_2hop_edgeidx = src_full_hop_info[:, idx_hop1_end:idx_hop2_end]
        oppo_neighbor_1hop_edgeidx = dst_full_hop_info[:, idx_hop1_start:idx_hop1_end]
        oppo_neighbor_2hop_edgeidx = dst_full_hop_info[:, idx_hop1_end:idx_hop2_end]

        user_1hop_edge_orgfeature = self.edgeidx_2_feat_dict[user_neighbor_1hop_edgeidx].astype(np.float32)
        user_2hop_edge_orgfeature = self.edgeidx_2_feat_dict[user_neighbor_2hop_edgeidx].astype(np.float32)
        oppo_1hop_edge_orgfeature = self.edgeidx_2_feat_dict[oppo_neighbor_1hop_edgeidx].astype(np.float32)
        oppo_2hop_edge_orgfeature = self.edgeidx_2_feat_dict[oppo_neighbor_2hop_edgeidx].astype(np.float32)


        return {
            'label': torch.from_numpy(label),
            'edge_feat': torch.from_numpy(edge_feat),
            'trip_feat': torch.from_numpy(trip_feat),
            'pair_feat': torch.from_numpy(pair_feat),
            'label_feat': torch.from_numpy(label_feat),
            'trip_feat_extra_b': torch.from_numpy(trip_feat_extra_b),
            'pair_feat_extra_b': torch.from_numpy(pair_feat_extra_b),
            'src_feat_extra_b': torch.from_numpy(src_feat_extra_b),
            'dst_feat_extra_b': torch.from_numpy(dst_feat_extra_b),
            'src_feat': torch.from_numpy(src_feat),
            'dst_feat': torch.from_numpy(dst_feat),
            'eid': idx,

            # graph
            'labels_time': torch.from_numpy(labels_time),
            'user_graphfeat_initial': torch.from_numpy(src_full_hop_info),
            'oppo_grapgfeat_initial': torch.from_numpy(dst_full_hop_info),
            'user_1hop_edge_orgfeature': torch.from_numpy(user_1hop_edge_orgfeature),
            'user_2hop_edge_orgfeature': torch.from_numpy(user_2hop_edge_orgfeature),
            'oppo_1hop_edge_orgfeature': torch.from_numpy(oppo_1hop_edge_orgfeature),
            'oppo_2hop_edge_orgfeature': torch.from_numpy(oppo_2hop_edge_orgfeature),
            'graph_src_id': torch.from_numpy(graph_src_id),
            'graph_dst_id': torch.from_numpy(graph_dst_id)

        }

    def __len__(self):
        # return len(self.positive_eids)
        return self.num
    pass


class DygDatasetTest(DygDataset):
    def __init__(self, config, split):
        super().__init__(config, 'train')

        folder = config['dataset_path']

        self.test_src_nodes = np.load(
            os.path.join(folder, f'{split}_src_nodes.npy'))
        self.test_dst_nodes = np.load(
            os.path.join(folder, f'{split}_dst_nodes.npy'))
        self.test_edge_types = np.load(
            os.path.join(folder, f'{split}_edge_types.npy'))
        
        self.test_start_timestamps = np.load(
            os.path.join(folder, f'{split}_start_timestamps.npy'))
        self.test_end_timestamps = np.load(
            os.path.join(folder, f'{split}_end_timestamps.npy'))

        if split == 'val':
            self.test_labels = np.load(
                os.path.join(folder, f'{split}_labels.npy'))
        else:
            self.test_labels = np.ones(len(self.test_src_nodes))

        with open(os.path.join(folder, 'edge_type_map.pk'), 'rb') as fin:
            self.edge_type_map = pickle.load(fin)
            self.num_edge_types = len(self.edge_type_map)
            pass

        with open(os.path.join(folder, 'node_map.pk'), 'rb') as fin:
            self.node_map = pickle.load(fin)
            self.num_nodes = len(self.node_map)
            pass

        self.node_map_reverse =  dict(zip(self.node_map.values(), self.node_map.keys()))
        self.edgetype_map_reverse = dict(zip(self.edge_type_map.values(), self.edge_type_map.keys()))


        #############graph#############
        self.folder_graph = config['graph_dataset_path']

        self.full_data_graph = pd.read_csv('{}/full_data_edge.csv'.format(self.folder_graph))
        self.edgeidx_2_feat_dict = np.load('{}/edgeidx_2_feat.npy'.format(config['graph_dataset_path'])).astype(
            np.float16)
        self.ngh_finder = get_neighbor_finder(self.full_data_graph, uniform=False)
        self.neighbors_num = config['neighbors']

        if split == 'val':
            self.index_start_graph = np.min(np.where(self.full_data_graph.train_flag == 2))
            self.index_end_graph = np.max(np.where(self.full_data_graph.train_flag == 2))
        elif split == 'test':
            self.index_start_graph = np.min(np.where(self.full_data_graph.train_flag == 3))
            self.index_end_graph = np.max(np.where(self.full_data_graph.train_flag == 3))

        with open(os.path.join(self.folder_graph, 'graph_node_map.pk'), 'rb') as fin:
            self.graph_node_map = pickle.load(fin)



        pass

    def get_point_data(self, idx) -> PointDataTest:
        src_node = self.test_src_nodes[idx]
        dst_node = self.test_dst_nodes[idx]
        edge_type = self.test_edge_types[idx]
        
        current_sample = [
            src_node, dst_node, edge_type, self.config['max_train_ts']]
        
        point_data = point_data_utils_seq.get_point_data(
            current_sample, self.src_nodes, self.dst_nodes, self.edge_types,
            self.timestamps, self.config, self.triplet_index,
            self.triplet_index_bilateral, self.pair_index, self.node_index,
            self.trip_search_index, self.trip_search_index_map,
            self.pair_search_index, self.pair_search_index_map,
            self.src_node_search_index, self.src_node_search_index_map,
            self.dst_node_search_index, self.dst_node_search_index_map)

        point_data_test = PointDataTest(
            src_node, dst_node, edge_type, point_data.max_history_ts,
            point_data.history_edges_triplet, point_data.history_edges_pair,
            point_data.history_edges_src, point_data.history_edges_dst,
            self.test_start_timestamps[idx],
            self.test_end_timestamps[idx])
        
        return point_data_test
    
    def __getitem__(self, idx):        
        config = self.config

        point_data = self.get_point_data(idx)
        edge_feat = self.get_edge_feat(point_data.src_node, point_data.dst_node,
                                       point_data.edge_type)
        history_edges_trip = point_data.history_edges_triplet
        history_edges_pair = point_data.history_edges_pair
        history_edges_src = point_data.history_edges_src
        history_edges_dst = point_data.history_edges_dst    
        
        start_ts = self.test_start_timestamps[idx]
        end_ts = self.test_end_timestamps[idx]

        label = np.array([self.test_labels[idx]])

        max_history_ts = point_data.max_history_ts
        trip_feat = self.encode_history_edges(
            history_edges_trip,
            max_history_ts)
        
        trip_feat = feat_utils_seq.merge_category(
            trip_feat, config['trip_feat_dim'])

        if 'node_feat_file' not in self.config:
            trip_feat_extra_b = self.get_history_edge_feat_b(
                history_edges_trip)
            pair_feat_extra_b = self.get_history_edge_feat_b(
                history_edges_pair)
            src_feat_extra_b = self.get_history_edge_feat_b(
                history_edges_src)
            dst_feat_extra_b = self.get_history_edge_feat_b(
                history_edges_dst)
            pass
        else:
            trip_feat_extra_b = np.zeros((len(history_edges_trip), 1), dtype='float32')
            pair_feat_extra_b = np.zeros((len(history_edges_pair), 1), dtype='float32')
            src_feat_extra_b = np.zeros((len(history_edges_src), 1), dtype='float32')
            dst_feat_extra_b = np.zeros((len(history_edges_dst), 1), dtype='float32')
            pass
                    
        pair_feat = self.encode_history_edges(
            history_edges_pair,
            max_history_ts)

        pair_feat_extra = self.get_pair_feat_extra(
            point_data.edge_type, history_edges_pair)

        pair_feat = np.concatenate(
            (pair_feat, pair_feat_extra),
            axis=-1)

        pair_feat = feat_utils_seq.merge_category(
            pair_feat, config['pair_feat_dim'])

        src_feat = self.encode_node_history_edges(
            history_edges_src, max_history_ts, point_data.edge_type,
            point_data.dst_node)
        dst_feat = self.encode_node_history_edges(
            history_edges_dst, max_history_ts, point_data.edge_type,
            point_data.src_node)
        
        label_bins, label_weights = label_utils_seq.get_predict_bins(
            start_ts, end_ts, config['label_bin_size'])
        label_feat = feat_utils_seq.get_label_feat(
            label_bins, config['label_bin_size'],
            max_history_ts,
            config['max_label_class']
        )
        label_feat = feat_utils_seq.merge_category(
            label_feat, config['label_feat_dim'])

        labels_time = label_bins * config['label_bin_size']
        labels_time = np.reshape(labels_time, [-1, 1])
        #######graph
        eid = idx + self.index_start_graph
        graph_src_id = self.graph_node_map[self.node_map_reverse[point_data.src_node]]
        graph_dst_id = self.graph_node_map[self.node_map_reverse[point_data.dst_node]]
        graph_timestamp = self.full_data_graph.st_timestamps[eid]
        graph_edge_type = self.edgetype_map_reverse[point_data.edge_type] + 1

        graph_src_id = np.array([graph_src_id])
        graph_dst_id = np.array([graph_dst_id])
        graph_edge_type = np.array([graph_edge_type])
        graph_timestamp = np.array([graph_timestamp])


        src_neighbor_one_hop = self.ngh_finder.get_temporal_neighbor(graph_src_id, graph_timestamp,
                                                                     n_neighbors=self.neighbors_num)
        src_one_hop_source = src_neighbor_one_hop[0].reshape([-1])
        src_one_hop_ts = src_neighbor_one_hop[2].reshape([-1])
        src_neighbor_two_hop = self.ngh_finder.get_temporal_neighbor(src_one_hop_source, src_one_hop_ts,
                                                                     n_neighbors=self.neighbors_num)
        src_two_hop_source = src_neighbor_two_hop[0].reshape([-1, self.neighbors_num * self.neighbors_num])
        src_two_hop_idx = src_neighbor_two_hop[1].reshape([-1, self.neighbors_num * self.neighbors_num])
        src_two_hop_ts = src_neighbor_two_hop[2].reshape([-1, self.neighbors_num * self.neighbors_num])
        src_full_hop_info = np.concatenate(
            [eid.reshape([-1, 1]), src_neighbor_one_hop[0], src_two_hop_source, src_neighbor_one_hop[1], src_two_hop_idx
                , src_neighbor_one_hop[2], src_two_hop_ts], axis=1)

        dst_neighbor_one_hop = self.ngh_finder.get_temporal_neighbor(graph_dst_id, graph_timestamp,
                                                                     n_neighbors=self.neighbors_num)
        dst_one_hop_source = dst_neighbor_one_hop[0].reshape([-1])
        dst_one_hop_ts = dst_neighbor_one_hop[2].reshape([-1])
        dst_neighbor_two_hop = self.ngh_finder.get_temporal_neighbor(dst_one_hop_source, dst_one_hop_ts,
                                                                     n_neighbors=self.neighbors_num)
        dst_two_hop_source = dst_neighbor_two_hop[0].reshape([-1, self.neighbors_num * self.neighbors_num])
        dst_two_hop_idx = dst_neighbor_two_hop[1].reshape([-1, self.neighbors_num * self.neighbors_num])
        dst_two_hop_ts = dst_neighbor_two_hop[2].reshape([-1, self.neighbors_num * self.neighbors_num])
        dst_full_hop_info = np.concatenate(
            [eid.reshape([-1, 1]), dst_neighbor_one_hop[0], dst_two_hop_source, dst_neighbor_one_hop[1], dst_two_hop_idx
                , dst_neighbor_one_hop[2], dst_two_hop_ts], axis=1)
        src_full_hop_info = src_full_hop_info.astype(int)
        dst_full_hop_info = dst_full_hop_info.astype(int)
        # proprocess edgeidx_2_feat
        idx_hop1_start = 1 + self.neighbors_num + self.neighbors_num * self.neighbors_num
        idx_hop1_end = 1 + 2 * self.neighbors_num + self.neighbors_num * self.neighbors_num
        idx_hop2_end = 1 + 2 * self.neighbors_num + 2 * self.neighbors_num * self.neighbors_num

        user_neighbor_1hop_edgeidx = src_full_hop_info[:, idx_hop1_start:idx_hop1_end]
        user_neighbor_2hop_edgeidx = src_full_hop_info[:, idx_hop1_end:idx_hop2_end]
        oppo_neighbor_1hop_edgeidx = dst_full_hop_info[:, idx_hop1_start:idx_hop1_end]
        oppo_neighbor_2hop_edgeidx = dst_full_hop_info[:, idx_hop1_end:idx_hop2_end]

        user_1hop_edge_orgfeature = self.edgeidx_2_feat_dict[user_neighbor_1hop_edgeidx].astype(np.float32)
        user_2hop_edge_orgfeature = self.edgeidx_2_feat_dict[user_neighbor_2hop_edgeidx].astype(np.float32)
        oppo_1hop_edge_orgfeature = self.edgeidx_2_feat_dict[oppo_neighbor_1hop_edgeidx].astype(np.float32)
        oppo_2hop_edge_orgfeature = self.edgeidx_2_feat_dict[oppo_neighbor_2hop_edgeidx].astype(np.float32)

        return {
            'label': torch.from_numpy(label),
            'label_bins': torch.from_numpy(label_bins),
            'label_weights': torch.from_numpy(label_weights),
            'edge_feat': torch.from_numpy(edge_feat),
            'trip_feat': torch.from_numpy(trip_feat),
            'pair_feat': torch.from_numpy(pair_feat),
            'label_feat': torch.from_numpy(label_feat),
            'trip_feat_extra_b': torch.from_numpy(trip_feat_extra_b),
            'pair_feat_extra_b': torch.from_numpy(pair_feat_extra_b),
            'src_feat_extra_b': torch.from_numpy(src_feat_extra_b),
            'dst_feat_extra_b': torch.from_numpy(dst_feat_extra_b),
            'src_feat': torch.from_numpy(src_feat),
            'dst_feat': torch.from_numpy(dst_feat),

            'eid': idx,

            # graph
            'labels_time': torch.from_numpy(labels_time),
            'user_graphfeat_initial': torch.from_numpy(src_full_hop_info),
            'oppo_grapgfeat_initial': torch.from_numpy(dst_full_hop_info),
            'user_1hop_edge_orgfeature': torch.from_numpy(user_1hop_edge_orgfeature),
            'user_2hop_edge_orgfeature': torch.from_numpy(user_2hop_edge_orgfeature),
            'oppo_1hop_edge_orgfeature': torch.from_numpy(oppo_1hop_edge_orgfeature),
            'oppo_2hop_edge_orgfeature': torch.from_numpy(oppo_2hop_edge_orgfeature),
            'graph_src_id': torch.from_numpy(graph_src_id),
            'graph_dst_id': torch.from_numpy(graph_dst_id)


        }

    def __len__(self):
        return len(self.test_src_nodes)
    pass


def collate_seq(feat_list):
    batch_size = len(feat_list)
    feat_max_len = np.max([feat.shape[0] for feat in feat_list])
    feat_dim = feat_list[0].shape[1]
    feat = torch.zeros(
        (batch_size, feat_max_len, feat_dim),
        dtype=feat_list[0].dtype)
    mask = torch.zeros((batch_size, feat_max_len))

    for i, ifeat in enumerate(feat_list):
        size = ifeat.shape[0]
        feat[i, :size, :] = ifeat
        mask[i, :size] = 1
        pass

    return feat, mask

def collate_seq_3d(feat_list):
    batch_size = len(feat_list)
    feat_len = (np.sum([feat.shape[0] for feat in feat_list]))
    feat_dim = feat_list[0].shape[1]
    feat_dim3 = feat_list[0].shape[2]
    feat = torch.zeros(
        (feat_len, feat_dim, feat_dim3),
        dtype=feat_list[0].dtype)
    #mask = torch.zeros((batch_size, feat_max_len))
    idx=0
    for i, ifeat in enumerate(feat_list):
        size = ifeat.shape[0]
        feat[idx:idx+size, :, :] = ifeat
        #mask[i, :size] = 1
        idx += size
        pass
    #feat = torch.reshape(feat, [batch_size * feat_max_len, feat_dim])

    return feat



def dyg_collate_fn(batch):
    edge_feat = torch.cat([b['edge_feat'] for b in batch], dim=0)
    label, label_mask = collate_seq(
        [b['label'][:, None] for b in batch])
    label = label.squeeze(-1)

    trip_feat, trip_mask = collate_seq([b['trip_feat'] for b in batch])
    pair_feat, pair_mask = collate_seq([b['pair_feat'] for b in batch])

    trip_feat_extra_b, _ = collate_seq([b['trip_feat_extra_b'] for b in batch])
    pair_feat_extra_b, _ = collate_seq([b['pair_feat_extra_b'] for b in batch])
    src_feat_extra_b, _ = collate_seq([b['src_feat_extra_b'] for b in batch])
    dst_feat_extra_b, _ = collate_seq([b['dst_feat_extra_b'] for b in batch])

    label_feat, label_mask = collate_seq([b['label_feat'] for b in batch])
    eids = [b['eid'] for b in batch]

    src_feat, src_mask = collate_seq([b['src_feat'] for b in batch])
    dst_feat, dst_mask = collate_seq([b['dst_feat'] for b in batch])

    user_graphfeat_initial = torch.cat([b['user_graphfeat_initial'] for b in batch], dim=0)
    oppo_grapgfeat_initial = torch.cat([b['oppo_grapgfeat_initial'] for b in batch], dim=0)
    user_1hop_edge_orgfeature = collate_seq_3d([b['user_1hop_edge_orgfeature'] for b in batch])
    user_2hop_edge_orgfeature = collate_seq_3d([b['user_2hop_edge_orgfeature'] for b in batch])
    oppo_1hop_edge_orgfeature = collate_seq_3d([b['oppo_1hop_edge_orgfeature'] for b in batch])
    oppo_2hop_edge_orgfeature = collate_seq_3d([b['oppo_2hop_edge_orgfeature'] for b in batch])
    graph_src_id = torch.cat(([b['graph_src_id'] for b in batch]), dim=0)
    graph_dst_id = torch.cat(([b['graph_dst_id'] for b in batch]), dim=0)
    labels_time, _ = collate_seq([b['labels_time'] for b in batch])


    return {
        'label': label,
        'edge_feat': edge_feat,
        'trip_feat': trip_feat,
        'trip_mask': trip_mask,
        'pair_feat': pair_feat,
        'pair_mask': pair_mask,
        'src_feat': src_feat,
        'dst_feat': dst_feat,
        'src_mask': src_mask,
        'dst_mask': dst_mask,
        'trip_feat_extra_b': trip_feat_extra_b,
        'pair_feat_extra_b': pair_feat_extra_b,
        'src_feat_extra_b': src_feat_extra_b,
        'dst_feat_extra_b': dst_feat_extra_b,
        'label_feat': label_feat,
        'label_mask': label_mask,
        'eid': eids,
        'user_graphfeat_initial': user_graphfeat_initial,
        'oppo_grapgfeat_initial': oppo_grapgfeat_initial,
        'user_1hop_edge_orgfeature': user_1hop_edge_orgfeature,
        'user_2hop_edge_orgfeature': user_2hop_edge_orgfeature,
        'oppo_1hop_edge_orgfeature': oppo_1hop_edge_orgfeature,
        'oppo_2hop_edge_orgfeature': oppo_2hop_edge_orgfeature,
        'graph_src_id': graph_src_id,
        'graph_dst_id': graph_dst_id,
        'labels_time': labels_time
    }
    pass


def dyg_test_collate_fn(batch):
    edge_feat = torch.cat([b['edge_feat'] for b in batch], dim=0)
    label_bins, label_mask = collate_seq(
        [b['label_bins'][:, None] for b in batch])
    label_bins = label_bins.squeeze(-1)
    label_weights, label_mask = collate_seq(
        [b['label_weights'][:, None] for b in batch])
    label_weights = label_weights.squeeze(-1)

    trip_feat, trip_mask = collate_seq([b['trip_feat'] for b in batch])
    pair_feat, pair_mask = collate_seq([b['pair_feat'] for b in batch])

    trip_feat_extra_b, _ = collate_seq([b['trip_feat_extra_b'] for b in batch])
    pair_feat_extra_b, _ = collate_seq([b['pair_feat_extra_b'] for b in batch])
    src_feat_extra_b, _ = collate_seq([b['src_feat_extra_b'] for b in batch])
    dst_feat_extra_b, _ = collate_seq([b['dst_feat_extra_b'] for b in batch])

    src_feat, src_mask = collate_seq([b['src_feat'] for b in batch])
    dst_feat, dst_mask = collate_seq([b['dst_feat'] for b in batch])

    label_feat, label_mask = collate_seq([b['label_feat'] for b in batch])
    eids = [b['eid'] for b in batch]

    user_graphfeat_initial = torch.cat([b['user_graphfeat_initial'] for b in batch], dim=0)
    oppo_grapgfeat_initial = torch.cat([b['oppo_grapgfeat_initial'] for b in batch], dim=0)
    user_1hop_edge_orgfeature = collate_seq_3d([b['user_1hop_edge_orgfeature'] for b in batch])
    user_2hop_edge_orgfeature = collate_seq_3d([b['user_2hop_edge_orgfeature'] for b in batch])
    oppo_1hop_edge_orgfeature = collate_seq_3d([b['oppo_1hop_edge_orgfeature'] for b in batch])
    oppo_2hop_edge_orgfeature = collate_seq_3d([b['oppo_2hop_edge_orgfeature'] for b in batch])
    graph_src_id = torch.cat(([b['graph_src_id'] for b in batch]), dim=0)
    graph_dst_id = torch.cat(([b['graph_dst_id'] for b in batch]), dim=0)
    labels_time, _ = collate_seq([b['labels_time'] for b in batch])


    return {
        'label': torch.cat([b['label'] for b in batch]),
        'label_bins': label_bins,
        'label_weights': label_weights,
        'edge_feat': edge_feat,
        'trip_feat': trip_feat,
        'trip_mask': trip_mask,
        'pair_feat': pair_feat,
        'pair_mask': pair_mask,
        'trip_feat_extra_b': trip_feat_extra_b,
        'pair_feat_extra_b': pair_feat_extra_b,
        'src_feat_extra_b': src_feat_extra_b,
        'dst_feat_extra_b': dst_feat_extra_b,
        'src_feat': src_feat,
        'dst_feat': dst_feat,
        'src_mask': src_mask,
        'dst_mask': dst_mask,
        'label_feat': label_feat,
        'label_mask': label_mask,
        'eid': eids,
        'user_graphfeat_initial': user_graphfeat_initial,
        'oppo_grapgfeat_initial': oppo_grapgfeat_initial,
        'user_1hop_edge_orgfeature': user_1hop_edge_orgfeature,
        'user_2hop_edge_orgfeature': user_2hop_edge_orgfeature,
        'oppo_1hop_edge_orgfeature': oppo_1hop_edge_orgfeature,
        'oppo_2hop_edge_orgfeature': oppo_2hop_edge_orgfeature,
        'graph_src_id': graph_src_id,
        'graph_dst_id': graph_dst_id,
        'labels_time': labels_time
    }
    pass
    

class RandomDropSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, dataset, drop_rate):
        self.dataset = dataset
        self.drop_rate = drop_rate
        self.drop_num = int(len(dataset) * drop_rate)

    def __iter__(self):
        arange = np.arange(len(self.dataset))
        np.random.shuffle(arange)
        indices = arange[: (1-self.drop_num)]
        return iter(np.sort(indices))
            
    def __len__(self):
        return len(self.dataset) - self.drop_num


if __name__ == '__main__':
    config = importlib.import_module('config_a').config
    a = DygDataset(config, 'train')
    #a = DygDatasetTest(config, 'val')
    c = a[1]
    #print(c)