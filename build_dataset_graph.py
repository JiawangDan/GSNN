import os
import math
import numpy as np
import pandas as pd
import sys
import importlib
from sklearn.preprocessing import OneHotEncoder
import operator
from dyg_utils import get_neighbor_finder
from config_a import config as cfg
import pickle


def reindex_node_name(file_name):
    dict = {}
    dict[-1] = 0  #人工添加节点

    init_node_id = 1
    with open(file_name) as f:
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            star_node = int(e[0])
            dest_node = int(e[1])
            if star_node not in dict:
                dict[star_node] = init_node_id
                init_node_id += 1
            if dest_node not in dict:
                dict[dest_node] = init_node_id
                init_node_id += 1
    return dict


def preprocess_node(node_dict, PATH_NODE_FEAT):
    max_idx = len(node_dict)
    node_feature = np.zeros((max_idx + 1, 8), dtype=np.int32) - 1

    with open(PATH_NODE_FEAT) as f:
        for e_idx, e_line in enumerate(f):
            e = e_line.strip().split(',')
            node_org_name = int(e[0])
            node_prod_name = node_dict[node_org_name]
            feat = np.array([int(x) for x in e[1:]])
            node_feature[node_prod_name] = feat

    #节点特征转换
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(node_feature)
    node_feature = (pd.DataFrame(ohe.fit_transform(node_feature).toarray())).values

    return node_feature

def preprocess_edge_feat(PATH_EDGE_FEAT):
    edge_feat_dict = {}
    edge_feat_nparray = []

    with open(PATH_EDGE_FEAT) as f:
        for e_idx, e_line in enumerate(f):
            e = e_line.strip().split(',')
            e_name = int(e[0]) + 1
            feat = np.array([int(x) for x in e[1:]])
            edge_feat_dict[e_name] = feat
            edge_feat_nparray.append(feat)
    # 填充一个空白边特征
    edge_feat_dict[0] = np.array([-1, -1, -1])

    edge_feat_nparray = np.array(edge_feat_nparray)

    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(edge_feat_nparray)

    for k, v in edge_feat_dict.items():
        v = v.reshape(1, -1)
        edge_feat_dict[k] = ((pd.DataFrame(ohe.transform(v).toarray())).values).reshape(-1)

    tensor_edge_feat_dict = dict(sorted(edge_feat_dict.items(), key=operator.itemgetter(0)))
    #tensor_edge_feat_np = np.array(list(tensor_edge_feat_dict.values()))

    return tensor_edge_feat_dict

def create_preprocess_node(node_dict, dim):
    max_idx = len(node_dict)
    node_feature = np.random.random((max_idx + 1, dim))
    return node_feature

def create_preprocess_edge(num_edge, dim):
    max_idx = num_edge
    edge_feature = np.random.random((max_idx + 1, dim))
    return edge_feature

def preprocess_edgeidx_type_feat(edge_idx, edge_type, edge_feat):
    edge_idx.insert(0, 0)
    edge_type.insert(0, 0)
    type_len = 1
    edge_feat.insert(0, -1)

    feat_len = 0
    for feat in edge_feat:
        if isinstance(feat, list) and len(feat)>feat_len:
            feat_len = len(feat)

    edge_2_feat = np.zeros([len(edge_idx), type_len+feat_len ], dtype=float)
    for ind, (sing_type, sing_feat) in enumerate(zip(edge_type, edge_feat)):
        temp_feat = np.zeros([feat_len],dtype=float)
        if isinstance(sing_feat, list):
            temp_feat = np.array(sing_feat)
        edge_2_feat[ind] = np.insert(temp_feat, 0, sing_type)

    return edge_2_feat

def read_test_label(file_name):
    label_list = []
    with open(file_name) as f:
        # s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            label = int(e[0])
            label_list.append(label)
    return label_list


class Data:
  def __init__(self, sources, destinations, st_timestamps, end_timestamps, edge_idxs, edge_type, train_flag, labels, edge_extra_feat):
    self.sources = sources
    self.destinations = destinations
    self.st_timestamps = st_timestamps
    self.end_timestamps = end_timestamps
    self.edge_idxs = edge_idxs
    self.edge_type = edge_type
    self.edge_extra_feat = edge_extra_feat
    self.train_flag = train_flag
    self.labels = labels
    self.n_interactions = len(sources)
    self.unique_nodes = set(sources) | set(destinations)
    self.n_unique_nodes = len(self.unique_nodes)
    self.unique_dest_nodes = set(destinations)
    self.edge_type_list = set(edge_type)


def read_edge(file_name, reindex_node_dict, idx_initial=1, config_file=''):
    u_list, i_list, etype_list = [], [], []
    ts_start_list = []
    ts_end_list = []
    label_list = []
    idx_list = []
    extra_feat_list = []
    #
    with open(file_name) as f:
        # s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = reindex_node_dict[int(e[0])]
            i = reindex_node_dict[int(e[1])]

            edge_type = int(e[2]) + 1  # 留一个空位 给填充
            ts_start = float(e[3])

            if len(e) > 4 and len(e) < 10 and len(e[4]) > 0:
                ts_end = float(e[4])
            else:
                ts_end = ts_start

            if len(e) > 5 and len(e) < 10 and len(e[5]) > 0:
                label = int(e[5])
            else:
                label = -1

            if len(e) > 10:
                extra_feat = e[4:]
                extra_feat = [float(feat.replace('"', '')) for feat in extra_feat]
            else:
                extra_feat = -1

            temp_idx = idx + idx_initial

            u_list.append(u)
            i_list.append(i)
            etype_list.append(edge_type)
            ts_start_list.append(ts_start)
            ts_end_list.append(ts_end)
            idx_list.append(temp_idx)
            label_list.append(label)
            extra_feat_list.append(extra_feat)


    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts_start': ts_start_list,
                         'ts_end':ts_end_list,
                         'etype': etype_list,
                         'idx': idx_list,
                         'label':label_list,
                         'extra_feat': extra_feat_list}), temp_idx


def constract_graph(file_train, file_val, file_test, reindex_node_dict, config_file):
    train_data_edge, train_idx = read_edge(file_train, reindex_node_dict, 1, config_file)
    val_data_edge, val_idx = read_edge(file_val, reindex_node_dict, train_idx + 1, config_file)
    test_data_dege, test_idx = read_edge(file_test, reindex_node_dict, val_idx + 1, config_file)

    sources = np.concatenate([train_data_edge['u'].values, val_data_edge['u'].values, test_data_dege['u'].values])
    destinations = np.concatenate([train_data_edge['i'].values, val_data_edge['i'].values, test_data_dege['i'].values,])
    st_timestamps = np.concatenate(
        [train_data_edge['ts_start'].values, val_data_edge['ts_start'].values, test_data_dege['ts_start'].values,])
    end_timestamps = np.concatenate(
        [train_data_edge['ts_end'].values, val_data_edge['ts_end'].values, test_data_dege['ts_end'].values])
    edge_idxs = np.concatenate(
        [train_data_edge['idx'].values, val_data_edge['idx'].values, test_data_dege['idx'].values])
    edge_type = np.concatenate(
        [train_data_edge['etype'].values, val_data_edge['etype'].values, test_data_dege['etype'].values])
    train_flag = np.concatenate(
        [np.ones_like(train_data_edge['u'].values), 2 + np.zeros_like(val_data_edge['u'].values),
         3 + np.zeros_like(test_data_dege['u'].values)])

    edge_extra_feat = np.concatenate(
        [train_data_edge['extra_feat'].values, val_data_edge['extra_feat'].values, test_data_dege['extra_feat'].values])


    labels = np.concatenate(
        [train_data_edge['label'].values, val_data_edge['label'].values, test_data_dege['label'].values])


    full_data = Data(sources, destinations, st_timestamps, end_timestamps, edge_idxs, edge_type, train_flag, labels, edge_extra_feat)

    # ngh_finder = get_neighbor_finder(full_data, uniform=False)

    if os.path.exists(cfg['graph_dataset_path']) is False:
        os.makedirs(cfg['graph_dataset_path'])
    #
    #
    # num_instance = len(full_data.sources)
    # num_batch = math.ceil(num_instance / 100)
    # for batch_idx in range(0, num_batch):
    #     start_idx = batch_idx * 100
    #     end_idx = min(num_instance, start_idx + 100)
    #     idx = np.arange(start_idx, end_idx)
    #
    #     a = full_data.sources[idx]
    #     b = full_data.st_timestamps[idx]
    #
    #     neighbor_one_hop = ngh_finder.get_temporal_neighbor(a, b, n_neighbors=cfg['neighbors'])
    #
    #
    #     one_hop_source = neighbor_one_hop[0].reshape([-1])
    #     one_hop_idx = neighbor_one_hop[1].reshape([-1])
    #     one_hop_ts = neighbor_one_hop[2].reshape([-1])
    #     neighbor_two_hop = ngh_finder.get_temporal_neighbor(one_hop_source, one_hop_ts, n_neighbors=cfg['neighbors'])
    #
    #
    #     two_hop_source = neighbor_two_hop[0].reshape([-1, cfg['neighbors'] * cfg['neighbors']])
    #     two_hop_idx = neighbor_two_hop[1].reshape([-1, cfg['neighbors'] * cfg['neighbors']])
    #     two_hop_ts = neighbor_two_hop[2].reshape([-1, cfg['neighbors'] * cfg['neighbors']])
    #
    #     idx = idx.reshape([-1, 1])
    #     full_hop_info = np.concatenate([idx + 1, neighbor_one_hop[0], two_hop_source, neighbor_one_hop[1], two_hop_idx
    #                                     , neighbor_one_hop[2], two_hop_ts], axis=1)
    #     save_np_path = os.path.join(cfg['graph_dataset_path'], 'graphfeature_user_{}.npy'.format(idx[0, 0] + 1) )
    #     np.save(save_np_path, full_hop_info)
    #
    #
    #     #被动方
    #     idx = idx.reshape([-1])
    #     neighbor_one_hop_opp = ngh_finder.get_temporal_neighbor(full_data.destinations[idx], full_data.st_timestamps[idx],
    #                                                         n_neighbors=cfg['neighbors'])
    #     one_hop_source_opp = neighbor_one_hop_opp[0].reshape([-1])
    #     one_hop_idx_opp = neighbor_one_hop_opp[1].reshape([-1])
    #     one_hop_ts_opp = neighbor_one_hop_opp[2].reshape([-1])
    #     neighbor_two_hop_opp = ngh_finder.get_temporal_neighbor(one_hop_source_opp, one_hop_ts_opp, n_neighbors=cfg['neighbors'])
    #     two_hop_source_opp = neighbor_two_hop_opp[0].reshape([-1, cfg['neighbors'] * cfg['neighbors']])
    #     two_hop_idx_opp = neighbor_two_hop_opp[1].reshape([-1, cfg['neighbors'] * cfg['neighbors']])
    #     two_hop_ts_opp = neighbor_two_hop_opp[2].reshape([-1, cfg['neighbors'] * cfg['neighbors']])
    #
    #     idx = idx.reshape([-1, 1])
    #     full_hop_info = np.concatenate([idx + 1, neighbor_one_hop_opp[0], two_hop_source_opp, neighbor_one_hop_opp[1], two_hop_idx_opp
    #                                        , neighbor_one_hop_opp[2], two_hop_ts_opp], axis=1)
    #     save_np_path = os.path.join(cfg['graph_dataset_path'],
    #                                 'graphfeature_oppo_{}.npy'.format(idx[0, 0] + 1))
    #     np.save(save_np_path, full_hop_info)
    #     if end_idx % 100 == 0:
    #         print(batch_idx,'/', num_batch, 'have finished pick')

    return full_data




if __name__=='__main__':
    config_file = sys.argv[1]
    cfg = importlib.import_module(config_file).config



    reindex_node_dict = reindex_node_name(cfg['train_file'])




    if config_file == 'config_a':
        node_initial_feat = preprocess_node(reindex_node_dict, cfg['node_feat_file'])
        edge_initial_feat = preprocess_edge_feat(cfg['edge_type_feat_file'])  # 边上被填充了一个空白特征
        edge_initial_feat = np.array([item for item in edge_initial_feat.values()])
    else:
        node_initial_feat = create_preprocess_node(reindex_node_dict, cfg['node_initial_dims'])
        edge_initial_feat = create_preprocess_edge(cfg['num_edge_type'], cfg['edge_initial_dims'])



    # #子图构建
    full_data = constract_graph(cfg['train_file'], cfg['val_file'], cfg['test_file'],
                    reindex_node_dict, config_file)

    with open(os.path.join(cfg['graph_dataset_path'], 'graph_node_map.pk'), 'wb') as fout:
        pickle.dump(reindex_node_dict, fout)

    edgeidx_2_type = preprocess_edgeidx_type_feat(full_data.edge_idxs.tolist(), full_data.edge_type.tolist(),
                                                  full_data.edge_extra_feat.tolist())

    full_data_pd = pd.DataFrame({'sources': full_data.sources,
                  'destinations': full_data.destinations,
                  'st_timestamps': full_data.st_timestamps,
                  'end_timestamps': full_data.end_timestamps,
                  'edge_idxs': full_data.edge_idxs,
                  'edge_type': full_data.edge_type,
                  'train_flag': full_data.train_flag,
                   'labels':  full_data.labels
                                 })

    save_fulldata_path = os.path.join(cfg['graph_dataset_path'],
                                      'full_data_edge.csv')
    full_data_pd.to_csv(save_fulldata_path)

    save_np_path = os.path.join(cfg['graph_dataset_path'],
                                'edgeidx_2_feat.npy')
    np.save(save_np_path, edgeidx_2_type)

    save_np_path = os.path.join(cfg['graph_dataset_path'],
                                'node_initial_feat.npy')
    np.save(save_np_path, node_initial_feat)

    save_np_path = os.path.join(cfg['graph_dataset_path'],
                                'edge_initial_feat.npy')
    np.save(save_np_path, edge_initial_feat)

    print('build dataset finished!')

    h = 1