import pandas as pd
import numpy as np
import scipy.sparse
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import label_utils_seq
import constants
import sys
import os
import pickle
import multiprocessing as mp
import tarfile
from io import BytesIO
from common_structures_seq import PointData, PointDataTest
import pandas as pd
import constants
import json
import importlib
import sqlite3

global_data = None


def dt2ts(dt):
    return datetime.strptime(dt, '%Y%m%d').timestamp()


def get_target_ts(
        triplet_index, eid, snode, dnode, etype, timestamps, interval,
        max_ts):
    current_ts = timestamps[eid]
    target_ts_pre = []
    target_ts_post = []

    next_eid = triplet_index[1, eid]
    while next_eid != -1:
        ts = timestamps[next_eid]
        ts_diff = current_ts - ts
        if ts_diff > interval:
            break

        target_ts_pre.append(ts)
        next_eid = triplet_index[1, next_eid]
        pass

    next_eid = triplet_index[0, eid]

    bench_ts = current_ts
    while next_eid != -1:
        ts = timestamps[next_eid]

        ts_diff = ts - bench_ts
        if ts_diff > interval:
            if len(target_ts_post) > 0:
                break
            else:
                bench_ts = current_ts
                pass
            pass

        target_ts_post.append(ts)
        next_eid = triplet_index[0, next_eid]
        pass

    if next_eid == -1:
        target_ts_post.append(max_ts)
        pass

    return np.array(
        target_ts_pre[::-1] + target_ts_post, dtype='int32')


def get_history_feat(
        index, eid, src_nodes, dst_nodes, edge_types, timestamps, max_time):
    reverse_key = (dst_nodes[eid], src_nodes[eid])

    pre_eid = index[eid]

    current_ts = timestamps[eid]
    current_day = current_ts // constants.SECONDS_PER_DAY
    counts = defaultdict(int)
    while pre_eid != -1:
        ts = timestamps[pre_eid]

        if current_ts - ts > max_time:
            break

        day = ts // constants.SECONDS_PER_DAY
        day_diff = current_day - day
        is_reverse = int(
            (src_nodes[pre_eid], dst_nodes[pre_eid]) == reverse_key)
        etype = edge_types[pre_eid]

        fkey = day_diff * constants.NUM_EDGE_TYPE * 2 + etype * 2 + is_reverse
        counts[fkey] += 1

        pre_eid = index[pre_eid]
        pass

    feat = np.zeros((len(counts), 2), dtype='int32')
    for i, (k, v) in enumerate(counts.items()):
        feat[i, :] = [k, v]
        pass

    return feat


def get_history_edges(
        index, eid, src_nodes, dst_nodes, edge_types, timestamps, max_num):
    key = (src_nodes[eid], dst_nodes[eid])
    reverse_key = (dst_nodes[eid], src_nodes[eid])

    if edge_types is not None:
        key += (edge_types[eid],)
        reverse_key += (edge_types[eid],)
        pass

    pre_eid = index[eid]

    history_edges = []
    while pre_eid != -1:
        compare_key = (src_nodes[pre_eid], dst_nodes[pre_eid])

        if edge_types is not None:
            compare_key += (edge_types[pre_eid],)
            pass

        reverse = int(compare_key == reverse_key)
        history_edges.append((pre_eid, reverse))

        if len(history_edges) > max_num:
            break

        pre_eid = index[pre_eid]

        pass

    return np.array(history_edges, dtype='int32')


def get_history_edges_node_single(
        index, eid, i, src_nodes, dst_nodes, timestamps, max_num):
    history_edges = []
    pre_eid = index[eid, i]
    if i == 0:
        current_node = src_nodes[eid]
    else:
        current_node = dst_nodes[eid]
        pass

    while pre_eid != -1:
        snode = src_nodes[pre_eid]
        dnode = dst_nodes[pre_eid]

        if snode == current_node:
            history_edges.append((pre_eid, 0))
            pre_eid = index[pre_eid, 0]
        elif dnode == current_node:
            history_edges.append((pre_eid, 1))
            pre_eid = index[pre_eid, 1]
        else:
            raise RuntimeError('index corrupted')

        if len(history_edges) > max_num:
            break
        pass

    return np.array(history_edges, dtype='int32')


def get_history_edges_node(
        index, eid, src_nodes, dst_nodes, timestamps, max_num):
    src_history_edges = get_history_edges_node_single(
        index, eid, 0, src_nodes, dst_nodes, timestamps, max_num)
    dst_history_edges = get_history_edges_node_single(
        index, eid, 1, src_nodes, dst_nodes, timestamps, max_num)

    return src_history_edges, dst_history_edges


def reorder_feature(idmap, data, default=-1):
    ids = data[:, 0]
    feat = data[:, 1:]

    num_ids = len(idmap)
    feat_ordered = np.zeros((num_ids, feat.shape[1]), dtype=feat.dtype)
    feat_ordered.fill(default)

    order = np.asarray([idmap[i] for i in ids])
    feat_ordered[order] = feat

    return feat_ordered


def build_dataset_numpy(config):
    build_dataset_numpy_train(config)
    build_dataset_numpy_val(config)
    build_dataset_numpy_test(config)
    pass


def build_dataset_numpy_train(config: dict):
    output_folder = config['dataset_path']
    df = pd.read_csv(config['train_file'], header=None)
    num_edges = df.shape[0]

    data = df[[0, 1, 2, 3]].values

    eids = np.arange(num_edges)
    timestamps = data[:, -1]
    print(data.dtype)

    raw_nodes = np.hstack((df.values[:, 0], df.values[:, 1]))
    val_and_test_nodes = np.unique(np.hstack((
        pd.read_csv(config['val_file'], header=None).values[:, :2].flatten(),
        pd.read_csv(config['test_file'], header=None).values[:, :2].flatten())))

    all_nodes = np.hstack((raw_nodes, val_and_test_nodes))
    unodes, nodes = np.unique(all_nodes, return_inverse=True)
    nodes = nodes[:len(raw_nodes)]
    node_map = {node: i for i, node in enumerate(unodes)}

    src_nodes = nodes[: num_edges]
    dst_nodes = nodes[num_edges:]

    raw_edge_types = data[:, 2]
    uniq_edge_types, edge_types = np.unique(
        raw_edge_types, return_inverse=True)

    np.save(os.path.join(output_folder, 'eids.npy'), eids)
    np.save(os.path.join(output_folder, 'timestamps.npy'), timestamps)
    np.save(os.path.join(output_folder, 'src_nodes.npy'), src_nodes)
    np.save(os.path.join(output_folder, 'dst_nodes.npy'), dst_nodes)
    np.save(os.path.join(output_folder, 'edge_types.npy'), edge_types)
    with open(os.path.join(output_folder, 'node_map.pk'), 'wb') as fout:
        pickle.dump(node_map, fout)
        pass

    edge_type_map = {et: i for i, et in enumerate(uniq_edge_types)}
    with open(os.path.join(output_folder, 'edge_type_map.pk'), 'wb') as fout:
        pickle.dump(edge_type_map, fout)
        pass

    if 'node_feat_file' in config:
        # it is dataset a
        edge_type_feat = pd.read_csv(config['edge_type_feat_file'],
                                     header=None).values

        edge_type_feat = reorder_feature(edge_type_map, edge_type_feat)

        node_feat = pd.read_csv(config['node_feat_file'],
                                header=None).values

        node_feat = reorder_feature(node_map, node_feat)

        np.save(os.path.join(output_folder, 'edge_type_feat.npy'), edge_type_feat)
        np.save(os.path.join(output_folder, 'node_feat.npy'), node_feat)
        pass
    else:
        # it is dataset b
        is_have_edge_feat = np.logical_not(pd.isnull(df[[4]]).values).flatten()
        edge_feat = df[[4]].values[is_have_edge_feat].flatten()
        edge_feat_ids = np.argwhere(is_have_edge_feat).flatten()

        edge_feat = np.array(
            [[float(v) for v in ef.split(',')] for ef in tqdm(edge_feat)],
            dtype='float32')

        np.save(os.path.join(output_folder, 'edge_feat_ids.npy'),
                edge_feat_ids)
        np.save(os.path.join(output_folder, 'edge_feat'), edge_feat)

        pass

    return node_map, edge_type_map


def read_idmaps(config):
    output_folder = config['dataset_path']

    with open(os.path.join(output_folder, 'node_map.pk'), 'rb') as fin:
        node_map = pickle.load(fin)
        pass
    with open(os.path.join(output_folder, 'edge_type_map.pk'), 'rb') as fin:
        edge_type_map = pickle.load(fin)
        pass

    return node_map, edge_type_map


def build_dataset_numpy_val(config):
    output_folder = config['dataset_path']
    data = pd.read_csv(config['val_file'], header=None).values

    node_map, edge_type_map = read_idmaps(config)
    val_labels = data[:, 5]
    np.save(os.path.join(output_folder, 'val_labels.npy'), val_labels)

    val_src_nodes = np.array([node_map[n] for n in data[:, 0]], dtype='int32')
    val_dst_nodes = np.array([node_map[n] for n in data[:, 1]], dtype='int32')
    val_edge_types = np.array([edge_type_map[t] for t in data[:, 2]], dtype='int32')
    val_start_timestamps = data[:, 3].astype('int32')
    val_end_timestamps = data[:, 4].astype('int32')
    np.save(os.path.join(output_folder, 'val_src_nodes.npy'), val_src_nodes)
    np.save(os.path.join(output_folder, 'val_dst_nodes.npy'), val_dst_nodes)
    np.save(os.path.join(output_folder, 'val_edge_types.npy'), val_edge_types)
    np.save(os.path.join(output_folder, 'val_start_timestamps.npy'), val_start_timestamps)
    np.save(os.path.join(output_folder, 'val_end_timestamps.npy'), val_end_timestamps)

    pass


def build_dataset_numpy_test(config):
    output_folder = config['dataset_path']
    data = pd.read_csv(config['test_file'], header=None).values

    node_map, edge_type_map = read_idmaps(config)
    test_src_nodes = np.array([node_map[n] for n in data[:, 0]], dtype='int32')
    test_dst_nodes = np.array([node_map[n] for n in data[:, 1]], dtype='int32')
    test_edge_types = np.array([edge_type_map[t] for t in data[:, 2]], dtype='int32')
    test_start_timestamps = data[:, 3].astype('int32')
    test_end_timestamps = data[:, 4].astype('int32')
    np.save(os.path.join(output_folder, 'test_src_nodes.npy'), test_src_nodes)
    np.save(os.path.join(output_folder, 'test_dst_nodes.npy'), test_dst_nodes)
    np.save(os.path.join(output_folder, 'test_edge_types.npy'), test_edge_types)
    np.save(os.path.join(output_folder, 'test_start_timestamps.npy'), test_start_timestamps)
    np.save(os.path.join(output_folder, 'test_end_timestamps.npy'), test_end_timestamps)

    pass


def build_index_triplet(output_folder, eids, src_nodes, dst_nodes, edge_types):
    triplet_index = -np.ones((2, len(eids)), dtype='int32')
    triplet_last_eid = dict()

    for eid in tqdm(eids):
        key = (src_nodes[eid], dst_nodes[eid], edge_types[eid])
        last_eid = triplet_last_eid.get(key)
        if last_eid is None:
            triplet_last_eid[key] = eid
            pass
        else:
            triplet_index[0, last_eid] = eid
            triplet_index[1, eid] = last_eid
            pass
        triplet_last_eid[key] = eid
        pass

    np.save(os.path.join(output_folder, 'triplet_index.npy'), triplet_index)
    pass


def build_index_triplet_bilateral(
        output_folder, eids, src_nodes, dst_nodes, edge_types):
    triplet_index_bilateral = -np.ones(len(eids), dtype='int32')
    triplet_last_eid = dict()

    for eid in tqdm(eids):
        snode = src_nodes[eid]
        dnode = dst_nodes[eid]
        etype = edge_types[eid]

        last_eid = triplet_last_eid.get((snode, dnode, etype))
        if last_eid is None:
            last_eid = triplet_last_eid.get((dnode, snode, etype))

            if last_eid is None:
                triplet_last_eid[(snode, dnode, etype)] = eid
                pass
            else:
                triplet_index_bilateral[eid] = last_eid
                triplet_last_eid[(dnode, snode, etype)] = eid
                pass
            pass
        else:
            triplet_index_bilateral[eid] = last_eid
            triplet_last_eid[(snode, dnode, etype)] = eid
            pass
        pass
    np.save(
        os.path.join(
            output_folder, 'triplet_index_bilateral.npy'),
        triplet_index_bilateral)

    with open(os.path.join(output_folder, 'triplet_last_eid'), 'wb') as fout:
        pickle.dump(triplet_last_eid, fout, protocol=pickle.HIGHEST_PROTOCOL)
        pass
    pass


def build_index_pair(
        output_folder, eids, src_nodes, dst_nodes):
    pair_index = -np.ones(len(eids), dtype='int32')
    pair_last_eid = dict()
    for eid in tqdm(eids):
        snode = src_nodes[eid]
        dnode = dst_nodes[eid]

        last_eid = pair_last_eid.get((snode, dnode))
        if last_eid is None:
            last_eid = pair_last_eid.get((dnode, snode))

            if last_eid is None:
                pair_last_eid[(snode, dnode)] = eid
                pass
            else:
                pair_index[eid] = last_eid
                pair_last_eid[(dnode, snode)] = eid
                pass
            pass
        else:
            pair_index[eid] = last_eid
            pair_last_eid[(snode, dnode)] = eid
            pass
        pass
    np.save(
        os.path.join(
            output_folder, 'pair_index.npy'),
        pair_index)

    with open(os.path.join(output_folder, 'pair_last_eid'), 'wb') as fout:
        pickle.dump(pair_last_eid, fout, protocol=pickle.HIGHEST_PROTOCOL)
        pass
    pass


def build_index_val(config, prefix):
    output_folder = config['dataset_path']
    src_nodes = np.load(os.path.join(output_folder, f'{prefix}_src_nodes.npy'))
    dst_nodes = np.load(os.path.join(output_folder, f'{prefix}_dst_nodes.npy'))
    edge_types = np.load(
        os.path.join(output_folder, f'{prefix}_edge_types.npy'))

    with open(os.path.join(output_folder, 'pair_last_eid'), 'rb') as fin:
        pair_last_eid = pickle.load(fin)
        pass

    with open(os.path.join(output_folder, 'triplet_last_eid'), 'rb') as fin:
        triplet_last_eid = pickle.load(fin)
        pass

    with open(os.path.join(output_folder, 'node_last_eid'), 'rb') as fin:
        node_last_eid = pickle.load(fin)
        pass

    index = -np.ones((len(src_nodes), 4), dtype='int32')

    for i in range(len(src_nodes)):
        snode = src_nodes[i]
        dnode = dst_nodes[i]
        etype = edge_types[i]

        last_eid = triplet_last_eid.get((snode, dnode, etype))
        if last_eid is None:
            last_eid = triplet_last_eid.get((dnode, snode, etype))
            if last_eid is None:
                last_eid = -1
            pass

        index[i, 1] = last_eid

        last_eid = pair_last_eid.get((snode, dnode))
        if last_eid is None:
            last_eid = pair_last_eid.get((dnode, snode))
            if last_eid is None:
                last_eid = -1
            pass
        index[i, 0] = last_eid

        last_eid = node_last_eid.get(snode)
        if last_eid is not None:
            index[i, 2] = last_eid
            pass

        last_eid = node_last_eid.get(dnode)
        if last_eid is not None:
            index[i, 3] = last_eid
            pass
        pass

    np.save(os.path.join(output_folder, f'{prefix}_index.npy'), index)
    pass


def build_index_node(output_folder, eids, src_nodes, dst_nodes):
    node_index = -np.ones((len(eids), 2), dtype='int32')
    node_last_eid = dict()
    for eid in tqdm(eids):
        snode = src_nodes[eid]
        dnode = dst_nodes[eid]

        last_eid_s = node_last_eid.get(snode)
        last_eid_d = node_last_eid.get(dnode)

        if last_eid_s is not None:
            node_index[eid, 0] = last_eid_s
            pass

        if last_eid_d is not None:
            node_index[eid, 1] = last_eid_d
            pass

        node_last_eid[snode] = eid
        node_last_eid[dnode] = eid
        pass

    np.save(
        os.path.join(
            output_folder, 'node_index.npy'),
        node_index)

    with open(os.path.join(output_folder, 'node_last_eid'), 'wb') as fout:
        pickle.dump(node_last_eid, fout, protocol=pickle.HIGHEST_PROTOCOL)
        pass
    pass

    pass


def build_index_sqlite(
        config, eids, src_nodes, dst_nodes, edge_types, timestamps):
    output_folder = config['dataset_path']

    conn = sqlite3.connect(os.path.join(output_folder, 'data.sqlite'))

    conn.executescript(
        '''
        drop table if exists edges;
        create table edges(
          eid integer,
          src_node integer,
          dst_node integer,
          edge_type integer,
          timestamp integer);
        ''')
    conn.commit()

    # eids = eids[:100]
    # dst_nodes = dst_nodes[:100]
    # src_nodes = src_nodes[:100]
    # edge_types = edge_types[:100]
    # timestamps = timestamps[:100]
    #
    # print(timestamps)
    # print(edge_types)
    conn.executemany(
        '''
        insert into edges values(?, ?, ?, ?, ?)
        ''',
        tqdm((tuple(int(i) for i in rec) for rec in
              zip(eids, src_nodes, dst_nodes, edge_types, timestamps)),
             total=len(eids)))
    conn.commit()

    conn.executescript(
        '''
        create index index_timestamp on edges(timestamp);
        create index index_ts_snode on edges(timestamp, src_node);
        create index index_ts_dnode on edges(timestamp, dst_node);
        create index index_pair on edges(timestamp, src_node, dst_node);
        create index index_trip on edges(timestamp, src_node, dst_node, edge_type);
        create index index_snode_ts on edges(src_node, timestamp);
        create index index_dnode_ts on edges(dst_node, timestamp);
        create index index_pair_ts on edges(src_node, dst_node, timestamp);
        create index index_trip_ts on edges(src_node, dst_node, edge_type, timestamp);

        '''
    )
    conn.commit()
    pass


def array2struct_array(arr):
    num = arr.shape[1]
    arr = [tuple(i) for i in arr]
    arr = np.ascontiguousarray(
        np.array(arr, np.dtype([(f'f{i}', 'int32') for i in range(num)])))

    sort_idx = np.argsort(arr, kind='stable')

    arr = arr[sort_idx]

    sort_idx = sort_idx.astype('int32')
    return arr, sort_idx


def build_index_search(
        config, eids, src_nodes, dst_nodes, edge_types, timestamps):
    output_folder = config['dataset_path']

    trip_search_index = np.vstack(
        (src_nodes, dst_nodes, edge_types, timestamps)).T
    trip_search_index, trip_search_index_map = array2struct_array(
        trip_search_index)

    np.save(os.path.join(output_folder, 'trip_search_index.npy'),
            trip_search_index)
    np.save(os.path.join(output_folder, 'trip_search_index_map.npy'),
            trip_search_index_map)

    pair_search_index = np.vstack((src_nodes, dst_nodes, timestamps)).T
    pair_search_index, pair_search_index_map = array2struct_array(
        pair_search_index)

    np.save(os.path.join(output_folder, 'pair_search_index.npy'),
            pair_search_index)
    np.save(os.path.join(output_folder, 'pair_search_index_map.npy'),
            pair_search_index_map)

    src_node_search_index = np.vstack((src_nodes, timestamps)).T
    src_node_search_index, src_node_search_index_map = array2struct_array(
        src_node_search_index)

    np.save(os.path.join(output_folder, 'src_node_search_index.npy'),
            src_node_search_index)
    np.save(os.path.join(output_folder, 'src_node_search_index_map.npy'),
            src_node_search_index_map)

    dst_node_search_index = np.vstack((dst_nodes, timestamps)).T
    dst_node_search_index, dst_node_search_index_map = array2struct_array(
        dst_node_search_index)

    np.save(os.path.join(output_folder, 'dst_node_search_index.npy'),
            dst_node_search_index)
    np.save(os.path.join(output_folder, 'dst_node_search_index_map.npy'),
            dst_node_search_index_map)

    pass


def build_index(config):
    output_folder = config['dataset_path']
    eids = np.load(os.path.join(output_folder, 'eids.npy'))
    src_nodes = np.load(os.path.join(output_folder, 'src_nodes.npy'))
    dst_nodes = np.load(os.path.join(output_folder, 'dst_nodes.npy'))
    edge_types = np.load(os.path.join(output_folder, 'edge_types.npy'))
    timestamps = np.load(os.path.join(output_folder, 'timestamps.npy'))

    build_index_triplet(output_folder, eids, src_nodes, dst_nodes, edge_types)
    build_index_triplet_bilateral(
        output_folder, eids, src_nodes, dst_nodes, edge_types)
    build_index_pair(output_folder, eids, src_nodes, dst_nodes)
    build_index_node(output_folder, eids, src_nodes, dst_nodes)

    build_index_val(config, 'val')
    build_index_val(config, 'test')

    # build_index_sqlite(
    #     config, eids, src_nodes, dst_nodes, edge_types, timestamps)

    build_index_search(
        config, eids, src_nodes, dst_nodes, edge_types, timestamps)
    pass


def build_point_data_process(param):
    eids = param
    src_nodes = global_data['src_nodes']
    dst_nodes = global_data['dst_nodes']
    edge_types = global_data['edge_types']
    timestamps = global_data['timestamps']
    triplet_index = global_data['triplet_index']
    triplet_index_bilateral = global_data['triplet_index_bilateral']
    pair_index = global_data['pair_index']
    node_index = global_data['node_index']
    train_folder = global_data['train_folder']
    train_start_ts = global_data['train_start_ts']
    config = global_data['config']

    # eids = param

    train_filename = os.path.join(
        train_folder, format(eids[0], '08d') + '.tar')

    tar_file = None
    for eid in eids:
        snode = src_nodes[eid]
        dnode = dst_nodes[eid]
        etype = edge_types[eid]

        current_ts = timestamps[eid]
        if current_ts < train_start_ts:
            continue

        target_ts = get_target_ts(
            triplet_index, eid, snode, dnode, etype, timestamps,
            config['neg_sample_interval'], config['max_train_ts'])

        history_edges_triplet = get_history_edges(
            triplet_index_bilateral, eid, src_nodes, dst_nodes, edge_types,
            timestamps, constants.MAX_HISTORY_NUM_TRIPLET
        )

        history_edges_pair = get_history_edges(
            pair_index, eid, src_nodes, dst_nodes, None, timestamps,
            constants.MAX_HISTORY_NUM_PAIR)

        history_edges_src, history_edges_dst = get_history_edges_node(
            node_index, eid, src_nodes, dst_nodes, timestamps,
            constants.MAX_HISTORY_NUM_NODE)

        point_data = PointData(
            eid, target_ts, history_edges_triplet, history_edges_pair,
            history_edges_src, history_edges_dst)

        temp_io = BytesIO()
        pickle.dump(point_data, temp_io, protocol=pickle.HIGHEST_PROTOCOL)

        buf = temp_io.getbuffer()

        if tar_file is None:
            tar_file = tarfile.open(train_filename, 'w')
            pass

        tinfo = tarfile.TarInfo(name=f'{eid:08d}.pk')
        tinfo.size = buf.nbytes
        temp_io2 = BytesIO(buf)
        tar_file.addfile(tarinfo=tinfo, fileobj=temp_io2)
        # tar_file.close()
        temp_io2.close()
        del buf
        temp_io.close()
        pass

    if tar_file is not None:
        tar_file.close()
        pass
    pass


def build_point_data(config):
    build_point_data_train(config)
    build_point_data_val(config, 'val')
    build_point_data_val(config, 'test')

    pass


def get_test_node_history_edges(eid, node, node_index, src_nodes, dst_nodes, timestamps, max_num):
    if eid != -1:
        if src_nodes[eid] == node:
            is_src = 0
        else:
            is_src = 1
            pass
        history_edges = get_history_edges_node_single(
            node_index, eid, is_src, src_nodes, dst_nodes, timestamps,
            max_num)
        history_edges = np.vstack((
            np.array([eid, is_src], dtype='int32'),
            history_edges.reshape(-1, 2)))
        pass
    else:
        history_edges = np.zeros((0, 2), dtype='int32')
        pass

    return history_edges


def build_point_data_val(config, prefix):
    output_folder = config['dataset_path']
    src_nodes = np.load(os.path.join(output_folder, 'src_nodes.npy'))
    dst_nodes = np.load(os.path.join(output_folder, 'dst_nodes.npy'))
    edge_types = np.load(os.path.join(output_folder, 'edge_types.npy'))
    timestamps = np.load(os.path.join(output_folder, 'timestamps.npy'))
    triplet_index_bilateral = np.load(os.path.join(output_folder, 'triplet_index_bilateral.npy'))
    pair_index = np.load(os.path.join(output_folder, 'pair_index.npy'))
    node_index = np.load(os.path.join(output_folder, 'node_index.npy'))

    index = np.load(os.path.join(output_folder, f'{prefix}_index.npy'))
    test_src_nodes = np.load(os.path.join(output_folder, f'{prefix}_src_nodes.npy'))
    test_dst_nodes = np.load(os.path.join(output_folder, f'{prefix}_dst_nodes.npy'))

    test_start_timestamps = np.load(
        os.path.join(output_folder, f'{prefix}_start_timestamps.npy'))
    test_end_timestamps = np.load(
        os.path.join(output_folder, f'{prefix}_end_timestamps.npy'))

    tar_file = tarfile.open(
        os.path.join(output_folder, f'{prefix}_data.tar'), 'w')

    for i in tqdm(range(len(test_src_nodes))):
        eid_triplet = index[i, 1]
        if eid_triplet != -1:
            history_edges_triplet = get_history_edges(
                triplet_index_bilateral, eid_triplet, src_nodes, dst_nodes, edge_types,
                timestamps, constants.MAX_HISTORY_NUM_TRIPLET
            )
            history_edges_triplet = np.vstack((
                np.array([eid_triplet, 0], dtype='int32'),
                history_edges_triplet.reshape(-1, 2)))
        else:
            history_edges_triplet = np.zeros((0, 2), dtype='int32')
            pass

        eid_pair = index[i, 0]
        if eid_pair != -1:
            history_edges_pair = get_history_edges(
                pair_index, eid_pair, src_nodes, dst_nodes, None, timestamps,
                constants.MAX_HISTORY_NUM_PAIR)
            history_edges_pair = np.vstack((
                np.array([eid_pair, 0], dtype='int32'),
                history_edges_pair.reshape(-1, 2)))
            pass
        else:
            history_edges_pair = np.zeros((0, 2), dtype='int32')
            pass

        history_edges_src = get_test_node_history_edges(
            index[i, 2], test_src_nodes[i], node_index, src_nodes, dst_nodes,
            timestamps, constants.MAX_HISTORY_NUM_NODE)

        history_edges_dst = get_test_node_history_edges(
            index[i, 3], test_dst_nodes[i], node_index, src_nodes, dst_nodes,
            timestamps, constants.MAX_HISTORY_NUM_NODE)

        point_data = PointDataTest(
            history_edges_triplet, history_edges_pair,
            history_edges_src, history_edges_dst,
            test_start_timestamps[i],
            test_end_timestamps[i])

        temp_io = BytesIO()
        pickle.dump(point_data, temp_io, protocol=pickle.HIGHEST_PROTOCOL)

        buf = temp_io.getbuffer()

        tinfo = tarfile.TarInfo(name=f'{i:08d}.pk')
        tinfo.size = buf.nbytes
        temp_io2 = BytesIO(buf)
        tar_file.addfile(tarinfo=tinfo, fileobj=temp_io2)
        # tar_file.close()
        temp_io2.close()
        del buf
        temp_io.close()

        pass
    tar_file.close()
    pass


def build_point_data_train(config):
    output_folder = config['dataset_path']
    eids = np.load(os.path.join(output_folder, 'eids.npy'))
    src_nodes = np.load(os.path.join(output_folder, 'src_nodes.npy'))
    dst_nodes = np.load(os.path.join(output_folder, 'dst_nodes.npy'))
    edge_types = np.load(os.path.join(output_folder, 'edge_types.npy'))
    timestamps = np.load(os.path.join(output_folder, 'timestamps.npy'))
    triplet_index = np.load(os.path.join(output_folder, 'triplet_index.npy'))
    triplet_index_bilateral = np.load(os.path.join(output_folder, 'triplet_index_bilateral.npy'))
    pair_index = np.load(os.path.join(output_folder, 'pair_index.npy'))
    node_index = np.load(os.path.join(output_folder, 'node_index.npy'))

    global global_data

    global_data = {
        'eids': eids,
        'src_nodes': src_nodes,
        'dst_nodes': dst_nodes,
        'edge_types': edge_types,
        'timestamps': timestamps,
        'triplet_index': triplet_index,
        'triplet_index_bilateral': triplet_index_bilateral,
        'pair_index': pair_index,
        'node_index': node_index,
        'train_folder': os.path.join(output_folder, 'data'),
        'train_start_ts': dt2ts(config['train_start']),
        'config': config
    }

    num_per_tar = constants.PACK_FILE_SIZE

    chunked_eids = [eids[i:i + num_per_tar] for i in
                    range(0, len(eids), num_per_tar)]
    pool = mp.Pool()
    list(pool.imap(
        build_point_data_process, tqdm(chunked_eids)))
    pool.close()
    pass


def read_tar_names(param):
    filename = param
    with tarfile.open(filename, 'r') as fin:
        names = fin.getnames()
        return names
    pass


def read_all_names_in_folder(folder):
    tar_files = [os.path.join(folder, f) for f in
                 sorted(os.listdir(folder))]

    pool = mp.Pool()
    names_list = pool.map(read_tar_names, tqdm(tar_files))
    pool.close()

    all_names = []
    for names in names_list:
        names_int = [int(n.split('.')[0]) for n in names]
        all_names.extend(names_int)
        pass

    return all_names


def build_dataset(config):
    output_folder = config['dataset_path']

    os.makedirs(output_folder, exist_ok=True)
    train_folder = os.path.join(output_folder, 'data')
    os.makedirs(train_folder, exist_ok=True)

    build_dataset_numpy(config)
    build_index(config)

    pass


if __name__ == '__main__':
    config_file = sys.argv[1]
    config = importlib.import_module(config_file).config
    build_dataset(config)
    pass
