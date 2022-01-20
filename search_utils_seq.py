import numpy as np
import os
import time


def get_history_edges(src_node, dst_node,
                      index, eid, src_nodes, dst_nodes, edge_types, timestamps, max_num):
    key = (src_node, dst_node)
    reverse_key = (dst_node, src_node)

    if edge_types is not None:
        key += (edge_types[eid],)
        reverse_key += (edge_types[eid],)
        pass

    # pre_eid = index[eid]
    pre_eid = eid

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

    if len(history_edges) == 0:
        return np.zeros((0, 2), dtype='int32')
    else:
        return np.array(history_edges, dtype='int32')
    pass


def search_tuple(t, search_index, search_index_map, shift=-1):
    dim = len(t)

    v = np.array(
        t, dtype=np.dtype([(f'f{i}', 'int32') for i in range(dim)]))

    # print('>> ', t)
    # idx = np.searchsorted(search_index, v, side='right') + shift
    idx = np.searchsorted(search_index, v, side='left') + shift
    idx = max(idx, 0)
    idx = min(idx, len(search_index) - 1)
    # print('<< ', search_index[idx])
    # print('<< ', search_index[idx+1])

    if t[:-1] != tuple(search_index[idx])[:-1]:
        return None
    else:
        return search_index_map[idx]
    pass


def get_max_exclue_none(values):
    values = [v for v in values if v is not None]

    if len(values) == 0:
        return None

    return max(values)


def get_trip_history(
        src_node, dst_node, edge_type, timestamp, trip_index_bilateral,
        trip_search_index, trip_search_index_map, src_nodes, dst_nodes,
        edge_types, timestamps, limit):
    eid1 = search_tuple(
        (src_node, dst_node, edge_type, timestamp),
        trip_search_index, trip_search_index_map)

    eid2 = search_tuple(
        (dst_node, src_node, edge_type, timestamp),
        trip_search_index, trip_search_index_map)
    # print(eid1, eid2)
    eid = get_max_exclue_none((eid1, eid2))
    if eid is None:
        return np.zeros((0, 2), dtype='int32')

    edges = get_history_edges(src_node, dst_node,
                              trip_index_bilateral, eid, src_nodes, dst_nodes, edge_types,
                              timestamps, limit)

    # edges = np.vstack((np.array([[eid, 0]], dtype='int32'), edges))

    return edges


def get_pair_history(
        src_node, dst_node, timestamp, pair_index, pair_search_index,
        pair_search_index_map, src_nodes, dst_nodes, timestamps, limit):
    eid1 = search_tuple(
        (src_node, dst_node, timestamp),
        pair_search_index, pair_search_index_map)

    eid2 = search_tuple(
        (dst_node, src_node, timestamp),
        pair_search_index, pair_search_index_map)

    eid = get_max_exclue_none((eid1, eid2))

    if eid is None:
        return np.zeros((0, 2), dtype='int32')

    edges = get_history_edges(src_node, dst_node,
                              pair_index, eid, src_nodes, dst_nodes, None,
                              timestamps, limit)

    # edges = np.vstack((np.array([[eid, 0]], dtype='int32'), edges))

    return edges


def get_history_edges_node_single(
        index, eid, i, src_nodes, dst_nodes, timestamps, max_num):
    history_edges = []
    # pre_eid = index[eid, i]
    pre_eid = eid
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

    if len(history_edges) == 0:
        return np.zeros((0, 2), dtype='int32')
    else:
        return np.array(history_edges, dtype='int32')


def get_node_history(
        node, timestamp, node_index, src_node_search_index,
        src_node_search_index_map, dst_node_search_index,
        dst_node_search_index_map, src_nodes, dst_nodes, timestamps, limit):
    eid1 = search_tuple(
        (node, timestamp), src_node_search_index, src_node_search_index_map)

    eid2 = search_tuple(
        (node, timestamp), dst_node_search_index, dst_node_search_index_map)

    eid = get_max_exclue_none((eid1, eid2))

    if eid is None:
        return np.zeros((0, 2), dtype='int32')

    if src_nodes[eid] == node:
        is_dst = 0
    else:
        is_dst = 1
        pass

    edges = get_history_edges_node_single(
        node_index, eid, is_dst, src_nodes, dst_nodes, timestamps, limit)

    # edges = np.vstack((np.array([[eid, is_dst]], dtype='int32'), edges))

    return edges


if __name__ == '__main__':
    output_folder = '../dataset_A'
    eids = np.load(os.path.join(output_folder, 'eids.npy'))
    src_nodes = np.load(os.path.join(output_folder, 'src_nodes.npy'))
    dst_nodes = np.load(os.path.join(output_folder, 'dst_nodes.npy'))
    edge_types = np.load(os.path.join(output_folder, 'edge_types.npy'))
    timestamps = np.load(os.path.join(output_folder, 'timestamps.npy'))
    trip_index_bilateral = np.load(os.path.join(
        output_folder, 'triplet_index_bilateral.npy'))
    trip_search_index = np.load(os.path.join(
        output_folder, 'trip_search_index.npy'))
    trip_search_index_map = np.load(os.path.join(
        output_folder, 'trip_search_index_map.npy'))

    pair_index = np.load(os.path.join(
        output_folder, 'pair_index.npy'))
    pair_search_index = np.load(os.path.join(
        output_folder, 'pair_search_index.npy'))
    pair_search_index_map = np.load(os.path.join(
        output_folder, 'pair_search_index_map.npy'))

    node_index = np.load(os.path.join(
        output_folder, 'node_index.npy'))
    src_node_search_index = np.load(os.path.join(
        output_folder, 'src_node_search_index.npy'))
    src_node_search_index_map = np.load(os.path.join(
        output_folder, 'src_node_search_index_map.npy'))
    dst_node_search_index = np.load(os.path.join(
        output_folder, 'dst_node_search_index.npy'))
    dst_node_search_index_map = np.load(os.path.join(
        output_folder, 'dst_node_search_index_map.npy'))

    start = time.time()
    eids = get_trip_history(
        507, 17549, 151, 1497473624,
        trip_index_bilateral,
        trip_search_index,
        trip_search_index_map,
        src_nodes,
        dst_nodes,
        edge_types,
        timestamps,
        64)
    print(eids)
    print(time.time() - start)

    start = time.time()
    eids = get_pair_history(
        507, 17549,
        1497473624,
        pair_index,
        pair_search_index,
        pair_search_index_map,
        src_nodes,
        dst_nodes,
        timestamps,
        64
    )
    print(eids)
    print(time.time() - start)

    start = time.time()
    eids = get_node_history(
        507,
        1497473624,
        node_index,
        src_node_search_index,
        src_node_search_index_map,
        dst_node_search_index,
        dst_node_search_index_map,
        src_nodes,
        dst_nodes,
        timestamps,
        64)
    print(eids)
    print(len(eids))
    print(time.time() - start)
    pass

