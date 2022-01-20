from common_structures_seq import PointData
import constants
import numpy as np
import search_utils_seq


def get_positive_ts(
        triplet_index, start_eid, max_history_ts, timestamps, interval):
    if start_eid is None:
        return np.array([], dtype='int32')

    first_ts = None
    next_eid = start_eid
    positive_ts = []
    while next_eid != -1:
        ts = timestamps[next_eid]

        if ts > max_history_ts:
            if first_ts is not None:
                ts_diff = ts - first_ts
                if ts_diff > interval:
                    break
                pass
            else:
                first_ts = ts
                pass

            positive_ts.append(ts)
            pass

        next_eid = triplet_index[0, next_eid]
        pass

    return np.array(positive_ts, dtype='int32')


def get_max_history_ts(edges_list, timestamps):
    max_ts = -1

    for edges in edges_list:
        if len(edges) == 0:
            continue

        max_ts = max(max_ts, np.max(timestamps[edges[:, 0]]))
        pass

    return max_ts


def get_point_data(
        current_sample, src_nodes, dst_nodes, edge_types, timestamps, config,
        triplet_index, triplet_index_bilateral, pair_index, node_index,
        trip_search_index, trip_search_index_map, pair_search_index,
        pair_search_index_map, src_node_search_index,
        src_node_search_index_map, dst_node_search_index,
        dst_node_search_index_map) \
        -> PointData:
    current_src_node, current_dst_node, current_edge_type, current_ts = \
        current_sample
    history_edges_triplet = search_utils_seq.get_trip_history(
        current_src_node,
        current_dst_node,
        current_edge_type,
        current_ts,
        triplet_index_bilateral,
        trip_search_index,
        trip_search_index_map,
        src_nodes,
        dst_nodes,
        edge_types,
        timestamps,
        constants.MAX_HISTORY_NUM_TRIPLET)

    history_edges_pair = search_utils_seq.get_pair_history(
        current_src_node,
        current_dst_node,
        current_ts,
        pair_index,
        pair_search_index,
        pair_search_index_map,
        src_nodes,
        dst_nodes,
        timestamps,
        constants.MAX_HISTORY_NUM_PAIR)

    history_edges_src = search_utils_seq.get_node_history(
        current_src_node,
        current_ts,
        node_index,
        src_node_search_index,
        src_node_search_index_map,
        dst_node_search_index,
        dst_node_search_index_map,
        src_nodes,
        dst_nodes,
        timestamps,
        constants.MAX_HISTORY_NUM_NODE)
    history_edges_dst = search_utils_seq.get_node_history(
        current_dst_node,
        current_ts,
        node_index,
        src_node_search_index,
        src_node_search_index_map,
        dst_node_search_index,
        dst_node_search_index_map,
        src_nodes,
        dst_nodes,
        timestamps,
        constants.MAX_HISTORY_NUM_NODE)

    max_history_ts = get_max_history_ts(
        [history_edges_triplet, history_edges_pair, history_edges_src,
         history_edges_dst],
        timestamps)


    # if len(history_edges_triplet) > 0:
    #     pos_search_start = history_edges_triplet[0, 0]
    # else:
    pos_search_start = search_utils_seq.search_tuple(
        (current_src_node, current_dst_node, current_edge_type,
         current_ts), trip_search_index, trip_search_index_map, shift=0)

    if max_history_ts == -1:
        #raise RuntimeError('program logical error')
        max_history_ts = timestamps[pos_search_start] - config['neg_sample_interval']
    # if pos_search_start is None:
    #     # there is no postive label for current sample
    #     pass
    # pass

    target_ts = get_positive_ts(
        triplet_index, pos_search_start, max_history_ts, timestamps,
        config['neg_sample_interval'])

    if len(target_ts) > 0:
        max_future_ts = target_ts[0] + config['neg_sample_interval']
    else:
        max_future_ts = config['max_train_ts']
        pass

    point_data = PointData(
        current_src_node, current_dst_node, current_edge_type, max_history_ts,
        max_future_ts, target_ts, history_edges_triplet, history_edges_pair,
        history_edges_src, history_edges_dst)

    return point_data