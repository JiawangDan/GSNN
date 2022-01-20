# from collections import namedtuple

# SequenceStatus = namedtuple(
#     'SequenceStatus', ['current_index', 'seq', 'direction'],
#     defaults=[None, None, None])

class SequenceStatus(object):
    __slots__ = ('current_index', 'seq')

    def __init__(self,):
        self.current_index = 0
        self.seq = list()
        pass


class PointData(object):
    __slots__ = (
        'src_node', 'dst_node', 'edge_type', 'max_history_ts', 'max_future_ts',
        'target_ts', 'history_edges_triplet',
        'history_edges_pair', 'history_edges_src', 'history_edges_dst'
    )

    def __init__(
            self, src_node, dst_node, edge_type, max_history_ts, max_future_ts,
            target_ts, history_edges_triplet, history_edges_pair,
            history_edges_src, history_edges_dst):
        self.src_node = src_node
        self.dst_node = dst_node
        self.edge_type = edge_type
        self.max_history_ts = max_history_ts
        self.max_future_ts = max_future_ts
        self.target_ts = target_ts
        self.history_edges_triplet = history_edges_triplet
        self.history_edges_pair = history_edges_pair
        self.history_edges_src = history_edges_src
        self.history_edges_dst = history_edges_dst
        pass

    pass


class PointDataTest(object):
    __slots__ = (
        'src_node', 'dst_node', 'edge_type', 'max_history_ts',
        'history_edges_triplet', 'history_edges_pair',
        'history_edges_src', 'history_edges_dst',
        'start_timestamp', 'end_timestamp')

    def __init__(
            self, src_node, dst_node, edge_type, max_history_ts,
            history_edges_triplet, history_edges_pair,
            history_edges_src, history_edges_dst,
            start_timestamp, end_timestamp):
        self.src_node = src_node
        self.dst_node = dst_node
        self.edge_type = edge_type
        self.max_history_ts = max_history_ts
        self.history_edges_triplet = history_edges_triplet
        self.history_edges_pair = history_edges_pair
        self.history_edges_src = history_edges_src
        self.history_edges_dst = history_edges_dst
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        pass

