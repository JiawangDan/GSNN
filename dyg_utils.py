import numpy as np


def get_neighbor_finder(data, uniform, max_node_idx=None):
    max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
    adj_list = [[] for _ in range(max_node_idx + 1)]
    for source, destination, edge_idx,edge_type, timestamp, train_flag in zip(data.sources, data.destinations,
                                                      data.edge_idxs, data.edge_type,
                                                      data.st_timestamps, data.train_flag):
        if train_flag==1:
            adj_list[source].append((destination, edge_idx, edge_type, timestamp))
            adj_list[destination].append((source, edge_idx, edge_type, timestamp))

    return NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:
    def __init__(self, adj_list, uniform=False, seed=None):
        self.node_to_neighbors = []
        self.node_to_edge_idxs = []
        self.node_to_edge_timestamps = []
        self.node_to_edge_type = []

        for neighbors in adj_list:
            # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
            # We sort the list based on timestamp
            sorted_neighhbors = sorted(neighbors, key=lambda x: x[3])
            self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
            self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
            self.node_to_edge_type.append(np.array([x[2] for x in sorted_neighhbors]))
            self.node_to_edge_timestamps.append(np.array([x[3] for x in sorted_neighhbors]))

        self.uniform = uniform

        if seed is not None:
          self.seed = seed
          self.random_state = np.random.RandomState(self.seed)


    def find_before(self, src_idx, cut_time):
        """
        Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

        Returns 3 lists: neighbors, edge_idxs, timestamps

        """
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

        if i > 20:
            start = i-20
        else:
            start=0

        return self.node_to_neighbors[src_idx][start:i], self.node_to_edge_idxs[src_idx][start:i], self.node_to_edge_timestamps[src_idx][start:i]

    def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
        """
        Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        #print(source_nodes)
        #print(timestamps)
        assert (len(source_nodes) == len(timestamps))

        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        # NB! All interactions described in these matrices are sorted in each row by time

        neighbors = np.zeros((len(source_nodes), tmp_n_neighbors), dtype=np.int32)
        edge_times = np.zeros((len(source_nodes), tmp_n_neighbors), dtype=np.float32)
        edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors), dtype=np.int32)

        for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
            source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

            if len(source_neighbors) > 0 and n_neighbors > 0:
                if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
                    sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

                    neighbors[i, :] = source_neighbors[sampled_idx]
                    edge_times[i, :] = source_edge_times[sampled_idx]
                    edge_idxs[i, :] = source_edge_idxs[sampled_idx]

                    # re-sort based on time
                    pos = edge_times[i, :].argsort()
                    neighbors[i, :] = neighbors[i, :][pos]
                    edge_times[i, :] = edge_times[i, :][pos]
                    edge_idxs[i, :] = edge_idxs[i, :][pos]
                else:
                    # Take most recent interactions
                    source_edge_times = source_edge_times[-n_neighbors:]
                    source_neighbors = source_neighbors[-n_neighbors:]
                    source_edge_idxs = source_edge_idxs[-n_neighbors:]

                    assert (len(source_neighbors) <= n_neighbors)
                    assert (len(source_edge_times) <= n_neighbors)
                    assert (len(source_edge_idxs) <= n_neighbors)

                    neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
                    edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
                    edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

        return neighbors, edge_idxs, edge_times