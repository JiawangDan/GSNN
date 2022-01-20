import numpy as np
import copy

def get_label(current_ts, max_past_time, min_future_time, edge_type, edge_type_list, bin_size, neg_interval, neg_num):
    current_bin = current_ts // bin_size + 1

    candidate_time_list = np.concatenate((np.arange((current_ts - neg_interval/2) // bin_size, current_bin ),
                                         np.arange(current_bin+1, (current_ts + neg_interval/2)// bin_size)) )

    candidate_edgetype_list = copy.deepcopy(edge_type_list)
    candidate_edgetype_list.discard(edge_type)
    candidate_edgetype_list = np.array(list(candidate_edgetype_list))



    candidate_time_list = candidate_time_list * bin_size
    if min_future_time > 0:
        require_idx = np.where(candidate_time_list > max_past_time) and np.where(candidate_time_list < min_future_time)
    else:
        require_idx = np.where(candidate_time_list > max_past_time)
    candidate_time_list = candidate_time_list[require_idx]
    candidate_time_list = candidate_time_list - 1 # 当天最后一秒
    current_time_out = current_bin * bin_size - 1 # 当天最后一秒

    neg_labels = np.zeros(len(candidate_time_list), dtype='float32')

    if len(candidate_time_list) > neg_num:
        selected_bins = np.random.choice(
            np.arange(len(candidate_time_list)),
            size=neg_num, replace=False)

        candidate_time_list = candidate_time_list[selected_bins]
        neg_labels = neg_labels[selected_bins]
        pass



    #再插入一个类型不同的负样本
    edge_type_out = np.tile(edge_type, len(neg_labels))

    time_out = np.insert(candidate_time_list, 0, current_time_out)
    label = np.insert(neg_labels, 0, 0)
    neg_type = np.random.choice(candidate_edgetype_list, size=1, replace=False)
    edge_type_out = np.insert(edge_type_out, 0, neg_type)


    #最后插入一个正样本
    time_out = np.insert(time_out, 0, current_time_out)
    label = np.insert(label, 0, 1)
    edge_type_out = np.insert(edge_type_out, 0, edge_type)

    return time_out, label, edge_type_out


def get_train_label(current_ts, max_past_time, target_ts, edge_type, edge_type_list,
                    bin_size, neg_interval, neg_num, randsampler, ngh_finder, neighbors_num):
    current_bin = current_ts // bin_size + 1

    ## 可选的时间范围、 可选边类型
    candidate_edgetype_list = copy.deepcopy(edge_type_list)
    candidate_edgetype_list.discard(edge_type)
    candidate_edgetype_list = np.array(list(candidate_edgetype_list))

    candidate_time_bin_list = np.concatenate((np.arange((current_ts - neg_interval) // bin_size, current_bin),
                                              np.arange(current_bin + 1, (current_ts + neg_interval) // bin_size)))

    # 正样本时间
    target_ts = target_ts[target_ts > max_past_time]
    target_ts = target_ts[target_ts < (current_ts + neg_interval)]
    target_bin = target_ts // bin_size + 1

    #排除非负、超越时间的样本
    candidate_time_bin_list = candidate_time_bin_list[candidate_time_bin_list >= (max_past_time // bin_size)]
    require_idx = np.where(candidate_time_bin_list in target_bin)
    #candidate_time_bin_list = candidate_time_bin_list[require_idx]
    candidate_time_bin_list = np.delete(candidate_time_bin_list, require_idx)
    candidate_time_list = candidate_time_bin_list * bin_size - 1
    current_time_out = current_bin * bin_size - 1

    # 标签修正
    neg_num = np.maximum(len(target_bin), neg_num)
    neg_labels = np.zeros(len(candidate_time_list), dtype='float32')

    if len(candidate_time_list) > neg_num:
        selected_bins = np.random.choice(
            np.arange(len(candidate_time_list)),
            size=neg_num, replace=False)

        candidate_time_list = candidate_time_list[selected_bins]
        neg_labels = neg_labels[selected_bins]
        pass

    # for times_ind in range(len(candidate_time_list)):
    #     if ( (candidate_time_list[times_ind]+ 1) // bin_size) in target_bin:
    #         neg_labels[times_ind] = 1
    #融合所有正样本
    candidate_time_list = np.concatenate([candidate_time_list, target_bin*bin_size - 1], axis=0)
    neg_labels = np.concatenate([neg_labels, [1]*len(target_bin)], axis=0)

    # 类型不同的负样本
    edge_type_out = np.tile(edge_type, len(neg_labels))
    neg_type_num = 1
    time_out = np.insert(candidate_time_list, 0, [current_time_out] * neg_type_num)
    label = np.insert(neg_labels, 0, [0] * neg_type_num)
    neg_type = np.random.choice(candidate_edgetype_list, size=neg_type_num, replace=False)
    edge_type_out = np.insert(edge_type_out, 0, neg_type)

    #不同节点的用户子图构建
    neg_num_for_dest = 1
    _, negatives_batch = randsampler.sample(neg_num_for_dest)
    neighbor_one_hop = ngh_finder.get_temporal_neighbor(negatives_batch, [max_past_time] * neg_num_for_dest, n_neighbors=neighbors_num)
    one_hop_source = neighbor_one_hop[0].reshape([-1])
    one_hop_idx = neighbor_one_hop[1].reshape([-1])
    one_hop_ts = neighbor_one_hop[2].reshape([-1])
    neighbor_two_hop = ngh_finder.get_temporal_neighbor(one_hop_source, one_hop_ts, n_neighbors=neighbors_num)

    two_hop_source = neighbor_two_hop[0].reshape([-1, neighbors_num * neighbors_num])
    two_hop_idx = neighbor_two_hop[1].reshape([-1, neighbors_num * neighbors_num])
    two_hop_ts = neighbor_two_hop[2].reshape([-1, neighbors_num * neighbors_num])
    idx = np.zeros([neg_num_for_dest, 1]) - 1
    full_hop_info = np.concatenate([idx, neighbor_one_hop[0], two_hop_source, neighbor_one_hop[1], two_hop_idx
                                       , neighbor_one_hop[2], two_hop_ts], axis=1)


    return time_out, label, edge_type_out, negatives_batch, full_hop_info


def get_test_label(start_ts, end_ts, bin_size, labels):
    prediction_time_list = np.arange(start_ts  // bin_size + 1, end_ts // bin_size + 1)
    if len(prediction_time_list) >= 1:
        prediction_time_list = prediction_time_list
    else:
        prediction_time_list = start_ts // bin_size

    candidate_time_list = prediction_time_list * bin_size - 1  # 当天最后一秒
    candidate_time_list = np.reshape(candidate_time_list, [-1])
    label_list = np.tile(labels.reshape([-1]), np.shape(candidate_time_list)[0])
    label_list = label_list.astype(np.float32)

    return candidate_time_list, label_list


class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:

      src_index = self.random_state.randint(0, len(self.src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]