

import numpy as np
import constants


def get_label(
        positive_ts, max_history_ts, max_future_ts, num, bin_size):
    
    max_history_bin = max_history_ts // bin_size
    max_future_bin = max_future_ts // bin_size

    positive_bins = np.unique(positive_ts // bin_size)

    if len(positive_bins) >= num:
        return positive_bins[:num], np.ones(num, dtype='float32')

    pos_num = len(positive_bins)
    neg_num = num - pos_num
    all_bins = np.arange(max_history_bin, max_future_bin + 1)
    negative_bins = np.setdiff1d(all_bins, positive_bins)

    if neg_num > len(negative_bins):
        neg_num = len(negative_bins)
        pass

    negative_bins = np.random.choice(negative_bins, size=neg_num, replace=False)

    label = np.concatenate((
        np.ones(len(positive_bins), dtype='float32'),
        np.zeros(len(negative_bins), dtype='float32')))
    label_bins = np.concatenate((positive_bins, negative_bins))

    return label_bins, label


def get_predict_bins(start_ts, end_ts, bin_size):
    start_bin = start_ts // bin_size
    end_bin = end_ts // bin_size + 1

    bins = np.arange(start_bin, end_bin)
    weights = np.ones(len(bins))

    if len(bins) > 1:
        weights[0] = ((start_bin + 1) * bin_size - start_ts) / bin_size
        weights[-1] = (end_ts - bins[-1] * bin_size) / bin_size
        pass
    else:
        weights[0] = 1
        pass

    return bins, weights
    pass
