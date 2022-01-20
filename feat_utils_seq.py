
import constants
from datetime import datetime
import numpy as np


def time_encoding(current_ts, ts):
    current_day = current_ts // constants.SECONDS_PER_DAY
    days = ts // constants.SECONDS_PER_DAY

    current_hour = current_ts // constants.SECONDS_PER_HOUR
    thours = ts // constants.SECONDS_PER_HOUR
    
    datetimes = [datetime.fromtimestamp(t) for t in ts]
    weekdays = np.asarray([t.weekday() for t in datetimes], dtype='int32')
    hours = np.asarray([t.hour for t in datetimes], dtype='int32')
    days_diff = current_day - days
    hours_diff = current_hour - thours

    days_diff = np.clip(days_diff, 0, constants.MAX_TE_DAYS_DIFF - 1)
    hours_diff = np.clip(hours_diff, 0, constants.MAX_TE_HOURS_DIFF - 1)

    return np.stack((days_diff, hours_diff, weekdays, hours)).T
    pass


def get_label_feat(label_bins, bin_size, max_history_ts, num_class):
    label_ts = label_bins * bin_size
    
    days_diff = (label_ts - max_history_ts) // constants.SECONDS_PER_DAY

    current_wd = datetime.fromtimestamp(max_history_ts).weekday()

    weekdays = (days_diff +  current_wd) % 7

    days_diff = np.clip(days_diff, 0, constants.MAX_TE_DAYS_DIFF - 1)

    hours = (label_ts // (60 * 60)) % 24
    
    return np.stack((np.clip(
        label_bins -  max_history_ts // bin_size, 0, num_class - 1), weekdays, days_diff, hours)).T
    # return np.reshape(np.clip(
    #     label_bins -  max_history_ts // bin_size, 0, num_class-1), (-1, 1))



def merge_category(feat, feat_nums):
    feat_nums_sum = np.cumsum(feat_nums)
    feat_nums_start = np.insert(feat_nums_sum, 0, 0)[:-1]

    return feat + feat_nums_start[None, :]
    
    pass

