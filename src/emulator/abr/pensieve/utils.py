import json
import math
import os
import random
import re

import numpy as np
from numba import jit

from pensieve.constants import (HD_REWARD, M_IN_K, REBUF_PENALTY,
                                SMOOTH_PENALTY, VIDEO_BIT_RATE)

NAMES = ['timestamp', 'bandwidth']


def load_traces(trace_dir):
    trace_files = sorted(os.listdir(trace_dir))
    all_ts = []
    all_bw = []
    all_file_names = []
    for trace_file in trace_files:
        # if trace_file in ["ferry.nesoddtangen-oslo-report.2011-02-01_1000CET.log", "trace_32551_http---www.amazon.com", "trace_5294_http---www.youtube.com", "trace_5642_http---www.youtube.com"]:
        #     continue
        if trace_file.startswith("."):
            continue
        file_path = os.path.join(trace_dir, trace_file)
        ts_list = []
        bw_list = []
        with open(file_path, 'r') as f:
            for line in f:
                if len(line.split()) > 2:
                    ts, bw, _ = line.split()
                else:
                    ts, bw = line.split()
                ts = float(ts)
                bw = float(bw)
                ts_list.append(ts)
                bw_list.append(bw)
        all_ts.append(ts_list)
        all_bw.append(bw_list)
        all_file_names.append(trace_file)

    return all_ts, all_bw, all_file_names


def adjust_traces(all_ts, all_bw, bw_noise=0, duration_factor=1):
    new_all_bw = []
    new_all_ts = []
    for trace_ts, trace_bw in zip(all_ts, all_bw):
        duration = trace_ts[-1]
        new_duration = duration_factor * duration
        new_trace_ts = []
        new_trace_bw = []
        for i in range(math.ceil(duration_factor)):
            for t, bw in zip(trace_ts, trace_bw):
                if (t + i * duration) <= new_duration:
                    new_trace_ts.append(t + i * duration)
                    new_trace_bw.append(bw+bw_noise)

        new_all_ts.append(new_trace_ts)
        new_all_bw.append(new_trace_bw)
    assert len(new_all_ts) == len(all_ts)
    assert len(new_all_bw) == len(all_bw)
    return new_all_ts, new_all_bw


def compute_cdf(data):
    """ Return the cdf of input data.

    Args
        data(list): a list of numbers.

    Return
        sorted_data(list): sorted list of numbers.

    """
    length = len(data)
    sorted_data = sorted(data)
    cdf = [i / length for i, val in enumerate(sorted_data)]
    return sorted_data, cdf


def adjust_traces_one_random(all_ts, all_bw, random_seed, robust_noise, sample_length):
    adjust_n_random_traces(all_ts, all_bw, random_seed,
                           robust_noise, sample_length, number_pick=1)
    # new_all_bw = all_bw.copy()
    # new_all_ts = all_ts.copy()
    # np.random.seed(random_seed)
    #
    # number_of_traces = len(all_ts)
    # random_trace_index = random.randint(0, number_of_traces - 1)
    # trace_bw = new_all_bw[random_trace_index]
    #
    # ########
    # # use your randomization code from the notebook on new_all_bw
    # ########
    # start_index = random.randint( 0, len( trace_bw ) - sample_length )
    # sublist = trace_bw[start_index: start_index + sample_length]
    # trace_bw[start_index:start_index + sample_length] = [i * float(1+robust_noise) for i in sublist]
    #
    # assert len(new_all_ts) == len(all_ts)
    # assert len(new_all_bw) == len(all_bw)
    #
    # return new_all_ts, new_all_bw


def adjust_n_random_traces(all_ts, all_bw, random_seed, robust_noise, sample_length, number_pick):
    new_all_bw = all_bw.copy()
    new_all_ts = all_ts.copy()
    random.seed(random_seed)
    np.random.seed(random_seed)

    number_of_traces = len(all_ts)

    # we need n random index numbers from the set
    # do this n times
    random_trace_indices = random.sample(
        range(0, number_of_traces - 1), number_pick)

    for ri in random_trace_indices:
        trace_bw = new_all_bw[ri]

        start_index = random.randint(0, len(trace_bw) - sample_length)
        sublist = trace_bw[start_index: start_index + sample_length]
        new_sublist = []
        for i in sublist:
            # add constant noise
            # i = i*float(1+robust_noise)
            # if i + robust_noise > 0:
            #     i = i + robust_noise
            # else:
            #     i = i
            # new_sublist.append(i)

            # add normal noise
            noise = np.random.normal(0, 0.1, 1)
            if noise < -0.5 or noise > 0.5:
                noise = 0
            delta = 1 + float(noise)
            new_sublist.append(i * delta)

        trace_bw[start_index:start_index + sample_length] = new_sublist

    assert len(new_all_ts) == len(all_ts)
    assert len(new_all_bw) == len(all_bw)

    return new_all_ts, new_all_bw


@jit(nopython=True)
def linear_reward(current_bitrate: int, last_bitrate: int, rebuffer: float):
    """ Return linear QoE metric.

    Args
        current_bitrate: current bit rate (kbps).
        last_bitrate: previous bit rate (kbps).
        rebuffer: rebuffering time (second).
    """
    reward = current_bitrate / M_IN_K - REBUF_PENALTY * rebuffer - \
        SMOOTH_PENALTY * np.abs(current_bitrate - last_bitrate) / M_IN_K
    return reward


def opposite_linear_reward(current_bitrate_idx, last_bitrate_idx, rebuffer):
    """Return linear reward which encourages rebuffering and bitrate switching.
    """
    current_bitrate = VIDEO_BIT_RATE[current_bitrate_idx]
    last_bitrate = VIDEO_BIT_RATE[last_bitrate_idx]
    reward = current_bitrate / M_IN_K + REBUF_PENALTY * rebuffer + \
        SMOOTH_PENALTY * np.abs(current_bitrate - last_bitrate) / M_IN_K
    return reward


def log_scale_reward(current_bitrate_idx, last_bitrate_idx, rebuffer):
    current_bitrate = VIDEO_BIT_RATE[current_bitrate_idx]
    last_bitrate = VIDEO_BIT_RATE[last_bitrate_idx]
    log_bit_rate = np.log(current_bitrate / VIDEO_BIT_RATE[-1])
    log_last_bit_rate = np.log(last_bitrate / VIDEO_BIT_RATE[-1])
    reward = log_bit_rate - REBUF_PENALTY * rebuffer - SMOOTH_PENALTY * \
        np.abs(log_bit_rate - log_last_bit_rate)
    return reward


def hd_reward(current_bitrate_idx, last_bitrate_idx, rebuffer):
    reward = HD_REWARD[current_bitrate_idx] - \
        REBUF_PENALTY * rebuffer - SMOOTH_PENALTY * \
        np.abs(HD_REWARD[current_bitrate_idx] - HD_REWARD[last_bitrate_idx])
    return reward


def natural_sort(arr):
    """Natural sort list."""
    def convert(text): return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(arr, key=alphanum_key)


def construct_bitrate_chunksize_map(video_size_file_dir):
    """Construct a dict mapping bitrate to video chunk size."""
    video_size = {}  # in bytes
    for bitrate in range(len(VIDEO_BIT_RATE)):
        video_size[bitrate] = []
        video_size_file = os.path.join(video_size_file_dir,
                                       'video_size_{}'.format(bitrate))
        with open(video_size_file, 'r') as f:
            for line in f:
                video_size[bitrate].append(int(line.split()[0]))
    return video_size


def read_json_file(filename):
    """Load json object from a file."""
    with open(filename, 'r') as f:
        content = json.load(f)
    return content


def write_json_file(filename, content):
    """Dump into a json file."""
    with open(filename, 'w') as f:
        json.dump(content, f, indent=4)
