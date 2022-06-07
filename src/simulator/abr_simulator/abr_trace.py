from typing import List

import numpy as np

from common.utils import read_json_file, write_json_file

from simulator.abr_simulator.constants import MILLISECONDS_IN_SECOND

class AbrTrace:
    def __init__(self, timestamps: List[float], bandwidths: List[float],
                 link_rtt: float, buffer_thresh: float, name: str = ""):
        """Network trace used ABR applications.

        Args
            timestamps: list of timestamps. Unit: Second.
            bandwidths: list of bandwidth values. Unit: Mbps
            link_rtt: network link base RTT. Unit: Millisecond.
            buffer_thresh: length of playback buffer in client video player.
                Unit: Second.
        """
        assert len(timestamps) == len(bandwidths)
        self.timestamps = timestamps
        self.bandwidths = bandwidths
        self.link_rtt = link_rtt
        self.buffer_thresh = buffer_thresh * MILLISECONDS_IN_SECOND
        self.name = name

    def dump(self, filename: str):
        """Save ABR trace into a json file."""
        data = {'timestamps': self.timestamps,
                'bandwidths': self.bandwidths,
                'link_rtt': self.link_rtt,
                'buffer_thresh': self.buffer_thresh,
                'name': self.name}
        write_json_file(filename, data)

    @staticmethod
    def load_from_file(filename: str):
        trace_data = read_json_file(filename)
        tr = AbrTrace(trace_data['timestamps'], trace_data['bandwidths'],
                   trace_data['link_rtt'], trace_data['buffer_thresh'],
                   trace_data['name'])
        return tr


def generate_bw_time_series(T_s, duration, min_bw, max_bw):
    """Generate a network bandwidth trace."""
    max_bw_low = max(max_bw-50, 1)
    max_bw = round(np.random.uniform(max_bw_low ,max_bw))
    flag = np.random.randint(0, 1)
    if flag == 0:
        min_bw = min_bw
    else:
        min_bw = round(np.random.uniform(min_bw, max_bw * 0.6), 2)
    round_digit = 2
    ts = 0 # timestamp
    cnt = 0
    trace_time = []
    trace_bw = []
    assert min_bw is not None
    assert max_bw is not None
    last_val = round(np.random.uniform(min_bw, max_bw), round_digit)

    while ts < duration:
        if cnt <= 0:
            bw_val = round(np.random.uniform(min_bw, max_bw), round_digit)
            if T_s + 1 == 1:
                cnt = np.random.randint(1, T_s + 1)
            else:
                cnt = 1

        elif cnt >= 1:
            bw_val = last_val
        else:
            bw_val = round(np.random.uniform(min_bw, max_bw), round_digit)

        cnt -= 1
        last_val = bw_val
        time_noise = np.random.uniform(0.1, 3.5)
        ts += time_noise
        ts = round(ts, 2)
        trace_time.append(ts)
        trace_bw.append(bw_val)

    return trace_time, trace_bw


def generate_trace(bw_change_interval, duration, min_bw, max_bw, link_rtt, buffer_thresh):
    trace_time, trace_bw = generate_bw_time_series(
        bw_change_interval, duration, min_bw, max_bw)
    return AbrTrace(trace_time, trace_bw, link_rtt, buffer_thresh)


def generate_trace_from_config(config) -> AbrTrace:
    weight_sum = 0
    weights = []
    for env_config in config:
        weight_sum += env_config['weight']
        weights.append(env_config['weight'])
    assert round(weight_sum, 1) == 1.0
    indices_sorted = sorted(range(len(weights)), key=weights.__getitem__)
    weights_sorted = sorted(weights)
    weight_cumsums = np.cumsum(np.array(weights_sorted))

    rand_num = float(np.random.uniform(0, 1))

    for i, weight_cumsum in zip(indices_sorted, weight_cumsums):
        if rand_num <= float(weight_cumsum):
            env_config = config[i]

            return generate_trace_from_ranges(
                    env_config['bw_change_interval'],
                    env_config['min_bw'],
                    env_config['max_bw'],
                    env_config['link_rtt'],
                    env_config['buffer_thresh'],
                    env_config['duration'])
    raise ValueError("This line should never be reached.")


def generate_trace_from_ranges(bw_change_interval_range, min_bw_range,
        max_bw_range, link_rtt_range, buffer_thresh_range, duration):
    assert len(bw_change_interval_range) == 2 and \
            bw_change_interval_range[0] <= bw_change_interval_range[1]
    assert len(min_bw_range) == 2 and min_bw_range[0] <= min_bw_range[1]
    assert len(max_bw_range) == 2 and max_bw_range[0] <= max_bw_range[1]
    assert len(link_rtt_range) == 2 and link_rtt_range[0] <= link_rtt_range[1]
    assert len(buffer_thresh_range) == 2 and \
            buffer_thresh_range[0] <= buffer_thresh_range[1]

    if bw_change_interval_range[0] == bw_change_interval_range[1]:
        bw_change_interval = bw_change_interval_range[0]
    else:
        bw_change_interval = np.random.uniform(
            bw_change_interval_range[0], bw_change_interval_range[1])

    if min_bw_range[0] == min_bw_range[1]:
        min_bw = min_bw_range[0]
    else:
        min_bw = np.random.uniform(min_bw_range[0], min_bw_range[1])
    max_bw = np.exp(np.random.uniform(np.log(max_bw_range[0]), np.log(max_bw_range[1])))

    if link_rtt_range[0] == link_rtt_range[1]:
        link_rtt = link_rtt_range[0]
    else:
        link_rtt = np.random.uniform(link_rtt_range[0], link_rtt_range[1])
    if buffer_thresh_range[0] == buffer_thresh_range[1]:
        buffer_thresh = buffer_thresh_range[0]
    else:
        buffer_thresh = np.random.uniform(buffer_thresh_range[0], buffer_thresh_range[1])
    return generate_trace(bw_change_interval, duration, min_bw, max_bw, link_rtt,
                          buffer_thresh)

def generate_trace_from_config_file(config_file: str):
    config = read_json_file(config_file)
    return generate_trace_from_config(config)
