import itertools
import multiprocessing as mp

import numpy as np
from numba import jit

from pensieve.agent_policy import BaseAgentPolicy
from pensieve.constants import (B_IN_MB, DEFAULT_QUALITY, M_IN_K,
                                MILLISECONDS_IN_SECOND, VIDEO_BIT_RATE,
                                VIDEO_CHUNK_LEN)
from pensieve.utils import linear_reward

CHUNK_COMBO_OPTIONS = np.array(
            [combo for combo in itertools.product(
                range(len(VIDEO_BIT_RATE)), repeat=5)])

def next_possible_bitrates(br):
    next_brs = [br - 1 ,br ,br + 1]
    next_brs = [a for a in next_brs if 0 <= a <= 5]
    return next_brs

def calculate_jump_action_combo(br):
    all_combos = CHUNK_COMBO_OPTIONS
    combos = np.empty((0, 5), np.int64)
    #combos = np.expand_dims( combos ,axis=0 )
    for combo in all_combos:
        br1 = combo[0]
        if br1 in next_possible_bitrates( br ):
            br2 = combo[1]
            if br2 in next_possible_bitrates( br1 ):
                br3 = combo[2]
                if br3 in next_possible_bitrates( br2 ):
                    br4 = combo[3]
                    if br4 in next_possible_bitrates( br3 ):
                        br5 = combo[4]
                        if br5 in next_possible_bitrates( br4 ):
                            combo = np.expand_dims( combo ,axis=0 )
                            combos = np.append(combos, combo, axis=0)

    return combos

class FastMPC():
    """Naive implementation of RobustMPC."""

    def __init__(self, mpc_future_chunk_cnt=4, accumulate_past_error=False):
        self.mpc_future_chunk_cnt = mpc_future_chunk_cnt

        # all possible combinations of 5 chunk bitrates (9^5 options)
        # iterate over list and for each, compute reward and store max
        # reward combination
        self.bitrate_options = np.array(VIDEO_BIT_RATE)

    def select_action(self, state, last_index, future_chunk_cnt, video_size,
                      bit_rate, buffer_size):
        # defualt assumes that this is the first request so error is 0
        # since we have never predicted bandwidth
        # pick bitrate according to MPC
        # first get harmonic mean of last 5 bandwidths
        past_bandwidths = state[0, 2, -5:]
        while past_bandwidths[0] == 0.0:
            past_bandwidths = past_bandwidths[1:]

        future_bandwidth = 1 / np.mean(1 / past_bandwidths)  #FastMPC

        jump_action_combos = calculate_jump_action_combo( bit_rate )
        bit_rate = predict_bitrate(
            future_chunk_cnt, buffer_size, bit_rate, last_index,
            future_bandwidth, video_size, jump_action_combos,
            self.bitrate_options)
        return bit_rate, future_bandwidth

    def evaluate(self, net_env):
        """Evaluate on a single net_env."""
        results = []

        net_env.reset()
        video_size = np.array([net_env.video_size[i]
                               for i in sorted(net_env.video_size)])
        time_stamp = 0
        bit_rate = DEFAULT_QUALITY
        future_bandwidth = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            state, reward, end_of_video, info = net_env.step(bit_rate)

            time_stamp += info['delay']  # in ms
            time_stamp += info['sleep_time']  # in ms

            results.append([time_stamp / M_IN_K,
                            self.bitrate_options[bit_rate],
                            info['buffer_size'], info['rebuf'],
                            info['video_chunk_size'], info['delay'], reward,
                            future_bandwidth])

            # future chunks length (try 4 if that many remaining)
            last_index = (net_env.total_video_chunk -
                          info['video_chunk_remain'] - 1)
            future_chunk_cnt = min(self.mpc_future_chunk_cnt,
                                   net_env.total_video_chunk - last_index - 1)

            bit_rate, future_bandwidth = self.select_action(
                state, last_index, future_chunk_cnt, video_size, bit_rate,
                info['buffer_size'])

            if end_of_video:
                break
        return results

    def evaluate_envs(self, net_envs, n_proc=mp.cpu_count()//2):
        """Evaluate multipe environment using multiprocessing."""
        arguments = [(net_env, ) for net_env in net_envs]
        with mp.Pool(processes=n_proc) as pool:
            results = pool.starmap(self.evaluate, arguments)
        return results


@jit(nopython=True)
def predict_bitrate(future_chunk_length, buffer_size, bit_rate, last_index,
                    future_bandwidth, video_size, chunk_combo_options,
                    bitrate_options):
    max_reward = np.NINF
    best_combo = ()
    start_buffer = buffer_size

    for full_combo in chunk_combo_options:
        combo = full_combo[0:int(future_chunk_length)]
        # calculate total rebuffer time for this combination (start with
        # start_buffer and subtract each download time and add 2 seconds in
        # that order)
        curr_buffer = start_buffer
        reward = 0
        last_quality = int(bit_rate)
        for position in range(0, len(combo)):
            chunk_quality = combo[position]
            # e.g., if last chunk is 3, then first iter is 3+0+1=4
            index = last_index + position + 1
            # this is MB/MB/s --> seconds
            row_len = len(video_size[chunk_quality])
            download_time = video_size[chunk_quality,
                                       int(index % (row_len))] / \
                B_IN_MB / future_bandwidth
            if (curr_buffer < download_time):
                rebuffer_time = (download_time - curr_buffer)
                curr_buffer = 0
            else:
                curr_buffer -= download_time
                rebuffer_time = 0
            curr_buffer += VIDEO_CHUNK_LEN / MILLISECONDS_IN_SECOND
            reward += linear_reward(bitrate_options[chunk_quality],
                                    bitrate_options[last_quality],
                                    rebuffer_time)
            last_quality = chunk_quality
        # compute reward for this combination (one reward per 5-chunk combo)
        # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in
        # Mbits/s

        if reward >= max_reward:
            best_combo = combo

            max_reward = reward
            # send data to html side (first chunk of best combo)
            # no combo had reward better than -1000000 (ERROR) so send 0
            send_data = 0
            if best_combo.size != 0:  # some combo was good
                send_data = best_combo[0]

    return send_data
