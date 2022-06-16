import csv
import os
from typing import List

from simulator.abr_simulator.abr_trace import AbrTrace
from simulator.abr_simulator.base_abr import BaseAbr
from simulator.abr_simulator.constants import (
        A_DIM, DEFAULT_QUALITY, M_IN_K, MILLISECONDS_IN_SECOND,
        VIDEO_BIT_RATE, VIDEO_CHUNK_LEN)
from simulator.abr_simulator.env import Environment
from simulator.abr_simulator.schedulers import TestScheduler
from simulator.abr_simulator.utils import plot_abr_log, linear_reward


RESEVOIR = 5  # BB
CUSHION = 10  # BB

class BBA(BaseAbr):
    abr_name = 'bba'

    def __init__(self, plot_flag: bool = False) -> None:
        self.plot_flag = plot_flag

    def test_on_traces(self, traces: List[AbrTrace], video_size_file_dir: str, save_dirs: List[str]):
        rewards = []
        for trace, save_dir in zip(traces, save_dirs):
            rewards.append(self.test(trace, video_size_file_dir, save_dir))
        return rewards

    def get_next_bitrate(self, buffer_size):
        if buffer_size < RESEVOIR:
            bit_rate = 0
        elif buffer_size >= RESEVOIR + CUSHION:
            bit_rate = A_DIM - 1
        else:
            bit_rate = (A_DIM - 1) * (buffer_size - RESEVOIR) / float(CUSHION)
        return int(bit_rate)

    def test(self, trace: AbrTrace, video_size_file_dir: str, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        if trace.name:
            log_name = os.path.join(save_dir, "{}_{}.csv".format(self.abr_name, trace.name))
        else:
            log_name = os.path.join(save_dir, "{}_log.csv".format(self.abr_name))
        abr_log = open(log_name, 'w')
        log_writer = csv.writer(abr_log, lineterminator='\n')
        log_writer.writerow(["timestamp", "bitrate", "buffer_size",
                             "rebuffering", "video_chunk_size", "delay",
                             "reward"])

        trace_scheduler = TestScheduler(trace)
        net_env = Environment(trace_scheduler, VIDEO_CHUNK_LEN / MILLISECONDS_IN_SECOND,
                              video_size_file_dir=video_size_file_dir)

        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        r_batch = []
        final_reward = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # reward is video quality - rebuffer penalty
            reward = linear_reward(VIDEO_BIT_RATE[bit_rate], 
                                   VIDEO_BIT_RATE[last_bit_rate], rebuf)

            r_batch.append(reward)

            last_bit_rate = bit_rate

            log_writer.writerow([time_stamp / M_IN_K, VIDEO_BIT_RATE[bit_rate],
                                 buffer_size, rebuf, video_chunk_size, delay,
                                 reward])

            bit_rate = self.get_next_bitrate(buffer_size)

            if end_of_video:

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here
                final_reward = sum(r_batch)
                r_batch = []

                break
        abr_log.close()
        if self.plot_flag:
            plot_abr_log(trace, log_name, save_dir)
        return final_reward
