import csv
import os
from typing import List


import numpy as np
from numba import jit

from simulator.abr_simulator.abr_trace import AbrTrace
from simulator.abr_simulator.env import Environment
from simulator.abr_simulator.constants import (
        A_DIM, DEFAULT_QUALITY, M_IN_K, MILLISECONDS_IN_SECOND,
        VIDEO_BIT_RATE, VIDEO_CHUNK_LEN, REBUF_PENALTY, SMOOTH_PENALTY,
        BUFFER_NORM_FACTOR, TOTAL_VIDEO_CHUNK)
from simulator.abr_simulator.utils import plot_abr_log  # linear_reward


class Oboe:
    abr_name = 'oboe'

    def __init__(self, plot_flag: bool = False) -> None:
        self.plot_flag = plot_flag
        pass

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
        size_video_array = np.array([net_env.video_size[i] for i in
                                     sorted(net_env.video_size)])





    def test_on_traces(self, traces: List[AbrTrace], video_size_file_dir: str, save_dirs: List[str]):
        rewards = []
        for trace, save_dir in zip(traces, save_dirs):
            rewards.append(self.test(trace, video_size_file_dir, save_dir))
        return rewards
