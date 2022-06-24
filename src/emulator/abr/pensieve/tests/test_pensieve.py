import time
import csv
import os

from pensieve.agent_policy import Pensieve
from pensieve.agent_policy.pensieve import compare_mpc_pensieve
from pensieve.environment import Environment
from pensieve.utils import load_traces

VIDEO_SIZE_FILE_DIR = '/data3/zxxia/pensieve/data/video_sizes'
# CONFIG_FILE = '/data3/zxxia/active-domainrand/pensieve/config/default.json'
CONFIG_FILE = '/data3/zxxia/active-domainrand/pensieve/config/rand_buff_thresh_link_rtt.json'
SEED = 42
SUMMARY_DIR = '/data3/zxxia/active-domainrand/pensieve/tests/pensieve_log'
TEST_TRACE_DIR = '/data3/zxxia/pensieve/data/test'
VAL_TRACE_DIR = '/data3/zxxia/pensieve/data/val'
TRAIN_TRACE_DIR = '/data3/zxxia/pensieve/data/train'

ACTOR_PATH = '/data3/zxxia/pensieve-pytorch/results_fix_pred/actor.pt'
CRITIC_PATH = '/data3/zxxia/pensieve-pytorch/results_fix_pred/actor.pt'


def main():
    pensieve_abr = Pensieve(16, SUMMARY_DIR, actor_path=ACTOR_PATH,
                            model_save_interval=10, batch_size=100)

    all_cooked_time, all_cooked_bw, all_file_names = load_traces(
        TRAIN_TRACE_DIR)
    net_envs = []
    for trace_idx, (trace_time, trace_bw, trace_filename) in enumerate(
            zip(all_cooked_time, all_cooked_bw, all_file_names)):
        net_env = Environment(VIDEO_SIZE_FILE_DIR, CONFIG_FILE, trace_idx,
                              trace_time=trace_time, trace_bw=trace_bw,
                              trace_file_name=trace_filename, fixed=False,
                              trace_video_same_duration_flag=True)
        net_envs.append(net_env)

    all_cooked_time, all_cooked_bw, all_file_names = load_traces(VAL_TRACE_DIR)
    val_envs = []
    for trace_idx, (trace_time, trace_bw, trace_filename) in enumerate(
            zip(all_cooked_time, all_cooked_bw, all_file_names)):
        net_env = Environment(VIDEO_SIZE_FILE_DIR, CONFIG_FILE, trace_idx,
                              trace_time=trace_time, trace_bw=trace_bw,
                              trace_file_name=trace_filename, fixed=True,
                              trace_video_same_duration_flag=True)
        val_envs.append(net_env)

    test_envs = []
    all_cooked_time, all_cooked_bw, all_file_names = load_traces(
        TEST_TRACE_DIR)
    for trace_idx, (trace_time, trace_bw, trace_filename) in enumerate(
            zip(all_cooked_time, all_cooked_bw, all_file_names)):
        net_env = Environment(VIDEO_SIZE_FILE_DIR, CONFIG_FILE, trace_idx,
                              trace_time=trace_time, trace_bw=trace_bw,
                              trace_file_name=trace_filename, fixed=True,
                              trace_video_same_duration_flag=True)
        test_envs.append(net_env)

    # test training
    # pensieve_abr.train(net_envs, val_envs=val_envs,
    #                    test_envs=test_envs, iters=1e4)

    # test evaluate and evalute_env
    # t_start = time.time()
    # single_proc_results = pensieve_abr.evaluate_envs(test_envs)
    # print('singleproc', time.time() - t_start)

    # t_start = time.time()
    # multi_proc_results = pensieve_abr.evaluate_envs(
    #     test_envs, os.path.join(SUMMARY_DIR, 'test_log'))
    # print('multiproc', time.time() - t_start)

    # t_start = time.time()
    # multi_proc_results = pensieve_abr.evaluate_envs_multi(test_envs)
    # print('multiproc startmap', time.time() - t_start)
    # assert len(single_proc_results) == len(multi_proc_results), \
    #     "video numbers are not equal"
    # for vid_logs_s, vid_logs_m in zip(single_proc_results, multi_proc_results):
    #     assert len(vid_logs_s) == len(vid_logs_m), \
    #         "video chunk numbers are not equal."
    #     for row_s, row_m in zip(vid_logs_s, vid_logs_m):
    #         for val_s, val_m in zip(row_s, row_m):
    #             assert round(float(val_s), 6) == round(float(val_m), 6), \
    #                 '{}!={}\n{}\n{}'.format(float(val_s), float(val_m),
    #                                         row_s, row_m)

    # test compare_mpc_pensieve
    param_ranges = {'buffer_threshold': (1, 600), 'link_rtt': (0, 1000)}
    compare_mpc_pensieve(pensieve_abr, val_envs, param_ranges)

    print('pass the test')


if __name__ == "__main__":
    main()
