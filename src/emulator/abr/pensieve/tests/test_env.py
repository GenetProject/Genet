import os

import numpy as np
from pensieve.utils import load_traces
from pensieve.environment import Environment, MultiEnv

VIDEO_SIZE_FILE_DIR = '/data3/zxxia/pensieve/data/video_sizes'
CONFIG_FILE = '/data3/zxxia/active-domainrand/pensieve/config/rand_buff_thresh.json'
SEED = 42
SUMMARY_DIR = '/data3/zxxia/active-domainrand/pensieve/tests/mpc_logs'
TEST_TRACE_DIR = '/data3/zxxia/pensieve/data/test'


def main():
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    np.random.seed(SEED)

    all_cooked_time, all_cooked_bw, all_file_names = load_traces(
        TEST_TRACE_DIR)

    test_envs = []
    for trace_idx, (trace_time, trace_bw, trace_filename) in enumerate(
            zip(all_cooked_time, all_cooked_bw, all_file_names)):
        net_env = Environment(VIDEO_SIZE_FILE_DIR, CONFIG_FILE, trace_idx,
                              trace_time=trace_time, trace_bw=trace_bw,
                              trace_file_name=trace_filename, fixed=True,
                              trace_video_same_duration_flag=True)
        test_envs.append(net_env)
    # import ipdb; ipdb.set_trace()
    nagents = 16
    menv = MultiEnv(nagents, test_envs[:nagents])

    # test step function
    state, reward, end_of_video, info = menv.step([1] * nagents)
    print(state.shape)
    print(reward.shape)
    print(end_of_video.shape)
    print(len(info))

    # test get_current_params function
    params = menv.get_current_params()
    print(params)
    params = menv.get_current_randomization_params()
    print(params)

    # test reset function
    menv.reset()

    # test randomization funciton
    menv.randomize([6]*nagents)
    # params = menv.get_current_randomization_params()
    params = menv.get_current_params()
    print(params)
    for _ in range(5):
        menv.randomize([None]*nagents)
        params = menv.get_current_randomization_params()
        print(params)


if __name__ == '__main__':
    main()
