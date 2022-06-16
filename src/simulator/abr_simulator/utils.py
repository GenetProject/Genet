import os
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit

from simulator.abr_simulator.constants import (HD_REWARD, M_IN_K, REBUF_PENALTY,
                                SMOOTH_PENALTY, VIDEO_BIT_RATE)
from simulator.abr_simulator.abr_trace import AbrTrace

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



def load_traces_for_train(cooked_trace_folder):
    # print("Loading traces from " + cooked_trace_folder)
    # cooked_files = os.listdir(cooked_trace_folder)
    # print("Found " + str(len(cooked_files)) + " trace files.")
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []

    newest_folder = max( glob.glob( os.path.join( cooked_trace_folder ,'*/' ) ) ,key=os.path.getmtime )
    for subdir ,dirs ,files in os.walk( cooked_trace_folder ):
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']
        # TODO: do I still need to 0.6 random sampling?
        # if subdir + '/' != newest_folder:
        #     # sample 0.6*original files out
        #     random_files = np.random.choice( files ,int( len( files ) * .6 ) )
        # else:
        random_files = files

        for file in random_files:
            #print (os.path.join(subdir, file))
            file_path = subdir + os.sep + file
            val_folder_name = os.path.basename( os.path.normpath( subdir ) )
            #print( val_folder_name, "-----")
            cooked_time = []
            cooked_bw = []
            with open(file_path, 'rb') as phile:
                #print(file_path)
                for line in phile:
                    parse = line.split()
                    cooked_time.append(float(parse[0]))
                    cooked_bw.append(float(parse[1]))
            all_cooked_time.append(cooked_time)
            all_cooked_bw.append(cooked_bw)
            all_file_names.append(val_folder_name + '_' + file)

    return all_cooked_time, all_cooked_bw, all_file_names


# for test
def load_traces(cooked_trace_folder):
    # print("Loading traces from " + cooked_trace_folder)
    # cooked_files = os.listdir(cooked_trace_folder)
    # print("Found " + str(len(cooked_files)) + " trace files.")
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for subdir ,dirs ,files in os.walk( cooked_trace_folder ):
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']
        for file in files:
            # print os.path.join(subdir, file)
            file_path = subdir + os.sep + file
            val_folder_name = os.path.basename( os.path.normpath( subdir ) )
            #print( val_folder_name, "-----")
            cooked_time = []
            cooked_bw = []
            with open(file_path, 'rb') as phile:
                #print(file_path)
                for line in phile:
                    parse = line.split()
                    cooked_time.append(float(parse[0]))
                    cooked_bw.append(float(parse[1]))
            all_cooked_time.append(cooked_time)
            all_cooked_bw.append(cooked_bw)
            all_file_names.append(val_folder_name + '_' + file)

    return all_cooked_time, all_cooked_bw, all_file_names


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



def map_lin_to_log(x, min_value=1, max_value=500):
    x_log = (np.log(x) - np.log(min_value)) / (np.log(max_value) - np.log(min_value))
    return x_log


# def map_log_to_lin(x):
#     x_lin = np.exp((np.log(MAX_BW)-np.log(MIN_BW))*x + np.log(MIN_BW))
#     return x_lin


def map_to_unnormalize(x, min_value, max_value):
    x_unnormalize = x*(max_value - min_value) + min_value
    x_unnormalize = round(x_unnormalize, 0)
    return x_unnormalize


# another log scale
def map_log_to_lin(x):
    x_lin = round(2**(10*x), 1)
    return x_lin


def latest_actor_from(path):
    """
    Returns latest tensorflow checkpoint file from a directory.
    Assumes files are named:
    nn_model_ep_<EPOCH#>.ckpt.meta
    """
    mtime = lambda f: os.stat( os.path.join( path ,f ) ).st_mtime
    files = list( sorted( os.listdir( path ) ,key=mtime ) )
    actors = [a for a in files if "nn_model_ep_" in a]
    if not actors:
        return None
    actor_path = str( path + '/' + actors[-1] )
    return os.path.splitext( actor_path )[0]


def plot_abr_log(trace: Optional[AbrTrace], log_file: str, save_dir: Optional[str]):
    fig, axes = plt.subplots(5, 1,  figsize=(12, 10))
    log_name = os.path.splitext(os.path.basename(log_file))[0]
    df = pd.read_csv(log_file)
    assert isinstance(df, pd.DataFrame)
    axes[0].plot(df['timestamp'], df['bitrate'], 'o-', ms=2, drawstyle='steps-pre',
                 label='bitrate, avg {:.3f}kbps'.format(df['bitrate'].mean()))

    # max_ts = trace.timestamps[-1]
    max_ts = df['timestamp'].iloc[len(df['timestamp'])-1]
    if trace:
        if trace.timestamps[-1] < max_ts:
            # max_ts = df['timestamp'][-1]
            pass
            # TODO: wrap around trace
        axes[0].plot(trace.timestamps, np.array(trace.bandwidths) * 1000, 'o-', ms=2,
                     label='bw, avg {:.3f}kbps'.format(1000*np.mean(trace.bandwidths)))

    axes[0].set_xlabel("Time(s)")
    axes[0].set_ylabel("kbps")
    axes[0].legend(loc='right')
    axes[0].set_ylim(0, )
    axes[0].set_xlim(0, max_ts)

    axes[1].plot(df['timestamp'], df['buffer_size'],
                 label='Buffer size avg {:.3f}s'.format(df['buffer_size'].mean()))
    axes[1].set_xlabel("Time(s)")
    axes[1].set_ylabel("Buffer size(s)")
    axes[1].legend(loc='right')
    axes[1].set_xlim(0, max_ts)
    axes[1].set_ylim(0, )

    axes[2].plot(df['timestamp'], df['rebuffering'],
                 label='Rebuffering avg {:.3f}s'.format(df['rebuffering'].mean()))
    axes[2].set_xlabel("Time(s)")
    axes[2].set_ylabel("Rebuffering(s)")
    axes[2].legend(loc='right')
    axes[2].set_xlim(0, max_ts)
    axes[2].set_ylim(0, )

    axes[3].plot(df['timestamp'], df['delay'] / 1000,
                 label='Delay avg {:.3f}'.format(df['delay'].mean()/1000))
    axes[3].set_xlabel("Time(s)")
    axes[3].set_ylabel("Delay (s)")
    axes[3].legend()
    axes[3].set_xlim(0, max_ts)

    axes[4].plot(df['timestamp'], df['reward'],
            label='rewards avg {:.3f}, sum {:.3f}'.format(df['reward'].mean(), df['reward'].sum()))
    axes[4].set_xlabel("Time(s)")
    axes[4].set_ylabel("Reward")
    axes[4].legend()
    axes[4].set_xlim(0, max_ts)

    plt.tight_layout()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "{}.jpg".format(log_name)))
    plt.close()

def construct_bitrate_chunksize_map(video_size_file_dir: str):
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
