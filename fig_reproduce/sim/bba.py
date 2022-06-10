import argparse
import numpy as np
import env
from utils.utils import adjust_traces ,load_traces
import os
from utils.constants import (REBUF_PENALTY)


S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
M_IN_K = 1000.0
#REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
RESEVOIR = 5  # BB
CUSHION = 10  # BB
SUMMARY_DIR = './results'
LOG_FILE = './results/log_sim_bb'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pensieve testing script." )
    parser.add_argument( "--test_trace_dir" ,type=str ,
                         # optional now because we have a default example
                         # required=True,
                         help='dir to all test traces.' )
    parser.add_argument( "--summary_dir" ,type=str ,
                         required=True ,help='output path.' )
    parser.add_argument( "--log_str" ,type=str ,required=True ,
                         help='str add to log' )
    parser.add_argument( "--random_seed" ,type=int ,default=11 )

    parser.add_argument( '--A_DIM' ,type=int ,default='3' ,help='' )
    parser.add_argument( '--BITRATE_DIM' ,type=int ,default='6' ,help='' )
    parser.add_argument( '--S_LEN' ,type=int ,default='6' ,help='' )

    return parser.parse_args()

def main():
    args = parse_args()
    summary_dir = args.summary_dir

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time ,all_cooked_bw ,all_file_names = load_traces( args.test_trace_dir )

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw, fixed=True)

    os.makedirs( summary_dir ,exist_ok=True )

    log_path = os.path.join( summary_dir ,
                             'log_sim_BBA_{}_{}'.format( args.log_str ,all_file_names[net_env.trace_idx] ) )
    log_file = open( log_path ,'w' )

    epoch = 0
    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    r_batch = []

    video_count = 0

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
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                           VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
        r_batch.append(reward)

        smoothness = np.abs( VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate] ) / M_IN_K

        last_bit_rate = bit_rate

        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write( str( time_stamp / M_IN_K ) + '\t' +
                        str( VIDEO_BIT_RATE[bit_rate] ) + '\t' +
                        str( buffer_size ) + '\t' +
                        str( rebuf ) + '\t' +
                        str( video_chunk_size ) + '\t' +
                        str( delay ) + '\t' +
                        str( smoothness ) + '\t' +
                        str( reward ) + '\n' )
        log_file.flush()

        if buffer_size < RESEVOIR:
            bit_rate = 0
        elif buffer_size >= RESEVOIR + CUSHION:
            bit_rate = A_DIM - 1
        else:
            bit_rate = (A_DIM - 1) * (buffer_size - RESEVOIR) / float(CUSHION)

        bit_rate = int(bit_rate)

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here
            r_batch = []

            print("video count", video_count)
            video_count += 1

            if video_count > len(all_file_names):
                break

            log_path = os.path.join( summary_dir ,
                                     'log_sim_BBA_{}_{}'.format( args.log_str ,all_file_names[net_env.trace_idx] ) )
            log_file = open( log_path ,'w' )


if __name__ == '__main__':
    main()