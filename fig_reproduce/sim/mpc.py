import argparse
import itertools
import os
import numba
from numba import jit

# import fixed_env as env
import env
# import load_trace
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import load_traces
from utils.constants import (REBUF_PENALTY, VIDEO_SIZE_FILE, TOTAL_VIDEO_CHUNK, CHUNK_TIL_VIDEO_END_CAP)


S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
MPC_FUTURE_CHUNK_COUNT = 5
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
# VIDEO_BIT_RATE = np.array([300, 1200, 2850, 6500, 33000, 165000])  # Kbps
VIDEO_BIT_RATE = np.array([300, 750, 1200, 1850, 2850, 4300])

BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
#CHUNK_TIL_VIDEO_END_CAP = 48.0
#TOTAL_VIDEO_CHUNKS = 48
M_IN_K = 1000.0
#REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 0  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
# SUMMARY_DIR = '../results/tmp'
# TEST_TRACES = '../data/test'
# LOG_FILE = './results/log_sim_mpc'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size
# download_time reward

#CHUNK_COMBO_OPTIONS = []

# past errors in bandwidth
past_errors = []
past_bandwidth_ests = []
#VIDEO_SIZE_FILE = '../data/video_sizes/video_size_'

CHUNK_COMBO_OPTIONS = np.array([combo for combo in itertools.product(
                range(6), repeat=5)])

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pensieve testing script.")
    parser.add_argument("--test_trace_dir", type=str,
                        required=True, help='dir to all test traces.')
    parser.add_argument("--summary_dir", type=str,
                        required=True, help='output path.')
    parser.add_argument( "--log_str" ,type=str ,required=True ,
                         help='str add to log' )
    parser.add_argument("--random_seed", type=int, default=11)
    return parser.parse_args()


@jit(nopython=True)
def next_possible_bitrates(br):
    next_brs = [br - 1 ,br ,br + 1]
    next_brs = [a for a in next_brs if 0 <= a <= 5]
    return next_brs

@jit(nopython=True)
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

@jit(nopython=True)
def get_chunk_size(quality, index, size_video_array):
    if (index < 0 or index > TOTAL_VIDEO_CHUNK):
        return 0
    # note that the quality and video labels are inverted (i.e., quality 4 is
    # highest and this pertains to video1)
    return size_video_array[quality, index]

@jit(nopython=True)
def calculate_rebuffer(size_video_array, future_chunk_length, buffer_size, bit_rate, last_index, future_bandwidth, jump_action_combos):
    max_reward = -100000000
    start_buffer = buffer_size

    #jump_action_combos = calculate_jump_action_combo(bit_rate)
    for full_combo in jump_action_combos:
        combo = full_combo[0:future_chunk_length]
        # calculate total rebuffer time for this combination (start with start_buffer and subtract
        # each download time and add 2 seconds in that order)
        curr_rebuffer_time = 0
        curr_buffer = start_buffer
        bitrate_sum = 0
        smoothness_diffs = 0
        last_quality = int( bit_rate )
        for position in range( 0, len( combo ) ):
            chunk_quality = combo[position]
            # e.g., if last chunk is 3, then first iter is 3+0+1=4
            index = last_index + position + 1
            # this is MB/MB/s --> seconds
            download_time = (get_chunk_size(chunk_quality, index, size_video_array) / 1000000.) / future_bandwidth
            if (curr_buffer < download_time):
                curr_rebuffer_time += (download_time - curr_buffer)
                curr_buffer = 0
            else:
                curr_buffer -= download_time
            curr_buffer += 4
            bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
            smoothness_diffs += abs(
                VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality] )
            last_quality = chunk_quality

        reward = (bitrate_sum / 1000.) - (REBUF_PENALTY *
                                          curr_rebuffer_time) - (smoothness_diffs / 1000.)
        if reward >= max_reward:
            best_combo = combo
            max_reward = reward
            send_data = 0
            if best_combo.size != 0:  # some combo was good
                send_data = best_combo[0]
    return send_data

def main():
    combo_dict = {'0': calculate_jump_action_combo( 0 ) ,
                  '1': calculate_jump_action_combo( 1 ) ,
                  '2': calculate_jump_action_combo( 2 ) ,
                  '3': calculate_jump_action_combo( 3 ) ,
                  '4': calculate_jump_action_combo( 4 ) ,
                  '5': calculate_jump_action_combo( 5 )}

    args = parse_args()

    os.makedirs(args.summary_dir, exist_ok=True)
    np.random.seed(RANDOM_SEED)

    size_video_array =[]
    for bitrate in range( 6 ):
        video_size = []
        with open( VIDEO_SIZE_FILE + str( bitrate ) ) as f:
            for line in f:
                video_size.append( int( line.split()[0] ) )
        size_video_array.append(video_size)

    size_video_array = np.array(np.squeeze(size_video_array))

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_traces(
        args.test_trace_dir)


    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw, fixed=True)

    log_path = os.path.join( args.summary_dir ,
                             'log_sim_RobustMPC_{}_{}'.format( args.log_str ,all_file_names[net_env.trace_idx] ) )

    log_file = open(log_path, 'w', 1)

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    entropy_record = []

    video_count = 0

    # make chunk combination options
    # for combo in itertools.product([0, 1, 2, 3, 4, 5], repeat=5):
    #     CHUNK_COMBO_OPTIONS.append(combo)

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

        # log scale reward
        # log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[0]))
        # log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[0]))

        # reward = log_bit_rate \
        #          - REBUF_PENALTY * rebuf \
        #          - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

        # reward = BITRATE_REWARD[bit_rate] \
        #          - 8 * rebuf - np.abs(BITRATE_REWARD[bit_rate] - BITRATE_REWARD[last_bit_rate])

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

        # retrieve previous state
        if len(s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(s_batch[-1], copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
            float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = rebuf
        state[3, -1] = float(video_chunk_size) / \
            float(delay) / M_IN_K  # kilo byte / ms
        state[4, -1] = np.minimum(video_chunk_remain,
                                  CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        # state[5: 10, :] = future_chunk_sizes / M_IN_K / M_IN_K

        # ================== MPC =========================
        curr_error = 0  # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
        if (len(past_bandwidth_ests) > 0):
            curr_error = abs(
                past_bandwidth_ests[-1]-state[3, -1])/float(state[3, -1])
        past_errors.append(curr_error)

        # pick bitrate according to MPC
        # first get harmonic mean of last 5 bandwidths
        past_bandwidths = state[3, -5:]
        while past_bandwidths[0] == 0.0:
            past_bandwidths = past_bandwidths[1:]
        # if ( len(state) < 5 ):
        #    past_bandwidths = state[3,-len(state):]
        # else:
        #    past_bandwidths = state[3,-5:]
        bandwidth_sum = 0
        for past_val in past_bandwidths:
            bandwidth_sum += (1/float(past_val))
        harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))

        # future bandwidth prediction
        # divide by 1 + max of last 5 (or up to 5) errors
        max_error = 0
        error_pos = -5
        if (len(past_errors) < 5):
            error_pos = -len(past_errors)
        max_error = float(max(past_errors[error_pos:]))
        #print(max_error, "-mpc")
        future_bandwidth = harmonic_bandwidth/(1+max_error)  # robustMPC here
        past_bandwidth_ests.append(harmonic_bandwidth)

        # future chunks length (try 4 if that many remaining)
        last_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain)
        future_chunk_length = MPC_FUTURE_CHUNK_COUNT
        if (TOTAL_VIDEO_CHUNK - last_index < 5):
            future_chunk_length = TOTAL_VIDEO_CHUNK - last_index

        # all possible combinations of 5 chunk bitrates (9^5 options)
        # iterate over list and for each, compute reward and store max reward combination
        # start = time.time()
        #chunk_combo_options = np.array( CHUNK_COMBO_OPTIONS )

        jump_action_combos = combo_dict[str(bit_rate)]

        bit_rate = calculate_rebuffer(size_video_array, future_chunk_length, buffer_size, bit_rate,
                                      last_index, future_bandwidth, jump_action_combos)

        # hack
        # if bit_rate == 1 or bit_rate == 2:
        #    bit_rate = 0

        # ================================================

        # Note: we need to discretize the probability into 1/RAND_RANGE steps,
        # because there is an intrinsic discrepancy in passing single state and batch states

        s_batch.append(state)

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(action_vec)
            entropy_record = []

            print("video count", video_count)
            video_count += 1

            if video_count >= len(all_file_names):
                break

            log_path = os.path.join( args.summary_dir ,
                                     'log_sim_RobustMPC_{}_{}'.format( args.log_str ,all_file_names[net_env.trace_idx] ) )

            log_file = open(log_path, 'w', 1)


if __name__ == '__main__':
    main()
