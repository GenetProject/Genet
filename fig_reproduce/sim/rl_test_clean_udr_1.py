import argparse
import os
from utils.utils import adjust_traces ,load_traces
import a3c
# import fixed_env as env
import env
import numpy as np
import tensorflow as tf
import subprocess
from utils.constants import (REBUF_PENALTY, VIDEO_BIT_RATE, CHUNK_TIL_VIDEO_END_CAP)


os.environ['CUDA_VISIBLE_DEVICES'] = ''

tf.logging.set_verbosity( tf.logging.ERROR )
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_INFO = 6
# S_LEN = 11  # take how many frames in the past
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
#VIDEO_BIT_RATE = [300, 1200, 2850, 6500, 33000, 165000]  # Kbps
#VIDEO_BIT_RATE = [300 ,750 ,1200 ,1850 ,2850 ,4300]

BUFFER_NORM_FACTOR = 10.0
#CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
#REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 0  # default video quality without agent
# RANDOM_SEED = 42
RAND_RANGE = 1000


# LOG_FILE = './test_results/log_sim_rl'
# TEST_TRACES = './cooked_test_traces/'
# TEST_TRACES = './test_sim_traces/'
# TEST_TRACES = '../data/val/'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward

# Strategy:

# Input for RL Testing should be:
#
# 1. a configuration from which test traces are generated
#   - load the configuration from json and create a TraceConfig to generate traces (later)
#   - create the traces from a configuration (refer to example) (priority)
#
# 2. a model checkpoint file to load and test against the traces (DONE)
# 3. Move TraceConfig outside of this file so it can be used elsewhere too. (later)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pensieve testing script." )
    parser.add_argument( "--test_trace_dir" ,type=str ,
                         # optional now because we have a default example
                         # required=True,
                         help='dir to all test traces.' )
    parser.add_argument( "--summary_dir" ,type=str ,
                         required=True ,help='output path.' )
    parser.add_argument( "--model_path" ,type=str ,required=True ,
                         help='model path' )
    parser.add_argument( "--log_str" ,type=str ,required=True ,
                         help='str add to log' )
    parser.add_argument( "--random_seed" ,type=int ,default=11 )
    parser.add_argument( '--A_DIM' ,type=int ,default='3' ,help='' )
    parser.add_argument( '--BITRATE_DIM' ,type=int ,default='6' ,help='' )
    parser.add_argument( '--S_LEN' ,type=int ,default='6' ,help='' )

    return parser.parse_args()


def calculate_from_selection(selected ,last_bit_rate):
    # selected_action is 0-5
    # naive step implementation
    if selected == 1:
        bit_rate = last_bit_rate
    elif selected == 2:
        bit_rate = last_bit_rate + 1
    else:
        bit_rate = last_bit_rate - 1
    # bound
    if bit_rate < 0:
        bit_rate = 0
    if bit_rate > 5:
        bit_rate = 5

    # print(bit_rate)
    return bit_rate


def given_string_mean_reward(plot_files ,test_dir ,str):
    matching = [s for s in plot_files if str in s]
    reward = []
    count = 0
    for log_file in matching:
        count += 1
        # print(log_file)
        with open( test_dir + '/' + log_file ,'r' ) as f:
            for line in f:
                parse = line.split()
                if len( parse ) <= 1:
                    break
                reward.append( float( parse[6] ) )
    print( count )
    return np.mean( reward[1:] )


class TraceConfig:
    def __init__(self ,
                 trace_dir ,
                 max_throughput=10):
        self.trace_dir = trace_dir
        self.max_throughput = max_throughput
        self.T_l = 0
        self.T_s = 3
        self.cov = 3
        self.duration = 250
        self.step = 0
        self.min_throughput = 0.2
        self.num_traces = 100


def example_trace_config():
    return TraceConfig( "../data/example_traces/" )


def generate_traces_with(config):
    """
    Generates traces based on the config
    """
    script = "trace_generator.py"
    command = "python {script} \"{config}\"".format( script=script ,config=vars( config ) )
    # alternatively call with os.system, but it doesn't print the result that way
    print( command )
    os.system( command )
    # output = subprocess.check_output(command, shell=True, text=True).strip()
    # print(output)


def main():
    args = parse_args()
    summary_dir = args.summary_dir
    nn_model = args.model_path

    # generate test traces
    # test_trace_dir = args.test_trace_dir

    # Just manually load the example .... as an example...
    # trace_config = example_trace_config()
    # generate_traces_with(trace_config)

    os.makedirs( summary_dir ,exist_ok=True )
    # np.random.seed(args.random_seed)

    # assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time ,all_cooked_bw ,all_file_names = load_traces( args.test_trace_dir )

    # print(len(all_cooked_time[-1]))

    net_env = env.Environment( all_cooked_time=all_cooked_time ,
                               all_cooked_bw=all_cooked_bw ,fixed=True )

    log_path = os.path.join( summary_dir ,
                             'log_sim_udr_1_{}_{}'.format( args.log_str ,all_file_names[net_env.trace_idx] ) )

    log_file = open( log_path ,'w' )

    with tf.Session() as sess:

        actor = a3c.ActorNetwork( sess ,
                                  state_dim=[S_INFO ,args.S_LEN] ,action_dim=args.A_DIM ,
                                  bitrate_dim=args.BITRATE_DIM
                                  )

        sess.run( tf.global_variables_initializer() )
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        if nn_model is not None:  # NN_MODEL is the path to file
            saver.restore( sess ,nn_model )
            print( "Testing model restored." )

        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros( args.A_DIM )
        action_vec[bit_rate] = 1

        s_batch = [np.zeros( (S_INFO ,args.S_LEN) )]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        video_count = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay ,sleep_time ,buffer_size ,rebuf , \
            video_chunk_size ,next_video_chunk_sizes , \
            end_of_video ,video_chunk_remain = \
                net_env.get_video_chunk( bit_rate )

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # reward is video quality - rebuffer penalty - smoothness
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * np.abs( VIDEO_BIT_RATE[bit_rate] -
                                                VIDEO_BIT_RATE[last_bit_rate] ) / M_IN_K

            r_batch.append( reward )

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

            # retrieve previous state
            if len( s_batch ) == 0:
                state = [np.zeros( (S_INFO ,args.S_LEN) )]
            else:
                state = np.array( s_batch[-1] ,copy=True )

            # dequeue history record
            state = np.roll( state ,-1 ,axis=1 )

            # this should be S_INFO number of terms
            state[0 ,-1] = VIDEO_BIT_RATE[bit_rate] / \
                           float( np.max( VIDEO_BIT_RATE ) )  # last quality
            state[1 ,-1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2 ,-1] = float( video_chunk_size ) / \
                           float( delay ) / M_IN_K  # kilo byte / ms
            state[3 ,-1] = float( delay ) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4 ,:args.BITRATE_DIM] = np.array(
                next_video_chunk_sizes ) / M_IN_K / M_IN_K  # mega byte
            state[5 ,-1] = np.minimum(
                video_chunk_remain ,
                CHUNK_TIL_VIDEO_END_CAP ) / float( CHUNK_TIL_VIDEO_END_CAP )

            action_prob = actor.predict( np.reshape( state ,(1 ,S_INFO ,args.S_LEN) ) )
            action_cumsum = np.cumsum( action_prob )
            selection = (action_cumsum > np.random.randint(
                1 ,RAND_RANGE ) / float( RAND_RANGE )).argmax()
            bit_rate = calculate_from_selection( selection ,last_bit_rate )
            # Note: we need to discretize the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            s_batch.append( state )

            entropy_record.append( a3c.compute_entropy( action_prob[0] ) )

            if end_of_video:
                log_file.write( '\n' )
                log_file.close()

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = np.zeros( args.A_DIM )
                action_vec[selection] = 1

                s_batch.append( np.zeros( (S_INFO ,args.S_LEN) ) )
                a_batch.append( action_vec )
                entropy_record = []

                video_count += 1

                if video_count >= len( all_file_names ):
                    break

                log_path = os.path.join( summary_dir ,
                                         'log_sim_udr_1_{}_{}'.format( args.log_str ,
                                                                       all_file_names[net_env.trace_idx] ) )
                log_file = open( log_path ,'w' )

            test_dir = summary_dir
            plot_files = os.listdir( test_dir )

        # reward_0 = given_string_mean_reward( plot_files ,test_dir ,str='0-5' )
        # reward_1 = given_string_mean_reward( plot_files ,test_dir ,str='5-100' )
        # reward_2 = given_string_mean_reward( plot_files ,test_dir ,str='100-250' )
        # reward_3 = given_string_mean_reward( plot_files ,test_dir ,str='250-450' )
        # reward_4 = given_string_mean_reward( plot_files ,test_dir ,str='450-1050' )
        reward_5 = given_string_mean_reward( plot_files ,test_dir ,str='' )

        # rl_mean_reward = {'0-5': reward_0 ,
        #                   '5-100': reward_1 ,
        #                   '100-250': reward_2 ,
        #                   '250-450': reward_3 ,
        #                   '450-1050': reward_4 ,
        #                   'FCC': reward_5}
        rl_mean_reward = {'FCC': reward_5}

        print( rl_mean_reward )


if __name__ == '__main__':
    main()
